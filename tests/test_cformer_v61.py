import torch
from torch.nn import functional as F

from cformer_v61 import (
    IVFConfig,
    IVFIndex,
    QuantizedVectorStore,
    exact_search,
    rerank,
)


def _random_bank(count: int, dimension: int = 64, seed: int = 61) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(count, dimension), dim=-1)


def test_ivf_with_full_probe_matches_bruteforce() -> None:
    bank = _random_bank(2048)
    index = IVFIndex(bank.shape[1], IVFConfig(n_centroids=64, n_iter=8))
    index.train(bank)
    index.add(bank, list(range(bank.shape[0])))
    query = bank[7]
    ann_scores, ann_ids = index.search(query, nprobe=64, topk=10)
    exact_scores, exact_ids = exact_search(query, bank, topk=10)
    assert torch.equal(ann_ids, exact_ids)
    assert torch.allclose(ann_scores, exact_scores, atol=1e-3)


def test_ivf_recall_improves_with_nprobe() -> None:
    bank = _random_bank(8192)
    index = IVFIndex(bank.shape[1], IVFConfig(n_centroids=256, n_iter=10))
    index.train(bank)
    index.add(bank, list(range(bank.shape[0])))
    queries = bank[:64]
    for nprobe in (8, 32, 128):
        hits = 0
        for position, query in enumerate(queries):
            _, ids = index.search(query, nprobe=nprobe, topk=256)
            hits += bool((ids == position).any())
        recall = hits / len(queries)
        if nprobe == 128:
            assert recall >= 0.99, f"recall@256 with nprobe=128 too low: {recall:.4f}"


def test_tombstone_removes_vector_from_search() -> None:
    bank = _random_bank(1024)
    index = IVFIndex(bank.shape[1], IVFConfig(n_centroids=32, n_iter=6))
    index.train(bank)
    index.add(bank, list(range(bank.shape[0])))
    index.remove([5, 9])
    _, ids = index.search(bank[5], nprobe=32, topk=100)
    assert 5 not in ids.tolist()
    assert 9 not in ids.tolist()
    assert index.count == 1024  # tombstone keeps storage, excludes search


def test_snapshot_restore_and_rollback() -> None:
    import io

    bank = _random_bank(1024)
    index = IVFIndex(bank.shape[1], IVFConfig(n_centroids=32, n_iter=6))
    index.train(bank)
    index.add(bank, list(range(bank.shape[0])))
    buffer = io.BytesIO()
    index.snapshot(buffer)
    version_at_snapshot = index.version
    buffer.seek(0)
    extra = _random_bank(128, bank.shape[1], seed=2)
    index.add(extra, list(range(1000, 1128)))
    assert index.count == 1152
    restored = IVFIndex.restore(buffer)
    assert restored.count == 1024
    assert restored.version == version_at_snapshot
    assert restored.ids == list(range(1024))


def test_quantized_stores_keep_cosine_fidelity() -> None:
    bank = _random_bank(2048)
    query = bank[3]
    for dtype, tolerance in (("fp16", 1e-3), ("int8", 0.02)):
        store = QuantizedVectorStore(bank.shape[1], dtype=dtype)
        store.add(bank)
        assert store.count == 2048  # single copy, no duplication
        candidate = store.vector(3)
        cosine = F.cosine_similarity(query.unsqueeze(0), candidate.unsqueeze(0)).item()
        assert cosine >= 1.0 - tolerance, (dtype, cosine)


def test_fixed_scale_int8_meets_32mib_gate_with_fidelity() -> None:
    bank = _random_bank(2048)
    store = QuantizedVectorStore(bank.shape[1], dtype="int8", fixed_scale=True)
    store.add(bank)
    assert store.count == 2048
    assert store.bytes_per_vector == bank.shape[1]  # 64 B, no scale column
    # 512K x 64 B = 32 MiB exactly (the roadmap INT8 gate).
    assert 524288 * store.bytes_per_vector == 33_554_432
    query = bank[3]
    cosine = F.cosine_similarity(query.unsqueeze(0), store.vector(3).unsqueeze(0)).item()
    assert cosine >= 0.99, cosine


def test_fixed_scale_rejects_fp16() -> None:
    import pytest

    with pytest.raises(ValueError):
        QuantizedVectorStore(64, dtype="fp16", fixed_scale=True)


def test_rerank_of_ann_candidates_matches_exact_top1() -> None:
    bank = _random_bank(4096)
    index = IVFIndex(bank.shape[1], IVFConfig(n_centroids=128, n_iter=8))
    index.train(bank)
    index.add(bank, list(range(bank.shape[0])))
    store = QuantizedVectorStore(bank.shape[1], dtype="fp16")
    store.add(bank)
    exact_scores, exact_ids = exact_search(bank[0], bank, topk=1)
    _, ann_ids = index.search(bank[0], nprobe=128, topk=256)
    _, rerank_ids = rerank(bank[0], ann_ids, store, topk=1)
    assert int(rerank_ids[0]) == int(exact_ids[0])
    assert exact_scores[0].item() - 1.0 < 1e-3  # self query scores ~1.0
