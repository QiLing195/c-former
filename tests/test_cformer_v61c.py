import torch

from cformer_v59 import CandidateLedger, CandidateStatus, EvidenceVerifier
from cformer_v61 import IVFConfig, IVFIndex
from cformer_v61c import (
    ObjectRecord,
    UnifiedObjectStore,
    UnifiedResolutionPipeline,
    normalize_surface,
)


class FakeEncoder:
    """Deterministic text->vector stub so tests control score geometry."""

    def __init__(self, object_ids: list[str], vectors: torch.Tensor):
        self.object_ids = object_ids
        self.vectors = vectors
        self.queries: dict[str, tuple[torch.Tensor, float]] = {}
        self.calls = 0

    def bind(self, text: str, vector: torch.Tensor, coverage: float = 1.0) -> None:
        self.queries[text] = (vector, coverage)

    def encode_query(self, text: str) -> tuple[torch.Tensor, float]:
        self.calls += 1
        return self.queries[text]

    def object_id_of(self, label: int) -> str:
        return self.object_ids[label]


def _build_world():
    vectors = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    ids = ["obj-a", "obj-b", "obj-c"]
    store = UnifiedObjectStore()
    for object_id, name in zip(ids, ["Alpha One", "Beta Two", "Gamma Three"]):
        store.upsert_object(
            ObjectRecord(object_id, name, {"名称": f"{name} 文档"}, {}),
            aliases=[name.replace(" ", "")],
        )
    index = IVFIndex(vectors.shape[1], IVFConfig(n_centroids=2, n_iter=2))
    index.train(vectors)
    index.add(vectors, list(range(len(ids))))
    encoder = FakeEncoder(ids, vectors)
    verifier = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08)
    ledger = CandidateLedger()
    pipeline = UnifiedResolutionPipeline(
        store, ledger, index, encoder, verifier, top_ann=16, top_rerank=16,
    )
    return store, ledger, index, encoder, pipeline


def test_exact_hit_bypasses_encoder_and_is_repeatable() -> None:
    store, _, _, encoder, pipeline = _build_world()
    first = pipeline.resolve("alpha one")
    assert first.status == CandidateStatus.SUPPORTED
    assert first.path == "exact"
    assert first.object_id == "obj-a"
    calls_after_first = encoder.calls
    second = pipeline.resolve("ALPHA ONE")
    assert second.path == "exact"
    assert encoder.calls == calls_after_first  # 精确命中不进神经计算


def test_ann_supported_ambiguous_unknown_statuses() -> None:
    _, _, _, encoder, pipeline = _build_world()
    a_vec = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mid_vec = torch.tensor([0.0, 0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0, 0.0])
    far_vec = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    encoder.bind("靠近A的查询", a_vec)
    encoder.bind("骑在AB中间的查询", mid_vec)
    encoder.bind("完全无关的查询xyz", far_vec, coverage=0.9)

    supported = pipeline.resolve("靠近A的查询", query_type="known")
    assert supported.status == CandidateStatus.SUPPORTED
    assert supported.path == "ann"
    assert supported.object_id == "obj-a"

    ambiguous = pipeline.resolve("骑在AB中间的查询")
    assert ambiguous.status == CandidateStatus.AMBIGUOUS

    unknown = pipeline.resolve("完全无关的查询xyz")
    assert unknown.status == CandidateStatus.UNKNOWN


def test_tombstone_hides_exact_and_index() -> None:
    store, _, index, encoder, pipeline = _build_world()
    store.remove_object("obj-a")
    index.remove([0])
    assert store.exact_lookup("Alpha One") is None
    encoder.bind("Alpha One", torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    result = pipeline.resolve("Alpha One")
    # 精确命中消失；索引墓碑生效后该向量不再返回 obj-a
    if result.path == "ann":
        assert result.object_id != "obj-a"
    assert store.version >= 4  # 墓碑写入推高版本号


def test_alias_lifecycle_requires_explicit_review_write() -> None:
    store, ledger, _, encoder, pipeline = _build_world()
    encoder.bind("全新别名 gpt-nine", torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), coverage=0.95)
    result = pipeline.resolve("全新别名 gpt-nine")
    assert result.status == CandidateStatus.UNKNOWN
    assert result.proposed_alias
    # 审核前：正式别名不可见
    assert store.exact_lookup("gpt-nine") is None
    proposals = ledger.connection.execute(
        "SELECT COUNT(*) FROM candidates WHERE status='proposed'"
    ).fetchone()[0]
    assert proposals >= 1
    # 重复查询不重复提案
    pipeline.resolve("全新别名 gpt-nine")
    again = ledger.connection.execute(
        "SELECT COUNT(*) FROM candidates WHERE status='proposed'"
    ).fetchone()[0]
    assert again == proposals
    # 外部审核通过后写入，精确命中可见
    store.add_verified_alias("gpt-nine", "obj-a")
    hit = pipeline.resolve("gpt-nine")
    assert hit.path == "exact"
    assert hit.object_id == "obj-a"


def test_long_questions_never_create_proposals() -> None:
    _, ledger, _, encoder, pipeline = _build_world()
    encoder.bind("请问一下这个模型是哪一年发布的呢", torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), coverage=0.6)
    result = pipeline.resolve("请问一下这个模型是哪一年发布的呢")
    assert result.status == CandidateStatus.UNKNOWN
    assert not result.proposed_alias
    count = ledger.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    assert count == 0


def test_structural_series_ambiguity_overrides_margin() -> None:
    _, _, _, encoder, pipeline = _build_world()
    # 三个对象都报告为同系列多成员；查询无选择标准措辞
    pipeline.series_size_of = lambda object_id: 3
    near_a = torch.tensor([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    encoder.bind("Kimi是哪一个模型", near_a)
    result = pipeline.resolve("Kimi是哪一个模型")
    assert result.status == CandidateStatus.AMBIGUOUS
    assert "structural" in result.reason

    # 有选择标准措辞 → 恢复神经 margin 判定（此处 margin 足够大 → supported）
    encoder.bind("Kimi最新的模型", near_a)
    latest = pipeline.resolve("Kimi最新的模型")
    assert latest.status == CandidateStatus.SUPPORTED


def test_fts_candidates_find_documents() -> None:
    store, _, _, _, _ = _build_world()
    hits = store.fts_candidates("Alpha")
    assert "obj-a" in hits


def test_ivf_search_matches_exhaustive_at_small_scale() -> None:
    _, _, index, encoder, _ = _build_world()
    exhaustive_scores = encoder.vectors @ encoder.vectors.T
    for label in range(len(encoder.object_ids)):
        scores, ids = index.search(encoder.vectors[label], nprobe=2, topk=3)
        assert int(ids[0]) == int(exhaustive_scores[label].argmax())
