"""Focused V6.1 latency benchmark at 64K (optimized vectorized search path).

The full 3-scale x 4-world correctness matrix is already complete in
artifacts/v61_results.json (recall/top1 do not change with search-path
optimization). This script only re-measures per-query latency at the
headline scale with the vectorized search.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from cformer_v60 import ChineseAliasWorld, ChineseTransformerConfig, TokenCFormerResolver
from cformer_v61 import IVFConfig, IVFIndex, rerank

ROOT = Path(__file__).resolve().parent
SCALE = 65536
QUERIES = 96
NPROBES = (16, 32, 64)


@torch.inference_mode()
def encode_bank(model, world, device) -> torch.Tensor:
    bank = torch.empty((world.scale, model.config.output_dimensions), dtype=torch.float16)
    for start in range(0, world.scale, 8192):
        stop = min(start + 8192, world.scale)
        tokens = world.encode_candidates(world.objects(range(start, stop))).to(device)
        bank[start:stop] = model.encode_candidate(tokens).cpu().half()
    return bank


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    tokenizer_size = ChineseAliasWorld(2048).tokenizer.size
    model = TokenCFormerResolver(ChineseTransformerConfig(tokenizer_size, layers=2)).to(device)
    model.load_state_dict(
        torch.load(ROOT / "artifacts" / "v60_strict_checkpoints" / "cformer_L2_seed601.pt",
                   map_location="cpu", weights_only=True)
    )
    model.eval()

    world = ChineseAliasWorld(SCALE, seed=60)
    bank = encode_bank(model, world, device)
    bank32 = bank.float().to(device)
    targets = world.heldout_objects(QUERIES)
    tokens, _ = world.encode_queries(targets, [index % 6 for index in range(QUERIES)])
    queries = torch.nn.functional.normalize(model.encode_query(tokens.to(device)), dim=-1)
    target_labels = torch.tensor([obj.label for obj in targets], device=device)

    # -- exact batched baseline --
    exact_started = time.perf_counter()
    exact_top1 = (queries @ bank32.T).topk(1).indices[:, 0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    exact_seconds = time.perf_counter() - exact_started
    exact_rate = float((exact_top1 == target_labels).float().mean())

    # -- IVF build (timed) --
    index = IVFIndex(bank32.shape[1], IVFConfig(n_centroids=1024, train_sample=32768))
    train_started = time.perf_counter()
    index.train(bank32[:32768])
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    add_started = time.perf_counter()
    index.add(bank32, list(range(SCALE)))
    if device.type == "cuda":
        torch.cuda.synchronize()
    add_seconds = time.perf_counter() - add_started

    rows = []
    for nprobe in NPROBES:
        latencies = []
        recall = 0
        ann_top1 = 0
        for position, query in enumerate(queries):
            started = time.perf_counter()
            _, candidate_ids = index.search(query.unsqueeze(0), nprobe=nprobe, topk=256)
            _, rerank_ids = rerank(query.unsqueeze(0), candidate_ids, bank32, topk=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
            recall += int((candidate_ids == target_labels[position]).any())
            ann_top1 += int(rerank_ids[0] == target_labels[position])
        latencies.sort()
        rows.append({
            "nprobe": nprobe,
            "recall_at_256": recall / QUERIES,
            "ann_top1": ann_top1 / QUERIES,
            "exact_top1": exact_rate,
            "p50_ms": latencies[QUERIES // 2],
            "p95_ms": latencies[int(QUERIES * 0.95)],
            "p99_ms": latencies[min(QUERIES - 1, int(QUERIES * 0.99))],
        })
        print(json.dumps({"phase": "latency", **rows[-1]}, ensure_ascii=False), flush=True)

    payload = {
        "scale": SCALE,
        "queries": QUERIES,
        "exact_batched_scan_ms_per_query": exact_seconds * 1000 / QUERIES,
        "ivf_train_seconds": train_seconds,
        "ivf_add_seconds": add_seconds,
        "index_bytes_fp16": SCALE * 2 * bank32.shape[1],
        "latency": rows,
    }
    (ROOT / "artifacts" / "v61_latency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"phase": "done", "payload": payload}, ensure_ascii=False))


if __name__ == "__main__":
    main()
