"""V6.1b: IVF scaling on real structured vectors beyond 64K.

The V6.0 ChineseAliasWorld caps at 65,536 objects, so V6.1 could only measure
IVF there (Recall@256 = 100%, "too easy"). This script retrains the V5.9
SemanticDualEncoder (open-alias world, up to ~929K objects) and re-measures
IVF Recall@256 / rerank Top-1 / latency / memory at 128K / 256K / 512K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from cformer_v59 import (
    DualEncoderConfig,
    OpenAliasWorld,
    SemanticDualEncoder,
)
from cformer_v61 import IVFConfig, IVFIndex, rerank

ROOT = Path(__file__).resolve().parent
SCALES = (131072, 262144, 524288)
NPROBES = (16, 32, 64, 128)
QUERIES = 128


def train_encoder(device, steps: int = 300, batch_size: int = 128, seed: int = 59):
    world = OpenAliasWorld(65536, seed=59)
    model = SemanticDualEncoder(DualEncoderConfig(world.tokenizer.size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for step in range(steps):
        objects = world.training_objects(batch_size, seed_offset=seed * 1000 + step)
        queries, _ = world.encode_queries(objects)
        positives = world.encode_candidates(objects)
        negatives = world.encode_candidates([world.hard_negative(obj) for obj in objects])
        optimizer.zero_grad(set_to_none=True)
        loss = model.contrastive_loss(
            queries.to(device), positives.to(device), negatives.to(device)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model, world


@torch.inference_mode()
def encode_bank(model, world, device, chunk: int = 8192) -> torch.Tensor:
    bank = torch.empty((world.scale, model.config.embedding_dimensions), dtype=torch.float16)
    for start in range(0, world.scale, chunk):
        stop = min(start + chunk, world.scale)
        tokens = world.encode_candidates(world.objects(range(start, stop))).to(device)
        bank[start:stop] = model.encode_candidate(tokens).cpu().half()
    return bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, default=SCALES)
    parser.add_argument("--nprobes", nargs="+", type=int, default=NPROBES)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v61_scale_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model, train_world = train_encoder(device, steps=args.steps)
    rows = []

    for scale in args.scales:
        world = OpenAliasWorld(scale, seed=59)
        bank = encode_bank(model, world, device).float().to(device)
        centroids = max(1024, scale // 64)
        index = IVFIndex(bank.shape[1], IVFConfig(n_centroids=centroids))
        sample = bank[:32768]
        train_started = time.perf_counter()
        index.train(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_started
        index.add(bank, list(range(scale)))

        targets = world.heldout_objects(QUERIES)
        tokens, _ = world.encode_queries(targets)
        queries = torch.nn.functional.normalize(model.encode_query(tokens.to(device)), dim=-1)
        target_labels = torch.tensor([obj.label for obj in targets], device=device)

        # exact top1 baseline (batched)
        exact_top1 = (queries @ bank.T).topk(1).indices[:, 0]
        exact_rate = float((exact_top1 == target_labels).float().mean())

        for nprobe in args.nprobes:
            recall_hits = 0
            ann_top1_hits = 0
            latencies = []
            for position, query in enumerate(queries):
                started = time.perf_counter()
                _, candidate_ids = index.search(query.unsqueeze(0), nprobe=nprobe, topk=256)
                _, rerank_ids = rerank(query.unsqueeze(0), candidate_ids, bank, topk=1)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - started) * 1000)
                recall_hits += int((candidate_ids == target_labels[position]).any())
                ann_top1_hits += int(rerank_ids[0] == target_labels[position])
            latencies.sort()
            row = {
                "scale": scale,
                "centroids": centroids,
                "nprobe": nprobe,
                "recall_at_256": recall_hits / QUERIES,
                "ann_top1": ann_top1_hits / QUERIES,
                "exact_top1": exact_rate,
                "top1_drop": exact_rate - ann_top1_hits / QUERIES,
                "p50_ms": latencies[QUERIES // 2],
                "p95_ms": latencies[int(QUERIES * 0.95)],
                "index_bytes_fp16": scale * 2 * bank.shape[1],
                "index_bytes_int8": scale * (bank.shape[1] + 2),
                "ivf_train_seconds": train_seconds,
            }
            rows.append(row)
            print(json.dumps({"phase": "row", **row}, ensure_ascii=False), flush=True)

    payload = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "settings": {key: str(value) for key, value in vars(args).items()},
        "encoder": "SemanticDualEncoder (V5.9), retrained 300 steps",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done"}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
