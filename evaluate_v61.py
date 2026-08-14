from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from cformer_v60 import ChineseAliasWorld, ChineseTransformerConfig, TokenCFormerResolver
from cformer_v60b import BlindSet
from cformer_v61 import IVFConfig, IVFIndex, QuantizedVectorStore, exact_search, rerank

ROOT = Path(__file__).resolve().parent

WORLD_SEEDS = (60, 161, 262, 363)
NPROBES = (16, 32, 64, 128)
TOP_ANN = 256
QUERIES_PER_WORLD = 96


@torch.inference_mode()
def encode_bank(model, world, device) -> torch.Tensor:
    bank = torch.empty((world.scale, model.config.output_dimensions), dtype=torch.float16)
    for start in range(0, world.scale, 8192):
        stop = min(start + 8192, world.scale)
        tokens = world.encode_candidates(world.objects(range(start, stop))).to(device)
        bank[start:stop] = model.encode_candidate(tokens).cpu().half()
    return bank


def measure(func, repeat: int = 1):
    started = time.perf_counter()
    for _ in range(repeat):
        result = func()
    return result, (time.perf_counter() - started) / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "v60_strict_checkpoints" / "cformer_L2_seed601.pt")
    parser.add_argument("--scales", nargs="+", type=int, default=(16384, 32768, 65536))
    parser.add_argument("--nprobes", nargs="+", type=int, default=NPROBES)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v61_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    tokenizer_size = ChineseAliasWorld(2048).tokenizer.size
    model = TokenCFormerResolver(ChineseTransformerConfig(tokenizer_size, layers=2)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    rows: list[dict] = []
    for scale in args.scales:
        centroids = max(256, scale // 64)
        for world_seed in WORLD_SEEDS:
            world = ChineseAliasWorld(scale, seed=world_seed)
            bank = encode_bank(model, world, device)
            bank32 = bank.float().to(device)

            targets = world.heldout_objects(QUERIES_PER_WORLD)
            tokens, _ = world.encode_queries(targets, [index % 6 for index in range(QUERIES_PER_WORLD)])
            queries = torch.nn.functional.normalize(model.encode_query(tokens.to(device)), dim=-1)

            # -- exact baseline ------------------------------------------------
            exact_started = time.perf_counter()
            exact_scores = queries @ bank32.T
            exact_top1 = exact_scores.topk(1).indices[:, 0]
            exact_seconds = time.perf_counter() - exact_started
            target_labels = torch.tensor([obj.label for obj in targets], device=device)
            exact_top1_rate = float((exact_top1 == target_labels).float().mean())

            # -- build IVF -------------------------------------------------------
            sample = bank32 if scale <= 32768 else bank32[:32768]
            index = IVFIndex(bank32.shape[1], IVFConfig(n_centroids=centroids, train_sample=min(32768, scale)))
            index.train(sample)
            index.add(bank32, list(range(scale)))

            for nprobe in args.nprobes:
                recall_hits = 0
                ann_top1_hits = 0
                latencies = []
                for position, query in enumerate(queries):
                    q = query.unsqueeze(0)
                    started = time.perf_counter()
                    _, candidate_ids = index.search(q, nprobe=nprobe, topk=TOP_ANN)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - started) * 1000)
                    recall_hits += int((candidate_ids == target_labels[position]).any())
                    _, rerank_ids = rerank(q, candidate_ids, bank32, topk=1)
                    ann_top1_hits += int(rerank_ids[0] == target_labels[position])
                latencies.sort()
                row = {
                    "scale": scale,
                    "world_seed": world_seed,
                    "centroids": centroids,
                    "nprobe": nprobe,
                    "recall_at_256": recall_hits / QUERIES_PER_WORLD,
                    "ann_top1": ann_top1_hits / QUERIES_PER_WORLD,
                    "exact_top1": exact_top1_rate,
                    "top1_drop": exact_top1_rate - ann_top1_hits / QUERIES_PER_WORLD,
                    "latency_ms_p50": latencies[len(latencies) // 2],
                    "latency_ms_p95": latencies[int(len(latencies) * 0.95)],
                    "exact_scan_seconds": exact_seconds,
                    "index_bytes_fp16": scale * 2 * bank32.shape[1],
                    "index_bytes_int8": scale * (bank32.shape[1] + 2),
                }
                rows.append(row)
                print(json.dumps({"phase": "row", **row}, ensure_ascii=False), flush=True)

    # -- blind set re-check at 64K (nprobe 64) ----------------------------------
    blind_rows: list[dict] = []
    if 65536 in args.scales:
        world = ChineseAliasWorld(65536, seed=60)
        bank = encode_bank(model, world, device)
        bank32 = bank.float().to(device)
        label_of = {world.object_at(label).values: label for label in range(65536)}
        blind = BlindSet()
        known = blind.by_expected("known")
        tokens = torch.stack(
            [world.tokenizer.encode(query.text, world.query_length)[0] for query in known]
        )
        queries = torch.nn.functional.normalize(model.encode_query(tokens.to(device)), dim=-1)
        targets = torch.tensor([label_of[query.target] for query in known], device=device)
        exact_top1 = (queries @ bank32.T).topk(1).indices[:, 0]
        exact_rate = float((exact_top1 == targets).float().mean())
        index = IVFIndex(bank32.shape[1], IVFConfig(n_centroids=1024, train_sample=32768))
        index.train(bank32[:32768])
        index.add(bank32, list(range(65536)))
        hits = 0
        recall_hits = 0
        for position, query in enumerate(queries):
            _, candidate_ids = index.search(query.unsqueeze(0), nprobe=64, topk=TOP_ANN)
            recall_hits += int((candidate_ids == targets[position]).any())
            _, rerank_ids = rerank(query.unsqueeze(0), candidate_ids, bank32, topk=1)
            hits += int(rerank_ids[0] == targets[position])
        blind_rows = {
            "exact_top1": exact_rate,
            "ann_top1": hits / len(known),
            "recall_at_256": recall_hits / len(known),
            "queries": len(known),
        }
        print(json.dumps({"phase": "blind", **blind_rows}, ensure_ascii=False), flush=True)

    def group(scale: int, nprobe: int) -> list[dict]:
        return [row for row in rows if row["scale"] == scale and row["nprobe"] == nprobe]

    aggregate = {}
    for scale in args.scales:
        for nprobe in args.nprobes:
            group_rows = group(scale, nprobe)
            key = f"{scale}_nprobe{nprobe}"
            aggregate[key] = {
                metric: {
                    "mean": statistics.mean(row[metric] for row in group_rows),
                    "min": min(row[metric] for row in group_rows),
                    "max": max(row[metric] for row in group_rows),
                }
                for metric in ("recall_at_256", "ann_top1", "exact_top1", "top1_drop", "latency_ms_p50", "latency_ms_p95")
            }

    payload = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "settings": {key: str(value) for key, value in vars(args).items()},
        "aggregate": aggregate,
        "blind_set_64k": blind_rows,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
