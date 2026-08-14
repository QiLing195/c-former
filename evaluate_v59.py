from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from cformer_v59 import (
    CandidateStatus,
    DualEncoderConfig,
    EvidenceVerifier,
    FlatTransformerDualEncoder,
    OpenAliasWorld,
    SemanticDualEncoder,
)


ROOT = Path(__file__).resolve().parent


def train_model(model, world: OpenAliasWorld, device: torch.device, *, steps: int, batch_size: int):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        objects = world.training_objects(batch_size, seed_offset=step)
        queries, _ = world.encode_queries(objects)
        positives = world.encode_candidates(objects)
        negatives = world.encode_candidates([world.hard_negative(obj) for obj in objects])
        optimizer.zero_grad(set_to_none=True)
        loss = model.contrastive_loss(
            queries.to(device, non_blocking=True),
            positives.to(device, non_blocking=True),
            negatives.to(device, non_blocking=True),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "initial_loss": statistics.mean(losses[: min(10, len(losses))]),
        "final_loss": statistics.mean(losses[-min(10, len(losses)) :]),
    }


@torch.inference_mode()
def encode_bank(model, world: OpenAliasWorld, device: torch.device, *, chunk_size: int):
    model.eval()
    bank = torch.empty((world.scale, 64), dtype=torch.float16)
    started = time.perf_counter()
    for start in range(0, world.scale, chunk_size):
        stop = min(start + chunk_size, world.scale)
        objects = world.objects(range(start, stop))
        tokens = world.encode_candidates(objects).to(device, non_blocking=True)
        bank[start:stop] = model.encode_candidate(tokens).cpu().to(torch.float16)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return bank, time.perf_counter() - started


@torch.inference_mode()
def retrieve(model, query_tokens, bank, device, *, topk=10, chunk_size=65536):
    query = model.encode_query(query_tokens.to(device)).float()
    best_scores = torch.full((query.shape[0], topk), -2.0, device=device)
    best_ids = torch.full((query.shape[0], topk), -1, dtype=torch.long, device=device)
    started = time.perf_counter()
    for start in range(0, bank.shape[0], chunk_size):
        candidates = bank[start : start + chunk_size].to(device).float()
        scores = query @ candidates.T
        local_scores, local_ids = scores.topk(min(topk, candidates.shape[0]), dim=-1)
        local_ids += start
        merged_scores = torch.cat((best_scores, local_scores), dim=-1)
        merged_ids = torch.cat((best_ids, local_ids), dim=-1)
        best_scores, order = merged_scores.topk(topk, dim=-1)
        best_ids = torch.gather(merged_ids, 1, order)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return best_scores.cpu(), best_ids.cpu(), time.perf_counter() - started


def evaluate_world(model, scale: int, seed: int, device: torch.device, *, queries: int, chunk_size: int):
    world = OpenAliasWorld(scale, seed=seed)
    bank, encoding_seconds = encode_bank(model, world, device, chunk_size=chunk_size)
    targets = world.heldout_objects(queries)
    query_tokens, coverage = world.encode_queries(targets)
    scores, ids, search_seconds = retrieve(model, query_tokens, bank, device)
    labels = torch.tensor([obj.label for obj in targets])
    top1 = ids[:, 0].eq(labels)
    recall10 = ids.eq(labels[:, None]).any(dim=-1)
    verifier = EvidenceVerifier()
    decisions = [
        verifier.decide(float(row[0]), float(row[1]), float(cov))
        for row, cov in zip(scores, coverage)
    ]

    ambiguous_targets = world.ambiguous_objects(queries)
    ambiguous_tokens, ambiguous_coverage = world.encode_queries(ambiguous_targets, omit_mode=True)
    ambiguous_scores, _, ambiguous_seconds = retrieve(model, ambiguous_tokens, bank, device)
    ambiguous_decisions = [
        verifier.decide(float(row[0]), float(row[1]), float(cov))
        for row, cov in zip(ambiguous_scores, ambiguous_coverage)
    ]
    unknown_tokens = torch.stack(
        [world.tokenizer.encode("unseen qxz nebula concept without registered evidence", world.query_length)[0]
         for _ in targets]
    )
    unknown_coverage = torch.tensor(
        [world.tokenizer.encode("unseen qxz nebula concept without registered evidence", world.query_length)[1]
         for _ in targets]
    )
    unknown_scores, _, unknown_seconds = retrieve(model, unknown_tokens, bank, device)
    unknown_decisions = [
        verifier.decide(float(row[0]), float(row[1]), float(cov))
        for row, cov in zip(unknown_scores, unknown_coverage)
    ]
    return {
        "scale": scale,
        "seed": seed,
        "queries": queries,
        "top1": float(top1.float().mean()),
        "recall_at_10": float(recall10.float().mean()),
        "false_merge_rate": float((~top1).float().mean()),
        "known_supported_rate": sum(d.status == CandidateStatus.SUPPORTED for d in decisions) / queries,
        "ambiguous_detection_rate": sum(d.status == CandidateStatus.AMBIGUOUS for d in ambiguous_decisions) / max(1, len(ambiguous_decisions)),
        "ambiguous_cases": len(ambiguous_decisions),
        "unknown_rejection_rate": sum(d.status == CandidateStatus.UNKNOWN for d in unknown_decisions) / queries,
        "bank_mib": bank.numel() * bank.element_size() / 2**20,
        "encode_seconds": encoding_seconds,
        "known_search_ms_per_query": search_seconds * 1000 / queries,
        "boundary_search_ms_per_query": (ambiguous_seconds + unknown_seconds) * 1000 / max(1, len(ambiguous_decisions) + queries),
        "auto_verified_writes": 0,
    }


def aggregate(rows):
    metrics = ("top1", "recall_at_10", "false_merge_rate", "known_supported_rate", "ambiguous_detection_rate", "unknown_rejection_rate", "known_search_ms_per_query")
    result = {}
    for scale in sorted({row["scale"] for row in rows}):
        group = [row for row in rows if row["scale"] == scale]
        result[str(scale)] = {
            metric: {
                "mean": statistics.mean(row[metric] for row in group),
                "min": min(row[metric] for row in group),
                "max": max(row[metric] for row in group),
            }
            for metric in metrics
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("correctness", "scale", "all"), default="all")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "v59_dual_encoder.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v59_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(5901)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(5901)
    training_world = OpenAliasWorld(8192, seed=59)
    config = DualEncoderConfig(training_world.tokenizer.size)
    model = SemanticDualEncoder(config)
    baseline = FlatTransformerDualEncoder(training_world.tokenizer.size)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        training = {"loaded_checkpoint": True}
    else:
        training = train_model(model, training_world, device, steps=args.steps, batch_size=args.batch_size)
        torch.save(model.state_dict(), args.checkpoint)
    model.to(device)
    baseline_training = train_model(
        baseline, training_world, device, steps=args.steps, batch_size=args.batch_size
    )

    rows = []
    baseline_rows = []
    correctness_scales = (2048, 8192)
    scale_scales = (65536, 131072, 262144, 512000)
    selected = correctness_scales if args.stage == "correctness" else scale_scales if args.stage == "scale" else correctness_scales + scale_scales
    for scale in selected:
        for world_index in range(args.worlds):
            seed = 59 + world_index * 101
            row = evaluate_world(model, scale, seed, device, queries=args.queries, chunk_size=args.chunk_size)
            rows.append(row)
            print(json.dumps({"model": "cformer_v59", **row}, ensure_ascii=False), flush=True)
            if scale in correctness_scales:
                baseline_row = evaluate_world(
                    baseline, scale, seed, device, queries=args.queries, chunk_size=args.chunk_size
                )
                baseline_rows.append(baseline_row)
                print(json.dumps({"model": "transformer", **baseline_row}, ensure_ascii=False), flush=True)
        if scale == 8192:
            gate = [row for row in rows if row["scale"] == 8192]
            if statistics.mean(row["recall_at_10"] for row in gate) < 0.95:
                raise RuntimeError("correctness gate failed at 8K; scale expansion stopped")

    payload = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
        },
        "parameters": {"cformer_v59": model.parameter_count(), "transformer": baseline.parameter_count()},
        "training": {"cformer_v59": training, "transformer": baseline_training},
        "worlds": rows,
        "transformer_worlds": baseline_rows,
        "aggregate": aggregate(rows),
        "transformer_aggregate": aggregate(baseline_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
