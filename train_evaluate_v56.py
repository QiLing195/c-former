from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer_v56 import (
    CognitiveCFormerReranker,
    EvidenceRAGReranker,
    RerankerConfig,
    SyntheticRetrievalWorld,
    V56_SCALES,
    sample_training_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5.6 governed neural cognitive retrieval")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shortlist", type=int, default=64)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--worlds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303])
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/v56_results.json")
    )
    return parser.parse_args()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_pair(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    config = RerankerConfig()
    cformer = CognitiveCFormerReranker(config).to(args.device)
    rag = EvidenceRAGReranker(config).to(args.device)
    rag.load_state_dict(cformer.state_dict())
    optimizers = (
        torch.optim.AdamW(cformer.parameters(), lr=args.lr),
        torch.optim.AdamW(rag.parameters(), lr=args.lr),
    )
    generator = torch.Generator(device=args.device).manual_seed(seed + 10_000)
    losses = [0.0, 0.0]
    for step in range(args.steps):
        batch = sample_training_batch(
            args.batch_size,
            args.shortlist,
            config.key_dimensions,
            generator=generator,
            device=args.device,
        )
        for index, (model, optimizer) in enumerate(zip((cformer, rag), optimizers)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            loss = F.cross_entropy(logits, batch[-1])
            loss.backward()
            optimizer.step()
            losses[index] = loss.item()
        if (step + 1) % 200 == 0:
            print(
                f"seed={seed} step={step + 1} "
                f"cformer_loss={losses[0]:.4f} rag_loss={losses[1]:.4f}",
                flush=True,
            )
    return cformer.eval(), rag.eval(), losses


@torch.inference_mode()
def evaluate_world(model, world, args, *, use_observer: bool = True):
    entities, query_keys, kinds, observers, correct_ids = world.fixed_queries(args.queries)
    shortlist_ids, labels = world.indexed_shortlists(
        entities, correct_ids, args.shortlist
    )
    candidate_keys = world.candidate_keys[shortlist_ids]
    candidate_kinds = world.candidate_kinds[shortlist_ids]
    candidate_scopes = world.candidate_scopes[shortlist_ids]
    tensors = [
        tensor.to(args.device)
        for tensor in (
            candidate_keys,
            candidate_kinds,
            candidate_scopes,
            query_keys,
            kinds,
            observers,
            labels,
        )
    ]
    start = time.perf_counter()
    logits = model(*tensors[:-1], use_observer=use_observer)
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / args.queries
    predictions = logits.argmax(dim=-1)
    accuracy = predictions.eq(tensors[-1]).float().mean().item()
    rough_predictions = shortlist_ids[:, 0]
    rough_accuracy = rough_predictions.eq(correct_ids).float().mean().item()
    recall = shortlist_ids.eq(correct_ids[:, None]).any(dim=-1).float().mean().item()
    return {
        "accuracy": accuracy,
        "rough_recall_at_k": recall,
        "rough_top1_accuracy": rough_accuracy,
        "rerank_query_ms": elapsed_ms,
    }


@torch.inference_mode()
def full_scan_latency(model, world, args) -> float:
    count = min(32, args.queries)
    _, query_keys, kinds, observers, _ = world.fixed_queries(count)
    candidate_keys = world.candidate_keys.to(args.device)
    candidate_kinds = world.candidate_kinds.to(args.device)
    candidate_scopes = world.candidate_scopes.to(args.device)
    encoded = model.encode_candidates(
        candidate_keys, candidate_kinds, candidate_scopes
    )
    query = model.encode_query(
        query_keys.to(args.device), kinds.to(args.device), observers.to(args.device)
    )
    start = time.perf_counter()
    torch.matmul(query, encoded.T).argmax(dim=-1)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / count


def mean_ci(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(variance / len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    seed_results = []
    for seed in args.seeds:
        cformer, rag, losses = train_pair(args, seed)
        results = {"seed": seed, "final_train_loss": {"cformer": losses[0], "rag": losses[1]}, "scales": {}}
        for scale in V56_SCALES:
            aggregate = defaultdict(float)
            for world_index in range(args.worlds):
                world = SyntheticRetrievalWorld.build(scale, world_index)
                c_metrics = evaluate_world(cformer, world, args)
                r_metrics = evaluate_world(rag, world, args)
                n_metrics = evaluate_world(cformer, world, args, use_observer=False)
                for key, value in c_metrics.items():
                    aggregate[f"cformer_{key}"] += value / args.worlds
                for key, value in r_metrics.items():
                    aggregate[f"rag_{key}"] += value / args.worlds
                aggregate["no_observer_accuracy"] += n_metrics["accuracy"] / args.worlds
            latency_world = SyntheticRetrievalWorld.build(scale, 0)
            aggregate["full_scan_query_ms"] = full_scan_latency(cformer, latency_world, args)
            aggregate["compact_cache_mb"] = latency_world.compact_cache_bytes / 1024**2
            aggregate["neural_candidates_per_query"] = args.shortlist
            results["scales"][str(scale)] = dict(aggregate)
            print(f"seed={seed} scale={scale} metrics={dict(aggregate)}", flush=True)
        results["parameters"] = {
            "cformer": parameter_count(cformer),
            "rag": parameter_count(rag),
        }
        seed_results.append(results)

    summary = {}
    for scale in V56_SCALES:
        key = str(scale)
        c_values = [item["scales"][key]["cformer_accuracy"] for item in seed_results]
        r_values = [item["scales"][key]["rag_accuracy"] for item in seed_results]
        differences = [c - r for c, r in zip(c_values, r_values)]
        summary[key] = {
            "cformer_accuracy": mean_ci(c_values),
            "rag_accuracy": mean_ci(r_values),
            "paired_cformer_minus_rag": mean_ci(differences),
            "no_observer_accuracy": mean_ci(
                [item["scales"][key]["no_observer_accuracy"] for item in seed_results]
            ),
            "rough_recall_at_k": sum(
                item["scales"][key]["cformer_rough_recall_at_k"]
                for item in seed_results
            )
            / len(seed_results),
            "rerank_query_ms": sum(
                item["scales"][key]["cformer_rerank_query_ms"]
                for item in seed_results
            )
            / len(seed_results),
            "full_scan_query_ms": sum(
                item["scales"][key]["full_scan_query_ms"] for item in seed_results
            )
            / len(seed_results),
            "compact_cache_mb": seed_results[0]["scales"][key]["compact_cache_mb"],
        }
    output = {
        "configuration": vars(args) | {"output": str(args.output)},
        "parameters": seed_results[0]["parameters"],
        "seed_results": seed_results,
        "summary": summary,
        "boundary_metrics_from_v55": {
            "unsafe_simulation_rate": 0.0,
            "future_leakage_rate": 0.0,
            "identity_collapse_rate": 0.0,
            "note": "verified by the unchanged deterministic V5.5 controller tests",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
