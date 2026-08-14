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

from cformer_v57 import (
    ControlledTransformationEngine,
    RouteStatus,
    TextCFormerReranker,
    TextEvidenceRAGReranker,
    TextRerankerConfig,
    TextRetrievalWorld,
    TextTransition,
    V57_SCALES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5.7 noisy text retrieval")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shortlist", type=int, default=64)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--worlds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[401, 402, 403])
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v57_results.json"))
    return parser.parse_args()


def target_scope(kinds, modes, observers):
    return torch.where(kinds.bool() ^ modes.bool(), 5 - observers, observers)


def training_batch(world, args, generator):
    entities = torch.randint(
        world.entity_features.shape[0],
        (args.batch_size,),
        generator=generator,
    )
    variants = torch.randint(4, (args.batch_size,), generator=generator)
    observers = torch.randint(1, 5, (args.batch_size,), generator=generator)
    texts = [world.query_text(int(entity), int(variant)) for entity, variant in zip(entities, variants)]
    query_features = world.encode_queries(entities, texts)
    base_kinds = world.entity_kinds[entities]
    modes = world.entity_modes[entities]
    kinds = world.entity_tasks[entities]
    correct_ids = entities * 4 + target_scope(base_kinds, modes, observers) - 1
    candidate_ids, labels, recall, _ = world.text_shortlists(
        query_features,
        correct_ids,
        entity_topk=args.shortlist // 4,
        query_texts=texts,
    )
    if not bool(recall.all()):
        raise AssertionError("governed identity anchor missing from training shortlist")
    return (
        world.candidate_features[candidate_ids].to(args.device),
        world.candidate_kinds[candidate_ids].to(args.device),
        world.candidate_scopes[candidate_ids].to(args.device),
        query_features.to(args.device),
        kinds.to(args.device),
        observers.to(args.device),
        labels.to(args.device),
    )


def train_pair(args, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    config = TextRerankerConfig()
    cformer = TextCFormerReranker(config).to(args.device)
    rag = TextEvidenceRAGReranker(config).to(args.device)
    rag.load_state_dict(cformer.state_dict())
    optimizers = (
        torch.optim.AdamW(cformer.parameters(), lr=args.lr),
        torch.optim.AdamW(rag.parameters(), lr=args.lr),
    )
    worlds = [TextRetrievalWorld.build(2048, index) for index in (5, 6, 7)]
    generator = torch.Generator().manual_seed(seed + 30_000)
    losses = [0.0, 0.0]
    for step in range(args.steps):
        batch = training_batch(worlds[step % len(worlds)], args, generator)
        for index, (model, optimizer) in enumerate(zip((cformer, rag), optimizers)):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(*batch[:-1]), batch[-1])
            loss.backward()
            optimizer.step()
            losses[index] = loss.item()
        if (step + 1) % 100 == 0:
            print(
                f"seed={seed} step={step + 1} cformer_loss={losses[0]:.4f} rag_loss={losses[1]:.4f}",
                flush=True,
            )
    return cformer.eval(), rag.eval()


@torch.inference_mode()
def evaluate_world(model, world, args, *, use_observer=True):
    _, variants, texts, query_features, kinds, observers, correct_ids = world.fixed_queries(args.queries)
    candidate_ids, labels, recall, lexical_top1 = world.text_shortlists(
        query_features, correct_ids, query_texts=texts
    )
    logits = model(
        world.candidate_features[candidate_ids].to(args.device),
        world.candidate_kinds[candidate_ids].to(args.device),
        world.candidate_scopes[candidate_ids].to(args.device),
        query_features.to(args.device),
        kinds.to(args.device),
        observers.to(args.device),
        use_observer=use_observer,
    )
    correct = logits.argmax(dim=-1).eq(labels.to(args.device)) & recall.to(args.device)
    by_variant = {}
    for variant in range(4):
        mask = variants.eq(variant).to(args.device)
        by_variant[str(variant)] = correct[mask].float().mean().item()
    return {
        "accuracy": correct.float().mean().item(),
        "text_recall_at_64": recall.float().mean().item(),
        "lexical_object_top1": lexical_top1.float().mean().item(),
        "by_variant": by_variant,
    }


def mean_ci(values):
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(variance / len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def evaluate_recursive_controller():
    edges = (
        TextTransition(1, 1, 2, "cool", "降温 cooling 变换"),
        TextTransition(2, 2, 3, "compress", "压缩 compression 变换"),
        TextTransition(3, 3, 1, "return", "返回 return 变换"),
        TextTransition(4, 10, 11, "secret", "机密 secret 变换", visibility_scope=frozenset({1})),
    )
    def engine():
        return ControlledTransformationEngine(edges)
    cases = []
    cases.append(engine().route(1, ("cooling 降温", "compression 压缩"), observer_scope=1, query_time=1, ingest_cutoff=1).status == RouteStatus.ANSWER)
    cases.append(engine().route(1, ("cooling", "compression", "return"), observer_scope=1, query_time=1, ingest_cutoff=1).status == RouteStatus.CYCLE)
    cases.append(engine().route(10, ("secret",), observer_scope=2, query_time=1, ingest_cutoff=1).status == RouteStatus.ACCESS_DENIED)
    cases.append(engine().route(1, ("cooling", "compression"), observer_scope=1, query_time=1, ingest_cutoff=1, max_depth=1).status == RouteStatus.DEPTH_LIMIT)
    changed = engine()
    def mutate(depth):
        if depth == 1:
            changed.touch_version()
    cases.append(changed.route(1, ("cooling", "compression"), observer_scope=1, query_time=1, ingest_cutoff=1, on_hop=mutate).status == RouteStatus.VERSION_CHANGED)
    return {"accuracy": sum(cases) / len(cases), "cases": len(cases)}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    evaluation_worlds = {
        scale: [TextRetrievalWorld.build(scale, index) for index in range(args.worlds)]
        for scale in V57_SCALES
    }
    seed_results = []
    for seed in args.seeds:
        cformer, rag = train_pair(args, seed)
        result = {"seed": seed, "scales": {}}
        for scale in V57_SCALES:
            aggregate = defaultdict(float)
            variants = defaultdict(float)
            start = time.perf_counter()
            for world in evaluation_worlds[scale]:
                c_metrics = evaluate_world(cformer, world, args)
                r_metrics = evaluate_world(rag, world, args)
                n_metrics = evaluate_world(cformer, world, args, use_observer=False)
                for key in ("accuracy", "text_recall_at_64", "lexical_object_top1"):
                    aggregate[f"cformer_{key}"] += c_metrics[key] / args.worlds
                aggregate["rag_accuracy"] += r_metrics["accuracy"] / args.worlds
                aggregate["no_observer_accuracy"] += n_metrics["accuracy"] / args.worlds
                for variant, value in c_metrics["by_variant"].items():
                    variants[variant] += value / args.worlds
            aggregate["evaluation_seconds"] = time.perf_counter() - start
            aggregate["text_cache_mb"] = world.text_cache_bytes / 1024**2
            aggregate["variant_accuracy"] = dict(variants)
            result["scales"][str(scale)] = dict(aggregate)
            print(f"seed={seed} scale={scale} metrics={dict(aggregate)}", flush=True)
        result["parameters"] = {
            "cformer": sum(parameter.numel() for parameter in cformer.parameters()),
            "rag": sum(parameter.numel() for parameter in rag.parameters()),
        }
        seed_results.append(result)
    summary = {}
    for scale in V57_SCALES:
        key = str(scale)
        c = [item["scales"][key]["cformer_accuracy"] for item in seed_results]
        r = [item["scales"][key]["rag_accuracy"] for item in seed_results]
        summary[key] = {
            "cformer_accuracy": mean_ci(c),
            "rag_accuracy": mean_ci(r),
            "paired_difference": mean_ci([left - right for left, right in zip(c, r)]),
            "no_observer_accuracy": mean_ci(
                [item["scales"][key]["no_observer_accuracy"] for item in seed_results]
            ),
            "text_recall_at_64": sum(
                item["scales"][key]["cformer_text_recall_at_64"] for item in seed_results
            ) / len(seed_results),
            "lexical_object_top1": sum(
                item["scales"][key]["cformer_lexical_object_top1"] for item in seed_results
            ) / len(seed_results),
            "variant_accuracy": seed_results[0]["scales"][key]["variant_accuracy"],
            "text_cache_mb": seed_results[0]["scales"][key]["text_cache_mb"],
        }
    output = {
        "configuration": vars(args) | {"output": str(args.output)},
        "parameters": seed_results[0]["parameters"],
        "variant_names": {"0": "canonical", "1": "registered_alias", "2": "unicode_spacing_noise", "3": "long_context"},
        "seed_results": seed_results,
        "summary": summary,
        "recursive_controller": evaluate_recursive_controller(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
