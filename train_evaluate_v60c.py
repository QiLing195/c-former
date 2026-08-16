from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import (
    ChineseAliasWorld,
    ChineseSemanticObject,
    ChineseTransformerConfig,
    TokenCFormerResolver,
)
from cformer_v60b import BlindSet
from cformer_v60c import REGION_NEAR_GROUPS, RegionAugmentedWorld
from train_evaluate_v60 import encode_bank, evaluate_world, retrieve

ROOT = Path(__file__).resolve().parent
VERIFIER = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)

REGION_SPECIAL_VALUES = [
    (1, 2, region, mode)
    for region in (4, 6, 7, 8, 12, 13, 10, 11)
    for mode in (3, 8)
]


def train_model(
    model,
    world: RegionAugmentedWorld,
    device,
    *,
    steps: int,
    batch_size: int,
    seed: int,
):
    torch.manual_seed(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        objects = world.world.training_objects(batch_size, seed_offset=seed * 1000 + step)
        variants = [(step * batch_size + index) % 6 for index in range(batch_size)]
        queries, positives, negatives = world.training_batch(
            objects, variants, seed_offset=seed * 7919 + step
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            loss = model.contrastive_loss(
                queries.to(device, non_blocking=True),
                positives.to(device, non_blocking=True),
                [negatives[0].to(device, non_blocking=True), negatives[1].to(device, non_blocking=True)],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "initial_loss": statistics.mean(losses[: min(10, len(losses))]),
        "final_loss": statistics.mean(losses[-min(10, len(losses)) :]),
    }


@torch.inference_mode()
def evaluate_blind_set(model, world: ChineseAliasWorld, device) -> dict:
    """V6.0b blind set with the same frozen verifier thresholds."""
    blind = BlindSet()
    bank, _ = encode_bank(model, world, device, dimensions=64, chunk_size=2048)
    bank = bank.to(device)
    counts: dict[str, int] = {}
    correct_hits = {"known": 0, "disambiguated": 0, "hard": 0}
    ambiguous_hits = 0
    unknown_hits = 0
    unknown_not_supported = 0
    errors = []
    for query in blind.queries:
        tokens, coverage = world.tokenizer.encode(query.text, world.query_length)
        scores, ids, _ = retrieve(model, tokens[None], bank, device, topk=2)
        decision = VERIFIER.decide(float(scores[0, 0]), float(scores[0, 1]), float(coverage))
        top1_values = world.object_at(int(ids[0, 0])).values
        correct = query.target is not None and tuple(top1_values) == tuple(query.target)
        category = query.expected
        counts[category] = counts.get(category, 0) + 1
        if category in correct_hits:
            correct_hits[category] += int(correct)
        elif category == "ambiguous":
            ambiguous_hits += int(decision.status == CandidateStatus.AMBIGUOUS)
        elif category == "unknown":
            unknown_hits += int(decision.status == CandidateStatus.UNKNOWN)
            unknown_not_supported += int(decision.status != CandidateStatus.SUPPORTED)
        if not correct and category in ("known", "disambiguated", "hard"):
            errors.append({"text": query.text, "expected": category, "note": query.note,
                           "target": query.target, "got": tuple(top1_values),
                           "status": decision.status.value, "margin": decision.margin})
        if category == "unknown" and decision.status == CandidateStatus.SUPPORTED:
            errors.append({"text": query.text, "expected": "unknown", "note": query.note,
                           "status": decision.status.value, "score": decision.score})
    return {
        "known_top1": correct_hits["known"] / max(1, counts.get("known", 0)),
        "disambiguated_top1": correct_hits["disambiguated"] / max(1, counts.get("disambiguated", 0)),
        "hard_top1": correct_hits["hard"] / max(1, counts.get("hard", 0)),
        "ambiguous_rate": ambiguous_hits / max(1, counts.get("ambiguous", 0)),
        "unknown_rejection": unknown_hits / max(1, counts.get("unknown", 0)),
        "unknown_not_supported": unknown_not_supported / max(1, counts.get("unknown", 0)),
        "errors": errors,
    }


@torch.inference_mode()
def evaluate_region_special(model, world: ChineseAliasWorld, device, bank) -> dict:
    label_of = {world.object_at(label).values: label for label in range(world.scale)}
    targets = []
    queries = []
    for index, values in enumerate(REGION_SPECIAL_VALUES):
        if values not in label_of:
            raise AssertionError(f"region-special values not in world: {values}")
        targets.append(label_of[values])
        queries.append(world.query_text(values, index % 6))
    tokens = torch.stack(
        [world.tokenizer.encode(text, world.query_length)[0] for text in queries]
    )
    scores, ids, _ = retrieve(model, tokens, bank, device, topk=1)
    correct = ids[:, 0].eq(torch.tensor(targets)).float().mean().item()
    return {"region_special_top1": correct, "queries": len(targets)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=(701, 702, 703))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=65536)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts" / "v60c_checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v60c_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer_size = ChineseAliasWorld(2048).tokenizer.size
    train_world = RegionAugmentedWorld(args.scale, seed=60)
    eval_world = ChineseAliasWorld(args.scale, seed=60)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args_augmentation = {
        "typo_rate": 0.15,
        "region_rate": 0.4,
        "typo_chars": "content-only (alias chars)",
        "region_groups": [list(group) for group in REGION_NEAR_GROUPS],
        "negatives_per_sample": 2,
        "negative_mix": "region_swap_or_standard + standard",
        "conflict_negative": "data primitive only, excluded after iteration 1 regression",
    }

    per_seed: dict[str, dict] = {}
    for seed in args.seeds:
        model = TokenCFormerResolver(ChineseTransformerConfig(tokenizer_size, layers=2))
        checkpoint = args.checkpoint_dir / f"cformer_L2_seed{seed}.pt"
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            training = {"loaded_checkpoint": True}
        else:
            training = train_model(
                model, train_world, device, steps=args.steps, batch_size=args.batch_size, seed=seed
            )
            torch.save(model.state_dict(), checkpoint)
        model.to(device).eval()

        blind = evaluate_blind_set(model, eval_world, device)
        blind_metrics = {key: value for key, value in blind.items() if key != "errors"}

        # V6.0 formal protocol at 64K: shared bank + 4 fixed worlds.
        shared_bank, shared_seconds = encode_bank(model, eval_world, device, dimensions=64, chunk_size=2048)
        label_lookup = {eval_world.object_at(label).values: label for label in range(args.scale)}
        formal_rows = []
        for world_index in range(4):
            row = evaluate_world(
                model,
                scale=args.scale,
                world_seed=60 + world_index * 101,
                device=device,
                queries=96,
                dimensions=64,
                chunk_size=2048,
                bank_override=shared_bank,
                label_lookup=label_lookup,
                shared_encode_seconds=shared_seconds,
            )
            formal_rows.append(row)
        formal = {
            "formal_top1": statistics.mean(row["top1"] for row in formal_rows),
            "formal_known_supported": statistics.mean(row["known_supported_rate"] for row in formal_rows),
            "formal_ambiguous_detection": statistics.mean(row["ambiguous_detection_rate"] for row in formal_rows),
            "formal_unknown_rejection": statistics.mean(row["unknown_rejection_rate"] for row in formal_rows),
        }
        region = evaluate_region_special(model, eval_world, device, shared_bank)

        per_seed[str(seed)] = {**blind_metrics, **formal, **region, "training": training}
        print(json.dumps({"phase": "seed", "seed": seed, **per_seed[str(seed)]}, ensure_ascii=False), flush=True)
        # Persist incrementally so a mid-run interruption never loses completed seeds.
        partial = {
            "environment": {"device": str(device), "torch": torch.__version__},
            "settings": {key: str(value) for key, value in vars(args).items()},
            "augmentation": args_augmentation,
            "per_seed": per_seed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = (
        "known_top1", "disambiguated_top1", "hard_top1", "ambiguous_rate",
        "unknown_rejection", "unknown_not_supported",
        "formal_top1", "formal_known_supported", "formal_ambiguous_detection",
        "formal_unknown_rejection", "region_special_top1",
    )
    aggregate = {
        metric: {
            "mean": statistics.mean(per_seed[str(seed)][metric] for seed in args.seeds),
            "min": min(per_seed[str(seed)][metric] for seed in args.seeds),
            "max": max(per_seed[str(seed)][metric] for seed in args.seeds),
        }
        for metric in metrics
    }
    payload = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
        },
        "settings": {key: str(value) for key, value in vars(args).items()},
        "augmentation": args_augmentation,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "errors": per_seed[str(args.seeds[0])]["errors"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
