from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import (
    MAX_WORLD_SIZE,
    ChineseAliasWorld,
    ChineseTransformerConfig,
    FlatTransformerResolver,
    MeanPoolMLPResolver,
    TokenCFormerResolver,
)


ROOT = Path(__file__).resolve().parent


def make_model(kind: str, config: ChineseTransformerConfig):
    if kind == "cformer":
        return TokenCFormerResolver(config)
    if kind == "mlp":
        return MeanPoolMLPResolver(config)
    if kind == "flat":
        return FlatTransformerResolver(config)
    raise ValueError(kind)


def train_model(
    model,
    world: ChineseAliasWorld,
    device: torch.device,
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
        objects = world.training_objects(batch_size, seed_offset=seed * 1000 + step)
        variants = [(step * batch_size + index) % 6 for index in range(batch_size)]
        queries, _ = world.encode_queries(objects, variants)
        positives = world.encode_candidates(objects)
        negatives = world.encode_candidates(
            [
                world.hard_negative(obj, variant)
                for obj, variant in zip(objects, variants)
            ]
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
                negatives.to(device, non_blocking=True),
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
def encode_bank(model, world, device, *, dimensions: int, chunk_size: int):
    model.eval()
    bank = torch.empty((world.scale, dimensions), dtype=torch.float16)
    started = time.perf_counter()
    for start in range(0, world.scale, chunk_size):
        stop = min(start + chunk_size, world.scale)
        tokens = world.encode_candidates(world.objects(range(start, stop))).to(device)
        bank[start:stop] = model.encode_candidate(tokens).cpu().to(torch.float16)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return bank, time.perf_counter() - started


@torch.inference_mode()
def retrieve(model, tokens, bank, device, *, topk=10, chunk_size=65536):
    query = model.encode_query(tokens.to(device)).float()
    best_scores = torch.full((query.shape[0], topk), -2.0, device=device)
    best_ids = torch.full((query.shape[0], topk), -1, dtype=torch.long, device=device)
    started = time.perf_counter()
    for start in range(0, bank.shape[0], chunk_size):
        candidates = bank[start : start + chunk_size].to(device).float()
        local_scores, local_ids = (query @ candidates.T).topk(
            min(topk, candidates.shape[0]), dim=-1
        )
        local_ids += start
        scores = torch.cat((best_scores, local_scores), dim=-1)
        ids = torch.cat((best_ids, local_ids), dim=-1)
        best_scores, order = scores.topk(topk, dim=-1)
        best_ids = torch.gather(ids, 1, order)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return best_scores.cpu(), best_ids.cpu(), time.perf_counter() - started


def evaluate_world(
    model,
    *,
    scale: int,
    world_seed: int,
    device: torch.device,
    queries: int,
    dimensions: int,
    chunk_size: int,
    bank_override: torch.Tensor | None = None,
    label_lookup: dict[tuple[int, int, int, int], int] | None = None,
    shared_encode_seconds: float | None = None,
):
    world = ChineseAliasWorld(scale, seed=world_seed)
    if bank_override is None:
        bank, encode_seconds = encode_bank(
            model, world, device, dimensions=dimensions, chunk_size=chunk_size
        )
    else:
        bank = bank_override
        encode_seconds = float(shared_encode_seconds or 0.0)
    targets = world.heldout_objects(queries)
    variants = [index % 6 for index in range(len(targets))]
    tokens, coverage = world.encode_queries(targets, variants)
    scores, ids, search_seconds = retrieve(model, tokens, bank, device)
    labels = torch.tensor(
        [
            obj.label if label_lookup is None else label_lookup[obj.values]
            for obj in targets
        ]
    )
    top1 = ids[:, 0].eq(labels)
    recall10 = ids.eq(labels[:, None]).any(dim=-1)
    negation_mask = torch.tensor([variant in (4, 5) for variant in variants])
    regular_mask = ~negation_mask
    verifier = EvidenceVerifier(minimum_margin=0.08)
    decisions = [
        verifier.decide(float(row[0]), float(row[1]), float(cov))
        for row, cov in zip(scores, coverage)
    ]

    ambiguous_targets = world.ambiguous_objects(queries)
    partial, partial_coverage = world.encode_partial_queries(ambiguous_targets)
    ambiguous_scores, _, ambiguous_seconds = retrieve(model, partial, bank, device)
    ambiguous_decisions = [
        verifier.decide(float(row[0]), float(row[1]), float(cov))
        for row, cov in zip(ambiguous_scores, partial_coverage)
    ]

    unknown_text = "qxz nebula unsupported identity"
    unknown_row, unknown_coverage_value = world.tokenizer.encode(
        unknown_text, world.query_length
    )
    unknown_tokens = unknown_row[None].repeat(queries, 1)
    unknown_scores, _, unknown_seconds = retrieve(model, unknown_tokens, bank, device)
    unknown_decisions = [
        verifier.decide(float(row[0]), float(row[1]), unknown_coverage_value)
        for row in unknown_scores
    ]
    return {
        "scale": scale,
        "world_seed": world_seed,
        "queries": queries,
        "top1": float(top1.float().mean()),
        "recall_at_10": float(recall10.float().mean()),
        "regular_top1": float(top1[regular_mask].float().mean()),
        "negation_top1": float(top1[negation_mask].float().mean()),
        "false_merge_rate": float((~top1).float().mean()),
        "known_supported_rate": sum(
            decision.status == CandidateStatus.SUPPORTED for decision in decisions
        )
        / queries,
        "ambiguous_detection_rate": sum(
            decision.status == CandidateStatus.AMBIGUOUS
            for decision in ambiguous_decisions
        )
        / max(1, len(ambiguous_decisions)),
        "ambiguous_cases": len(ambiguous_decisions),
        "unknown_rejection_rate": sum(
            decision.status == CandidateStatus.UNKNOWN for decision in unknown_decisions
        )
        / queries,
        "bank_mib": bank.numel() * bank.element_size() / 2**20,
        "encode_seconds": encode_seconds,
        "known_search_ms_per_query": search_seconds * 1000 / queries,
        "boundary_search_ms_per_query": (
            ambiguous_seconds + unknown_seconds
        )
        * 1000
        / max(1, len(ambiguous_decisions) + queries),
        "auto_verified_writes": 0,
    }


def aggregate(rows):
    metrics = (
        "top1",
        "recall_at_10",
        "regular_top1",
        "negation_top1",
        "false_merge_rate",
        "known_supported_rate",
        "ambiguous_detection_rate",
        "unknown_rejection_rate",
        "encode_seconds",
        "known_search_ms_per_query",
    )
    result = {}
    keys = sorted({(row["kind"], row["layers"], row["scale"]) for row in rows})
    for kind, layers, scale in keys:
        group = [
            row
            for row in rows
            if row["kind"] == kind
            and row["layers"] == layers
            and row["scale"] == scale
        ]
        name = f"{kind}_L{layers}_{scale}"
        result[name] = {
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
    parser.add_argument("--kinds", nargs="+", choices=("cformer", "mlp", "flat"), default=("cformer", "mlp", "flat"))
    parser.add_argument("--layers", nargs="+", type=int, default=(2, 4, 6))
    parser.add_argument("--flat-layers", nargs="+", type=int, default=(4,))
    parser.add_argument("--seeds", nargs="+", type=int, default=(601, 602, 603))
    parser.add_argument("--scales", nargs="+", type=int, default=(2048, 8192, 65536))
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--queries", type=int, default=72)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=768)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v60_results.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts" / "v60_checkpoints")
    args = parser.parse_args()

    if 4 * ChineseAliasWorld.field_length > 128:
        raise RuntimeError("flat candidate text exceeds Transformer maximum")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer_size = ChineseAliasWorld(2048).tokenizer.size
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    training_rows = []

    configurations = []
    if "cformer" in args.kinds:
        configurations.extend(("cformer", layers) for layers in args.layers)
    if "mlp" in args.kinds:
        configurations.append(("mlp", 0))
    if "flat" in args.kinds:
        configurations.extend(("flat", layers) for layers in args.flat_layers)

    for kind, layers in configurations:
        effective_layers = max(1, layers)
        for seed in args.seeds:
            torch.manual_seed(seed)
            config = ChineseTransformerConfig(
                tokenizer_size,
                layers=effective_layers,
                d_model=args.d_model,
                heads=args.heads,
                ffn_dimensions=args.ffn,
                output_dimensions=args.dimensions,
            )
            model = make_model(kind, config)
            checkpoint = args.checkpoint_dir / f"{kind}_L{layers}_seed{seed}.pt"
            if checkpoint.exists():
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                training = {"loaded_checkpoint": True}
            else:
                training = train_model(
                    model,
                    ChineseAliasWorld(8192, seed=60),
                    device,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    seed=seed,
                )
                torch.save(model.state_dict(), checkpoint)
            model.to(device)
            training_row = {
                "kind": kind,
                "layers": layers,
                "seed": seed,
                "parameters": model.parameter_count(),
                **training,
            }
            training_rows.append(training_row)
            print(json.dumps({"phase": "training", **training_row}, ensure_ascii=False), flush=True)
            for scale in args.scales:
                shared_bank = None
                label_lookup = None
                shared_encode_seconds = None
                if scale == MAX_WORLD_SIZE:
                    bank_world = ChineseAliasWorld(scale, seed=60)
                    shared_bank, shared_encode_seconds = encode_bank(
                        model,
                        bank_world,
                        device,
                        dimensions=args.dimensions,
                        chunk_size=args.chunk_size,
                    )
                    label_lookup = {
                        bank_world.object_at(label).values: label
                        for label in range(scale)
                    }
                for world_index in range(args.worlds):
                    result = evaluate_world(
                        model,
                        scale=scale,
                        world_seed=60 + world_index * 101,
                        device=device,
                        queries=args.queries,
                        dimensions=args.dimensions,
                        chunk_size=args.chunk_size,
                        bank_override=shared_bank,
                        label_lookup=label_lookup,
                        shared_encode_seconds=shared_encode_seconds,
                    )
                    row = {
                        "kind": kind,
                        "layers": layers,
                        "train_seed": seed,
                        **result,
                    }
                    rows.append(row)
                    print(json.dumps({"phase": "evaluation", **row}, ensure_ascii=False), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda"
            else 0,
        },
        "settings": vars(args) | {"output": str(args.output), "checkpoint_dir": str(args.checkpoint_dir)},
        "training": training_rows,
        "worlds": rows,
        "aggregate": aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
