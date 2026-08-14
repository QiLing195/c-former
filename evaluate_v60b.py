from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import ChineseAliasWorld, ChineseTransformerConfig, TokenCFormerResolver
from cformer_v60b import BlindQuery, BlindSet

ROOT = Path(__file__).resolve().parent

# Frozen V6.0 boundary thresholds; intentionally not tuned here.
VERIFIER = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)

GATED_CATEGORIES = ("known", "ambiguous", "unknown", "disambiguated")


def make_model(tokenizer_size: int) -> TokenCFormerResolver:
    # V6.0 frozen default: 2 layers.
    config = ChineseTransformerConfig(tokenizer_size, layers=2)
    return TokenCFormerResolver(config)


@torch.inference_mode()
def encode_bank(model, world, device) -> torch.Tensor:
    bank = torch.empty((world.scale, model.config.output_dimensions), dtype=torch.float16)
    for start in range(0, world.scale, 8192):
        stop = min(start + 8192, world.scale)
        tokens = world.encode_candidates(world.objects(range(start, stop))).to(device)
        bank[start:stop] = model.encode_candidate(tokens).cpu().half()
    return bank


@torch.inference_mode()
def resolve(model, bank, world, query: BlindQuery, device) -> dict:
    tokens, coverage = world.tokenizer.encode(query.text, world.query_length)
    query_vector = model.encode_query(tokens[None].to(device)).float()
    scores = query_vector @ bank.float().to(device).T
    top_scores, top_ids = torch.topk(scores, 2, dim=-1)
    decision = VERIFIER.decide(
        float(top_scores[0, 0]), float(top_scores[0, 1]), float(coverage)
    )
    top1_label = int(top_ids[0, 0])
    top1_values = tuple(world.object_at(top1_label).values)
    return {
        "text": query.text,
        "expected": query.expected,
        "note": query.note,
        "target": query.target,
        "coverage": float(coverage),
        "status": decision.status.value,
        "score": decision.score,
        "margin": decision.margin,
        "top1_label": top1_label,
        "top1_values": top1_values,
        "top1_correct": query.target is not None and top1_values == tuple(query.target),
    }


def aggregate(rows: list[dict]) -> dict:
    def mean_of(category: str, key: str, default: float = float("nan")) -> float:
        group = [row for row in rows if row["expected"] == category]
        if not group:
            return default
        return statistics.mean(row[key] for row in group)

    return {
        "counts": {cat: sum(1 for r in rows if r["expected"] == cat) for cat in GATED_CATEGORIES},
        "known_top1": mean_of("known", "top1_correct"),
        "disambiguated_top1": mean_of("disambiguated", "top1_correct"),
        "hard_top1": mean_of("hard", "top1_correct"),
        "ambiguous_rate": mean_of("ambiguous", "is_ambiguous"),
        "unknown_rejection": mean_of("unknown", "is_unknown"),
        "unknown_not_supported": mean_of("unknown", "not_supported"),
        "conflict_statuses": {
            status: sum(1 for r in rows if r["expected"] == "conflict" and r["status"] == status)
            for status in ("supported", "ambiguous", "unknown")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts" / "v60_strict_checkpoints")
    parser.add_argument("--scale", type=int, default=65536)
    parser.add_argument("--seeds", nargs="+", type=int, default=(601, 602, 603))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v60b_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = ChineseAliasWorld(args.scale, seed=60)
    blind = BlindSet()
    tokenizer_size = world.tokenizer.size

    all_rows: list[dict] = []
    per_seed: dict[str, dict] = {}
    for seed in args.seeds:
        checkpoint = args.checkpoint_dir / f"cformer_L2_seed{seed}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = make_model(tokenizer_size).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.eval()
        bank = encode_bank(model, world, device).to(device)
        rows = []
        for query in blind.queries:
            row = resolve(model, bank, world, query, device)
            row["is_ambiguous"] = row["status"] == CandidateStatus.AMBIGUOUS.value
            row["is_unknown"] = row["status"] == CandidateStatus.UNKNOWN.value
            row["not_supported"] = row["status"] != CandidateStatus.SUPPORTED.value
            rows.append(row)
        all_rows.extend(rows)
        per_seed[str(seed)] = aggregate(rows)
        print(json.dumps({"phase": "seed", "seed": seed, **per_seed[str(seed)]}, ensure_ascii=False), flush=True)
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3-seed aggregation with min/max.
    metrics = (
        "known_top1",
        "disambiguated_top1",
        "hard_top1",
        "ambiguous_rate",
        "unknown_rejection",
        "unknown_not_supported",
    )
    aggregate_three = {
        metric: {
            "mean": statistics.mean(per_seed[str(seed)][metric] for seed in args.seeds),
            "min": min(per_seed[str(seed)][metric] for seed in args.seeds),
            "max": max(per_seed[str(seed)][metric] for seed in args.seeds),
        }
        for metric in metrics
    }

    # Calibration sweep over margin (risk-coverage curve on known-like queries).
    calibration: list[dict] = []
    known_rows = [row for row in all_rows if row["expected"] in ("known", "disambiguated", "hard")]
    for margin in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
        verifier = EvidenceVerifier(minimum_score=0.50, minimum_margin=margin, minimum_coverage=0.60)
        decisions = [
            verifier.decide(row["score"], row["score"] - row["margin"], row["coverage"])
            for row in known_rows
        ]
        supported = [row for row, dec in zip(known_rows, decisions) if dec.status == CandidateStatus.SUPPORTED]
        coverage = len(supported) / max(1, len(known_rows))
        risk = 1.0 - (
            statistics.mean(row["top1_correct"] for row in supported) if supported else float("nan")
        )
        calibration.append({"margin": margin, "coverage": coverage, "supported_risk": risk})

    errors = [
        row
        for row in all_rows
        if (row["expected"] in ("known", "disambiguated", "hard") and not row["top1_correct"])
        or (row["expected"] == "ambiguous" and not row["is_ambiguous"])
        or (row["expected"] == "unknown" and row["status"] == CandidateStatus.SUPPORTED.value)
    ]

    payload = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "verifier": {
            "minimum_score": VERIFIER.minimum_score,
            "minimum_margin": VERIFIER.minimum_margin,
            "minimum_coverage": VERIFIER.minimum_coverage,
        },
        "per_seed": per_seed,
        "aggregate": aggregate_three,
        "calibration": calibration,
        "error_samples": errors,
        "queries": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate_three}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
