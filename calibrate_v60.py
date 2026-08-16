from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import ChineseAliasWorld, ChineseTransformerConfig, TokenCFormerResolver
from cformer_v60b import BlindSet, CalibrationSet
from train_evaluate_v60 import encode_bank, retrieve

ROOT = Path(__file__).resolve().parent

BASE = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)
MARGINS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)


def collect(model, world, queries, device, bank) -> list[dict]:
    rows = []
    for query in queries:
        tokens, coverage = world.tokenizer.encode(query.text, world.query_length)
        scores, ids, _ = retrieve(model, tokens[None], bank, device, topk=2)
        decision = BASE.decide(float(scores[0, 0]), float(scores[0, 1]), float(coverage))
        top1_values = world.object_at(int(ids[0, 0])).values
        rows.append({
            "expected": query.expected,
            "score": float(scores[0, 0]),
            "margin": float(scores[0, 0] - scores[0, 1]),
            "coverage": float(coverage),
            "status": decision.status.value,
            "top1_correct": query.target is not None and tuple(top1_values) == tuple(query.target),
        })
    return rows


def sweep(rows: list[dict], target_risk: float) -> dict:
    """Risk-coverage over margin on queries that carry a target (known+typo)."""
    targeted = [row for row in rows if row["expected"] in ("known", "typo")]
    curve = []
    for margin in MARGINS:
        supported = [
            row for row in targeted
            if row["coverage"] >= BASE.minimum_coverage
            and row["score"] >= BASE.minimum_score
            and row["margin"] >= margin
        ]
        coverage = len(supported) / max(1, len(targeted))
        risk = 1.0 - statistics.mean(row["top1_correct"] for row in supported) if supported else float("nan")
        curve.append({"margin": margin, "coverage": coverage, "risk": risk})
    recommended = None
    for point in curve:
        if point["risk"] == point["risk"] and point["risk"] <= target_risk:
            recommended = point
            break
    return {"curve": curve, "recommended": recommended, "total": len(targeted)}


def apply_threshold(rows: list[dict], margin: float) -> dict:
    decision_map = {}
    risk_hits = 0
    for row in rows:
        if row["coverage"] < BASE.minimum_coverage or row["score"] < BASE.minimum_score:
            status = CandidateStatus.UNKNOWN.value
        elif row["margin"] < margin:
            status = CandidateStatus.AMBIGUOUS.value
        else:
            status = CandidateStatus.SUPPORTED.value
            if row["expected"] in ("known", "typo"):
                risk_hits += int(not row["top1_correct"])
        if status == CandidateStatus.SUPPORTED.value:
            decision_map[row["expected"]] = decision_map.get(row["expected"], 0) + 1
    return decision_map, risk_hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts" / "v60_strict_checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=(601, 602, 603))
    parser.add_argument("--scale", type=int, default=65536)
    parser.add_argument("--target-risk", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v60b_calibration.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = ChineseAliasWorld(args.scale, seed=60)
    calibration = CalibrationSet()
    blind = BlindSet()
    tokenizer_size = world.tokenizer.size

    per_seed: dict[str, dict] = {}
    for seed in args.seeds:
        checkpoint = args.checkpoint_dir / f"cformer_L2_seed{seed}.pt"
        model = TokenCFormerResolver(ChineseTransformerConfig(tokenizer_size, layers=2)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        bank, _ = encode_bank(model, world, device, dimensions=64, chunk_size=2048)
        bank = bank.to(device)

        cal_rows = collect(model, world, calibration.queries, device, bank)
        blind_rows = collect(model, world, blind.queries, device, bank)

        sweep_result = sweep(cal_rows, args.target_risk)
        recommended = sweep_result["recommended"]
        margin_star = recommended["margin"] if recommended else BASE.minimum_margin
        blind_support, blind_risk_hits = apply_threshold(blind_rows, margin_star)
        frozen_support, _ = apply_threshold(blind_rows, BASE.minimum_margin)

        per_seed[str(seed)] = {
            "calibration_sweep": sweep_result["curve"],
            "recommended_margin": margin_star,
            "recommended_coverage": recommended["coverage"] if recommended else None,
            "recommended_risk": recommended["risk"] if recommended else None,
            "blind_supported_under_star": blind_support,
            "blind_supported_under_frozen": frozen_support,
        }
        print(json.dumps({"phase": "seed", "seed": seed, "margin_star": margin_star,
                          "coverage": recommended["coverage"] if recommended else None,
                          "risk": recommended["risk"] if recommended else None}, ensure_ascii=False), flush=True)
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "settings": {key: str(value) for key, value in vars(args).items()},
        "target_risk": args.target_risk,
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done"}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
