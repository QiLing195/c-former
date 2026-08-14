from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


METRICS = (
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


def summarize(rows: list[dict]) -> dict:
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
        result[f"{kind}_L{layers}_{scale}"] = {
            metric: {
                "mean": statistics.mean(row[metric] for row in group),
                "min": min(row[metric] for row in group),
                "max": max(row[metric] for row in group),
            }
            for metric in METRICS
        }
    return result


def main() -> None:
    strict = json.loads(
        (ARTIFACTS / "v60_strict_results_final.json").read_text(encoding="utf-8")
    )
    l6 = json.loads(
        (ARTIFACTS / "v60_strict_l6_results.json").read_text(encoding="utf-8")
    )
    flat2 = json.loads(
        (ARTIFACTS / "v60_strict_flat2_results.json").read_text(encoding="utf-8")
    )
    rebench = json.loads(
        (ARTIFACTS / "v60_c2_seed601_rebench.json").read_text(encoding="utf-8")
    )
    corrected_seconds = rebench["worlds"][0]["encode_seconds"]
    corrected_rows = []
    for original in strict["worlds"]:
        row = dict(original)
        if (
            row["kind"] == "cformer"
            and row["layers"] == 2
            and row["train_seed"] == 601
            and row["scale"] == 65536
        ):
            row["encode_seconds"] = corrected_seconds
        corrected_rows.append(row)
    corrected_rows.extend(flat2["worlds"])

    training = {}
    all_training = strict["training"] + flat2["training"]
    for kind, layers in (
        ("cformer", 2),
        ("cformer", 4),
        ("mlp", 0),
        ("flat", 2),
        ("flat", 4),
    ):
        group = [
            row
            for row in all_training
            if row["kind"] == kind and row["layers"] == layers
        ]
        training[f"{kind}_L{layers}"] = {
            "parameters": group[0]["parameters"],
            "seconds_mean": statistics.mean(row["seconds"] for row in group),
            "final_loss_mean": statistics.mean(row["final_loss"] for row in group),
            "seeds": [row["seed"] for row in group],
        }

    payload = {
        "selected_default": {
            "kind": "cformer",
            "token_transformer_layers": 2,
            "reason": "Best scale-efficiency trade-off: exact retrieval quality and substantially faster bank encoding than the same-depth flat Transformer; verifier support calibration remains follow-up work",
        },
        "environment": strict["environment"],
        "test_counts": {
            "strict_world_rows": len(corrected_rows),
            "strict_queries_per_row": 288,
            "strict_total_queries": len(corrected_rows) * 288,
            "l6_world_rows": len(l6["worlds"]),
            "l6_total_queries": len(l6["worlds"]) * 288,
            "grand_total_queries": (len(corrected_rows) + len(l6["worlds"])) * 288,
        },
        "training": training,
        "aggregate": summarize(corrected_rows),
        "l6_training": l6["training"],
        "l6_aggregate": summarize(l6["worlds"]),
        "timing_correction": {
            "field": "cformer L2 seed601 scale65536 encode_seconds",
            "reason": "desktop/task suspension contaminated wall-clock duration",
            "replacement_seconds": corrected_seconds,
            "accuracy_metrics_changed": False,
        },
        "source_artifacts": (
            "v60_strict_results_final.json",
            "v60_strict_l6_results.json",
            "v60_strict_flat2_results.json",
            "v60_c2_seed601_rebench.json",
        ),
    }
    output = ARTIFACTS / "v60_summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
