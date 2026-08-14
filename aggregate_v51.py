from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch


SCALES = (128, 512, 2048)
SYSTEMS = {
    "cformer": "cformer_v5_system",
    "rag": "evidence_rag_v5_system",
}
T_CRITICAL_95 = {
    1: float("inf"),
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate V5.1 multi-seed results")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203, 204, 205])
    parser.add_argument("--output", type=Path, default=Path("artifacts/v51_summary.json"))
    parser.add_argument("--report", type=Path, default=Path("V51_STATISTICAL_REPORT.md"))
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired_ci95(values: list[float]) -> dict[str, float | bool]:
    summary = describe(values)
    if len(values) == 1:
        margin = float("inf")
    else:
        critical = T_CRITICAL_95.get(len(values), 1.96)
        margin = critical * summary["sample_std"] / math.sqrt(len(values))
    lower = summary["mean"] - margin
    upper = summary["mean"] + margin
    return {
        **summary,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "significantly_above_zero": lower > 0.0,
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    raw: dict[int, dict] = {}
    for seed in args.seeds:
        path = args.artifact_root / f"v51_seed_{seed}" / "v5_results.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing result for seed {seed}: {path}")
        raw[seed] = torch.load(path, map_location="cpu", weights_only=True)

    summary: dict = {"seeds": args.seeds, "scales": {}}
    report_rows = []
    per_seed_rows = []
    for scale in SCALES:
        scale_summary: dict = {}
        answer_values: dict[str, list[float]] = {}
        for short_name, result_name in SYSTEMS.items():
            values = [
                raw[seed][result_name][scale]["answer_normal_answer"]
                for seed in args.seeds
            ]
            answer_values[short_name] = values
            scale_summary[short_name] = {
                "answer_normal_answer": describe(values),
                "hallucination_rate": describe(
                    [raw[seed][result_name][scale]["hallucination_rate"] for seed in args.seeds]
                ),
                "false_conflict_rate": describe(
                    [raw[seed][result_name][scale]["false_conflict_rate"] for seed in args.seeds]
                ),
            }
        differences = [
            cformer - rag
            for cformer, rag in zip(answer_values["cformer"], answer_values["rag"])
        ]
        scale_summary["paired_cformer_minus_rag"] = paired_ci95(differences)
        scale_summary["paired_cformer_minus_rag"]["meets_2pp_target"] = (
            scale_summary["paired_cformer_minus_rag"]["mean"] >= 0.02 - 1e-12
        )
        scale_summary["per_seed"] = {
            str(seed): {
                "cformer": answer_values["cformer"][index],
                "rag": answer_values["rag"][index],
                "difference": differences[index],
            }
            for index, seed in enumerate(args.seeds)
        }
        summary["scales"][str(scale)] = scale_summary

        cformer_stats = scale_summary["cformer"]["answer_normal_answer"]
        rag_stats = scale_summary["rag"]["answer_normal_answer"]
        paired = scale_summary["paired_cformer_minus_rag"]
        report_rows.append(
            "| {scale} | {cm} ± {cs} | {rm} ± {rs} | {dm} | [{lo}, {hi}] | {sig} |".format(
                scale=scale,
                cm=percent(cformer_stats["mean"]),
                cs=percent(cformer_stats["sample_std"]),
                rm=percent(rag_stats["mean"]),
                rs=percent(rag_stats["sample_std"]),
                dm=percent(paired["mean"]),
                lo=percent(paired["ci95_lower"]),
                hi=percent(paired["ci95_upper"]),
                sig="是" if paired["significantly_above_zero"] else "否",
            )
        )
        for index, seed in enumerate(args.seeds):
            per_seed_rows.append(
                f"| {seed} | {scale} | {percent(answer_values['cformer'][index])} | "
                f"{percent(answer_values['rag'][index])} | {percent(differences[index])} |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    all_safe = all(
        summary["scales"][str(scale)][system][metric]["mean"] == 0.0
        for scale in SCALES
        for system in SYSTEMS
        for metric in ("hallucination_rate", "false_conflict_rate")
    )
    large_world = summary["scales"]["2048"]["paired_cformer_minus_rag"]
    significance_verdict = (
        "2048 世界中观察点融合具有统计显著优势。"
        if large_world["significantly_above_zero"]
        else "2048 世界中的平均优势尚未排除随机波动，不能宣称统计显著优于 RAG。"
    )
    target_verdict = (
        "2048 世界的平均配对差值达到预设的 +2 个百分点目标。"
        if large_world["meets_2pp_target"]
        else "2048 世界的平均配对差值未达到预设的 +2 个百分点目标。"
    )
    report = "\n".join(
        [
            "# V5.1 多种子统计复验报告",
            "",
            f"固定种子：{', '.join(map(str, args.seeds))}。每个模型每个种子训练 6,300 步；比较双方参数量、训练预算与证据监督保持一致。",
            "",
            "## 普通问答准确率",
            "",
            "| 世界规模 | C-Former 均值 ± 标准差 | Evidence-RAG 均值 ± 标准差 | 配对差值 | 差值 95% CI | 显著高于 0 |",
            "|---:|---:|---:|---:|---:|:---:|",
            *report_rows,
            "",
            "## 每个种子的原始准确率",
            "",
            "| 种子 | 世界规模 | C-Former | Evidence-RAG | 配对差值 |",
            "|---:|---:|---:|---:|---:|",
            *per_seed_rows,
            "",
            "## 边界指标",
            "",
            f"全部规模、全部种子中，双方的幻觉率和误报冲突率均为 0：{'是' if all_safe else '否'}。",
            "",
            "## 结论",
            "",
            target_verdict,
            significance_verdict,
            "该结论只适用于当前合成小世界与固定控制器，不等同于真实开放域语言模型能力。",
            "",
            f"机器可读统计：`{args.output.as_posix()}`。",
        ]
    )
    args.report.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"saved_summary={args.output}")
    print(f"saved_report={args.report}")


if __name__ == "__main__":
    main()
