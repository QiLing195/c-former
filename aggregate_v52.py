from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch


SCALES = (2048, 8192, 32768)
T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    15: 2.145,
    20: 2.093,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate V5.2 multi-seed stress results")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(201, 211)))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v52_summary.json"))
    parser.add_argument("--report", type=Path, default=Path("V52_FINAL_REPORT.md"))
    return parser.parse_args()


def result_dir(root: Path, seed: int) -> Path:
    candidates = [root / f"v51_seed_{seed}", root / f"v52_seed_{seed}"]
    for candidate in candidates:
        if (candidate / "v52_results.pt").exists():
            return candidate
    raise FileNotFoundError(f"missing V5.2 result for seed {seed}: {candidates}")


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def paired(values: list[float]) -> dict[str, float | bool]:
    result = describe(values)
    critical = T95.get(len(values), 1.96)
    margin = critical * result["sample_std"] / math.sqrt(len(values)) if len(values) > 1 else math.inf
    result["ci95_lower"] = result["mean"] - margin
    result["ci95_upper"] = result["mean"] + margin
    result["significantly_above_zero"] = result["ci95_lower"] > 0.0
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def mean_metric(raw: dict[int, dict], scale: int, system: str, metric: str) -> dict[str, float]:
    return describe([raw[seed]["stress"][scale]["metrics"][system][metric] for seed in raw])


def main() -> None:
    args = parse_args()
    raw = {}
    dirs = {}
    for seed in args.seeds:
        directory = result_dir(args.artifact_root, seed)
        dirs[seed] = directory
        raw[seed] = torch.load(directory / "v52_results.pt", map_location="cpu", weights_only=True)

    summary: dict = {"seeds": args.seeds, "result_directories": {str(k): str(v) for k, v in dirs.items()}, "scales": {}}
    scale_rows = []
    ablation_rows = []
    performance_rows = []
    per_seed_rows = []
    for scale in SCALES:
        c_values = [raw[s]["stress"][scale]["metrics"]["cformer_full"]["end_to_end_accuracy"] for s in args.seeds]
        r_values = [raw[s]["stress"][scale]["metrics"]["rag"]["end_to_end_accuracy"] for s in args.seeds]
        c_head = [raw[s]["stress"][scale]["metrics"]["cformer_full"]["answer_head_accuracy"] for s in args.seeds]
        c_refusal = [raw[s]["stress"][scale]["metrics"]["cformer_full"]["false_refusal_rate"] for s in args.seeds]
        observer_gain = [
            raw[s]["stress"][scale]["metrics"]["cformer_full"]["accuracy_observer_belief"]
            - raw[s]["stress"][scale]["metrics"]["cformer_no_observer"]["accuracy_observer_belief"]
            for s in args.seeds
        ]
        retrieval_vs_generation = [
            raw[s]["stress"][scale]["metrics"]["cformer_retrieval_only"]["end_to_end_accuracy"]
            - raw[s]["stress"][scale]["metrics"]["cformer_generation_only"]["end_to_end_accuracy"]
            for s in args.seeds
        ]
        paired_cr = paired([c - r for c, r in zip(c_values, r_values)])
        query_overhead = [
            raw[s]["stress"][scale]["performance"]["cformer_full"]["query_36_ms"]
            / raw[s]["stress"][scale]["performance"]["rag"]["query_36_ms"]
            - 1.0
            for s in args.seeds
        ]
        scale_summary = {
            "cformer_end_to_end": describe(c_values),
            "rag_end_to_end": describe(r_values),
            "cformer_answer_head": describe(c_head),
            "cformer_false_refusal": describe(c_refusal),
            "paired_cformer_minus_rag": paired_cr,
            "observer_belief_full_minus_no_observer": paired(observer_gain),
            "retrieval_only_minus_generation_only": paired(retrieval_vs_generation),
            "query_time_overhead": describe(query_overhead),
            "cache_mb": raw[args.seeds[0]]["stress"][scale]["performance"]["cformer"]["cache_mb"],
            "per_seed_cformer_minus_rag": {
                str(seed): c_values[index] - r_values[index]
                for index, seed in enumerate(args.seeds)
            },
        }
        summary["scales"][str(scale)] = scale_summary
        scale_rows.append(
            f"| {scale:,} | {pct(scale_summary['cformer_end_to_end']['mean'])} | "
            f"{pct(scale_summary['rag_end_to_end']['mean'])} | {pct(paired_cr['mean'])} | "
            f"[{pct(paired_cr['ci95_lower'])}, {pct(paired_cr['ci95_upper'])}] | "
            f"{pct(scale_summary['cformer_answer_head']['mean'])} | {pct(scale_summary['cformer_false_refusal']['mean'])} |"
        )
        ablation_rows.append(
            f"| {scale:,} | {pct(scale_summary['observer_belief_full_minus_no_observer']['mean'])} | "
            f"{pct(scale_summary['retrieval_only_minus_generation_only']['mean'])} |"
        )
        performance_rows.append(
            f"| {scale:,} | {scale_summary['cache_mb']:.2f} MB | {pct(scale_summary['query_time_overhead']['mean'])} |"
        )
        for index, seed in enumerate(args.seeds):
            per_seed_rows.append(
                f"| {seed} | {scale:,} | {pct(c_values[index])} | {pct(r_values[index])} | "
                f"{pct(c_values[index] - r_values[index])} |"
            )

    boundary = {}
    for system in ("cformer", "rag"):
        boundary[system] = {
            metric: describe([raw[s]["boundary"][system][metric] for s in args.seeds])
            for metric in (
                "end_to_end_answer_accuracy",
                "hallucination_rate",
                "false_conflict_rate",
                "status_accuracy",
            )
        }
    summary["boundary"] = boundary

    large = summary["scales"]["32768"]
    criteria = {
        "large_world_gain_at_least_2pp": large["paired_cformer_minus_rag"]["mean"] >= 0.02 - 1e-12,
        "large_world_gain_ci_lower_above_zero": large["paired_cformer_minus_rag"]["ci95_lower"] > 0.0,
        "observer_belief_gain_at_least_3pp": large["observer_belief_full_minus_no_observer"]["mean"] >= 0.03,
        "hallucination_not_worse_than_rag": boundary["cformer"]["hallucination_rate"]["mean"] <= boundary["rag"]["hallucination_rate"]["mean"],
        "query_overhead_at_most_15pct": large["query_time_overhead"]["mean"] <= 0.15,
        "false_refusal_at_most_5pct": large["cformer_false_refusal"]["mean"] <= 0.05,
    }
    summary["acceptance"] = criteria
    summary["ready_for_v6"] = all(criteria.values())

    criteria_rows = [
        f"| {name} | {'通过' if passed else '未通过'} |" for name, passed in criteria.items()
    ]
    report = "\n".join(
        [
            "# V5.2 泛化、消融与压力测试报告",
            "",
            f"随机种子：{', '.join(map(str, args.seeds))}。每个种子双方各训练 6,300 步。压力世界包含 2,048 个有效事实；8,192/32,768 的其余条目是确定性精确副本和近键干扰项，不代表同等数量的独立知识。",
            "",
            "## 压力世界结果",
            "",
            "| 规模 | C-Former 端到端 | RAG 端到端 | 配对差 | 差值 95% CI | C-Former 答案头 | C-Former 错误拒答 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *scale_rows,
            "",
            "## 逐种子端到端结果",
            "",
            "| 种子 | 规模 | C-Former | RAG | 配对差 |",
            "|---:|---:|---:|---:|---:|",
            *per_seed_rows,
            "",
            "## 观察点消融",
            "",
            "| 规模 | belief：完整减无观察点 | 检索路径减生成路径 |",
            "|---:|---:|---:|",
            *ablation_rows,
            "",
            "## 缓存与延迟",
            "",
            "| 规模 | 双方单世界 FP32 缓存 | C-Former 相对 RAG 查询耗时 |",
            "|---:|---:|---:|",
            *performance_rows,
            "",
            "## 新世界边界",
            "",
            f"C-Former：幻觉率 {pct(boundary['cformer']['hallucination_rate']['mean'])}，误报冲突率 {pct(boundary['cformer']['false_conflict_rate']['mean'])}。",
            f"RAG：幻觉率 {pct(boundary['rag']['hallucination_rate']['mean'])}，误报冲突率 {pct(boundary['rag']['false_conflict_rate']['mean'])}。",
            "",
            "## 验收",
            "",
            "| 条件 | 结果 |",
            "|---|:---:|",
            *criteria_rows,
            "",
            f"是否可直接进入 V6：{'是' if summary['ready_for_v6'] else '否'}。",
            "",
            "机器可读汇总：`artifacts/v52_summary.json`。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"saved_summary={args.output}")
    print(f"saved_report={args.report}")


if __name__ == "__main__":
    main()
