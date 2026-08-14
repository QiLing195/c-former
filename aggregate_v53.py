from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch


SEEDS = tuple(range(201, 211))
SCALES = (2048, 8192, 32768)
T95_N10 = 2.262


def result_dir(root: Path, seed: int) -> Path:
    return root / (f"v51_seed_{seed}" if seed <= 205 else f"v52_seed_{seed}")


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def paired(values: list[float]) -> dict[str, float | bool]:
    result = describe(values)
    margin = T95_N10 * result["sample_std"] / math.sqrt(len(values))
    result["ci95_lower"] = result["mean"] - margin
    result["ci95_upper"] = result["mean"] + margin
    result["significantly_above_zero"] = result["ci95_lower"] > 0.0
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    root = Path("artifacts")
    before = {}
    after = {}
    for seed in SEEDS:
        directory = result_dir(root, seed)
        before[seed] = torch.load(directory / "v52_results.pt", weights_only=True)
        after[seed] = torch.load(directory / "v53_results.pt", weights_only=True)

    summary: dict = {"seeds": list(SEEDS), "scales": {}}
    rows = []
    performance_rows = []
    for scale in SCALES:
        old_c = [before[s]["stress"][scale]["metrics"]["cformer_full"]["end_to_end_accuracy"] for s in SEEDS]
        new_c = [after[s]["stress"][scale]["metrics"]["cformer_full"]["end_to_end_accuracy"] for s in SEEDS]
        old_r = [before[s]["stress"][scale]["metrics"]["rag"]["end_to_end_accuracy"] for s in SEEDS]
        new_r = [after[s]["stress"][scale]["metrics"]["rag"]["end_to_end_accuracy"] for s in SEEDS]
        false_refusal = [after[s]["stress"][scale]["metrics"]["cformer_full"]["false_refusal_rate"] for s in SEEDS]
        c_query = [after[s]["stress"][scale]["performance"]["cformer_full"]["query_36_ms"] for s in SEEDS]
        r_query = [after[s]["stress"][scale]["performance"]["rag"]["query_36_ms"] for s in SEEDS]
        query_overhead = [c / r - 1.0 for c, r in zip(c_query, r_query)]
        c_minus_r = paired([c - r for c, r in zip(new_c, new_r)])
        improvement = paired([new - old for new, old in zip(new_c, old_c)])
        scale_summary = {
            "before_cformer": describe(old_c),
            "after_cformer": describe(new_c),
            "before_rag": describe(old_r),
            "after_rag": describe(new_r),
            "cformer_improvement": improvement,
            "after_cformer_minus_rag": c_minus_r,
            "after_cformer_false_refusal": describe(false_refusal),
            "cformer_query_36_ms": describe(c_query),
            "rag_query_36_ms": describe(r_query),
            "query_overhead": describe(query_overhead),
            "cache_mb": after[SEEDS[0]]["stress"][scale]["performance"]["cformer"]["cache_mb"],
        }
        summary["scales"][str(scale)] = scale_summary
        rows.append(
            f"| {scale:,} | {pct(scale_summary['before_cformer']['mean'])} | "
            f"{pct(scale_summary['after_cformer']['mean'])} | {pct(improvement['mean'])} | "
            f"{pct(scale_summary['after_rag']['mean'])} | {pct(c_minus_r['mean'])} | "
            f"[{pct(c_minus_r['ci95_lower'])}, {pct(c_minus_r['ci95_upper'])}] | "
            f"{pct(scale_summary['after_cformer_false_refusal']['mean'])} |"
        )
        performance_rows.append(
            f"| {scale:,} | {scale_summary['cache_mb']:.2f} MB | "
            f"{scale_summary['cformer_query_36_ms']['mean']:.2f} ms | "
            f"{scale_summary['rag_query_36_ms']['mean']:.2f} ms | "
            f"{pct(scale_summary['query_overhead']['mean'])} |"
        )

    boundary = {}
    for system in ("cformer", "rag"):
        boundary[system] = {
            metric: describe([after[s]["boundary"][system][metric] for s in SEEDS])
            for metric in ("hallucination_rate", "false_conflict_rate", "status_accuracy")
        }
    summary["boundary"] = boundary
    small = summary["scales"]["2048"]
    large = summary["scales"]["32768"]
    criteria = {
        "32k_accuracy_drop_at_most_2pp": large["after_cformer"]["mean"] >= small["after_cformer"]["mean"] - 0.02,
        "32k_gain_over_rag_at_least_2pp": large["after_cformer_minus_rag"]["mean"] >= 0.02,
        "32k_gain_ci_lower_above_zero": large["after_cformer_minus_rag"]["ci95_lower"] > 0.0,
        "32k_false_refusal_at_most_5pct": large["after_cformer_false_refusal"]["mean"] <= 0.05,
        "hallucination_not_worse_than_rag": boundary["cformer"]["hallucination_rate"]["mean"] <= boundary["rag"]["hallucination_rate"]["mean"],
        "query_overhead_at_most_15pct": large["query_overhead"]["mean"] <= 0.15,
    }
    summary["acceptance"] = criteria
    summary["all_gates_pass"] = all(criteria.values())
    criteria_rows = [f"| {name} | {'通过' if passed else '未通过'} |" for name, passed in criteria.items()]

    report = "\n".join(
        [
            "# V5.3 等价证据可靠性修复报告",
            "",
            "修复不增加参数：可靠性间隔从“最高分减第二高分”改为“最高证据减 Top-8 中最佳非等价证据”。重复的同一事实被视为一致支持，不再被误判为冲突或低置信度。使用 V5.2 的 10 个冻结检查点重新评测，没有重新训练。",
            "",
            "## 修复前后",
            "",
            "| 规模 | 修复前 C-Former | 修复后 C-Former | 提升 | 修复后 RAG | C-Former-RAG | 差值 95% CI | 错误拒答 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 缓存与性能",
            "",
            "| 规模 | FP32 缓存 | C-Former 36 查询 | RAG 36 查询 | 相对开销 |",
            "|---:|---:|---:|---:|---:|",
            *performance_rows,
            "",
            "## 边界安全",
            "",
            f"C-Former 幻觉率 {pct(boundary['cformer']['hallucination_rate']['mean'])}，误报冲突率 {pct(boundary['cformer']['false_conflict_rate']['mean'])}；RAG 对应指标均为 {pct(boundary['rag']['hallucination_rate']['mean'])}。",
            "",
            "## 验收",
            "",
            "| 条件 | 结果 |",
            "|---|:---:|",
            *criteria_rows,
            "",
            f"全部门槛通过：{'是' if summary['all_gates_pass'] else '否'}。",
            "",
            "注意：8K/32K 仍是 2,048 个有效事实加精确副本和近键干扰事实的压力测试，不是同等数量的独立知识。",
        ]
    )
    Path("artifacts/v53_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path("V53_FIX_REPORT.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
