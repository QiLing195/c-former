# -*- coding: utf-8 -*-
"""检索层技术升级验证：关键词锚定 vs LLM 语义检索（#1）+ 覆盖判定（#2）。

技术问题：
  #1 关键词无语义——用户问"什么时候会被开除"，条款写"予以劝退" → 关键词 miss；
  #2 空白识别靠人工 probe_rules——每域手写规则，脆且不通用。

本脚本用**措辞远离条款**的真实问法，对照两种模式：
  A. 关键词模式（无 LLM）：检索靠词面重合 + probe 词表判定覆盖；
  B. 语义模式（LLM）：LLM 判断哪些条款语义相关 + 是否明确回答了问题。

用法（需要 DeepSeek API key 才能跑语义模式）：
    $env:DEEPSEEK_API_KEY = "sk-..."
    D:/conda/envs/cformer-gpu/python.exe eval_semantic_gov.py
结果：artifacts/semantic_gov_results.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cformer_v63.governance import build_govlayer
from cformer_v63.semantic import LLMSemanticBackend

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gov_employee_rules.json"

# 测试问法：(问题, 期望命中的条款 id 或 None, 期望判定, 说明)
# 注意：多数问法**刻意避开条款原词**，用来暴露关键词检索的短板；
# 期望判定已考虑「员工角色」的权限截断（restricted = 制度有写但员工无权限）。
CASES = [
    # 措辞远离条款（关键词模式预期 miss，语义模式应命中）
    ("什么时候会被开除？", "penalty", "restricted", "条款写'予以劝退'；劝退标准仅HR可见→员工应得restricted"),
    ("上班时间是几点到几点？", "work-hours", "covered", "含'上班时间'（关键词也应命中，作对照）"),
    ("生病了要交什么材料？", "leave", "covered", "条款写'病假需医院证明'，无'生病/材料'原词"),
    ("我要出去办点私事，怎么走流程？", "leave", "covered", "口语化'私事'≈事假，措辞远"),
    ("早上晚到了会有什么后果？", "penalty", "covered", "'晚到'≈迟到，无原词"),
    ("连着好几天不来上班会怎样？", "penalty", "restricted", "'不来上班'≈旷工；旷工处罚细则仅HR可见→restricted"),
    # 制度空白（应判 gap/out_of_scope，不能编造）
    ("公司给员工配股票吗？", None, "no_answer", "制度未提及股权激励"),
    ("在家办公怎么申请？", None, "no_answer", "制度未提及远程办公"),
    # 权限相关的内部问题（员工角色）
    ("劝退的具体标准是什么？", "penalty", "restricted", "命中条款但内部字段对员工截断→restricted"),
]


def run_mode(name: str, backend) -> dict:
    layer = build_govlayer(DATA, semantic_backend=backend)
    rows = []
    hit_ok = verdict_ok = 0
    for question, expect_clause, expect_verdict, note in CASES:
        ans = layer.answer(question, "employee")
        # 命中判定：期望条款在命中列表里（None 表示期望"无相关条款"）
        if expect_clause is None:
            clause_hit = len(ans.hit_ids) == 0 or ans.verdict in ("gap", "out_of_scope")
        else:
            clause_hit = expect_clause in ans.hit_ids
        # 判定命中：no_answer 表示"不可回答"（gap/out_of_scope/restricted 任一皆可，核心是不编造）
        if expect_verdict == "no_answer":
            verdict_hit = ans.verdict in ("gap", "out_of_scope", "restricted")
        else:
            verdict_hit = (ans.verdict == expect_verdict)
        hit_ok += 1 if clause_hit else 0
        verdict_ok += 1 if verdict_hit else 0
        rows.append({
            "question": question, "note": note,
            "expect_clause": expect_clause, "got_hits": ans.hit_ids,
            "expect_verdict": expect_verdict, "got_verdict": ans.verdict,
            "clause_hit_ok": bool(clause_hit), "verdict_ok": bool(verdict_hit),
            "gap_reason": ans.gap_reason,
            "visible": {k: v[:50] for k, v in ans.visible_texts.items()},
        })
    return {"mode": name, "n": len(CASES),
            "clause_hit_accuracy": round(hit_ok / len(CASES), 3),
            "verdict_accuracy": round(verdict_ok / len(CASES), 3),
            "rows": rows}


def main() -> None:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    report = {"cases": len(CASES)}

    print("=" * 72)
    print("A. 关键词模式（无 LLM）")
    print("=" * 72)
    kw = run_mode("keyword", None)
    report["keyword"] = {k: v for k, v in kw.items() if k != "rows"}
    print(f"条款命中率 {kw['clause_hit_accuracy']} | 判定准确率 {kw['verdict_accuracy']}")
    for r in kw["rows"]:
        mark = "OK " if (r["clause_hit_ok"] and r["verdict_ok"]) else "XX "
        print(f"{mark}{r['question']}  → 命中{r['got_hits']} 判定{r['got_verdict']}  ({r['note']})")

    if not key:
        print("\n[跳过语义模式] 未设置 DEEPSEEK_API_KEY")
        report["semantic"] = None
    else:
        print("\n" + "=" * 72)
        print("B. 语义模式（LLM 检索 + 覆盖判定）")
        print("=" * 72)
        backend = LLMSemanticBackend()
        sem = run_mode("semantic", backend)
        report["semantic"] = {k: v for k, v in sem.items() if k != "rows"}
        report["semantic"]["llm_calls"] = backend.calls
        report["semantic"]["cache_hits"] = backend.cache_hits
        print(f"条款命中率 {sem['clause_hit_accuracy']} | 判定准确率 {sem['verdict_accuracy']} "
              f"| LLM 调用 {backend.calls} 次")
        for r in sem["rows"]:
            mark = "OK " if (r["clause_hit_ok"] and r["verdict_ok"]) else "XX "
            extra = f" | 缺:{r['gap_reason']}" if r["gap_reason"] else ""
            print(f"{mark}{r['question']}  → 命中{r['got_hits']} 判定{r['got_verdict']}{extra}  ({r['note']})")

    out = ROOT / "artifacts" / "semantic_gov_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report, "keyword_rows": kw["rows"],
                               "semantic_rows": (sem["rows"] if key else [])},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
