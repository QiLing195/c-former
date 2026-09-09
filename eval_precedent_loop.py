# -*- coding: utf-8 -*-
"""toB 先例案例库闭环 demo：制度空白 → 升级 HR → 裁决沉淀案例 → 下次复用。

对应企业知识四层模型的第 3 层：人情世故不可规则化，但每次人工裁决可**记录为先例**，
让"找难搞的人问"变成"先查历史先例、带着记录去问"。

四阶段演示：
  S1 空白识别：员工问"出差2天变5天" → 制度无明文 → GAP → 升级 HR（复用 eval_rule_gap 逻辑）
  S2 HR 裁决：HR 给出裁决 → 一键生成**案例记录**（问题/情境/裁决/理由/审批人/日期）→ 入案例库
  S3 先例复用：另一员工问同类问题（不同措辞）→ 检索案例库 → 命中 → 返回"过往案例仅供参考"
  S4 无先例提示：问一个无明文且无先例的敏感问题 → 明确提示"无先例，建议当面沟通"

案例库持久化：data/precedent_cases.json（可增量积累，模拟"隐性规则显性化"过程）。

用法：D:/conda/envs/cformer-gpu/python.exe eval_precedent_loop.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE_DB = ROOT / "data" / "precedent_cases.json"
CASE_SEED = [
    # 预置历史案例（演示"已有先例"的复用）
    {
        "case_id": "C-2026-001",
        "topic": "出差超期",
        "question": "出差原定2天结果待了5天，超出报备天数怎么处理？",
        "context": "部门A员工，市场活动出差，因客户临时加需求延长3天",
        "ruling": "按实际出差核销，需补交出差变更说明并由部门经理签字，超3天部分报备总经理知情。",
        "reasoning": "制度未规定出差超期，参照请假续假精神：超3天需上级批准；念及临时性工作延长，不按旷工处理。",
        "approver": "HR经理（王）", "department": "人力资源部",
        "date": "2026-03-12", "status": "closed", "reference_only": True,
    },
]


def load_cases() -> list[dict]:
    if CASE_DB.exists():
        return json.loads(CASE_DB.read_text(encoding="utf-8"))
    return list(CASE_SEED)


def save_cases(cases: list[dict]) -> None:
    CASE_DB.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")


def search_cases(question: str, cases: list[dict]) -> list[dict]:
    """案例检索：按主题词重合（简单关键词，生产可换语义）。"""
    hits = []
    for case in cases:
        # 主题词 = topic + question 里的关键片段
        topic_kws = {"出差": "出差", "超期": "出差超期", "延长": "出差超期",
                     "忘打卡": "忘打卡", "补卡": "忘打卡", "申诉": "扣款申诉",
                     "扣款": "扣款申诉", "冤枉": "扣款申诉"}
        score = 0
        for kw, topic in topic_kws.items():
            if kw in question and kw in case["topic"] + case["question"]:
                score += 1
        if score > 0:
            hits.append((score, case))
    hits.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in hits]


def rule_lookup(question: str) -> tuple[str | None, bool]:
    """制度明文查找（简化版）：命中关键词 → 返回条款 id + 是否覆盖。
    覆盖判定用探针：这里直接映射几个已知明文/空白主题。"""
    # (问法关键词, 条款, 是否明文可答)
    KNOWN = [
        (["几点上班", "作息"], "work-hours", True),
        (["扣多少", "扣钱", "迟到一次"], "penalty", True),
        (["病假", "医院证明"], "leave", True),
    ]
    for kws, clause, covered in KNOWN:
        if any(k in question for k in kws):
            return clause, covered
    # 命中出差/忘打卡/申诉类 → 命中条款但按探针判空白
    if any(k in question for k in ["出差", "超期", "延长"]):
        return "travel", False
    if any(k in question for k in ["忘打卡", "补卡"]):
        return "punch", False
    if any(k in question for k in ["申诉", "冤枉", "扣款"]):
        return "penalty", False
    return None, False


def main() -> None:
    cases = load_cases()
    print(f"案例库现有: {len(cases)} 条 (含预置 {len(CASE_SEED)} 条)\n")

    print("=" * 70)
    print("S1 空白识别：员工问制度外问题")
    print("=" * 70)
    q1 = "出差原定2天结果待了5天，超出报备天数怎么处理？"
    clause, covered = rule_lookup(q1)
    print(f"问: {q1}")
    print(f"命中条款: {clause} | 明文覆盖: {covered}")
    if not covered:
        print("→ 判定 GAP：制度未覆盖 → 升级 HR（生成升级记录，等待裁决）")

    print("\n" + "=" * 70)
    print("S2 HR 裁决并沉淀案例")
    print("=" * 70)
    # 模拟 HR 处理该升级：产生一条裁决 → 案例化
    new_case = {
        "case_id": f"C-2026-{100 + len(cases):03d}",
        "topic": "出差超期",
        "question": q1,
        "context": "（与 C-2026-001 类似的新案例：本次为部门B员工）",
        "ruling": "同意按实际出差核销，要求补交变更说明经部门经理签字，超3天部分报总经理知情（与 C-2026-001 一致）。",
        "reasoning": "与 2026-03 先例一致：出差超期按变更说明+上级签字处理，不按旷工。",
        "approver": "HR经理（王）", "department": "人力资源部",
        "date": time.strftime("%Y-%m-%d"), "status": "closed", "reference_only": True,
    }
    cases.append(new_case)
    save_cases(cases)
    print(f"HR 裁决已生成案例 {new_case['case_id']} 并入库（裁决：{new_case['ruling'][:50]}…）")
    print(f"案例库现有: {len(cases)} 条 → 隐性规则在逐步显性化")

    print("\n" + "=" * 70)
    print("S3 先例复用：另一员工问同类问题（不同措辞）")
    print("=" * 70)
    q2 = "我出差延长了3天，超出原报备时间怎么办？"
    hits = search_cases(q2, cases)
    print(f"问: {q2}")
    if hits:
        case = hits[0]
        print(f"→ 命中过往案例 {case['case_id']}（{case['date']}，{case['approver']} 裁决）")
        print(f"  裁决: {case['ruling']}")
        print(f"  理由: {case['reasoning'][:60]}…")
        print("  ⚠️ 附注：案例仅供参考，具体以直属上级确认为准")
    else:
        print("→ 无先例命中")

    print("\n" + "=" * 70)
    print("S4 无先例风险提示：敏感且无据可依的问题")
    print("=" * 70)
    q3 = "我和部门经理关系不好，他会不会故意卡我的请假？"
    clause3, covered3 = rule_lookup(q3)
    hits3 = search_cases(q3, cases)
    print(f"问: {q3}")
    print(f"命中条款: {clause3} | 明文覆盖: {covered3} | 先例: {len(hits3)} 条")
    if not covered3 and not hits3:
        print("→ 高风险提示（AI 边界声明）：")
        print("  此问题涉及人际判断，制度无明文且无过往先例，AI 无法也不应代为判断。")
        print("  建议：1) 按制度流程正常提交请假（制度保障你的权利）；")
        print("       2) 若遇流程阻碍，可向 HR 反映，系统已记录本次查询供跟进。")

    report = {
        "case_db": str(CASE_DB),
        "total_cases": len(cases),
        "s1_question": q1, "s1_verdict": "GAP→升级HR",
        "s2_new_case": new_case["case_id"],
        "s3_question": q2, "s3_hit": hits[0]["case_id"] if hits else None,
        "s4_question": q3, "s4_verdict": "high-risk, no-precedent → boundary notice",
    }
    out = ROOT / "artifacts" / "precedent_loop_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out} | 案例库: {CASE_DB}")


if __name__ == "__main__":
    main()
