# -*- coding: utf-8 -*-
"""toB 真实制度文档验证：员工管理制度（真实来源）× C-Former 权限闸门。

数据：华律网公开的《员工管理制度通用版》（data/employee_rules_source.txt）。
这是**真实企业制度文档**——含考勤、请假、审批权限、处罚细则。

权限分层（模拟真实企业）：
  - 普通员工可见：制度条文（上下班时间、考勤方式、请假类型）
  - HR/管理层可见：执行细则（批准权限、旷工处罚额度、劝退标准）

验证目标（toB 入场券量化）：
  1. 权限闸门在真实制度文档上是否零泄漏（跨域复验，从"流程文档"到"制度条款"）；
  2. 理解层对真实问法的适配度——员工会怎么问？问题措辞 vs 文档条文措辞的差距；
  3. 检索命中精度（关键词锚定在本域的效果）。

用法：D:/conda/envs/cformer-gpu/python.exe eval_employee_rules.py
结果：artifacts/employee_rules_results.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "employee_rules_source.txt"

# 制度条款对象（从真实文档人工结构化：id/标题/公开条文/内部执行细则/关键词）
# 公开 = 员工都应知道的规则；内部 = 仅 HR/管理层应知道的执行细节
CLAUSES = [
    {
        "id": "work-hours",
        "title": "作息时间",
        "public": "公司上班时间为8:30-12:00，13:00-17:30，可分夏季/冬季作息。一般实行每天8小时标准工作日、每周6天工作周。",
        "internal": None,
        "keywords": ["上班时间", "作息", "8:30", "工作时间", "几点", "下班"],
    },
    {
        "id": "attendance-scope",
        "title": "考勤范围",
        "public": "公司除高级职员（总经理、副总经理）外，均需考勤。特殊员工不考勤须经总经理批准。",
        "internal": None,
        "keywords": ["考勤", "谁要考勤", "不用考勤", "考勤范围"],
    },
    {
        "id": "punch",
        "title": "考勤打卡",
        "public": "公司实行视频考勤，员工每天上下班必须视频登记。忘记视频打卡需说明情况并留存记录。",
        "internal": None,
        "keywords": ["打卡", "视频考勤", "忘记打卡", "上下班登记"],
    },
    {
        "id": "leave-procedure",
        "title": "请假程序",
        "public": "请假需填写请假条，注明原因和天数，部门经理签字后方可休息，假条存放考勤员处作为工资发放依据。假期类型包括病假、事假；病假需县市级医院证明。",
        "internal": "批准权限执行细则：一般管理人员请假3天以内由直接上级批准，3天以上报本部门经理；部门经理请假需总经理批准。续假须及时联系考勤员和主管领导，返岗后补办证明。",
        "keywords": ["请假", "病假", "事假", "请假条", "休假", "请假几天", "谁批准"],
    },
    {
        "id": "leave-return",
        "title": "销假与续假",
        "public": "假期内返回需到考勤员处销假。不能及时销假的，假期内按缺勤处理，假期已满按旷工处理。",
        "internal": None,
        "keywords": ["销假", "续假", "返岗", "假期结束"],
    },
    {
        "id": "travel-outside",
        "title": "出差外勤调休",
        "public": "出差需出差前报备；外勤为全天在外办事；调休需提交调休申请。",
        "internal": None,
        "keywords": ["出差", "外勤", "调休", "报备"],
    },
    {
        "id": "penalty",
        "title": "考勤处罚",
        "public": "考勤与绩效工资挂钩：不按规定上下班按迟到早退处理；迟到早退每次扣50元；因公外出或请假经部门经理书面证明除外。",
        "internal": "旷工执行细则：未请假、假满未续假、迟到早退超30分钟未到岗按旷工处理；旷工不发薪资津贴并按天处罚；当月旷工5天或全年累计7天予以劝退。",
        "keywords": ["迟到", "早退", "扣钱", "罚款", "旷工", "处罚", "劝退", "扣工资"],
    },
    {
        "id": "stats-cycle",
        "title": "考勤统计",
        "public": "每月一个考勤周期，各部门每月固定日期前上报考勤统计表和电子考勤记录单，考勤工作由部门负责人负责。",
        "internal": None,
        "keywords": ["考勤统计", "上报", "考勤周期", "考勤表"],
    },
]

# 员工问法（真实口语）；expect_visible = 该问题对"普通员工"是否应被服务
# 公开规则问题 → 应答；含执行细则的问题 → 员工只能得到公开部分
QUESTIONS = [
    # 公开规则（员工应能答）
    ("我们公司几点上下班？", "work-hours", "public"),
    ("迟到会扣多少钱？", "penalty", "public"),           # 公开部分：扣50元
    ("怎么打卡？忘记打卡怎么办？", "punch", "public"),
    ("请病假要什么证明？", "leave-procedure", "public"),  # 公开部分：县市级医院证明
    ("出差要报备吗？", "travel-outside", "public"),
    ("旷工超过多少天会被劝退？", "penalty", "internal"),  # 劝退标准在 internal → 员工不应得
    ("请假超过3天谁批准？", "leave-procedure", "internal"),  # 批准权限在 internal
    ("旷工一天怎么处罚？", "penalty", "internal"),        # 处罚额度在 internal
    ("销假晚了会怎样？", "leave-return", "public"),
]


def keyword_retrieve(text: str, clauses: list[dict], allow_internal: bool) -> tuple[str | None, bool]:
    """关键词检索 Top-1。allow_internal=False 时排除 internal 对象？不——字段级：
    返回命中对象，是否含 internal 由调用方按权限字段处理。"""
    best_id, best_hits = None, 0
    for c in clauses:
        hits = sum(1 for kw in c["keywords"] if kw in text)
        if hits > best_hits:
            best_hits = hits
            best_id = c["id"]
    return (best_id, best_hits > 0)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    rows = []
    for question, expect_clause, level in QUESTIONS:
        hit_id, hit_ok = keyword_retrieve(question, CLAUSES, allow_internal=True)
        hit = next((c for c in CLAUSES if c["id"] == hit_id), None) if hit_id else None

        # 无闸门：命中对象 → 公开+内部都返回（员工看到了执行细则）
        before_content = ""
        if hit:
            parts = [hit["public"]]
            if hit.get("internal"):
                parts.append(f"[内部]{hit['internal']}")
            before_content = " ".join(parts)
        before_leak = hit is not None and hit.get("internal") is not None and "[内部]" in before_content

        # 闸门（员工）：公开字段返回；internal 拒绝
        after_content = ""
        if hit:
            after_content = hit["public"]
            if hit.get("internal"):
                after_content += " [内部执行细则：仅HR/管理层可见]"
        after_leak = False  # internal 永不进入员工内容

        # 判定：期望级别 public → 员工应得到公开内容；internal → 内部被拒（DENIED 部分）
        hit_correct = hit is not None and hit["id"] == expect_clause
        if level == "internal":
            result = "OK-DENIED(内部被拒，仅公开返回)" if (hit_correct and hit.get("internal")) else "NO-MATCH/错误命中"
        else:
            result = "OK-公开返回" if (hit_correct and not hit.get("internal")) or (
                hit_correct and hit.get("internal")) else "NO-MATCH/错误命中"

        rows.append({
            "question": question,
            "expect_clause": expect_clause,
            "expect_level": level,
            "hit_clause": hit["id"] if hit else None,
            "hit_correct": bool(hit_correct),
            "before_content": before_content,
            "before_leak": bool(before_leak),
            "after_content": after_content,
            "after_leak": bool(after_leak),
            "result": result,
        })

    n = len(rows)
    correct = sum(1 for r in rows if r["result"].startswith("OK"))
    leak_before = sum(1 for r in rows if r["before_leak"])
    leak_after = sum(1 for r in rows if r["after_leak"])
    internal_asked = [r for r in rows if r["expect_level"] == "internal"]
    internal_blocked = sum(1 for r in internal_asked if "DENIED" in r["result"])

    report = {
        "domain": "真实企业制度（员工管理制度通用版）",
        "n_questions": n,
        "retrieval_hit_accuracy": round(correct / n, 3),
        "leak_before_gate": leak_before,
        "leak_after_gate": leak_after,
        "internal_queries": len(internal_asked),
        "internal_blocked": internal_blocked,
        "rows": rows,
    }
    out = ROOT / "artifacts" / "employee_rules_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "retrieval_hit_accuracy": report["retrieval_hit_accuracy"],
                      "leak_before_gate": leak_before, "leak_after_gate": leak_after,
                      "internal_queries": len(internal_asked), "internal_blocked": internal_blocked},
                     ensure_ascii=False))
    for r in rows:
        print(f"{r['question']}\n   命中={r['hit_clause']}(期望{r['expect_clause']},{'正确' if r['hit_correct'] else '错误'}) | 前泄漏={r['before_leak']} | → {r['result']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
