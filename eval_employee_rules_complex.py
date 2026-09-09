# -*- coding: utf-8 -*-
"""toB 复杂场景验证：三级角色权限 + 组合/含噪/边界问法 × C-Former 字段级闸门。

角色（三级，模拟真实企业）：
  - employee  普通员工：公开规则
  - manager   部门经理：+ 批准权限细则（3天内批准/销假处理）
  - hr        HR/总经理：+ 处罚额度/劝退标准/高级职员豁免

复杂问法类型：
  A. 组合问题：一问多条款（请假流程+证明+批准）
  B. 多级权限：同一问题不同角色答案不同（旷工处罚）
  C. 含噪口语：忘打卡/算不算旷工/会不会劝退
  D. 边界判断：病假事假扣钱一样吗 / 迟到半小时算旷工吗

字段级权限模型：每条款的公开/经理/HR 三个级别内容，角色可见 = 该角色级别及以下。
用法：D:/conda/envs/cformer-gpu/python.exe eval_employee_rules_complex.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 三级字段内容（来自真实制度文档）
ROLE_LEVEL = {"employee": 0, "manager": 1, "hr": 2}

CLAUSES = [
    {
        "id": "work-hours",
        "title": "作息时间",
        "levels": {
            0: "公司上班时间8:30-12:00，13:00-17:30，一般每天8小时、每周6天。",
        },
        "keywords": ["上班时间", "作息", "几点", "下班", "8:30", "工作时间"],
    },
    {
        "id": "punch",
        "title": "考勤打卡",
        "levels": {
            0: "实行视频考勤，每天上下班必须视频登记；忘记打卡需说明情况并留存记录。",
            1: "忘记打卡的说明记录由部门经理核验；频繁漏打卡需面谈提醒。",
        },
        "keywords": ["打卡", "视频考勤", "忘打卡", "漏打卡"],
    },
    {
        "id": "leave",
        "title": "请假",
        "levels": {
            0: "请假需填请假条注明原因天数，经理签字后休息；病假需县市级医院证明；事假填请假单经批准。",
            1: "批准权限：一般员工请假3天内直接上级批准，3天以上报本部门经理；部门经理请假需总经理批准。",
            2: "请假记录作为工资发放依据存放考勤员处；HR 定期抽查请假条与医院证明一致性。",
        },
        "keywords": ["请假", "病假", "事假", "请假条", "休假", "医院证明", "批准", "请假单"],
    },
    {
        "id": "return",
        "title": "销假续假",
        "levels": {
            0: "假期内返回需到考勤员处销假；不及时销假按缺勤/旷工处理；假期结束不能返岗需联系考勤员和主管申请续假。",
        },
        "keywords": ["销假", "续假", "返岗", "假期结束", "回来"],
    },
    {
        "id": "travel",
        "title": "出差外勤调休",
        "levels": {
            0: "出差需出差前报备；外勤为全天在外办事；调休需提交调休申请。",
        },
        "keywords": ["出差", "外勤", "调休", "报备"],
    },
    {
        "id": "penalty",
        "title": "考勤处罚",
        "levels": {
            0: "迟到早退每次扣日工资50元；因公外出或经部门经理书面证明除外。",
            1: "部门经理对本部门考勤处罚有复核权，可对书面证明情况豁免扣款。",
            2: "旷工不发薪资津贴并按天处罚；当月旷工5天或全年累计7天予以劝退；迟到超30分钟未到岗按旷工处理；高级职员（总经理/副总）不考勤。",
        },
        "keywords": ["迟到", "早退", "扣", "旷工", "劝退", "处罚", "扣工资", "罚款", "缺勤", "30分钟"],
    },
    {
        "id": "stats",
        "title": "考勤统计",
        "levels": {
            0: "每月一个考勤周期，各部门每月固定日前上报考勤统计表。",
            2: "考勤统计由部门负责人全权负责，HR 汇总核查。",
        },
        "keywords": ["考勤统计", "上报", "考勤周期", "考勤表"],
    },
]

# 复杂问题：(问法, 期望条款列表, 期望角色可见级别上限, 问法类型)
QUESTIONS = [
    # A. 组合问题（一问多条款）
    ("请假流程怎么走？要医院证明吗？超过3天找谁批？",
     ["leave"], 1, "A-combo"),
    # B. 多级权限（同问不同角色答案不同）
    ("旷工会怎么处罚？", ["penalty"], 0, "B-multilevel"),
    ("旷工怎么处罚？会被劝退吗？", ["penalty"], 2, "B-multilevel"),
    # C. 含噪口语
    ("我上周忘打卡了，会算旷工吗？", ["punch", "penalty"], 0, "C-noisy"),
    ("迟到半小时没到岗，会不会被劝退啊？", ["penalty"], 2, "C-noisy"),
    # D. 边界判断
    ("病假和事假扣钱一样吗？", ["leave", "penalty"], 0, "D-boundary"),
    ("旷工一天和迟到扣款有什么区别？", ["penalty"], 0, "D-boundary"),
    # 公开常规
    ("几点上班几点下班？", ["work-hours"], 0, "public"),
    ("出差要不要提前报备？", ["travel"], 0, "public"),
    ("销假晚了会怎么样？", ["return"], 0, "public"),
]


def retrieve(text: str, clauses: list[dict]) -> list[str]:
    """关键词检索：按命中数排序，返回命中>0的条款 id（可多条款=组合问题）。"""
    scored = []
    for c in clauses:
        hits = sum(1 for kw in c["keywords"] if kw in text)
        if hits > 0:
            scored.append((hits, c["id"]))
    scored.sort(reverse=True)
    return [cid for _, cid in scored]


def visible_content(clause: dict, role_level: int) -> str:
    """字段级权限：返回 role_level 及以下级别的字段内容。"""
    parts = []
    for level in range(role_level + 1):
        if level in clause["levels"]:
            parts.append(clause["levels"][level])
    return " ".join(parts)


def main() -> None:
    rows = []
    for question, expect_clauses, expect_level, qtype in QUESTIONS:
        hits = retrieve(question, CLAUSES)
        # 期望条款是否都被检索到（组合问题要求多命中）
        expect_set = set(expect_clauses)
        hit_set = set(hits)
        retrieved_expected = expect_set.issubset(hit_set)

        # 对每个角色展示可见内容
        role_views = {}
        for role, level in ROLE_LEVEL.items():
            role_views[role] = {}
            for cid in hit_set:
                clause = next(c for c in CLAUSES if c["id"] == cid)
                role_views[role][cid] = visible_content(clause, level)

        # 判定：检索到期望条款 = 检索 OK；权限分级是否生效（级别越高内容越全）
        higher_has_more = True
        for cid in expect_set & hit_set:
            clause = next(c for c in CLAUSES if c["id"] == cid)
            len_e = len(visible_content(clause, 0))
            len_h = len(visible_content(clause, 2))
            if len_h <= len_e and clause.get("levels", {}).get(2):
                higher_has_more = False

        rows.append({
            "type": qtype,
            "question": question,
            "expect_clauses": expect_clauses,
            "hit_clauses": hits,
            "retrieved_expected": bool(retrieved_expected),
            "role_views": {role: {cid: v[:60] for cid, v in views.items()}
                           for role, views in role_views.items()},
            "permission_graded": bool(higher_has_more),
        })

    n = len(rows)
    retrieval_ok = sum(1 for r in rows if r["retrieved_expected"])
    graded_ok = sum(1 for r in rows if r["permission_graded"])
    report = {
        "n_questions": n,
        "retrieval_ok": retrieval_ok,
        "permission_graded_ok": graded_ok,
        "rows": rows,
    }
    out = ROOT / "artifacts" / "employee_rules_complex_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "n": n, "retrieval_ok": retrieval_ok,
                      "permission_graded_ok": graded_ok}, ensure_ascii=False))
    for r in rows:
        mark = "OK " if r["retrieved_expected"] else "XX "
        print(f"{mark}[{r['type']}] {r['question']}")
        print(f"   命中={r['hit_clauses']} 期望={r['expect_clauses']} 分级生效={r['permission_graded']}")
        for role in ("employee", "manager", "hr"):
            view = " | ".join(f"{k}:{v[:40]}" for k, v in r["role_views"][role].items())
            print(f"   {role:9s}: {view[:120]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
