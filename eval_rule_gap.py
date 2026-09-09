# -*- coding: utf-8 -*-
"""toB 制度空白识别测试：明文可答 vs 制度空白（需升级 HR）。

痛点背景：企业真实提问大量是"制度没写"的边界情况（出差2天变5天、
忘打卡补卡、特殊人群安排）——答案在"难搞的人"手里，不在文档里。
AI 若乱编会毁信任；正确行为是**诚实识别空白并升级**。

判定逻辑（POC 用词表探针，生产可换 LLM 判定）：
  1. 关键词检索命中条款（travel/leave/...）；
  2. 检查条款内容是否覆盖问题诉求：问法标注 answer_probe（答案应出现的词/短语）；
     - 条款内容含 probe → 明文可答（COVERED）
     - 不含 → 制度空白（GAP → NEEDS_ESCALATION：升级 HR + 生成升级记录）
  3. 无条款命中 → OUT_OF_SCOPE（不在制度范围，直接升级）

升级记录（模拟决策留痕，为"隐性规则沉淀"铺垫）：每条 GAP 生成一条
escalation record：{问题, 命中条款, 缺失主题, 升级对象:HR, 状态:pending}。

用法：D:/conda/envs/cformer-gpu/python.exe eval_rule_gap.py
结果：artifacts/rule_gap_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 条款库（复用复杂场景结构：id/标题/levels[0]=员工可见公开内容/关键词）
CLAUSES = [
    {"id": "work-hours", "title": "作息时间",
     "levels": {0: "公司上班时间8:30-12:00，13:00-17:30，一般每天8小时、每周6天。"},
     "keywords": ["上班时间", "作息", "几点", "下班", "8:30"]},
    {"id": "punch", "title": "考勤打卡",
     "levels": {0: "实行视频考勤，每天上下班必须视频登记；忘记打卡需说明情况并留存记录。"},
     "keywords": ["打卡", "视频考勤", "忘打卡", "漏打卡", "补卡"]},
    {"id": "leave", "title": "请假",
     "levels": {0: "请假需填请假条注明原因天数，经理签字后休息；病假需县市级医院证明；事假填请假单经批准。"},
     "keywords": ["请假", "病假", "事假", "医院证明", "批准"]},
    {"id": "return", "title": "销假续假",
     "levels": {0: "假期内返回需到考勤员处销假；不及时销假按缺勤/旷工处理。"},
     "keywords": ["销假", "续假", "返岗"]},
    {"id": "travel", "title": "出差外勤",
     "levels": {0: "出差需出差前报备；外勤为全天在外办事；调休需提交调休申请。"},
     "keywords": ["出差", "外勤", "调休", "报备"]},
    {"id": "penalty", "title": "考勤处罚",
     "levels": {0: "迟到早退每次扣日工资50元；因公外出或经部门经理书面证明除外。"},
     "keywords": ["迟到", "早退", "扣", "旷工", "劝退", "50"]},
]

# 测试：(问法, 期望命中条款, answer_probe[答案应出现的词], 期望判定)
# expect: covered(明文可答) / gap(制度空白) / out_of_scope(不属制度)
TESTS = [
    # ---- 明文可答（制度写了）----
    ("我们几点上班？", "work-hours", ["8:30"], "covered"),
    ("迟到一次扣多少钱？", "penalty", ["50"], "covered"),
    ("请病假要医院证明吗？", "leave", ["医院证明"], "covered"),
    ("忘打卡要不要说明？", "punch", ["说明"], "covered"),
    ("出差前要报备吗？", "travel", ["报备"], "covered"),
    # ---- 制度空白（命中条款但没写答案）----
    ("出差原定2天结果待了5天，超出报备天数怎么处理？", "travel", ["变更", "延期", "超期", "延长"], "gap"),
    ("忘打卡了，补卡之后还算迟到吗？", "punch", ["补卡", "补签", "不算迟到"], "gap"),
    ("请假超期没回来，能不能算自动续假？", "return", ["自动续假", "超期自动"], "gap"),
    ("孕妇产检请假有没有特殊安排？", "leave", ["产检", "孕妇", "特殊"], "gap"),
    ("迟到被扣了50元，觉得冤枉能申诉吗？", "penalty", ["申诉", "复议", "异议"], "gap"),
    # ---- 制度范围外（不在任何条款）----
    ("离职当月社保怎么交？", None, ["社保", "离职"], "out_of_scope"),
    ("加班有加班费吗？", None, ["加班费", "加班"], "out_of_scope"),
]


def retrieve(text: str) -> str | None:
    best_id, best_hits = None, 0
    for c in CLAUSES:
        hits = sum(1 for kw in c["keywords"] if kw in text)
        if hits > best_hits:
            best_hits, best_id = hits, c["id"]
    return best_id if best_hits > 0 else None


def main() -> None:
    rows = []
    for question, expect_clause, probes, expect in TESTS:
        hit_id = retrieve(question)
        hit = next((c for c in CLAUSES if c["id"] == hit_id), None) if hit_id else None

        if hit is None:
            verdict = "out_of_scope"
        else:
            content = hit["levels"].get(0, "")
            covered = any(probe in content for probe in probes)
            verdict = "covered" if covered else "gap"

        correct = (verdict == expect)
        escalation = None
        if verdict in ("gap", "out_of_scope"):
            escalation = {
                "question": question,
                "hit_clause": hit_id,
                "missing_topic": ", ".join(probes),
                "escalate_to": "HR",
                "status": "pending",
                "note": "制度未覆盖，需人工裁决并留痕沉淀为案例",
            }

        rows.append({
            "question": question,
            "expect": expect,
            "verdict": verdict,
            "correct": bool(correct),
            "hit_clause": hit_id,
            "escalation": escalation,
        })

    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    # 分类准确率
    by_type = {}
    for r in rows:
        by_type.setdefault(r["expect"], {"n": 0, "ok": 0})
        by_type[r["expect"]]["n"] += 1
        by_type[r["expect"]]["ok"] += 1 if r["correct"] else 0
    by_type_rate = {k: round(v["ok"] / v["n"], 3) for k, v in by_type.items()}

    report = {
        "n": n,
        "overall_accuracy": round(correct / n, 3),
        "by_type": by_type_rate,
        "escalations_generated": sum(1 for r in rows if r["escalation"]),
        "rows": rows,
    }
    out = ROOT / "artifacts" / "rule_gap_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "overall_accuracy": report["overall_accuracy"],
                      "by_type": by_type_rate,
                      "escalations": report["escalations_generated"]}, ensure_ascii=False))
    for r in rows:
        mark = "OK " if r["correct"] else "XX "
        esc = f" → 升级HR[{r['escalation']['missing_topic']}]" if r["escalation"] else ""
        print(f"{mark}期望={r['expect']:11s} 判定={r['verdict']:11s} 命中={r['hit_clause']} | {r['question']}{esc}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
