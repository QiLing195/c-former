# -*- coding: utf-8 -*-
"""GovLayer 通用性演示：同一治理框架跑 3 个域（toB 企业 / 流程 / toC 家庭）。

证明：治理逻辑（检索+字段级权限+空白识别+先例）域无关——
toB 与 toC 的差别只是 dataset 内容 + 角色定义，核心 GovLayer 零改动。

用法：
  1. D:/conda/envs/cformer-gpu/python.exe build_gov_datasets.py
  2. D:/conda/envs/cformer-gpu/python.exe build_gov_home.py
  3. D:/conda/envs/cformer-gpu/python.exe demo_govlayer.py
结果：artifacts/gov_demo_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

from cformer_v63.governance import GovLayer, build_govlayer

ROOT = Path(__file__).resolve().parent

# 每个域的演示查询：(dataset, 角色, 问题, 备注)
DEMOS = [
    # ---- toB：员工制度（三级角色）----
    ("gov_employee_rules.json", "employee",
     "迟到会扣多少钱？", "公开规则"),
    ("gov_employee_rules.json", "employee",
     "旷工超过多少天会被劝退？", "内部细则→员工应只见公开部分"),
    ("gov_employee_rules.json", "hr",
     "旷工超过多少天会被劝退？", "内部细则→HR 可见完整"),
    ("gov_employee_rules.json", "employee",
     "出差超期了，原定2天结果待了5天怎么处理？", "制度空白→先例参考"),
    # ---- toB：迎新流程（两级）----
    ("gov_registration.json", "student",
     "证件照有什么要求？", "公开"),
    ("gov_registration.json", "student",
     "照片背景不是纯蓝会被退回吗？", "内部复核标准→学生只见公开"),
    ("gov_registration.json", "admin",
     "照片背景不是纯蓝会被退回吗？", "内部复核标准→管理员可见"),
    # ---- toC：家庭（私人化定制示例）----
    ("gov_home.json", "child",
     "零花钱怎么算？考90分以上有奖励吗？", "孩子角色"),
    ("gov_home.json", "parent",
     "家庭每周采购预算是多少？", "家长角色"),
    ("gov_home.json", "child",
     "家里的储蓄账户有多少钱？", "孩子问敏感财务→应只见自己级别"),
    ("gov_home.json", "finance",
     "家里的储蓄账户有多少钱？", "财务管家→可见"),
]


def main() -> None:
    # 预构建 dataset 文件
    import build_gov_datasets  # noqa: F401
    import build_gov_home  # noqa: F401

    layers: dict[str, GovLayer] = {}
    rows = []
    for dataset_name, role, question, note in DEMOS:
        if dataset_name not in layers:
            layers[dataset_name] = build_govlayer(ROOT / "data" / dataset_name)
        layer = layers[dataset_name]
        ans = layer.answer(question, role)
        rows.append({
            "dataset": dataset_name, "role": role, "question": question,
            "note": note,
            "hit": ans.hit_ids,
            "verdict": ans.verdict,
            "visible": {k: v[:60] for k, v in ans.visible_texts.items()},
            "denied_fields": ans.denied_fields,
            "precedents": [c["case_id"] + ":" + c["ruling"][:30]
                           for c in ans.precedents],
            "boundary_note": ans.boundary_note,
        })

    report = {"n": len(rows), "rows": rows}
    out = ROOT / "artifacts" / "gov_demo_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "n": len(rows),
                      "datasets": sorted({r["dataset"] for r in rows})}, ensure_ascii=False))
    for r in rows:
        print(f"\n[{r['dataset']} | {r['role']}] {r['question']}")
        print(f"  ({r['note']}) 命中={r['hit']} 判定={r['verdict']} 隐藏字段={r['denied_fields']}")
        for k, v in r["visible"].items():
            print(f"  可见[{k}]: {v}…")
        for p in r["precedents"]:
            print(f"  先例: {p}")
        if r["boundary_note"]:
            print(f"  边界: {r['boundary_note']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
