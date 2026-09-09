# -*- coding: utf-8 -*-
"""toC 示例 dataset：家庭共享知识 + 个人角色权限。

证明 GovLayer 的通用性：toB（企业制度）与 toC（个人知识）共用同一治理框架，
差异只是 dataset 内容与角色定义。

家庭场景：
  - 成员（孩子）：可见 0 级（日常家规/零花钱规则）
  - 家长：可见 1 级（+ 预算/采购明细）
  - 财务管家（家长2）：可见 2 级（+ 账户/储蓄/敏感决策）

用法：作为 dataset 被 demo_govlayer.py 加载；也可单独 build 写盘。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build() -> dict:
    objects = [
        {"id": "house-rules", "title": "家庭日常规则",
         "keywords": ["家务", "作息", "零花钱", "规则", "手机", "作业"],
         "levels": {
             0: "家务分工：孩子负责自己房间整理与倒垃圾；周一到周五晚9点前完成作业可看30分钟动画。",
             1: "零花钱规则：每周50元，成绩单科≥90 加 10 元；家务额外任务按次计酬。",
         }},
        {"id": "shopping", "title": "家庭采购",
         "keywords": ["采购", "购物", "买菜", "超市", "预算"],
         "levels": {
             0: "每周日家庭采购日，孩子可列一个心愿零食（预算内）。",
             1: "采购预算每周 800 元，超支需记录原因；大宗采购（>500）需双方确认。",
             2: "家庭月度总预算 3500 元；储蓄目标每月 1000 元，采购不得挤占储蓄。",
         }},
        {"id": "vacation", "title": "假期计划",
         "keywords": ["假期", "旅行", "夏令营", "出游"],
         "levels": {
             0: "假期安排会提前两周和大家商量；孩子可选择一次兴趣活动。",
             1: "假期预算与行程由家长共同规划；孩子独自外出需报备去向与同伴。",
         }},
        {"id": "savings", "title": "家庭财务",
         "keywords": ["储蓄", "账户", "存款", "学费", "大额"],
         "levels": {
             2: "家庭应急账户余额、教育储蓄计划、投资明细为敏感信息，仅财务管家可见。",
         }},
    ]
    roles = {"child": 0, "parent": 1, "finance": 2}
    probe_rules = [
        (["零花钱", "家务"], ["零花钱", "家务"]),
        (["作息", "作业", "手机"], ["作业", "动画"]),
        (["采购", "买菜", "购物"], ["预算"]),
        (["假期", "旅行"], ["假期"]),
    ]
    precedent_cases = [
        {"case_id": "C-H-001", "topic": "零花钱争议",
         "question": "孩子考了95分但家务没做完，能拿奖励吗？",
         "context": "成绩达标但家务未完成",
         "ruling": "成绩奖励与家务是两条规则，可分别结算：成绩奖励照发，家务未完成则从下周零花钱扣对应部分。",
         "reasoning": "规则之间不互相抵消，避免把奖励变成要挟。",
         "approver": "家长", "department": "家庭",
         "date": "2026-06-01", "status": "closed", "reference_only": True},
    ]
    return {"dataset": "home", "description": "家庭共享知识（toC 示例）",
            "objects": objects, "roles": roles,
            "probe_rules": probe_rules, "precedent_cases": precedent_cases}


if __name__ == "__main__":
    out = ROOT / "data" / "gov_home.json"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
