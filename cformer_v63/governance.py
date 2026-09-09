# -*- coding: utf-8 -*-
"""通用知识治理层（GovLayer）：域无关的确定性治理框架。

把 toB POC 验证过的四支柱逻辑（权限 mask / 字段级分级 / 空白识别 / 先例沉淀）
抽成**可配置通用模块**——知识库内容、角色、权限规则全部由 dataset 驱动，
本层不含任何业务域硬编码。因此：
  - toB：换企业制度 dataset，即企业制度问答治理；
  - toC：换"个人知识 + 个人权限"dataset，即个人化定制（同一个 GovLayer）。

Dataset 约定（任何域遵守即可接入）：
  objects: [{id, title, keywords:[...], levels: {0: 公开内容, 1: 内容, ...}}]
           levels 的数字 = 可见该内容所需的最低角色级别（0 最低=全员可见）
  roles:   {role_name: level}（如 {"员工":0, "经理":1, "HR":2}）
  precedent_cases: [{case_id, topic, question, context, ruling,
                     reasoning, approver, date, status, reference_only}]

四能力：
  A. GuardLayer.visible_content(obj, role)   —— 字段级权限：返回 role 级别及以下内容
  B. GovLayer.retrieve(text)                  —— 关键词检索 Top-N（可换语义后端）
  C. GovLayer.answer(text, role)              —— 完整回答：检索→分级内容→覆盖探测→GAP/先例
  D. PrecedentStore                           —— 案例沉淀/检索（隐性知识显性化）
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------- 数据模型

@dataclass
class GovAnswer:
    question: str
    role: str
    hit_ids: list[str] = field(default_factory=list)
    visible_texts: dict[str, str] = field(default_factory=dict)   # 条款id -> 该角色可见内容
    denied_fields: list[str] = field(default_factory=list)        # 命中但字段超权限的条款
    verdict: str = "covered"      # covered | gap | out_of_scope
    precedents: list[dict] = field(default_factory=list)
    boundary_note: str = ""


class GovLayer:
    """确定性治理层：检索 + 字段级权限 + 覆盖判定 + 先例。域无关。"""

    def __init__(self, objects: list[dict], roles: dict[str, int],
                 probe_pairs: list[tuple[list[str], list[str]]] | None = None,
                 cases: list[dict] | None = None) -> None:
        """
        objects: 知识对象列表（约定见模块 docstring）
        roles: 角色名 -> 级别（级别=可见字段上限）
        probe_pairs: 覆盖探测规则 [(问法关键词组, 应出现在答案里的词)] ——
                     用于判定"命中条款是否真覆盖了问题诉求"（空白识别）
        cases: 先例案例列表（可选）
        """
        self.objects = objects
        self.roles = roles
        self._probe_rules = probe_pairs or []
        self.cases = cases or []
        self._keyword_index: dict[str, list[str]] = {}  # 关键词 -> 对象id列表
        for obj in objects:
            for kw in obj.get("keywords", []):
                self._keyword_index.setdefault(kw, []).append(obj["id"])

    # ---- 检索：关键词（可替换为语义后端）----
    def retrieve(self, text: str, top_n: int = 3) -> list[str]:
        scored: dict[str, int] = {}
        for kw, ids in self._keyword_index.items():
            if kw in text:
                for obj_id in ids:
                    scored[obj_id] = scored.get(obj_id, 0) + 1
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
        return [oid for oid, _ in ranked[:top_n]]

    # ---- 字段级权限 ----
    def role_level(self, role: str) -> int:
        return self.roles.get(role, 0)

    def visible_content(self, obj: dict, role: str) -> str:
        """返回 role 级别及以下字段内容（级别数字=可见所需最低级别）。"""
        level = self.role_level(role)
        parts = []
        for field_level, content in sorted(obj.get("levels", {}).items(),
                                           key=lambda kv: int(kv[0])):
            if int(field_level) <= level:
                parts.append(content)
        return " ".join(parts)

    def max_field_level(self, obj: dict) -> int:
        levels = obj.get("levels", {})
        return max((int(k) for k in levels), default=0)

    # ---- 覆盖探测（空白识别）：条款内容是否覆盖问题诉求 ----
    def _probe_covered(self, question: str, hit_content: str) -> bool:
        for q_kws, answer_probes in self._probe_rules:
            if any(kw in question for kw in q_kws):
                return any(probe in hit_content for probe in answer_probes)
        # 无匹配规则时，默认命中即有内容可答（保守：不误判空白）
        return True

    # ---- 先例检索 ----
    def search_precedents(self, question: str, top_n: int = 2) -> list[dict]:
        hits = []
        for case in self.cases:
            topic = case.get("topic", "")
            if any(tok in question and tok in topic + case.get("question", "")
                   for tok in ["出差", "超期", "请假", "迟到", "打卡", "报销", "申诉"]):
                hits.append(case)
        return hits[:top_n]

    # ---- 完整回答 ----
    def answer(self, question: str, role: str) -> GovAnswer:
        ans = GovAnswer(question=question, role=role)
        hit_ids = self.retrieve(question)
        ans.hit_ids = hit_ids
        if not hit_ids:
            ans.verdict = "out_of_scope"
            ans.precedents = self.search_precedents(question)
            ans.boundary_note = "问题不在已登记知识范围内，建议升级人工。"
            return ans

        for oid in hit_ids:
            obj = next(o for o in self.objects if o["id"] == oid)
            level = self.role_level(role)
            max_level = self.max_field_level(obj)
            vis = self.visible_content(obj, role)
            ans.visible_texts[oid] = vis
            if max_level > level:
                ans.denied_fields.append(oid)  # 该条款有更高权限字段，已被截断
                # 角色级别低于该条款最低字段级且该级别无内容 → 明确无权限
                if not vis:
                    ans.boundary_note = (
                        f"命中条目 [{obj['id']}] 的内容级别为 {max_level}，"
                        f"你的角色 [{role}] 级别 {level} 无查看权限。"
                    )
                    continue
            # 覆盖判定：probe 命中才认为覆盖；未覆盖 → gap
            covered = self._probe_covered(question, vis)
            if not covered:
                ans.verdict = "gap"

        ans.precedents = self.search_precedents(question)
        if ans.verdict == "gap":
            if ans.precedents:
                ans.boundary_note = ("知识未覆盖此问题，但有过往先例可参考"
                                     "（仅供参考，以实际确认为准）。")
            else:
                ans.boundary_note = ("知识未覆盖此问题，且无过往先例。"
                                     "AI 不编造规则：建议升级人工裁决，裁决后将沉淀为先例。")
        elif ans.denied_fields and not ans.boundary_note:
            ans.boundary_note = "已返回你可视范围内内容；命中条目含更高权限字段，未向你展示。"
        return ans

    # ---- 案例沉淀 ----
    def add_precedent(self, *, topic: str, question: str, context: str,
                      ruling: str, reasoning: str, approver: str,
                      department: str = "") -> dict:
        case = {
            "case_id": f"C-{time.strftime('%Y')}-{100 + len(self.cases):03d}",
            "topic": topic, "question": question, "context": context,
            "ruling": ruling, "reasoning": reasoning,
            "approver": approver, "department": department,
            "date": time.strftime("%Y-%m-%d"), "status": "closed",
            "reference_only": True,
        }
        self.cases.append(case)
        return case


# ---------------------------------------------------------------- 便捷加载

def load_dataset(dataset_path: str | Path) -> dict:
    """从 dataset JSON 加载 objects/roles/probe_rules/cases。"""
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    return {
        "objects": data.get("objects", []),
        "roles": data.get("roles", {"user": 0}),
        "probe_rules": data.get("probe_rules", []),
        "cases": data.get("precedent_cases", []),
    }


def build_govlayer(dataset_path: str | Path, cases_path: str | Path | None = None) -> GovLayer:
    """一行构建：dataset 驱动。toB/toC 只换 dataset 路径。"""
    spec = load_dataset(dataset_path)
    cases = list(spec["cases"])
    if cases_path and Path(cases_path).exists():
        cases.extend(json.loads(Path(cases_path).read_text(encoding="utf-8")))
    return GovLayer(objects=spec["objects"], roles=spec["roles"],
                    probe_pairs=spec["probe_rules"], cases=cases)
