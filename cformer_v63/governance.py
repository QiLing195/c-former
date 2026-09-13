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

# 常见错别字/同音字纠正（真实用户输入容错）
# 关键词检索对错字零容忍，但真实用户打错字是常态——先纠正再检索，
# 并在回答里透明提示"已将 X 理解为 Y"（不偷偷改，保持可审计）。
DEFAULT_TYPO_MAP = {
    "矿工": "旷工", "旷功": "旷工",
    "请加": "请假", "清假": "请假",
    "打刻": "打卡", "打咖": "打卡",
    "迟道": "迟到",
    "销加": "销假",
    "出拆": "出差",
    "绿通": "绿色通道", "助学": "绿色通道",
}


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
    typo_corrections: list[str] = field(default_factory=list)     # 纠错记录（透明可审计）
    semantic_used: bool = False                                    # 本次是否走了 LLM 语义检索
    gap_reason: str = ""                                           # 空白原因（LLM 指出缺什么）


class GovLayer:
    """确定性治理层：检索 + 字段级权限 + 覆盖判定 + 先例。域无关。"""

    def __init__(self, objects: list[dict], roles: dict[str, int],
                 probe_pairs: list[tuple[list[str], list[str]]] | None = None,
                 cases: list[dict] | None = None,
                 typo_map: dict[str, str] | None = None,
                 semantic_backend=None) -> None:
        """
        objects: 知识对象列表（约定见模块 docstring）
        roles: 角色名 -> 级别（级别=可见字段上限）
        probe_pairs: 覆盖探测规则（**仅当无语义后端时**作为回退使用）
        cases: 先例案例列表（可选）
        typo_map: 错别字纠正表（默认内置常见词；传 {} 可关闭）
        semantic_backend: LLMSemanticBackend 实例（None = 纯关键词模式，零依赖）
        """
        self.objects = objects
        self.roles = roles
        self._probe_rules = probe_pairs or []
        self.cases = cases or []
        self.typo_map = DEFAULT_TYPO_MAP if typo_map is None else typo_map
        self.semantic_backend = semantic_backend
        self._keyword_index: dict[str, list[str]] = {}  # 关键词 -> 对象id列表
        for obj in objects:
            for kw in obj.get("keywords", []):
                self._keyword_index.setdefault(kw, []).append(obj["id"])

    # ---- 错别字容错（真实用户输入）----
    def correct_typos(self, text: str) -> tuple[str, list[str]]:
        """纠正常见错别字，返回（纠正后文本, 纠错记录）。"""
        corrections = []
        corrected = text
        for wrong, right in self.typo_map.items():
            if wrong in corrected and wrong != right:
                corrected = corrected.replace(wrong, right)
                corrections.append(f"{wrong}→{right}")
        return corrected, corrections

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

    # ---- 语义候选：只把【该角色可见的内容】交给 LLM ----
    def _visible_candidates(self, role: str, corrected_question: str) -> list[dict]:
        """构造语义检索候选——权限过滤在前，LLM 只看得到可见内容（零泄漏前提）。"""
        candidates = []
        for obj in self.objects:
            vis = self.visible_content(obj, role)
            if vis:  # 该角色无可见内容的条款不进候选（不交给 LLM）
                candidates.append({"id": obj["id"], "title": obj.get("title", obj["id"]),
                                   "content": vis})
        # 大库先用关键词粗筛，控制 prompt 规模（可扩展性）
        backend = self.semantic_backend
        if backend is not None and len(candidates) > backend.max_candidates:
            kw = set(self.retrieve(corrected_question, top_n=backend.max_candidates))
            filtered = [c for c in candidates if c["id"] in kw]
            if filtered:
                return filtered
            return candidates[: backend.max_candidates]
        return candidates

    # ---- 完整回答 ----
    def answer(self, question: str, role: str) -> GovAnswer:
        ans = GovAnswer(question=question, role=role)
        # 错别字容错：先纠正再检索（透明记录，不偷偷改）
        corrected, corrections = self.correct_typos(question)
        ans.typo_corrections = corrections

        # 1) 检索：优先语义（LLM），不可用则回退关键词
        semantic_result = None
        if self.semantic_backend is not None and self.semantic_backend.available:
            candidates = self._visible_candidates(role, corrected)
            semantic_result = self.semantic_backend.match(corrected, candidates)
        if semantic_result is not None:
            known_ids = {o["id"] for o in self.objects}
            # 过滤模型可能编造的 id（只保留真实条款）
            hit_ids = [i for i in semantic_result.relevant_ids if i in known_ids]
            ans.semantic_used = True
        else:
            hit_ids = self.retrieve(corrected)

        ans.hit_ids = hit_ids
        if not hit_ids:
            # 空命中时区分：真空白 vs 相关条款存在但超出该角色权限
            raw_hits = self.retrieve(corrected, top_n=3)
            restricted_hits = [
                oid for oid in raw_hits
                if self.max_field_level(next(o for o in self.objects if o["id"] == oid))
                > self.role_level(role)
            ]
            if restricted_hits:
                ans.verdict = "restricted"
                ans.hit_ids = restricted_hits
                ans.denied_fields = restricted_hits
                for oid in restricted_hits:
                    ans.visible_texts[oid] = self.visible_content(
                        next(o for o in self.objects if o["id"] == oid), role)
                ans.boundary_note = ("该问题涉及的信息超出你的角色权限范围，未向你展示；"
                                     "如需了解请联系 HR 查询。")
                if corrections:
                    ans.boundary_note += " 已将输入中的 " + "、".join(corrections) + " 按制度用词理解。"
                return ans
            ans.verdict = "out_of_scope"
            ans.precedents = self.search_precedents(corrected)
            ans.boundary_note = "问题不在已登记知识范围内，建议升级人工。"
            return ans

        # 2) 字段级权限 + 覆盖判定
        for oid in hit_ids:
            obj = next(o for o in self.objects if o["id"] == oid)
            level = self.role_level(role)
            max_level = self.max_field_level(obj)
            vis = self.visible_content(obj, role)
            ans.visible_texts[oid] = vis
            if max_level > level:
                ans.denied_fields.append(oid)  # 该条款有更高权限字段，已被截断
                if not vis:
                    ans.boundary_note = (
                        f"命中条目 [{obj['id']}] 的内容级别为 {max_level}，"
                        f"你的角色 [{role}] 级别 {level} 无查看权限。"
                    )
                    continue
            # 覆盖判定：语义后端结果优先；否则回退词表探针
            if semantic_result is not None:
                covered = semantic_result.covered
                if not covered and semantic_result.missing:
                    ans.gap_reason = semantic_result.missing
            else:
                covered = self._probe_covered(corrected, vis)
            if not covered:
                ans.verdict = "gap"

        ans.precedents = self.search_precedents(corrected)
        if ans.verdict == "gap":
            # 关键区分：知识空白 vs 权限截断（真实场景语义完全不同）
            if ans.denied_fields:
                # 命中条款但（部分）内容超出该角色权限 → 不是"制度没写"，而是"你无权限"
                ans.verdict = "restricted"
                ans.boundary_note = ("制度中有相关规定，但超出你的角色权限范围，"
                                     "未向你展示；如需了解请联系 HR 查询。")
            else:
                reason = f"（缺少：{ans.gap_reason}）" if ans.gap_reason else ""
                if ans.precedents:
                    ans.boundary_note = (f"知识未覆盖此问题{reason}，但有过往先例可参考"
                                         "（仅供参考，以实际确认为准）。")
                else:
                    ans.boundary_note = (f"知识未覆盖此问题{reason}，且无过往先例。"
                                         "AI 不编造规则：建议升级人工裁决，裁决后将沉淀为先例。")
        elif ans.denied_fields and not ans.boundary_note:
            ans.boundary_note = "已返回你可视范围内内容；命中条目含更高权限字段，未向你展示。"
        # 纠错透明提示（可审计：让用户知道你理解成了什么）
        if corrections:
            hint = "已将输入中的 " + "、".join(corrections) + " 按制度用词理解。"
            ans.boundary_note = (ans.boundary_note + " " + hint).strip()
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


def build_govlayer(dataset_path: str | Path, cases_path: str | Path | None = None,
                   semantic_backend=None) -> GovLayer:
    """一行构建：dataset 驱动。toB/toC 只换 dataset 路径。
    semantic_backend 非空时启用 LLM 语义检索 + 覆盖判定（None = 关键词回退）。"""
    spec = load_dataset(dataset_path)
    cases = list(spec["cases"])
    if cases_path and Path(cases_path).exists():
        cases.extend(json.loads(Path(cases_path).read_text(encoding="utf-8")))
    return GovLayer(objects=spec["objects"], roles=spec["roles"],
                    probe_pairs=spec["probe_rules"], cases=cases,
                    semantic_backend=semantic_backend)
