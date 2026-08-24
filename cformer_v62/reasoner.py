# -*- coding: utf-8 -*-
"""V6.2 WorldReasoner: deterministic cross-candidate selection over retrieval.

超级指代（"系列最新/最早"）需要跨候选比较发布年份——dual-encoder 对候选独立
打分，原理上做不了（见 README 泛化消融负结果）。本模块把"选择标准"从神经
分数中拿出来，用证据字段里的结构化线索（年份、同系列序号）做可审计的比较：

    查询含方向词（最新→max / 最早→min）
        → 取检索 Top 候选中与榜首同系列的成员
        → 比较成员年份，平局用 series_index（系列内定义顺序）
        → 输出选择 + 完整决策轨迹

确定性控制器优先于神经网络继续推理（路线图 §13）。本模块不做检索、
不训练、不依赖 torch；无法给出结论时返回 None，回退神经排序。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")

AS_OF_PATTERNS = (
    re.compile(r"截至\s*((?:19|20)\d{2})\s*年"),
    re.compile(r"截止\s*((?:19|20)\d{2})\s*年"),
    re.compile(r"到\s*((?:19|20)\d{2})\s*年(?:为止|为止的话)?"),
    re.compile(r"((?:19|20)\d{2})\s*年(?:的)?时候"),
)

# 方向词 → 极值方向；两组同时出现视为方向冲突，放弃裁决
MIN_WORDS = ("最早", "初代", "第一代", "首个", "第一个")
MAX_WORDS = ("最新", "最近发布", "最新一代", "最新款", "最新版")


@dataclass
class TemporalNoMember:
    """时间过滤后无存活成员：必须显式返回，禁止回退神经路径（未来事实泄漏）。"""

    as_of: int


@dataclass
class ReasonedChoice:
    label: int
    direction: str            # max | min
    year: int
    neural_score: float
    trace: list[str] = field(default_factory=list)


def parse_direction(text: str) -> str | None:
    has_min = any(word in text for word in MIN_WORDS)
    has_max = any(word in text for word in MAX_WORDS)
    if has_min and has_max:
        return None  # 方向冲突，交回神经层
    return "min" if has_min else ("max" if has_max else None)


def parse_as_of(text: str) -> int | None:
    for pattern in AS_OF_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def extract_year(text: str) -> int | None:
    match = MAX_YEAR_PATTERN.search(text)
    return int(match.group()) if match else None


class WorldReasoner:
    """Callables keep this module decoupled from storage and tokenizer.

    series_key_from_text：词法锚定——查询文本通常显式包含公司/系列名
    （"{公司} 的 {系列} 系列"）。神经榜首在训练外查询上可能整个系列都错，
    所以词法锚定优先，神经榜首锚点只作回退。
    """

    def __init__(
        self,
        *,
        series_key_of,          # label -> (company, series) | None
        evidence_text_of,       # label -> 四证据拼接文本（用于抽年份）
        series_index_of,        # label -> int（系列内顺序，平局裁决）
        series_key_from_text=None,  # text -> (company, series) | None（词法锚定）
    ) -> None:
        self.series_key_of = series_key_of
        self.evidence_text_of = evidence_text_of
        self.series_index_of = series_index_of
        self.series_key_from_text = series_key_from_text

    def _anchor_series(self, text: str, ranked_labels: list[int]):
        """返回 (mode, value, source)；mode ∈ {series, company, none}。"""
        if self.series_key_from_text is not None:
            anchored = self.series_key_from_text(text)
            if anchored is not None:
                mode, value = anchored
                if mode in ("series", "company"):
                    return mode, value, "lexical"
        if ranked_labels:
            key = self.series_key_of(ranked_labels[0])
            if key is not None:
                return "series", key, "neural_top"
        return "none", None, "none"

    def select(
        self,
        text: str,
        ranked_labels: list[int],
        ranked_scores: list[float],
        *,
        max_trace: int = 6,
        as_of: int | None = None,
    ) -> ReasonedChoice | TemporalNoMember | None:
        direction = parse_direction(text)
        if direction is None or not ranked_labels:
            return None

        mode, value, anchor_source = self._anchor_series(text, ranked_labels)
        if mode == "none":
            return None

        def belongs(label: int) -> bool:
            key = self.series_key_of(label)
            if key is None:
                return False
            if mode == "series":
                return key == value
            return str(key[0]).lower() == value

        members = [
            (label, score)
            for label, score in zip(ranked_labels, ranked_scores)
            if belongs(label)
        ]
        if not members:
            return None

        dated: list[tuple[int, float, int]] = []
        for label, score in members:
            year = extract_year(self.evidence_text_of(label))
            if year is not None:
                dated.append((label, score, year))

        trace_prefix = [f"anchor={anchor_source}"]

        # 单成员且无年份：唯一成员即答案（无时间语义可过滤，保持既有行为）
        if len(members) == 1 and not dated:
            label, score = members[0]
            return ReasonedChoice(
                label=label, direction=direction, year=0,
                neural_score=float(score),
                trace=trace_prefix + ["single_member_series"],
            )
        if not dated:
            return None  # 年份缺失过多，回退神经排序

        # 时间快照：as_of 之后的事实一律不可见（未来事实泄漏防线）。
        # 过滤后为空必须显式返回 TemporalNoMember，禁止回退神经路径。
        if as_of is not None:
            future_count = sum(1 for item in dated if item[2] > as_of)
            dated = [item for item in dated if item[2] <= as_of]
            trace_prefix.append(f"as_of={as_of};filtered_future={future_count}")
            if not dated:
                return TemporalNoMember(as_of=as_of)

        if len(dated) == 1:
            label, score, year = dated[0]
            if len(members) == 1:
                trace_prefix.append("single_member_series")
            return ReasonedChoice(
                label=label, direction=direction, year=year,
                neural_score=float(score),
                trace=trace_prefix + [f"{label}:year={year}"],
            )

        # 主键：年份极值；平局：series_index 极值（定义顺序即时间顺序）
        pick_index = (
            max(range(len(dated)), key=lambda i: (dated[i][2], self.series_index_of(dated[i][0])))
            if direction == "max"
            else min(range(len(dated)), key=lambda i: (dated[i][2], -self.series_index_of(dated[i][0])))
        )
        label, score, year = dated[pick_index]
        trace = trace_prefix + [
            f"{candidate_label}:year={candidate_year}"
            for candidate_label, _, candidate_year in dated[:max_trace]
        ]
        return ReasonedChoice(
            label=label, direction=direction, year=year,
            neural_score=float(score), trace=trace,
        )
