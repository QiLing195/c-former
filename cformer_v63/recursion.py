# -*- coding: utf-8 -*-
"""V6.3 Transformation 递归：确定性多跳行走 + 安全控制器。

数据来源：对象"关系"证据中的「前一代是 P」构成 predecessor 边
（P --succeeded_by--> 当前对象），无需人工标注即可成图。

查询意图（词法）：
    「{对象} 的前一代」            → 后退 1 跳
    「{对象} 往前数三代」          → 后退 N 跳
    「{对象} 往后数2代」           → 前进 N 跳（successor 方向）

安全控制器（路线图 §9，全部确定性、先于任何推理）：
    cycle detection（visited 集）/ depth limit（MAX_HOPS=8）/
    chain end（出边不存在）/ 时间单调由 as-of 机制另行负责。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cformer_real import AIModelWorld

MAX_HOPS = 8

_PREDECESSOR_PATTERN = re.compile(r"前一代是\s*(.+?)\s*(?:，|。|$)")
_BACKWARD_HOPS = re.compile(r"往前(?:数|第)?\s*([一两二三四五六七八九十\d]+)\s*代")
_FORWARD_HOPS = re.compile(r"往后(?:数|第)?\s*([一两二三四五六七八九十\d]+)\s*代")
_SINGLE_BACK = re.compile(r"的前一代(?:是什么|叫什么|是哪个)?")
_SINGLE_FORWARD = re.compile(r"的(?:下一代|后一代)(?:是什么|叫什么|是哪个)?")
_NUMERAL = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


@dataclass
class WalkResult:
    status: str                 # ok | chain_end | depth_limit | cycle | no_anchor
    object_id: str | None       # 终点对象（ok 时非空）
    path: list[str] = field(default_factory=list)
    hops: int = 0


def parse_hop_count(raw: str) -> int:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return _NUMERAL.get(raw, 0)


def normalize_name(text: str) -> str:
    return " ".join(text.split()).lower()


class TransformationGraph:
    """Predecessor/successor 图：从四证据的关系字段零标注构建。"""

    def __init__(self, world: AIModelWorld) -> None:
        self.name_to_id: dict[str, str] = {}
        for obj in world.objects:
            self.name_to_id[normalize_name(obj.name)] = obj.object_id
        # 别名变体（去空格形）也进名字表
        for obj in world.objects:
            if " " in obj.name:
                self.name_to_id.setdefault(normalize_name(obj.name.replace(" ", "")),
                                           obj.object_id)

        self.predecessor: dict[str, str] = {}     # id -> 前一代 id
        name_at_build = dict(self.name_to_id)
        for obj in world.objects:
            relation_text = obj.evidence[2]        # FIELDS=("名称","属性","关系","变化")
            match = _PREDECESSOR_PATTERN.search(relation_text)
            if not match:
                continue
            prev_id = name_at_build.get(normalize_name(match.group(1)))
            if prev_id is not None and prev_id != obj.object_id:
                self.predecessor[obj.object_id] = prev_id

        self.successor: dict[str, str] = {
            prev: child for child, prev in self.predecessor.items()
        }

    def anchor(self, text: str) -> tuple[str, str] | None:
        """从查询中锚定起始对象：最长名字优先。返回 (object_id, matched_name)。"""
        lowered = normalize_name(text)
        best: tuple[str, str] | None = None
        for name, object_id in self.name_to_id.items():
            if name and name in lowered:
                if best is None or len(name) > len(best[1]):
                    best = (object_id, name)
        return best


def walk(graph: TransformationGraph, start_id: str, hops: int,
         direction: str) -> WalkResult:
    """direction: backward（前一代方向）| forward（后代方向）。"""
    if hops <= 0:
        return WalkResult("no_anchor", None)
    if hops > MAX_HOPS:
        return WalkResult("depth_limit", None, [start_id], 0)
    edge = graph.predecessor if direction == "backward" else graph.successor
    current = start_id
    visited = {current}
    path = [current]
    for hop in range(hops):
        next_id = edge.get(current)
        if next_id is None:
            return WalkResult("chain_end", None, path, hop)
        if next_id in visited:
            return WalkResult("cycle", None, path, hop)
        visited.add(next_id)
        current = next_id
        path.append(current)
    return WalkResult("ok", current, path, hops)


@dataclass
class HopIntent:
    start_id: str
    hops: int
    direction: str
    matched_name: str


class MultiHopResolver:
    """词法解析多跳意图；无法解析返回 None，交回后续链路。"""

    def __init__(self, graph: TransformationGraph) -> None:
        self.graph = graph

    def resolve_intent(self, text: str) -> HopIntent | None:
        anchored = self.graph.anchor(text)
        if anchored is None:
            return None
        start_id, matched_name = anchored
        rest = normalize_name(text).replace(matched_name, "", 1)

        backward = _BACKWARD_HOPS.search(rest)
        if backward:
            hops = parse_hop_count(backward.group(1))
            if hops:
                return HopIntent(start_id, hops, "backward", matched_name)
        forward = _FORWARD_HOPS.search(rest)
        if forward:
            hops = parse_hop_count(forward.group(1))
            if hops:
                return HopIntent(start_id, hops, "forward", matched_name)
        if _SINGLE_BACK.search(rest):
            return HopIntent(start_id, 1, "backward", matched_name)
        if _SINGLE_FORWARD.search(rest):
            return HopIntent(start_id, 1, "forward", matched_name)
        return None

    def run(self, text: str) -> WalkResult:
        intent = self.resolve_intent(text)
        if intent is None:
            return WalkResult("no_anchor", None)
        return walk(self.graph, intent.start_id, intent.hops, intent.direction)
