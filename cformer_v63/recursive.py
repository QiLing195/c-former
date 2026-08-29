# -*- coding: utf-8 -*-
"""V6.3 受控递归层：关系图的确定性遍历 + 四重控制（循环/深度/时间/版本）。

从真实数据的结构化 predecessor 字段构建「前代 → 后继」关系图，
对 latest（沿链走到末尾）与 predecessor（前代查表）查询做确定性推理，
并用 cycle/depth/time/version 控制保证终止与快照一致性。

本层是确定性控制器（v1 无神经网络），与身份解析层（V6.0 冻结）解耦：
身份层确认对象集合，递归层在其上的关系图上推理。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecursiveResult:
    ok: bool
    reason: str  # ok | no_head | cycle | depth_exceeded | time_violation | not_found | version_stale
    path: list[str] = field(default_factory=list)
    answer_id: str | None = None


class RelationGraph:
    """由对象的 predecessor 字段构建的有向图（前代 → 后继）。"""

    def __init__(self, objects: list[dict]) -> None:
        self.nodes: dict[str, dict] = {}
        self.successors: dict[str, list[str]] = {}
        self.predecessors: dict[str, list[str]] = {}
        self.series_members: dict[str, list[str]] = {}
        for obj in objects:
            object_id = obj["id"]
            self.nodes[object_id] = obj
            pred = obj.get("predecessor")
            self.predecessors.setdefault(object_id, [])
            if pred and pred in self.nodes:
                self.predecessors[object_id].append(pred)
            series = obj.get("series", "")
            self.series_members.setdefault(series, []).append(object_id)
        # 第二遍：建立 successors（需要所有节点已注册）
        for obj in objects:
            object_id = obj["id"]
            pred = obj.get("predecessor")
            if pred and pred in self.nodes:
                self.successors.setdefault(pred, []).append(object_id)
        for chain in self.successors.values():
            chain.sort()

    def successors_of(self, object_id: str) -> list[str]:
        return list(self.successors.get(object_id, []))

    def predecessors_of(self, object_id: str) -> list[str]:
        return list(self.predecessors.get(object_id, []))

    def series_heads(self, series: str) -> list[str]:
        """链头：该系列中没有前驱的对象。"""
        members = self.series_members.get(series, [])
        return [m for m in members if not self.predecessors_of(m)]

    def year_of(self, object_id: str) -> int:
        return int(self.nodes[object_id].get("year", 0))


class RecursiveResolver:
    """受控递归：latest / predecessor / 多跳链，带 cycle/depth/time/version 控制。"""

    def __init__(self, graph: RelationGraph, max_depth: int = 4, max_steps: int = 16) -> None:
        self.graph = graph
        self.max_depth = max_depth
        self.max_steps = max_steps

    def latest_of_series(self, series: str, *, world_version: int | None = None) -> RecursiveResult:
        """沿后继走到链尾（最新）；受控：循环检测、步数上限、时间单调、版本固定。"""
        heads = self.graph.series_heads(series)
        if not heads:
            return RecursiveResult(False, "no_head", [])
        results: list[RecursiveResult] = []
        for head in heads:
            results.append(self._walk(head, world_version=world_version))
        ok_results = [r for r in results if r.ok]
        if not ok_results:
            return results[0] if results else RecursiveResult(False, "not_found", [])
        # 多链取年份最大者为最新（时间单调约束下）
        best = max(ok_results, key=lambda r: self.graph.year_of(r.answer_id))
        return best

    def predecessor_of(self, object_id: str) -> RecursiveResult:
        preds = self.graph.predecessors_of(object_id)
        if not preds:
            return RecursiveResult(False, "not_found", [object_id])
        return RecursiveResult(True, "ok", [preds[0], object_id], answer_id=preds[0])

    def chain(self, start_id: str, hops: int, *, world_version: int | None = None) -> RecursiveResult:
        """从 start 向前走 hops 跳；hops 超过 max_depth 或遇到循环/时间回退即拒绝。"""
        if hops > self.max_depth:
            return RecursiveResult(False, "depth_exceeded", [start_id])
        return self._walk(start_id, max_hops=hops, world_version=world_version)

    def _walk(self, start_id: str, *, max_hops: int | None = None, world_version: int | None = None) -> RecursiveResult:
        current = start_id
        path = [current]
        visited = {current}
        step = 0
        while True:
            if step >= self.max_steps:
                return RecursiveResult(False, "depth_exceeded", path)
            if max_hops is not None and step >= max_hops:
                break
            successors = self.graph.successors_of(current)
            if world_version is not None:
                successors = [s for s in successors if self.graph.year_of(s) <= world_version]
            if not successors:
                break
            # 时间单调：后继年份必须 >= 当前年份
            nxt = successors[0]
            if self.graph.year_of(nxt) < self.graph.year_of(current):
                return RecursiveResult(False, "time_violation", path + [nxt])
            if nxt in visited:
                return RecursiveResult(False, "cycle", path + [nxt])
            visited.add(nxt)
            path.append(nxt)
            current = nxt
            step += 1
        return RecursiveResult(True, "ok", path, answer_id=current)
