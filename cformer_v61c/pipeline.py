# -*- coding: utf-8 -*-
"""V6.1c unified resolution chain.

    查询 → normalize → 精确别名命中（直接返回，不进神经计算）
        → 未命中：ANN 粗召回 Top-256（FTS 候选并入去重）
        → 全精度重排 Top-16
        → EvidenceVerifier（分类型 margin）
        → supported / ambiguous / unknown
        → 未知短文本 → CandidateLedger propose（审核前不写正式身份）

关键原则：精确命中即返回；模型只能 supported，外部审核才 verified；
对象向量在 IVF 索引中只存一份。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from cformer_v59 import (
    CandidateLedger,
    CandidateStatus,
    EvidenceVerifier,
    VerificationDecision,
)

# 存在选择标准的措辞：出现时才信任神经 margin 做成员选择。
# 裸系列指代（"X 是哪一个模型？"）没有选择标准，属于结构性歧义，
# 由确定性规则判定，不依赖神经分数（治理优先于模型）。
SELECTION_PHRASES = (
    "最新", "最早", "初代", "上一代", "前一代",
    "第一", "第二", "第三", "第四", "第五",
    "旗舰", "顶配", "轻量", "入门", "最强", "最快",
    "推理", "编程", "代码", "写代码", "多模态", "端侧", "商用",
)


def has_selection_phrase(text: str) -> bool:
    return any(phrase in text for phrase in SELECTION_PHRASES)


def is_alias_like(surface: str) -> bool:
    """Conservative heuristic for which unknowns deserve a ledger proposal.

    Short noun-ish surfaces (candidate aliases) are proposed; sentences,
    questions and chit-chat are not.
    """
    stripped = surface.strip()
    if not stripped or len(stripped) > 24:
        return False
    if stripped.endswith(("？", "?", "。", ".", "！", "!", "呢", "吗", "吧")):
        return False
    if any(marker in stripped for marker in ("请问", "哪", "什么", "怎么", "多少", "帮我", "介绍")):
        return False
    return any(ch.isalnum() for ch in stripped)


@dataclass
class PipelineResult:
    status: CandidateStatus
    path: str                      # exact | ann
    object_id: str | None
    score: float
    runner_up: float
    reason: str
    coverage: float
    stage_ms: dict = field(default_factory=dict)
    ann_candidates: int = 0
    rerank_candidates: int = 0
    proposed_alias: bool = False


class UnifiedResolutionPipeline:
    def __init__(
        self,
        store,
        ledger: CandidateLedger,
        index,
        encoder,
        verifier: EvidenceVerifier,
        *,
        nprobe: int = 4,
        top_ann: int = 256,
        top_rerank: int = 16,
        series_size_of: Callable[[str], int] | None = None,
        reasoner=None,
        access_gate=None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.index = index
        self.encoder = encoder          # encode_query(text)->(d,), id_of(label)->object_id
        self.verifier = verifier
        self.nprobe = nprobe
        self.top_ann = top_ann
        self.top_rerank = top_rerank
        self.series_size_of = series_size_of  # object_id -> 同系列存活成员数（含自身）
        self.reasoner = reasoner              # V6.2 WorldReasoner 或 None
        self.access_gate = access_gate        # V6.2 ObserverGate 或 None
        self._proposed_surfaces: set[str] = set()

    def _finalize_access(self, observer_frame, object_id: str) -> bool:
        """True=放行；False=已生成拒绝结论（确定性掩码，身份解析之后才注入观测点）。"""
        if self.access_gate is None:
            return True
        return self.access_gate.check(observer_frame, object_id).allowed

    def resolve(
        self, text: str, query_type: str | None = None, observer_frame=None
    ) -> PipelineResult:
        stage_ms = {}
        started = time.perf_counter()

        exact_id = self.store.exact_lookup(text)
        stage_ms["exact"] = (time.perf_counter() - started) * 1000
        if exact_id is not None:
            if not self._finalize_access(observer_frame, exact_id):
                return PipelineResult(
                    CandidateStatus.ACCESS_DENIED, "exact", None, 1.0, 0.0,
                    f"access_denied:{self.access_gate.check(observer_frame, exact_id).reason}",
                    1.0, stage_ms,
                )
            return PipelineResult(
                CandidateStatus.SUPPORTED, "exact", exact_id, 1.0, 0.0,
                "exact_alias_hit", 1.0, stage_ms,
            )

        token_started = time.perf_counter()
        query_vector, coverage = self.encoder.encode_query(text)
        stage_ms["encode"] = (time.perf_counter() - token_started) * 1000

        ann_started = time.perf_counter()
        scores, labels = self.index.search(query_vector, self.nprobe, self.top_ann)
        fts_ids = set(self.store.fts_candidates(text)) & {
            self.encoder.object_id_of(int(label)) for label in labels.tolist()
        } if len(labels) else set()
        stage_ms["ann"] = (time.perf_counter() - ann_started) * 1000
        ann_candidates = int(len(labels))

        rerank_started = time.perf_counter()
        if len(labels):
            order = torch.argsort(scores, descending=True)[: self.top_rerank]
            top_scores = scores[order].float().tolist()
            top_labels = labels[order].tolist()
        else:
            top_scores, top_labels = [], []
        stage_ms["rerank"] = (time.perf_counter() - rerank_started) * 1000

        # V6.2 世界推理块：超级指代需要跨候选比较，确定性裁决优先于神经 margin；
        # 无法裁决（方向冲突/年份缺失/单成员）时返回 None，走原神经路径。
        if self.reasoner is not None and len(labels):
            reason_started = time.perf_counter()
            choice = self.reasoner.select(
                text, labels.tolist(), scores.float().tolist(),
            )
            stage_ms["reason"] = (time.perf_counter() - reason_started) * 1000
            if choice is not None:
                if not self._finalize_access(observer_frame, self.encoder.object_id_of(choice.label)):
                    return PipelineResult(
                        CandidateStatus.ACCESS_DENIED, "reasoned", None,
                        choice.neural_score, 0.0,
                        f"access_denied:{self.access_gate.check(observer_frame, self.encoder.object_id_of(choice.label)).reason}",
                        coverage, stage_ms, ann_candidates, len(labels),
                    )
                return PipelineResult(
                    CandidateStatus.SUPPORTED, "reasoned",
                    self.encoder.object_id_of(choice.label),
                    choice.neural_score, 0.0,
                    f"cross_candidate_{choice.direction}_year={choice.year}"
                    f";trace={';'.join(choice.trace)}",
                    coverage, stage_ms, ann_candidates, len(labels),
                )

        if not top_labels:
            return PipelineResult(
                CandidateStatus.UNKNOWN, "ann", None, 0.0, 0.0,
                "no_candidates", coverage, stage_ms, ann_candidates, 0,
            )

        decision = self.verifier.decide(
            float(top_scores[0]), float(top_scores[1]) if len(top_scores) > 1 else 0.0,
            float(coverage), query_type,
        )
        object_id = self.encoder.object_id_of(int(top_labels[0]))

        # 观测点门控：身份确认后注入观测点，被掩对象不得以 supported 暴露
        if (
            decision.status == CandidateStatus.SUPPORTED
            and not self._finalize_access(observer_frame, object_id)
        ):
            return PipelineResult(
                CandidateStatus.ACCESS_DENIED, "ann", None,
                decision.score, decision.margin,
                f"access_denied:{self.access_gate.check(observer_frame, object_id).reason}",
                coverage, stage_ms, ann_candidates, len(top_labels),
            )

        # 结构性歧义（确定性规则，先于神经结论生效）：多成员系列 + 查询无任何
        # 选择标准措辞 → 无论 margin 多大都不支持单一成员。
        if (
            decision.status == CandidateStatus.SUPPORTED
            and self.series_size_of is not None
            and not has_selection_phrase(text)
            and self.series_size_of(object_id) >= 2
        ):
            decision = VerificationDecision(
                CandidateStatus.AMBIGUOUS, decision.score, decision.margin,
                "structural_series_ambiguity",
            )

        proposed = False
        if decision.status == CandidateStatus.UNKNOWN and is_alias_like(text):
            surface = text.strip()
            if surface not in self._proposed_surfaces:
                self.ledger.propose(surface, int(top_labels[0]), actor="model")
                self._proposed_surfaces.add(surface)
                proposed = True

        return PipelineResult(
            decision.status, "ann", object_id if decision.status == CandidateStatus.SUPPORTED else None,
            decision.score, decision.margin, decision.reason, coverage,
            stage_ms, ann_candidates, len(top_labels), proposed,
        )
