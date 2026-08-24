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

import torch

from cformer_v59 import CandidateLedger, CandidateStatus, EvidenceVerifier


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
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.index = index
        self.encoder = encoder          # encode_query(text)->(d,), id_of(label)->object_id
        self.verifier = verifier
        self.nprobe = nprobe
        self.top_ann = top_ann
        self.top_rerank = top_rerank
        self._proposed_surfaces: set[str] = set()

    def resolve(self, text: str, query_type: str | None = None) -> PipelineResult:
        stage_ms = {}
        started = time.perf_counter()

        exact_id = self.store.exact_lookup(text)
        stage_ms["exact"] = (time.perf_counter() - started) * 1000
        if exact_id is not None:
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
