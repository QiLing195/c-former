# -*- coding: utf-8 -*-
"""V6.5 UnifiedCFormer：把 v59–v63 全部组件串成一个入口的集成门面。

    UnifiedCFormer(dataset).resolve(text, observer_frame=None)
        → Answer(status, path, object_id, evidence, trace, stage_ms)

组装（全部复用已冻结组件，无新算法）：
    精确别名 → 多跳递归(v63) → ANN 粗召回+重排(v61/v61c)
    → 世界推理块：结构歧义规则 / as-of 快照(v62)
    → 分类型 margin 校验(v59/v60b) → 观测点掩码(v62)
    → CandidateLedger 提案(v59)；向量单副本贯穿。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_v62 import ObserverGate
from cformer_v63 import MultiHopResolver, TransformationGraph
from cformer_real import AIModelWorld, query_variants
from evaluate_v61c import WorldEncoder, build_chain
from evaluate_v62 import make_reasoner
from train_eval_real import split_known, train


@dataclass
class Answer:
    status: str
    path: str
    object_id: str | None
    score: float
    reason: str
    coverage: float
    evidence: dict = field(default_factory=dict)   # 四证据文档（supported 时）
    trace: list[str] = field(default_factory=list)
    stage_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class UnifiedCFormer:
    """One-line integration entry over the frozen component stack."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        seed: int = 601,
        steps: int = 400,
        minimum_score: float = 0.40,
        minimum_coverage: float = 0.60,
        known_margin: float = 0.01,
        device: str | None = None,
    ) -> None:
        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.set_float32_matmul_precision("high")
        self.world = AIModelWorld(dataset_path)
        raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        self._id_meta = {obj["id"]: obj.get("meta") or {} for obj in raw["objects"]}

        config = ChineseTransformerConfig(
            self.world.tokenizer.size, layers=2, d_model=64, heads=4,
            ffn_dimensions=128, output_dimensions=32,
        )
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        known_train, heldout = split_known(self.world.known_queries())
        self.heldout = heldout
        entries_spec = []
        for query in known_train:
            variants = query_variants(query["text"], query.get("meta"))
            target = self.world.target_label(query["target_id"])
            entries_spec.append({
                "variants": [variants[i] for i in (0, 2, 3)] if len(variants) >= 5 else [variants[0]],
                "target": target,
            })
        train(model, self.world, resolved_device, entries=entries_spec,
              steps=steps, lr=1e-3, batch_size=16, hard_k=0, seed=seed)

        encoder = WorldEncoder(self.world, model, resolved_device)
        store, ledger, index, vectors, pipeline = build_chain(
            self.world, encoder,
            minimum_score=minimum_score,
            minimum_coverage=minimum_coverage,
            known_margin=known_margin,
        )
        pipeline.reasoner = make_reasoner(self.world)
        pipeline.multihop = MultiHopResolver(TransformationGraph(self.world))
        pipeline.access_gate = ObserverGate(
            company_of=lambda oid: self._id_meta.get(oid, {}).get("company"),
            region_of=lambda oid: self._id_meta.get(oid, {}).get("region"),
        )
        self.store = store
        self.ledger = ledger
        self.index = index
        self.pipeline = pipeline
        self._documents = {
            rec.object_id: rec.document for rec, _ in store.live_records()
        }

    def resolve(self, text: str, observer_frame=None,
                query_type: str | None = None) -> Answer:
        result = self.pipeline.resolve(text, query_type=query_type,
                                       observer_frame=observer_frame)
        evidence = {}
        if result.status == CandidateStatus.SUPPORTED and result.object_id:
            evidence = self._documents.get(result.object_id, {})
        trace = [part.removeprefix("trace=") for part in result.reason.split(";") if part]
        return Answer(
            status=result.status.value,
            path=result.path,
            object_id=result.object_id if result.status == CandidateStatus.SUPPORTED else None,
            score=result.score,
            reason=result.reason,
            coverage=result.coverage,
            evidence=evidence,
            trace=trace,
            stage_ms=result.stage_ms,
        )
