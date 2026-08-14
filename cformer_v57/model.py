from __future__ import annotations

from dataclasses import dataclass

import torch

from cformer_v56.model import (
    CognitiveCFormerReranker,
    EvidenceRAGReranker,
    RerankerConfig,
)


@dataclass(frozen=True)
class TextRerankerConfig(RerankerConfig):
    key_dimensions: int = 264
    d_model: int = 40
    d_hidden: int = 72
    kind_count: int = 4


class TextCFormerReranker(CognitiveCFormerReranker):
    def __init__(self, config: TextRerankerConfig = TextRerankerConfig()) -> None:
        super().__init__(config)

    def forward(self, candidate_keys, candidate_kinds, candidate_scopes, query_keys, query_kinds, observers, *, use_observer=True):
        neural = super().forward(
            candidate_keys,
            candidate_kinds,
            candidate_scopes,
            query_keys,
            query_kinds,
            observers,
            use_observer=use_observer,
        )
        lexical = torch.einsum("bd,bkd->bk", query_keys, candidate_keys)
        return neural + 20.0 * lexical


class TextEvidenceRAGReranker(EvidenceRAGReranker):
    def __init__(self, config: TextRerankerConfig = TextRerankerConfig()) -> None:
        super().__init__(config)

    def forward(self, candidate_keys, candidate_kinds, candidate_scopes, query_keys, query_kinds, observers, *, use_observer=True):
        neural = super().forward(
            candidate_keys,
            candidate_kinds,
            candidate_scopes,
            query_keys,
            query_kinds,
            observers,
            use_observer=use_observer,
        )
        lexical = torch.einsum("bd,bkd->bk", query_keys, candidate_keys)
        return neural + 20.0 * lexical
