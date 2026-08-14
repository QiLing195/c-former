from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RerankerConfig:
    key_dimensions: int = 8
    d_model: int = 48
    d_hidden: int = 96
    observer_count: int = 4
    kind_count: int = 2


class CognitiveCFormerReranker(nn.Module):
    """Observer-gated neural reranker; receives governed candidates only."""

    def __init__(self, config: RerankerConfig = RerankerConfig()) -> None:
        super().__init__()
        self.config = config
        self.key_encoder = nn.Sequential(
            nn.Linear(config.key_dimensions, config.d_hidden),
            nn.GELU(),
            nn.Linear(config.d_hidden, config.d_model),
        )
        self.kind_embedding = nn.Embedding(config.kind_count, config.d_model)
        self.observer_embedding = nn.Embedding(
            config.observer_count + 1, config.d_model, padding_idx=0
        )
        self.candidate_norm = nn.LayerNorm(config.d_model)
        self.query_norm = nn.LayerNorm(config.d_model)
        self.observer_transform = nn.Linear(config.d_model, config.d_model)
        self.fusion_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.final_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_candidates(
        self,
        candidate_keys: torch.Tensor,
        candidate_kinds: torch.Tensor,
        candidate_scopes: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.key_encoder(candidate_keys)
            + self.kind_embedding(candidate_kinds)
            + self.observer_embedding(candidate_scopes)
        )
        return F.normalize(self.candidate_norm(hidden), dim=-1)

    def encode_query(
        self,
        query_keys: torch.Tensor,
        query_kinds: torch.Tensor,
        observers: torch.Tensor,
        *,
        use_observer: bool = True,
    ) -> torch.Tensor:
        query = self.key_encoder(query_keys) + self.kind_embedding(query_kinds)
        observer = self.observer_embedding(observers if use_observer else torch.zeros_like(observers))
        gate = torch.sigmoid(self.fusion_gate(torch.cat((query, observer), dim=-1)))
        query = query + gate * self.observer_transform(observer)
        return F.normalize(self.final_projection(self.query_norm(query)), dim=-1)

    def forward(
        self,
        candidate_keys: torch.Tensor,
        candidate_kinds: torch.Tensor,
        candidate_scopes: torch.Tensor,
        query_keys: torch.Tensor,
        query_kinds: torch.Tensor,
        observers: torch.Tensor,
        *,
        use_observer: bool = True,
    ) -> torch.Tensor:
        candidates = self.encode_candidates(
            candidate_keys, candidate_kinds, candidate_scopes
        )
        query = self.encode_query(
            query_keys, query_kinds, observers, use_observer=use_observer
        )
        return self.logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bd,bkd->bk", query, candidates
        )


class EvidenceRAGReranker(CognitiveCFormerReranker):
    """Equal-parameter additive observer baseline."""

    def encode_query(
        self,
        query_keys: torch.Tensor,
        query_kinds: torch.Tensor,
        observers: torch.Tensor,
        *,
        use_observer: bool = True,
    ) -> torch.Tensor:
        query = self.key_encoder(query_keys) + self.kind_embedding(query_kinds)
        observer = self.observer_embedding(observers if use_observer else torch.zeros_like(observers))
        # Gate/transform remain allocated for exact parameter parity but are not used.
        query = query + observer
        return F.normalize(self.final_projection(self.query_norm(query)), dim=-1)
