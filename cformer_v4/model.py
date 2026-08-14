from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cformer_v3.data import ANSWER_CLASSES

from .data import STATUS_NAMES, TOKEN_VOCAB_SIZE


@dataclass
class V4Config:
    token_vocab_size: int = TOKEN_VOCAB_SIZE
    d_model: int = 48
    d_index: int = 24
    n_heads: int = 4
    d_ff: int = 96
    dense_layers: int = 2
    dropout: float = 0.0
    reliability_topk: int = 8


class SharedFieldEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, config: V4Config, fields: int) -> None:
        super().__init__()
        self.embedding = embedding
        self.field_embedding = nn.Parameter(torch.empty(1, 1, fields, config.d_model))
        self.projection = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        nn.init.normal_(self.field_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).unsqueeze(-1)
        embedded = (self.embedding(tokens) + self.field_embedding[:, :, : tokens.shape[-1]]) * mask
        pooled = embedded.sum(dim=-2) / mask.sum(dim=-2).clamp_min(1)
        return pooled + self.projection(pooled)


class RecurrentQueryBlock(nn.Module):
    def __init__(self, config: V4Config) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(config.d_model)
        self.m_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, config.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, query: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.q_norm(query)[:, None], self.m_norm(evidence), self.m_norm(evidence), need_weights=False
        )
        query = query + attended[:, 0]
        return query + self.ffn(self.ffn_norm(query))


class ReliableCFormer(nn.Module):
    def __init__(self, config: V4Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.token_vocab_size, config.d_model, padding_idx=0)
        self.fact_encoder = SharedFieldEncoder(self.embedding, config, 4)
        self.question_encoder = SharedFieldEncoder(self.embedding, config, 4)
        self.observer_transform = nn.Linear(config.d_model, config.d_model)
        self.fusion_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.index_key = nn.Linear(config.d_model, config.d_index)
        self.index_query = nn.Linear(config.d_model, config.d_index)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.hop_embedding = nn.Parameter(torch.empty(2, config.d_model))
        self.query_block = RecurrentQueryBlock(config)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.answer_head = nn.Linear(config.d_model, ANSWER_CLASSES)
        self.status_head = nn.Linear(config.d_model, len(STATUS_NAMES))
        self.retrieval_status_head = nn.Linear(4, len(STATUS_NAMES), bias=False)
        nn.init.zeros_(self.retrieval_status_head.weight)
        nn.init.normal_(self.hop_embedding, std=0.02)

    def encode_world(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        exact = self.fact_encoder(memory)
        index = F.normalize(self.index_key(exact), dim=-1)
        return exact, index

    def _query(self, question: torch.Tensor, observer: torch.Tensor) -> torch.Tensor:
        query = self.question_encoder(question[:, None]).squeeze(1)
        observer_vector = self.embedding(observer)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((query, observer_vector), dim=-1)))
        return query + gate * self.observer_transform(observer_vector)

    def _retrieve(
        self,
        query: torch.Tensor,
        index: torch.Tensor,
        allowed: torch.Tensor,
        hop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indexed_query = F.normalize(self.index_query(query + self.hop_embedding[hop]), dim=-1)
        scores = self.logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bd,bnd->bn", indexed_query, index
        )
        scores = scores.masked_fill(~allowed, -1e4)
        selected = scores.argmax(dim=-1)
        return scores, selected

    def _reliability_features(
        self, scores: list[torch.Tensor], exact: torch.Tensor
    ) -> torch.Tensor:
        """Measure separation from the best non-equivalent evidence.

        Repeated copies of the same encoded fact are agreement, not ambiguity. The
        previous top-1/top-2 margin collapsed to zero for duplicate evidence and
        caused false refusals as memory grew.
        """
        features = []
        for hop_scores in scores:
            k = min(self.config.reliability_topk, hop_scores.shape[-1])
            top_values, top_indices = hop_scores.topk(k, dim=-1)
            candidate_exact = exact.gather(
                1, top_indices.unsqueeze(-1).expand(-1, -1, exact.shape[-1])
            )
            top_exact = candidate_exact[:, :1]
            non_equivalent = (
                candidate_exact[:, 1:] - top_exact
            ).abs().amax(dim=-1).gt(1e-6)
            runner_up = top_values[:, 1:].masked_fill(~non_equivalent, -1e4).max(dim=-1).values
            has_runner_up = non_equivalent.any(dim=-1)
            margin = top_values[:, 0] - runner_up
            # More than k equivalent copies are strong repeated support. Saturating
            # the downstream tanh avoids treating their zero raw margin as conflict.
            margin = torch.where(has_runner_up, margin, torch.full_like(margin, 100.0))
            features.extend((top_values[:, 0], margin))
        return torch.tanh(torch.stack(features, dim=-1) / 10.0)

    def answer_from_cache(
        self,
        cache: tuple[torch.Tensor, torch.Tensor],
        question: torch.Tensor,
        observer: torch.Tensor,
        allowed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        exact, index = cache
        query = self._query(question, observer)
        scores: list[torch.Tensor] = []
        selected: list[torch.Tensor] = []
        evidence: list[torch.Tensor] = []
        rows = torch.arange(query.shape[0], device=query.device)
        for hop in range(2):
            hop_scores, hop_selected = self._retrieve(query, index, allowed, hop)
            scores.append(hop_scores)
            selected.append(hop_selected)
            evidence.append(exact[rows, hop_selected])
            query = self.query_block(query, torch.stack(evidence, dim=1))
        hidden = self.final_norm(query)
        reliability = self._reliability_features(scores, exact)
        return {
            "answer_logits": self.answer_head(hidden),
            "status_logits": self.status_head(hidden) + self.retrieval_status_head(reliability),
            "scores_first": scores[0],
            "scores_second": scores[1],
            "selected_first": selected[0],
            "selected_second": selected[1],
        }

    def forward(self, memory, question, observer, allowed):
        return self.answer_from_cache(self.encode_world(memory), question, observer, allowed)


class EvidenceRAGTransformer(ReliableCFormer):
    """Evidence-supervised RAG control without observer-specific gated fusion."""

    def _query(self, question: torch.Tensor, observer: torch.Tensor) -> torch.Tensor:
        query = self.question_encoder(question[:, None]).squeeze(1)
        # Conventional conditioning: observer identity is added as another query feature.
        # The inherited gate/transform parameters remain allocated but are deliberately
        # unused so the parameter budget exactly matches ReliableCFormer.
        return query + self.embedding(observer)


class ReliableDenseTransformer(nn.Module):
    def __init__(self, config: V4Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.token_vocab_size, config.d_model, padding_idx=0)
        self.fact_encoder = SharedFieldEncoder(self.embedding, config, 4)
        self.question_encoder = SharedFieldEncoder(self.embedding, config, 4)
        self.cls = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.type_embedding = nn.Parameter(torch.empty(1, 3, config.d_model))
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.dense_layers, nn.LayerNorm(config.d_model)
        )
        self.answer_head = nn.Linear(config.d_model, ANSWER_CLASSES)
        self.status_head = nn.Linear(config.d_model, len(STATUS_NAMES))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(self, memory, question, observer, allowed):
        batch = memory.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        observer_encoded = self.embedding(observer)[:, None] + self.type_embedding[:, 0:1]
        question_encoded = self.question_encoder(question[:, None]) + self.type_embedding[:, 1:2]
        facts = self.fact_encoder(memory) + self.type_embedding[:, 2:3]
        tokens = torch.cat((cls, observer_encoded, question_encoded, facts), dim=1)
        prefix_allowed = torch.ones(batch, 3, dtype=torch.bool, device=allowed.device)
        key_padding_mask = ~torch.cat((prefix_allowed, allowed), dim=1)
        hidden = self.encoder(tokens, src_key_padding_mask=key_padding_mask)[:, 0]
        return {
            "answer_logits": self.answer_head(hidden),
            "status_logits": self.status_head(hidden),
        }
