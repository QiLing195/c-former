from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import ANSWER_CLASSES, TOKEN_VOCAB_SIZE


@dataclass
class V3Config:
    token_vocab_size: int = TOKEN_VOCAB_SIZE
    d_model: int = 48
    d_index: int = 32
    n_heads: int = 4
    d_ff: int = 96
    dense_layers: int = 2
    top_k: int = 1
    dropout: float = 0.0


class FieldEncoder(nn.Module):
    def __init__(self, config: V3Config, fields: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.token_vocab_size, config.d_model, padding_idx=0)
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


class ExactQueryBlock(nn.Module):
    def __init__(self, config: V3Config) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, config.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, query: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(query)[:, None, :]
        memory = self.memory_norm(selected)
        attended, _ = self.attention(normalized, memory, memory, need_weights=False)
        query = query + attended[:, 0]
        return query + self.ffn(self.ffn_norm(query))


class HierarchicalCFormer(nn.Module):
    """Low-dimensional global index plus Top-K exact observer-conditioned reasoning."""

    def __init__(self, config: V3Config) -> None:
        super().__init__()
        self.config = config
        self.fact_encoder = FieldEncoder(config, 4)
        self.question_encoder = FieldEncoder(config, 3)
        self.observer_embedding = nn.Embedding(config.token_vocab_size, config.d_model)
        self.observer_transform = nn.Linear(config.d_model, config.d_model)
        self.fusion_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.index_key = nn.Linear(config.d_model, config.d_index)
        self.first_index_query = nn.Linear(config.d_model, config.d_index)
        self.second_index_query = nn.Linear(config.d_model, config.d_index)
        # Cosine scores in [-1, 1] are too flat for hundreds of candidates.
        # A learned CLIP-style temperature lets retrieval become selective.
        self.first_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.second_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.first_block = ExactQueryBlock(config)
        self.second_block = ExactQueryBlock(config)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, ANSWER_CLASSES)

    def encode_world(self, memory_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        exact = self.fact_encoder(memory_tokens)
        index = F.normalize(self.index_key(exact), dim=-1)
        return exact, index

    def _initial_query(self, question: torch.Tensor, observer_tokens: torch.Tensor) -> torch.Tensor:
        query = self.question_encoder(question[:, None, :]).squeeze(1)
        observer = self.observer_embedding(observer_tokens)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((query, observer), dim=-1)))
        return query + gate * self.observer_transform(observer)

    @staticmethod
    def _select(exact: torch.Tensor, scores: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = scores.topk(min(top_k, scores.shape[-1]), dim=-1).indices
        selected = exact.gather(1, indices[:, :, None].expand(-1, -1, exact.shape[-1]))
        return selected, indices

    def answer_from_cache(
        self,
        cache: tuple[torch.Tensor, torch.Tensor],
        question: torch.Tensor,
        observer_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        exact, index = cache
        query = self._initial_query(question, observer_tokens)
        score_first = self.first_logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bd,bnd->bn", F.normalize(self.first_index_query(query), dim=-1), index
        )
        selected_first, top_first = self._select(exact, score_first, self.config.top_k)
        query = self.first_block(query, selected_first)

        score_second = self.second_logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bd,bnd->bn", F.normalize(self.second_index_query(query), dim=-1), index
        )
        selected_second, top_second = self._select(exact, score_second, self.config.top_k)
        # Preserve first-hop evidence while adding second-hop candidates.
        query = self.second_block(query, torch.cat((selected_first, selected_second), dim=1))
        return {
            "logits": self.classifier(self.final_norm(query)),
            "score_first": score_first,
            "score_second": score_second,
            "top_first": top_first,
            "top_second": top_second,
        }

    def forward(
        self, memory: torch.Tensor, question: torch.Tensor, observer_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.answer_from_cache(self.encode_world(memory), question, observer_tokens)


class DenseConcatTransformer(nn.Module):
    """Standard baseline that re-encodes every fact together with every query."""

    def __init__(self, config: V3Config) -> None:
        super().__init__()
        self.fact_encoder = FieldEncoder(config, 4)
        self.question_encoder = FieldEncoder(config, 3)
        # A standard language Transformer shares token identity across context,
        # question and observer positions. Tying these embeddings gives the
        # dense baseline that same advantage and avoids a deliberately weak control.
        self.question_encoder.embedding = self.fact_encoder.embedding
        self.observer_embedding = self.fact_encoder.embedding
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
        self.classifier = nn.Linear(config.d_model, ANSWER_CLASSES)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(
        self, memory: torch.Tensor, question: torch.Tensor, observer_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch = memory.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        observer = self.observer_embedding(observer_tokens)[:, None] + self.type_embedding[:, 0:1]
        question_encoded = self.question_encoder(question[:, None, :]) + self.type_embedding[:, 1:2]
        facts = self.fact_encoder(memory) + self.type_embedding[:, 2:3]
        encoded = self.encoder(torch.cat((cls, observer, question_encoded, facts), dim=1))
        return {"logits": self.classifier(encoded[:, 0])}
