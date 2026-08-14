from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .data import TOKEN_VOCAB_SIZE, WorldTask


@dataclass
class V2Config:
    token_vocab_size: int = TOKEN_VOCAB_SIZE
    d_model: int = 64
    n_heads: int = 4
    memory_layers: int = 2
    query_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class StructuredTokenEncoder(nn.Module):
    def __init__(self, config: V2Config, fields: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.token_vocab_size, config.d_model, padding_idx=0)
        self.field_embedding = nn.Parameter(torch.empty(1, 1, fields, config.d_model))
        self.network = nn.Sequential(
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
        return pooled + self.network(pooled)


class MemoryBackbone(nn.Module):
    def __init__(self, config: V2Config) -> None:
        super().__init__()
        self.fact_encoder = StructuredTokenEncoder(config, WorldTask.fact_fields)
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.memory_layers, nn.LayerNorm(config.d_model))

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.fact_encoder(memory_tokens))


class ObserverCrossAttention(nn.Module):
    def __init__(self, config: V2Config, use_observer: bool) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.use_observer = use_observer
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.observer_q = nn.Linear(config.d_model, config.d_model)
        self.relation_k = nn.Linear(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, observer: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.q(query)[:, None, :])
        k = self._heads(self.k(memory))
        v = self._heads(self.v(memory))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.use_observer:
            oq = self._heads(self.observer_q(observer)[:, None, :])
            rk = self._heads(self.relation_k(memory))
            scores = scores + torch.matmul(oq, rk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = scores.softmax(dim=-1)
        value = torch.matmul(weights, v).transpose(1, 2).reshape(query.shape[0], -1)
        return self.out(value)


class QueryBlock(nn.Module):
    def __init__(self, config: V2Config, use_observer: bool) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.attention = ObserverCrossAttention(config, use_observer)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor, observer: torch.Tensor) -> torch.Tensor:
        query = query + self.attention(
            self.query_norm(query), self.memory_norm(memory), observer
        )
        return query + self.ffn(self.ffn_norm(query))


class SharedMemoryQA(nn.Module):
    """Encode a world once, then issue many question/observer queries against it."""

    def __init__(self, config: V2Config, use_observer: bool = True) -> None:
        super().__init__()
        self.config = config
        self.use_observer = use_observer
        self.memory_backbone = MemoryBackbone(config)
        self.question_encoder = StructuredTokenEncoder(config, WorldTask.question_fields)
        self.observer_embedding = nn.Embedding(config.token_vocab_size, config.d_model)
        self.observer_transform = nn.Linear(config.d_model, config.d_model)
        self.fusion_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.blocks = nn.ModuleList(
            [QueryBlock(config, use_observer) for _ in range(config.query_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, WorldTask.answer_classes)

    def encode_memory(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        return self.memory_backbone(memory_tokens)

    def answer_from_memory(
        self,
        encoded_memory: torch.Tensor,
        question_tokens: torch.Tensor,
        observer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        query = self.question_encoder(question_tokens[:, None, :]).squeeze(1)
        observer = self.observer_embedding(observer_tokens)
        if self.use_observer:
            gate = torch.sigmoid(self.fusion_gate(torch.cat((query, observer), dim=-1)))
            query = query + gate * self.observer_transform(observer)
        else:
            observer = torch.zeros_like(observer)
        for block in self.blocks:
            query = block(query, encoded_memory, observer)
        return self.classifier(self.final_norm(query))

    def forward(
        self,
        memory_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        observer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.answer_from_memory(
            self.encode_memory(memory_tokens), question_tokens, observer_tokens
        )


class ConcatTransformerQA(nn.Module):
    """Baseline that re-encodes world, observer and question together for every query."""

    def __init__(self, config: V2Config) -> None:
        super().__init__()
        self.config = config
        self.fact_encoder = StructuredTokenEncoder(config, WorldTask.fact_fields)
        self.question_encoder = StructuredTokenEncoder(config, WorldTask.question_fields)
        self.observer_embedding = nn.Embedding(config.token_vocab_size, config.d_model)
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
            layer,
            config.memory_layers + config.query_layers,
            nn.LayerNorm(config.d_model),
        )
        self.classifier = nn.Linear(config.d_model, WorldTask.answer_classes)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(
        self,
        memory_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        observer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = memory_tokens.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        observer = self.observer_embedding(observer_tokens)[:, None, :] + self.type_embedding[:, 0:1]
        question = self.question_encoder(question_tokens[:, None, :]) + self.type_embedding[:, 1:2]
        memory = self.fact_encoder(memory_tokens) + self.type_embedding[:, 2:3]
        hidden = self.encoder(torch.cat((cls, observer, question, memory), dim=1))
        return self.classifier(hidden[:, 0])

