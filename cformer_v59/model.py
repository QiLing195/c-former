from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    mask = tokens.ne(0).unsqueeze(-1)
    return (values * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1)


@dataclass(frozen=True)
class DualEncoderConfig:
    vocabulary_size: int
    d_model: int = 64
    embedding_dimensions: int = 64
    evidence_fields: int = 4
    temperature: float = 0.07


class SemanticDualEncoder(nn.Module):
    """Observer-independent identity resolver with gated multi-evidence fusion."""

    def __init__(self, config: DualEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.d_model, padding_idx=0)
        self.query_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.embedding_dimensions),
        )
        self.evidence_type = nn.Embedding(config.evidence_fields, config.d_model)
        self.evidence_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.embedding_dimensions),
        )
        self.evidence_gate = nn.Linear(config.d_model, 1)
        self.candidate_projection = nn.Linear(
            config.embedding_dimensions, config.embedding_dimensions, bias=False
        )

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = _masked_mean(self.embedding(tokens), tokens)
        return F.normalize(self.query_projection(pooled), dim=-1)

    def encode_candidate(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, evidence, token]
        embedded = self.embedding(tokens)
        pooled = _masked_mean(embedded, tokens)
        types = self.evidence_type.weight[: tokens.shape[1]].unsqueeze(0)
        typed = pooled + types
        evidence = self.evidence_projection(typed)
        gates = torch.softmax(self.evidence_gate(typed).squeeze(-1), dim=-1)
        fused = torch.sum(evidence * gates.unsqueeze(-1), dim=1)
        return F.normalize(self.candidate_projection(fused), dim=-1)

    def contrastive_loss(
        self,
        query_tokens: torch.Tensor,
        positive_tokens: torch.Tensor,
        hard_negative_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.encode_query(query_tokens)
        positive = self.encode_candidate(positive_tokens)
        logits = query @ positive.T / self.config.temperature
        if hard_negative_tokens is not None:
            hard = self.encode_candidate(hard_negative_tokens)
            hard_score = torch.sum(query * hard, dim=-1, keepdim=True)
            logits = torch.cat((logits, hard_score / self.config.temperature), dim=-1)
        labels = torch.arange(query.shape[0], device=query.device)
        return F.cross_entropy(logits, labels)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class FlatTransformerDualEncoder(nn.Module):
    """Vanilla Transformer dual-encoder baseline over flattened evidence text."""

    def __init__(
        self,
        vocabulary_size: int,
        *,
        d_model: int = 64,
        embedding_dimensions: int = 64,
        layers: int = 2,
        heads: int = 4,
        temperature: float = 0.07,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.position = nn.Parameter(torch.empty(max_length, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 2, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.projection = nn.Linear(d_model, embedding_dimensions)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3:
            tokens = tokens.flatten(1)
        mask = tokens.eq(0)
        values = self.embedding(tokens) + self.position[: tokens.shape[1]]
        values = self.encoder(values, src_key_padding_mask=mask)
        pooled = _masked_mean(values, tokens)
        return F.normalize(self.projection(pooled), dim=-1)

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._encode(tokens)

    def encode_candidate(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._encode(tokens)

    def contrastive_loss(self, query_tokens, positive_tokens, hard_negative_tokens=None):
        query = self.encode_query(query_tokens)
        positive = self.encode_candidate(positive_tokens)
        logits = query @ positive.T / self.temperature
        if hard_negative_tokens is not None:
            hard = self.encode_candidate(hard_negative_tokens)
            logits = torch.cat(
                (logits, torch.sum(query * hard, dim=-1, keepdim=True) / self.temperature),
                dim=-1,
            )
        labels = torch.arange(query.shape[0], device=query.device)
        return F.cross_entropy(logits, labels)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

