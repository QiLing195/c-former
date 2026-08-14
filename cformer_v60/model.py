from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    mask = tokens.ne(0).unsqueeze(-1)
    return (values * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1)


@dataclass(frozen=True)
class ChineseTransformerConfig:
    vocabulary_size: int
    layers: int = 4
    d_model: int = 256
    heads: int = 8
    ffn_dimensions: int = 768
    output_dimensions: int = 64
    max_length: int = 128
    evidence_fields: int = 4
    temperature: float = 0.07


class SharedTokenTransformer(nn.Module):
    def __init__(self, config: ChineseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.d_model, padding_idx=0)
        self.position = nn.Parameter(torch.empty(config.max_length, config.d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.heads,
            config.ffn_dimensions,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.layers, norm=nn.LayerNorm(config.d_model)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] > self.config.max_length:
            raise ValueError("token sequence exceeds configured maximum")
        padding = tokens.eq(0)
        values = self.embedding(tokens) + self.position[: tokens.shape[-1]]
        values = self.encoder(values, src_key_padding_mask=padding)
        return _masked_mean(values, tokens)


class ContrastiveResolverMixin:
    """Shared in-batch contrastive loss and parameter counting for all resolvers."""

    def contrastive_loss(self, query_tokens, positive_tokens, hard_negative_tokens=None):
        query = self.encode_query(query_tokens)
        positive = self.encode_candidate(positive_tokens)
        logits = query @ positive.T / self.config.temperature
        if hard_negative_tokens is not None:
            hard = self.encode_candidate(hard_negative_tokens)
            logits = torch.cat(
                (
                    logits,
                    torch.sum(query * hard, dim=-1, keepdim=True)
                    / self.config.temperature,
                ),
                dim=-1,
            )
        return F.cross_entropy(
            logits, torch.arange(query.shape[0], device=query.device)
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TokenCFormerResolver(ContrastiveResolverMixin, nn.Module):
    """Shared token Transformer plus explicit four-evidence candidate fusion."""

    def __init__(self, config: ChineseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SharedTokenTransformer(config)
        self.query_projection = nn.Linear(config.d_model, config.output_dimensions)
        self.evidence_type = nn.Embedding(config.evidence_fields, config.d_model)
        self.evidence_projection = nn.Linear(config.d_model, config.output_dimensions)
        self.evidence_gate = nn.Linear(config.d_model, 1)
        self.candidate_projection = nn.Linear(
            config.output_dimensions, config.output_dimensions, bias=False
        )

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_projection(self.backbone(tokens)), dim=-1)

    def encode_candidate(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, fields, length = tokens.shape
        pooled = self.backbone(tokens.reshape(batch * fields, length)).reshape(
            batch, fields, self.config.d_model
        )
        typed = pooled + self.evidence_type.weight[:fields].unsqueeze(0)
        evidence = self.evidence_projection(typed)
        gates = torch.softmax(self.evidence_gate(typed).squeeze(-1), dim=-1)
        fused = torch.sum(evidence * gates.unsqueeze(-1), dim=1)
        return F.normalize(self.candidate_projection(fused), dim=-1)


class MeanPoolMLPResolver(ContrastiveResolverMixin, nn.Module):
    """V5.9-style order-insensitive baseline using the same Chinese tokens."""

    def __init__(self, config: ChineseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.d_model, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.output_dimensions),
        )
        self.evidence_type = nn.Embedding(config.evidence_fields, config.d_model)
        self.evidence_gate = nn.Linear(config.d_model, 1)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return _masked_mean(self.embedding(tokens), tokens)

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self._pool(tokens)), dim=-1)

    def encode_candidate(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, fields, length = tokens.shape
        pooled = self._pool(tokens.reshape(batch * fields, length)).reshape(
            batch, fields, self.config.d_model
        )
        typed = pooled + self.evidence_type.weight[:fields].unsqueeze(0)
        gates = torch.softmax(self.evidence_gate(typed).squeeze(-1), dim=-1)
        fused = torch.sum(pooled * gates.unsqueeze(-1), dim=1)
        return F.normalize(self.projection(fused), dim=-1)


class FlatTransformerResolver(ContrastiveResolverMixin, nn.Module):
    """Fair shared-Transformer baseline without explicit evidence gating."""

    def __init__(self, config: ChineseTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SharedTokenTransformer(config)
        self.projection = nn.Linear(config.d_model, config.output_dimensions)

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(tokens)), dim=-1)

    def encode_candidate(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(tokens.flatten(1))), dim=-1)

