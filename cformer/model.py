from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CFormerConfig:
    token_vocab_size: int = 16
    observer_vocab_size: int = 5
    max_seq_len: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0
    use_ego_projection: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


class ObserverEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, observer_tokens: torch.Tensor) -> torch.Tensor:
        mask = observer_tokens.ne(0).unsqueeze(-1)
        embedded = self.embedding(observer_tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.network(pooled)


class OPModulatedAttention(nn.Module):
    """Multi-head attention whose Q/K feature geometry is conditioned on observer o."""

    def __init__(self, config: CFormerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.modulator = nn.Linear(config.d_model, 3 * config.d_model)
        # Exact identity at initialization: Q'=Q and K'=K.
        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, observer: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        gamma_q, beta_q, gamma_k = self.modulator(observer).chunk(3, dim=-1)

        q = q * (1.0 + torch.tanh(gamma_q[:, None, :])) + beta_q[:, None, :]
        k = k * (1.0 + torch.tanh(gamma_k[:, None, :]))

        q, k, v = map(self._split_heads, (q, k, v))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        batch, _, length, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out(attended)


class SafeEgoProjection(nn.Module):
    """Remove a gated primary direction from non-primary tokens only."""

    def __init__(self) -> None:
        super().__init__()
        # sigmoid(-4) ~= 0.018, so projection begins nearly disabled.
        self.raw_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor, primary_positions: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        row = torch.arange(batch, device=x.device)
        primary = x[row, primary_positions]
        direction = F.normalize(primary, dim=-1, eps=1e-6)
        component = (x * direction[:, None, :]).sum(dim=-1, keepdim=True)
        component = component * direction[:, None, :]

        non_primary = torch.ones(batch, length, 1, device=x.device, dtype=x.dtype)
        non_primary[row, primary_positions] = 0.0
        return x - torch.sigmoid(self.raw_gate) * component * non_primary


class CFormerBlock(nn.Module):
    def __init__(self, config: CFormerConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = OPModulatedAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.ego_projection = SafeEgoProjection() if config.use_ego_projection else None

    def forward(
        self,
        x: torch.Tensor,
        observer: torch.Tensor,
        primary_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), observer))
        if self.ego_projection is not None:
            if primary_positions is None:
                raise ValueError("primary_positions are required when ego projection is enabled")
            x = self.ego_projection(x, primary_positions)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class CFormer(nn.Module):
    def __init__(self, config: CFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.token_vocab_size, config.d_model)
        self.cls = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.max_seq_len + 1, config.d_model)
        )
        self.observer_encoder = ObserverEncoder(config.observer_vocab_size, config.d_model)
        self.blocks = nn.ModuleList([CFormerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.token_vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        observer_tokens: torch.Tensor,
        primary_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = tokens.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds {self.config.max_seq_len}")

        observer = self.observer_encoder(observer_tokens)
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat((cls, self.token_embedding(tokens)), dim=1)
        x = x + self.position_embedding[:, : length + 1]

        # External token positions gain one because [CLS] is prepended.
        internal_primary = primary_positions + 1 if primary_positions is not None else None
        for block in self.blocks:
            x = block(x, observer, internal_primary)
        return self.classifier(self.final_norm(x[:, 0]))

    @torch.no_grad()
    def contrast_logits(
        self,
        tokens: torch.Tensor,
        custom_observer_tokens: torch.Tensor,
        default_observer_tokens: torch.Tensor,
        tau: float = 0.3,
    ) -> torch.Tensor:
        custom = self(tokens, custom_observer_tokens)
        default = self(tokens, default_observer_tokens)
        return custom + tau * (custom - default)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

