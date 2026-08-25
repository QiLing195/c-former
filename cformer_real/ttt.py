# -*- coding: utf-8 -*-
"""TTT-Linear 查询编码器（Test-Time Training, Sun et al. 2024 思想）。

核心：隐藏状态 W（D x D 线性映射）在输入序列上逐 token 做内层梯度下降
（重构损失 ||W k_t - v_t||^2），使表示在测试时自适应到当前输入的模式。
这里只把 TTT 用在「查询」路径，候选路径仍用共享 Transformer——
对齐 C-Former「身份判定保持稳定，只有查询表示可自适应」的原则。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cformer_v60 import ChineseTransformerConfig, SharedTokenTransformer
from cformer_v60.model import ContrastiveResolverMixin


class TTTLinearLayer(nn.Module):
    def __init__(self, d_model: int, inner_lr: float = 0.05, detach_inner: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.inner_lr = inner_lr
        self.detach_inner = detach_inner
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        # 初始状态 W0：单位阵缩放（从"不变换"开始学习）
        self.init_w = nn.Parameter(torch.eye(d_model) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D)
        batch, length, dim = x.shape
        w = self.init_w.unsqueeze(0).expand(batch, dim, dim).clone()
        outputs = []
        for t in range(length):
            xt = x[:, t]
            k = self.k_proj(xt)
            v = self.v_proj(xt)
            if self.detach_inner:
                k, v = k.detach(), v.detach()
            pred = torch.bmm(w, k.unsqueeze(-1)).squeeze(-1)  # (B, D)
            # 内层梯度：d/dW ||W k - v||^2 = 2 (W k - v) k^T
            grad = 2.0 * torch.bmm((pred - v).unsqueeze(-1), k.unsqueeze(1))  # (B, D, D)
            w = w - self.inner_lr * grad
            out = torch.bmm(w, self.q_proj(xt).unsqueeze(-1)).squeeze(-1)
            outputs.append(out)
        return torch.stack(outputs, dim=1).mean(dim=1)  # (B, D)


class TTTResolver(ContrastiveResolverMixin, nn.Module):
    """查询路径 = TTT-Linear（测试时自适应）；候选路径 = 共享 Transformer（稳定）。"""

    def __init__(self, config: ChineseTransformerConfig, inner_lr: float = 0.05,
                 detach_inner: bool = False) -> None:
        super().__init__()
        self.config = config
        self.inner_lr = inner_lr
        self.backbone = SharedTokenTransformer(config)  # 候选路径（稳定）
        self.query_embedding = nn.Embedding(config.vocabulary_size, config.d_model, padding_idx=0)
        self.query_position = nn.Parameter(torch.empty(config.max_length, config.d_model))
        nn.init.normal_(self.query_position, std=0.02)
        self.ttt = TTTLinearLayer(config.d_model, inner_lr=inner_lr, detach_inner=detach_inner)
        self.query_projection = nn.Linear(config.d_model, config.output_dimensions)
        # 候选融合（与 TokenCFormerResolver 相同）
        self.evidence_type = nn.Embedding(config.evidence_fields, config.d_model)
        self.evidence_projection = nn.Linear(config.d_model, config.output_dimensions)
        self.evidence_gate = nn.Linear(config.d_model, 1)
        self.candidate_projection = nn.Linear(
            config.output_dimensions, config.output_dimensions, bias=False
        )

    def encode_query(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.query_embedding(tokens) + self.query_position[: tokens.shape[-1]]
        pooled = self.ttt(x)
        return F.normalize(self.query_projection(pooled), dim=-1)

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
