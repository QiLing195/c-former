# -*- coding: utf-8 -*-
"""V6.3 神经递归块（共享参数，v2 实验件）。

确定性递归层（v1）在关系图上已拿 100%，本文件实现设计文档 §5 的「参数闸门」
对照：**共享参数递归块 vs 直接堆叠**。核心问题——"最新成员识别"这类跨候选
比较，能否用**共享权重的迭代传播**（同一组块重复 hops 次）在相近准确率下
比直接堆叠（每层独立参数）节省 ≥30% 推理层参数。

任务设定（链条目，可程序化构造）：
- 输入：一个系列的成员特征序列（按链序：前代 → 后继），长度 L；
- 目标：预测**链尾索引**（= 系列最新成员，与 V6.3 确定性 latest 同语义）；
- 架构 A（共享递归）：一层传播块 W 重复 hops 次，参数 = W 一次；
- 架构 B（直接堆叠）：hops 个独立块，参数 = hops × W；
- 数据：真实链（AI/电影/国家三域的关系链），链序正确标注；
- 闸门：A ≈ B 准确率，且 A 参数 < 0.7 × B 参数（省 ≥30%）。

注意：这是「表示传播能否学会链尾语义」的架构对照，不是要替代 v1 确定性层
（v1 已 100% 且零参数）；若 A 显著差于 B，说明跨候选比较确实需要独立参数，
即为确定性层 v1 的架构必要性提供反向证据。成员特征用固定随机向量（两架构
共用、冻结），专注纯架构差异。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA_PATHS = [
    ROOT / "data" / "ai_models_dataset.json",
    ROOT / "data" / "movies_dataset.json",
    ROOT / "data" / "countries_recursion.json",
]


def load_chains() -> list[dict]:
    """从多域数据集抽取「有前代链的系列」→ [{series, ids, years}]，链序 = 前代→后继。"""
    chains: list[dict] = []
    for path in DATA_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        objs = payload["objects"]
        by_series: dict[str, list[dict]] = {}
        for obj in objs:
            by_series.setdefault(obj["series"], []).append(obj)
        for series, members in by_series.items():
            members_with_pred = [m for m in members if m.get("predecessor")]
            if not members_with_pred:
                continue  # 无前代链的系列（genre 组/单对象）跳过
            ids = [m["id"] for m in members]
            by_id = {m["id"]: m for m in members}
            # 链头 = 自己没有前驱（predecessor 为 None 或指向成员外）
            heads = [i for i in ids if by_id[i].get("predecessor") not in ids]
            if len(heads) != 1:
                continue  # 只取单链头系列（避免多链）
            ordered: list[str] = []
            current = heads[0]
            while current is not None and current not in ordered:
                ordered.append(current)
                current = next((m["id"] for m in members if m.get("predecessor") == current), None)
            # 有效链：所有有前驱的成员都在 ordered 中（变体 predecessor=None 不在链上，允许缺席）
            if len(ordered) < 2 or not all(m["id"] in ordered for m in members_with_pred):
                continue
            chains.append({"series": series, "ids": ordered,
                           "years": [int(by_id[i]["year"]) for i in ordered]})
    return chains


class SharedRecurBlock(nn.Module):
    """单层传播块：h_i' = h_i + gating·tanh(W[h_i; h_{i-1}])，同组权重跨 hop 复用。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        inp = torch.cat([h, h_prev], dim=-1)
        g = torch.sigmoid(self.gate(inp))
        return h + g * torch.tanh(self.proj(inp))


class ChainHeadResolver(nn.Module):
    """链尾识别器：成员特征 → 传播块迭代 hops 次 → 预测链尾索引。

    shared=True：1 个块复用 hops 次（参数 O(1)）；shared=False：hops 个独立块。
    """

    def __init__(self, member_dim: int, hidden: int, hops: int, shared: bool) -> None:
        super().__init__()
        self.hops = hops
        self.member_proj = nn.Linear(member_dim, hidden)
        self.head_proj = nn.Linear(hidden, 1)
        n_blocks = 1 if shared else hops
        self.blocks = nn.ModuleList([SharedRecurBlock(hidden) for _ in range(n_blocks)])

    def forward(self, members: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.member_proj(members))  # (B, L, hidden)
        for _ in range(self.hops):
            for block in self.blocks:
                h = block(h)
        return self.head_proj(h).squeeze(-1)  # (B, L)


def build_batches(chains: list[dict], dim: int, max_len: int,
                  device, seed: int, semantic: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """成员特征：固定随机向量（两架构共用、冻结）；semantic=True 时追加年份归一化通道
    （模型可学「年份最大 = 链尾」），对照纯随机特征下的可学性差异。"""
    rng = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for chain in chains:
        n = len(chain["ids"])
        if n < 2 or n > max_len:
            continue
        feat = torch.randn(n, dim, generator=rng)
        if semantic:
            years = torch.tensor(chain["years"], dtype=torch.float32)
            ymin, ymax = years.min(), years.max()
            year_norm = (years - ymin) / max(1.0, (ymax - ymin).item())
            feat = torch.cat([feat, year_norm.unsqueeze(-1)], dim=-1)
        xs.append(feat)
        ys.append(n - 1)
    L = max(x.shape[0] for x in xs)
    padded = torch.zeros(len(xs), L, xs[0].shape[-1], device=device)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x
    targets = torch.tensor(ys, device=device, dtype=torch.long)
    return padded, targets


def run(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chains = load_chains()
    print(f"chains={len(chains)}")

    report = {"note": "共享递归 vs 直接堆叠：相近准确率下参数节省≥30% 闸门（设计文档§5）"}
    for semantic in (False, True):
        dim = args.dim + (1 if semantic else 0)
        X, y = build_batches(chains, args.dim, args.max_len, device, args.seed, semantic)
        n = X.shape[0]
        split = int(n * 0.8)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]
        print(f"semantic={semantic} samples={n} train={split} test={n - split}")

        results = {}
        for shared in (True, False):
            torch.manual_seed(args.seed)
            model = ChainHeadResolver(dim, args.hidden, args.hops, shared).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            for step in range(args.steps):
                model.train()
                opt.zero_grad()
                loss = F.cross_entropy(model(X_tr), y_tr)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                acc_tr = (model(X_tr).argmax(-1) == y_tr).float().mean().item()
                acc_te = (model(X_te).argmax(-1) == y_te).float().mean().item()
            name = "shared" if shared else "stacked"
            results[name] = {"params": n_params, "train_acc": round(acc_tr, 4),
                             "test_acc": round(acc_te, 4)}
            print(json.dumps({"semantic": semantic, "arch": name, **results[name]}, ensure_ascii=False))

        shared, stacked = results["shared"], results["stacked"]
        saving = 1.0 - shared["params"] / stacked["params"]
        gate = saving >= 0.30 and shared["test_acc"] >= stacked["test_acc"] - 0.05
        report[f"semantic={semantic}"] = {
            "shared": shared, "stacked": stacked,
            "param_saving": round(saving, 4),
            "gate_params_saving_30pct": bool(saving >= 0.30),
            "gate_accuracy_close": bool(shared["test_acc"] >= stacked["test_acc"] - 0.05),
            "gate_passed": bool(gate),
        }
    output = ROOT / "artifacts" / "v63_neural_recursion.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "report": report}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    run(parser.parse_args())
