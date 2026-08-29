# -*- coding: utf-8 -*-
"""跨域迁移实验（轻量快速版）：在 AI 模型域训练，在「国家」域评测。

为在 4GB 笔记本上几分钟内跑完：
- 小批量训练（batch=64，替代全批 1100 条）；
- d=64（替代 256）；
- 300 步 / 3 种子。

验证核心问题：模型学到的是「从四证据推断身份」的通用能力，还是只背 AI 模型规律？
国家域 identity_top1 显著高于随机基线（1/对象数）即迁移成立。

用法：
    D:/conda/envs/cformer-gpu/python.exe train_eval_cross.py --steps 300 --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld
from train_eval_real import evaluate

ROOT = Path(__file__).resolve().parent
AI_DATA = ROOT / "data" / "ai_models_dataset.json"
COUNTRIES_DATA = ROOT / "data" / "countries_dataset.json"
AI_DOMAIN = "ai_models_dataset"
COUNTRIES_DOMAIN = "countries_dataset"


def train_minibatch(model, world: AIModelWorld, device, *, steps: int, lr: float,
                    batch_size: int = 64, reject_weight: float = 2.0,
                    margin_weight: float = 1.0, margin_target: float = 0.05,
                    reject_target: float = 0.45, score_floor: float = 0.50,
                    score_weight: float = 1.5, domain: str | None = None,
                    seed: int = 1) -> dict:
    rng = random.Random(seed * 7919)
    known = world.known_queries("train", domain=domain)
    ambiguous = world.ambiguous_queries("train", domain=domain)
    unknown = world.unknown_queries("train", domain=domain)
    bank = world.encode_candidates(world.objects)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device).train()
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        bank_vec = F.normalize(model.encode_candidate(bank.to(device)), dim=-1)

        sample = rng.sample(known, min(batch_size, len(known)))
        kq = torch.stack([world.encode_query(q["text"])[0] for q in sample])
        kp = world.encode_candidates(
            [world.objects[world.target_label(q["target_id"])] for q in sample]
        )

        optimizer.zero_grad(set_to_none=True)
        loss = model.contrastive_loss(kq.to(device), kp.to(device))

        if unknown:
            uq = torch.stack([world.encode_query(q["text"])[0] for q in unknown])
            uvec = F.normalize(model.encode_query(uq.to(device)), dim=-1)
            loss = loss + reject_weight * torch.relu((uvec @ bank_vec.T).max(dim=-1).values - reject_target).mean()

        if ambiguous:
            aq = torch.stack([world.encode_query(q["text"])[0] for q in ambiguous])
            avec = F.normalize(model.encode_query(aq.to(device)), dim=-1)
            scores = avec @ bank_vec.T
            top2 = torch.topk(scores, 2, dim=-1).values
            loss = loss + margin_weight * torch.relu(top2[:, 0] - top2[:, 1] - margin_target).mean()
            loss = loss + score_weight * torch.relu(score_floor - top2[:, 0]).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "initial_loss": statistics.mean(losses[: min(10, len(losses))]),
        "final_loss": statistics.mean(losses[-min(10, len(losses)):]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=Path,
                        default=[AI_DATA, COUNTRIES_DATA, ROOT / "data" / "movies_dataset.json"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-domain", type=str, default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "cross_domain_results.json")
    args = parser.parse_args()

    train_domain = None if args.train_domain == "all" else args.train_domain

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(*args.datasets)  # 合并：共享 tokenizer、域标签
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=args.d_model, heads=4,
        ffn_dimensions=args.d_model * 2, output_dimensions=32,
    )
    domains = sorted({obj.domain for obj in world.objects})
    print(f"合并对象数: {len(world.objects)}  "
          f"各域: {[(d, len(world.objects_by_domain(d))) for d in domains]}", flush=True)

    per_seed = {}
    checkpoint_dir = ROOT / "artifacts" / "cross_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        training = train_minibatch(model, world, device, steps=args.steps, lr=args.lr,
                                   batch_size=args.batch_size, domain=train_domain, seed=seed)
        torch.save(model.state_dict(), checkpoint_dir / f"cross_seed{seed}.pt")
        per_domain = {d: evaluate(model, world, device, domain=d)["heldout"] for d in domains}
        per_seed[str(seed)] = {"training_seconds": training["seconds"], "per_domain": per_domain}
        summary = {"phase": "seed", "seed": seed, "train_seconds": round(training["seconds"], 1)}
        for d in domains:
            summary[f"{d}_identity_top1"] = per_domain[d].get("identity_top1")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def mean_of(key: str, domain: str) -> float:
        return statistics.mean(
            per_seed[str(seed)]["per_domain"][domain].get(key, float("nan"))
            for seed in args.seeds
        )

    aggregate = {
        "per_domain": {
            d: {
                "identity_top1": mean_of("identity_top1", d),
                "known_top1": mean_of("known_top1", d),
                "ambiguous": mean_of("ambiguous_detected", d),
                "unknown_not_supported": mean_of("unknown_not_supported", d),
                "random_baseline": 1 / max(1, len(world.objects_by_domain(d))),
            }
            for d in domains
        },
        "mean_train_seconds_per_seed": statistics.mean(
            per_seed[str(seed)]["training_seconds"] for seed in args.seeds
        ),
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "settings": {key: str(value) for key, value in vars(args).items()},
        "note": "多域训练/评测：--train-domain=all 联合训练所有域；identity_top1 与各域随机基线对比",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
