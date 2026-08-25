# -*- coding: utf-8 -*-
"""TTT-Linear 查询编码对照实验：训练 + train/heldout 评测。

与 train_eval_real.py 的唯一区别：查询编码器换成 TTTResolver（测试时自适应），
其余（数据、边界损失、分割、评测）完全一致，保证可比性。

用法：
    D:/conda/envs/cformer-gpu/python.exe train_ttt_real.py --steps 600 --seeds 1 2 3
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

from cformer_v60 import ChineseTransformerConfig
from cformer_real import AIModelWorld
from cformer_real.ttt import TTTResolver
from train_eval_real import evaluate

ROOT = Path(__file__).resolve().parent


def train(model, world: AIModelWorld, device, *, steps: int, lr: float,
          batch_size: int = 64, reject_weight: float = 2.0, margin_weight: float = 1.0,
          seed: int = 1) -> dict:
    rng = random.Random(seed * 7919)
    known = world.known_queries("train")
    ambiguous = world.ambiguous_queries("train")
    unknown = world.unknown_queries("train")
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
            top_unknown = (uvec @ bank_vec.T).max(dim=-1).values
            loss = loss + reject_weight * torch.relu(top_unknown - 0.45).mean()

        if ambiguous:
            sample_amb = rng.sample(ambiguous, min(12, len(ambiguous)))
            aq = torch.stack([world.encode_query(q["text"])[0] for q in sample_amb])
            avec = F.normalize(model.encode_query(aq.to(device)), dim=-1)
            scores = avec @ bank_vec.T
            top2 = torch.topk(scores, 2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            loss = loss + margin_weight * torch.relu(margin - 0.05).mean()

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
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--detach-inner", action="store_true")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "ttt_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(args.data)
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=2,
        d_model=args.d_model,
        heads=4,
        ffn_dimensions=args.ffn,
        output_dimensions=32,
    )

    per_seed = {}
    checkpoint_dir = ROOT / "artifacts" / "ttt_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = TTTResolver(config, inner_lr=args.inner_lr, detach_inner=args.detach_inner)
        training = train(model, world, device, steps=args.steps, lr=args.lr, seed=seed)
        torch.save(model.state_dict(), checkpoint_dir / f"ttt_seed{seed}.pt")
        metrics = evaluate(model, world, device)
        per_seed[str(seed)] = {**training, **metrics}
        print(json.dumps({"phase": "seed", "seed": seed,
                          "heldout": metrics["heldout"]}, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {
        split: {
            metric: statistics.mean(per_seed[str(seed)][split][metric] for seed in args.seeds)
            for metric in ("known_top1", "ambiguous_detected", "unknown_not_supported", "unknown_rejected")
        }
        for split in ("train", "heldout")
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "settings": {key: str(value) for key, value in vars(args).items()},
        "note": "TTT-Linear 查询编码对照实验（查询路径自适应，候选路径保持共享 Transformer）",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
