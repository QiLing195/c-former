# -*- coding: utf-8 -*-
"""观测点（observer）查询评测：神经身份解析 + 确定性可见性过滤。

协议（对齐路线图 §8「先身份、后观测点」）：
- identity_text（不含观测点）只喂给神经身份模型；
- observer 只决定确定性可见性 mask，从不进入身份模型输入；
- permission 额外记录 mask_caught：不带 mask 时禁止对象会被选中的次数，
  量化确定性边界实际挡住的泄漏。

用法（先训练出检查点）：
    D:/conda/envs/cformer-gpu/python.exe train_eval_real.py --steps 600 --seeds 1 2 3
    D:/conda/envs/cformer-gpu/python.exe build_observer_queries.py
    D:/conda/envs/cformer-gpu/python.exe eval_observer_real.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "artifacts" / "real_checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--observer-data", type=Path, default=ROOT / "data" / "observer_queries.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "observer_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(args.data)
    observer_data = json.loads(args.observer_data.read_text(encoding="utf-8"))
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=2,
        d_model=args.d_model,
        heads=4,
        ffn_dimensions=args.ffn,
        output_dimensions=args.dimensions,
    )

    per_seed: dict[str, dict] = {}
    for seed in args.seeds:
        checkpoint = CHECKPOINT_DIR / f"real_seed{seed}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"请先运行 train_eval_real.py 生成 {checkpoint}")
        model = TokenCFormerResolver(config).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()

        bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))
        labels = torch.tensor([obj.label for obj in world.objects], device=device)

        stats = {
            kind: {"hits": 0, "total": 0, "leak": 0, "mask_caught": 0}
            for kind in ("selection", "invariance", "permission")
        }
        with torch.inference_mode():
            for query in observer_data["queries"]:
                kind = query["kind"]
                stats[kind]["total"] += 1
                # 身份解析只看到 identity_text（不含观测点），对齐「先身份、后观测点」
                identity_text = query.get("identity_text", query["text"])
                tokens, _ = world.encode_query(identity_text)
                scores = model.encode_query(tokens[None].to(device)) @ bank.T  # (1, N)
                raw_top1 = int(scores[0].argmax())

                # 确定性观测点过滤：不可见对象直接置 -inf
                visible = torch.tensor(query["visible_labels"], dtype=torch.long, device=device)
                if visible.numel() == 0:
                    continue
                mask = torch.full_like(scores[0], float("-inf"))
                mask[visible] = scores[0, visible]
                top1 = int(mask.argmax())

                if kind in ("selection", "invariance"):
                    if top1 == query["target_label"]:
                        stats[kind]["hits"] += 1
                elif kind == "permission":
                    # 禁止对象不得成为 top-1（即使它分数天然高）
                    if top1 == query["forbidden_label"]:
                        stats[kind]["leak"] += 1
                    else:
                        stats[kind]["hits"] += 1
                    # 不带 mask 时禁止对象是否会被选中（量化掩码实际挡住的泄漏）
                    if raw_top1 == query["forbidden_label"]:
                        stats[kind]["mask_caught"] += 1

        per_seed[str(seed)] = {
            kind: {
                "rate": s["hits"] / s["total"] if s["total"] else float("nan"),
                "leak": s["leak"],
                "mask_caught": s["mask_caught"],
                "total": s["total"],
            }
            for kind, s in stats.items()
        }
        print(json.dumps({"phase": "seed", "seed": seed, **per_seed[str(seed)]}, ensure_ascii=False), flush=True)

    aggregate = {
        kind: statistics.mean(per_seed[str(seed)][kind]["rate"] for seed in args.seeds)
        for kind in ("selection", "invariance", "permission")
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "settings": {key: str(value) for key, value in vars(args).items()},
        "note": "观测点评测：神经身份解析 + 确定性可见性过滤；selection/invariance 越高越好，permission 越高表示防泄漏越好",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
