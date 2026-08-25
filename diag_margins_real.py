# -*- coding: utf-8 -*-
"""margin 分布诊断：加载 d=256 检查点，打印已知/歧义/未知查询的 top-2 margin。

判断：
- 若 ambiguous 的 margin 明显低于 known（但仍 > 0.08）→ loss 生效、阈值过时，需校准 verifier；
- 若 ambiguous 与 known 的 margin 几乎相同 → margin loss 没起作用，需查梯度/权重。

用法：
    D:/conda/envs/cformer-gpu/python.exe diag_margins_real.py --checkpoint artifacts/real_checkpoints/real_seed1.pt
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts" / "real_checkpoints" / "real_seed1.pt")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = AIModelWorld(args.data)
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=args.d_model, heads=4,
        ffn_dimensions=args.ffn, output_dimensions=32,
    )
    model = TokenCFormerResolver(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))

    def margins(queries):
        values = []
        with torch.inference_mode():
            for query in queries:
                tokens, _ = world.encode_query(query["text"])
                scores = model.encode_query(tokens[None].to(device)) @ bank.T
                top2 = torch.topk(scores, 2, dim=-1).values[0]
                values.append(float(top2[0] - top2[1]))
        return values

    report = {}
    for split in ("train", "heldout"):
        for kind in ("known", "ambiguous", "unknown"):
            queries = {
                "known": world.known_queries(split),
                "ambiguous": world.ambiguous_queries(split),
                "unknown": world.unknown_queries(split),
            }[kind]
            if not queries:
                continue
            values = sorted(margins(queries))
            report[f"{split}_{kind}"] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": values[len(values) // 2],
                "p10": values[len(values) // 10],
                "p90": values[int(len(values) * 0.9)],
                "below_0.08": sum(1 for v in values if v < 0.08) / len(values),
            }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"verifier minimum_margin = 0.08（低于它的比例越高 = 歧义检出越容易）")


if __name__ == "__main__":
    main()
