from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cformer_v59 import FlatTransformerDualEncoder, OpenAliasWorld
from evaluate_v59 import aggregate, evaluate_world, train_model


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "v59_transformer_final.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "v59_transformer_scale_results.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(5901)
    world = OpenAliasWorld(8192, seed=59)
    model = FlatTransformerDualEncoder(world.tokenizer.size)
    if args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        training = {"loaded_checkpoint": True}
    else:
        training = train_model(
            model, world, device, steps=args.steps, batch_size=args.batch_size
        )
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
    model.to(device)

    rows = []
    for scale in (65536, 131072, 262144, 512000):
        for world_index in range(args.worlds):
            row = evaluate_world(
                model,
                scale,
                59 + world_index * 101,
                device,
                queries=args.queries,
                chunk_size=args.chunk_size,
            )
            rows.append(row)
            print(json.dumps({"model": "transformer", **row}, ensure_ascii=False), flush=True)

    payload = {
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda"
            else 0,
        },
        "parameters": model.parameter_count(),
        "training": training,
        "worlds": rows,
        "aggregate": aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
