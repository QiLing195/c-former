from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cformer import CFormer, CFormerConfig, OBSERVER_TEXTS, SyntheticViewTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-view C-Former inference")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cformer_demo.pt"))
    parser.add_argument("--tokens", type=int, nargs=6, default=[3, 8, 1, 12, 5, 9])
    parser.add_argument("--tau", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CFormer(CFormerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task = SyntheticViewTask()
    tokens = torch.tensor([args.tokens])
    default_observer = task.observer_tokens(torch.tensor([0]))
    positions = [0, task.seq_len // 2, task.seq_len - 1]
    print(f"input={args.tokens}")
    for observer_id, position in enumerate(positions):
        observer = task.observer_tokens(torch.tensor([observer_id]))
        primary = torch.tensor([position])
        logits = model(tokens, observer, primary)
        guided = model.contrast_logits(tokens, observer, default_observer, tau=args.tau)
        print(
            f"observer='{OBSERVER_TEXTS[observer_id]}' expected={args.tokens[position]} "
            f"prediction={logits.argmax(-1).item()} guided={guided.argmax(-1).item()}"
        )


if __name__ == "__main__":
    main()

