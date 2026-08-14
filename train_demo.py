from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer import CFormer, CFormerConfig, SyntheticViewTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal C-Former demo")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ego-projection", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cformer_demo.pt"))
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: CFormer,
    task: SyntheticViewTask,
    device: torch.device,
    batches: int = 20,
) -> dict[str, float]:
    model.eval()
    correct = torch.zeros(3, device=device)
    total = torch.zeros(3, device=device)
    for _ in range(batches):
        tokens, observer_tokens, observer_ids, labels = task.sample_batch(256, device)
        primary = task.selected_positions(observer_ids)
        predictions = model(tokens, observer_tokens, primary).argmax(dim=-1)
        for observer_id in range(3):
            mask = observer_ids.eq(observer_id)
            correct[observer_id] += predictions[mask].eq(labels[mask]).sum()
            total[observer_id] += mask.sum()
    names = ("left", "center", "right")
    scores = (correct / total.clamp_min(1)).cpu().tolist()
    result = {name: score for name, score in zip(names, scores)}
    result["mean"] = sum(scores) / len(scores)
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    task = SyntheticViewTask()
    config = CFormerConfig(use_ego_projection=args.ego_projection)
    model = CFormer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"device={device} parameters={model.parameter_count():,}")
    report_every = max(1, args.steps // 6)
    model.train()
    for step in range(1, args.steps + 1):
        tokens, observer_tokens, observer_ids, labels = task.sample_batch(args.batch_size, device)
        primary = task.selected_positions(observer_ids)
        logits = model(tokens, observer_tokens, primary)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == args.steps:
            scores = evaluate(model, task, device, batches=5)
            print(
                f"step={step:4d} loss={loss.item():.4f} "
                f"accuracy={scores['mean']:.3f} "
                f"L/C/R={scores['left']:.3f}/{scores['center']:.3f}/{scores['right']:.3f}"
            )
            model.train()

    final_scores = evaluate(model, task, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(config),
            "metrics": final_scores,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"saved={args.output} final_metrics={final_scores}")


if __name__ == "__main__":
    main()

