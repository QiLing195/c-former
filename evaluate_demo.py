from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer import CFormer, CFormerConfig, SyntheticViewTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether C-Former truly uses observers")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cformer_demo.pt"))
    parser.add_argument("--samples-per-view", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_model(path: Path) -> CFormer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CFormer(CFormerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint)
    task = SyntheticViewTask()

    counts = {
        "correct": 0,
        "wrong_observer": 0,
        "observer_blind": 0,
        "joint_view_correct": 0,
        "total": 0,
        "joint_total": 0,
    }
    tau_values = (0.0, 0.3, 1.0, 2.0, 5.0)
    tau_stats = {tau: {"correct": 0, "nll": 0.0, "confidence": 0.0} for tau in tau_values}

    remaining = args.samples_per_view
    while remaining > 0:
        size = min(args.batch_size, remaining)
        tokens = torch.randint(task.token_vocab_size, (size, task.seq_len))
        default_ids = torch.zeros(size, dtype=torch.long)
        default_tokens = task.observer_tokens(default_ids)
        default_logits = model(tokens, default_tokens, task.selected_positions(default_ids))

        per_view_predictions = []
        per_view_labels = []
        for observer_id in range(3):
            observer_ids = torch.full((size,), observer_id, dtype=torch.long)
            observer_tokens = task.observer_tokens(observer_ids)
            labels = task.labels(tokens, observer_ids)
            primary = task.selected_positions(observer_ids)
            logits = model(tokens, observer_tokens, primary)
            predictions = logits.argmax(dim=-1)
            per_view_predictions.append(predictions)
            per_view_labels.append(labels)
            counts["correct"] += predictions.eq(labels).sum().item()

            wrong_ids = torch.full((size,), (observer_id + 1) % 3, dtype=torch.long)
            wrong_tokens = task.observer_tokens(wrong_ids)
            wrong_logits = model(tokens, wrong_tokens, task.selected_positions(wrong_ids))
            counts["wrong_observer"] += wrong_logits.argmax(dim=-1).eq(labels).sum().item()
            counts["observer_blind"] += default_logits.argmax(dim=-1).eq(labels).sum().item()

            for tau in tau_values:
                guided = logits + tau * (logits - default_logits)
                probabilities = guided.softmax(dim=-1)
                tau_stats[tau]["correct"] += guided.argmax(dim=-1).eq(labels).sum().item()
                tau_stats[tau]["nll"] += F.cross_entropy(guided, labels, reduction="sum").item()
                tau_stats[tau]["confidence"] += probabilities.max(dim=-1).values.sum().item()

            counts["total"] += size

        predictions = torch.stack(per_view_predictions, dim=1)
        labels = torch.stack(per_view_labels, dim=1)
        counts["joint_view_correct"] += predictions.eq(labels).all(dim=1).sum().item()
        counts["joint_total"] += size
        remaining -= size

    total = counts["total"]
    print(f"checkpoint={args.checkpoint} samples={total}")
    print(f"correct_observer_accuracy={counts['correct'] / total:.6f}")
    print(f"cyclic_wrong_observer_accuracy={counts['wrong_observer'] / total:.6f}")
    print(f"observer_blind_accuracy={counts['observer_blind'] / total:.6f}")
    print(f"all_three_views_correct={counts['joint_view_correct'] / counts['joint_total']:.6f}")
    print("tau_sweep:")
    for tau in tau_values:
        stats = tau_stats[tau]
        print(
            f"  tau={tau:>3.1f} accuracy={stats['correct'] / total:.6f} "
            f"nll={stats['nll'] / total:.6f} confidence={stats['confidence'] / total:.6f}"
        )


if __name__ == "__main__":
    main()
