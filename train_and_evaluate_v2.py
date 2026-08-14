from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer_v2 import (
    TASK_NAMES,
    ConcatTransformerQA,
    SharedMemoryQA,
    V2Config,
    WorldTask,
    fixed_worlds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare C-Former V2 baselines")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("concat_transformer", "shared_question_only", "observer_cformer"),
        default=None,
        help="Train only selected models (default: all three)",
    )
    return parser.parse_args()


def build_models(config: V2Config) -> dict[str, torch.nn.Module]:
    return {
        "concat_transformer": ConcatTransformerQA(config),
        "shared_question_only": SharedMemoryQA(config, use_observer=False),
        "observer_cformer": SharedMemoryQA(config, use_observer=True),
    }


@torch.no_grad()
def evaluate_fixed_worlds(
    model: torch.nn.Module, task: WorldTask, device: torch.device
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    task_correct = torch.zeros(4, device=device)
    task_total = torch.zeros(4, device=device)
    worlds_result: dict[str, dict[str, float]] = {}
    for world in fixed_worlds():
        memory, questions, observers, labels, task_ids = task.fixed_world_tensors(world)
        memory = memory.to(device)
        questions, observers, labels, task_ids = (
            value.to(device) for value in (questions, observers, labels, task_ids)
        )
        if isinstance(model, SharedMemoryQA):
            encoded = model.encode_memory(memory[None])
            encoded = encoded.expand(questions.shape[0], -1, -1)
            logits = model.answer_from_memory(encoded, questions, observers)
        else:
            logits = model(memory[None].expand(questions.shape[0], -1, -1), questions, observers)
        predictions = logits.argmax(dim=-1)
        world_scores: dict[str, float] = {}
        for task_id, name in enumerate(TASK_NAMES):
            mask = task_ids.eq(task_id)
            correct = predictions[mask].eq(labels[mask]).sum()
            task_correct[task_id] += correct
            task_total[task_id] += mask.sum()
            world_scores[name] = (correct / mask.sum()).item()
        worlds_result[world.name] = world_scores
    scores = {
        name: (task_correct[index] / task_total[index]).item()
        for index, name in enumerate(TASK_NAMES)
    }
    scores["mean"] = (task_correct.sum() / task_total.sum()).item()
    return scores, worlds_result


def train_model(
    name: str,
    model: torch.nn.Module,
    task: WorldTask,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    report_every = max(1, args.steps // 5)
    for step in range(1, args.steps + 1):
        model.train()
        memory, question, observer, labels, _ = task.sample_batch(args.batch_size, device)
        loss = F.cross_entropy(model(memory, question, observer), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % report_every == 0 or step == args.steps:
            scores, _ = evaluate_fixed_worlds(model, task, device)
            print(f"{name} step={step:4d} loss={loss.item():.4f} fixed_mean={scores['mean']:.4f}")
    return evaluate_fixed_worlds(model, task, device)[0]


@torch.no_grad()
def benchmark_queries(
    model: torch.nn.Module,
    task: WorldTask,
    device: torch.device,
    repeats: int = 100,
) -> dict[str, float]:
    model.eval()
    memory, questions, observers, _, _ = task.fixed_world_tensors(fixed_worlds()[0])
    memory, questions, observers = memory.to(device), questions.to(device), observers.to(device)
    repeated_memory = memory[None].expand(questions.shape[0], -1, -1)

    for _ in range(5):
        model(repeated_memory, questions, observers)
    start = time.perf_counter()
    for _ in range(repeats):
        model(repeated_memory, questions, observers)
    uncached_ms = 1000 * (time.perf_counter() - start) / repeats

    result = {"full_query_batch_ms": uncached_ms}
    if isinstance(model, SharedMemoryQA):
        encoded = model.encode_memory(memory[None]).expand(questions.shape[0], -1, -1)
        for _ in range(5):
            model.answer_from_memory(encoded, questions, observers)
        start = time.perf_counter()
        for _ in range(repeats):
            model.answer_from_memory(encoded, questions, observers)
        result["cached_query_batch_ms"] = 1000 * (time.perf_counter() - start) / repeats
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    task = WorldTask()
    config = V2Config()
    models = build_models(config)
    if args.models is not None:
        models = {name: models[name] for name in args.models}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float]] = {}
    for index, (name, model) in enumerate(models.items()):
        torch.manual_seed(args.seed + index)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"\nMODEL={name} parameters={parameters:,}")
        scores = train_model(name, model, task, args, device)
        _, world_scores = evaluate_fixed_worlds(model, task, device)
        timing = benchmark_queries(model, task, device)
        summary[name] = {**scores, **timing, "parameters": float(parameters)}
        torch.save(
            {"model_state": model.state_dict(), "config": vars(config), "scores": scores},
            args.output_dir / f"{name}.pt",
        )
        print(f"scores={scores}")
        print(f"worlds={world_scores}")
        print(f"timing={timing}")

    print("\nFINAL_SUMMARY")
    for name, result in summary.items():
        print(name, result)


if __name__ == "__main__":
    main()
