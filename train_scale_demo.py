from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer_v3 import (
    SCALE_FACTS,
    TASK_NAMES,
    DenseConcatTransformer,
    HierarchicalCFormer,
    ScaleWorldTask,
    V3Config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale test for hierarchical C-Former")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v3"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("dense_transformer", "hierarchical_cformer"),
        default=("dense_transformer", "hierarchical_cformer"),
    )
    return parser.parse_args()


def model_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def training_batch_size(scale: int) -> int:
    return 24 if scale == 128 else 6


def training_scale(step: int) -> int:
    # Most updates use the smaller world, while every fourth update teaches scale robustness.
    return 512 if step % 4 == 0 else 128


def train_model(
    name: str,
    model: torch.nn.Module,
    task: ScaleWorldTask,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    report_every = max(1, args.steps // 8)
    for step in range(1, args.steps + 1):
        scale = training_scale(step)
        batch = task.sample_batch(training_batch_size(scale), scale, device)
        memory, question, observer, labels, _, evidence_first, evidence_second = batch
        output = model(memory, question, observer)
        answer_loss = F.cross_entropy(output["logits"], labels)
        loss = answer_loss
        retrieval_loss = torch.tensor(0.0, device=device)
        if isinstance(model, HierarchicalCFormer):
            retrieval_loss = 0.5 * (
                F.cross_entropy(output["score_first"], evidence_first)
                + F.cross_entropy(output["score_second"], evidence_second)
            )
            loss = answer_loss + 0.5 * retrieval_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % report_every == 0 or step == args.steps:
            print(
                f"{name} step={step:4d} scale={scale:3d} total={loss.item():.4f} "
                f"answer={answer_loss.item():.4f} retrieval={retrieval_loss.item():.4f}"
            )


def query_chunk_size(model: torch.nn.Module, scale: int) -> int:
    if isinstance(model, HierarchicalCFormer):
        return 36
    return {128: 18, 512: 4, 2048: 1}[scale]


@torch.no_grad()
def evaluate_scale(
    model: torch.nn.Module,
    task: ScaleWorldTask,
    scale: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = torch.zeros(4, device=device)
    total = torch.zeros(4, device=device)
    recall_first = 0
    recall_second = 0
    recall_total = 0

    for world_index in range(5):
        world = task.fixed_world(scale, world_index)
        memory = world.memory.to(device)
        questions = world.questions.to(device)
        observers = world.observers.to(device)
        labels = world.labels.to(device)
        tasks = world.tasks.to(device)
        evidence_first = world.evidence_first.to(device)
        evidence_second = world.evidence_second.to(device)
        chunk = query_chunk_size(model, scale)

        cache = None
        if isinstance(model, HierarchicalCFormer):
            cache = model.encode_world(memory[None])

        for start in range(0, questions.shape[0], chunk):
            end = min(start + chunk, questions.shape[0])
            size = end - start
            if isinstance(model, HierarchicalCFormer):
                expanded_cache = tuple(value.expand(size, -1, -1) for value in cache)
                output = model.answer_from_cache(
                    expanded_cache, questions[start:end], observers[start:end]
                )
                top_first = output["top_first"]
                top_second = output["top_second"]
                target_first = evidence_first[start:end, None]
                target_second = evidence_second[start:end, None]
                recall_first += top_first.eq(target_first).any(dim=1).sum().item()
                combined = torch.cat((top_first, top_second), dim=1)
                recall_second += combined.eq(target_second).any(dim=1).sum().item()
                recall_total += size
            else:
                output = model(
                    memory[None].expand(size, -1, -1),
                    questions[start:end],
                    observers[start:end],
                )
            predictions = output["logits"].argmax(dim=-1)
            for task_id in range(4):
                mask = tasks[start:end].eq(task_id)
                correct[task_id] += predictions[mask].eq(labels[start:end][mask]).sum()
                total[task_id] += mask.sum()

    result = {
        name: (correct[index] / total[index]).item()
        for index, name in enumerate(TASK_NAMES)
    }
    result["mean"] = (correct.sum() / total.sum()).item()
    if recall_total:
        k = model.config.top_k
        result[f"first_evidence_recall_at_{k}"] = recall_first / recall_total
        result[f"second_evidence_recall_at_{2 * k}"] = recall_second / recall_total
    return result


@torch.no_grad()
def benchmark_world(
    model: torch.nn.Module,
    task: ScaleWorldTask,
    scale: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    world = task.fixed_world(scale, 0)
    memory = world.memory.to(device)
    questions = world.questions.to(device)
    observers = world.observers.to(device)
    chunk = query_chunk_size(model, scale)
    repeats = {128: 20, 512: 10, 2048: 3}[scale]

    def dense_run() -> None:
        for start in range(0, questions.shape[0], chunk):
            end = min(start + chunk, questions.shape[0])
            size = end - start
            model(
                memory[None].expand(size, -1, -1),
                questions[start:end],
                observers[start:end],
            )

    if isinstance(model, HierarchicalCFormer):
        cache = model.encode_world(memory[None])

        def cached_run() -> None:
            expanded = tuple(value.expand(questions.shape[0], -1, -1) for value in cache)
            model.answer_from_cache(expanded, questions, observers)

        cached_run()
        start = time.perf_counter()
        for _ in range(repeats):
            cached_run()
        latency = 1000 * (time.perf_counter() - start) / repeats
        cache_bytes = sum(value.numel() * value.element_size() for value in cache)
        return {
            "query_36_ms": latency,
            "cache_fp32_mb": cache_bytes / (1024 * 1024),
            "cache_fp16_mb_estimate": cache_bytes / (2 * 1024 * 1024),
        }

    dense_run()
    start = time.perf_counter()
    for _ in range(repeats):
        dense_run()
    return {"query_36_ms": 1000 * (time.perf_counter() - start) / repeats}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    task = ScaleWorldTask()
    config = V3Config()
    available = {
        "dense_transformer": DenseConcatTransformer(config),
        "hierarchical_cformer": HierarchicalCFormer(config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_index, name in enumerate(args.models):
        torch.manual_seed(args.seed + model_index)
        model = available[name]
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
            print(f"resumed={args.resume}")
        print(f"\nMODEL={name} parameters={model_parameters(model):,}")
        train_model(name, model, task, args, device)
        all_results: dict[int, dict[str, float]] = {}
        timings: dict[int, dict[str, float]] = {}
        for scale in SCALE_FACTS:
            all_results[scale] = evaluate_scale(model, task, scale, device)
            timings[scale] = benchmark_world(model, task, scale, device)
            print(f"scale={scale} results={all_results[scale]}")
            print(f"scale={scale} performance={timings[scale]}")
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": vars(config),
                "results": all_results,
                "performance": timings,
            },
            args.output_dir / f"{name}.pt",
        )


if __name__ == "__main__":
    main()
