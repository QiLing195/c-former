from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer_v3 import SCALE_FACTS
from cformer_v4 import (
    EvidenceRAGTransformer,
    STATUS_NAMES,
    ReliableCFormer,
    ReliableDenseTransformer,
    ReliabilityTask,
    V4Config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train boundary-aware C-Former V4")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v4"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("dense_transformer", "evidence_rag", "reliable_cformer"),
        default=("dense_transformer", "reliable_cformer"),
    )
    return parser.parse_args()


def training_scale(step: int) -> int:
    return 512 if step % 4 == 0 else 128


def batch_size(scale: int, model) -> int:
    if isinstance(model, ReliableCFormer):
        return 32 if scale == 128 else 24
    return 24 if scale == 128 else 6


def train_model(model, name, task, args, device) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    report_every = max(1, args.steps // 10)
    for step in range(1, args.steps + 1):
        scale = training_scale(step)
        (
            memory,
            question,
            observer,
            allowed,
            answers,
            statuses,
            _,
            first,
            second,
        ) = task.sample_batch(
            batch_size(scale, model), scale, device
        )
        output = model(memory, question, observer, allowed)
        status_loss = F.cross_entropy(output["status_logits"], statuses)
        answer_mask = statuses.eq(0)
        answer_loss = (
            F.cross_entropy(output["answer_logits"][answer_mask], answers[answer_mask])
            if answer_mask.any()
            else status_loss.new_zeros(())
        )
        retrieval_loss = status_loss.new_zeros(())
        if isinstance(model, ReliableCFormer):
            evidence_mask = first.ge(0)
            if evidence_mask.any():
                retrieval_loss = 0.5 * (
                    F.cross_entropy(output["scores_first"][evidence_mask], first[evidence_mask])
                    + F.cross_entropy(output["scores_second"][evidence_mask], second[evidence_mask])
                )
        loss = status_loss + answer_loss + 0.5 * retrieval_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % report_every == 0 or step == args.steps:
            print(
                f"{name} step={step:4d} scale={scale} total={loss.item():.4f} "
                f"status={status_loss.item():.4f} answer={answer_loss.item():.4f} "
                f"retrieval={retrieval_loss.item():.4f}"
            )


def chunk_size(model, scale: int) -> int:
    if isinstance(model, ReliableCFormer):
        return 32
    return {128: 16, 512: 4, 2048: 1}[scale]


@torch.no_grad()
def evaluate_scale(
    model, task, scale: int, device: torch.device, answer_logit_bias: float = 0.0
) -> dict[str, float]:
    model.eval()
    status_correct = 0
    status_total = 0
    answer_correct = 0
    answer_total = 0
    end_to_end_correct = 0
    false_refusals = 0
    hallucinations = 0
    nonanswer_total = 0
    per_status_correct = torch.zeros(len(STATUS_NAMES), device=device)
    per_status_total = torch.zeros(len(STATUS_NAMES), device=device)
    per_status_predicted_answer = torch.zeros(len(STATUS_NAMES), device=device)
    retrieval_first = retrieval_second = retrieval_total = 0

    for world_index in range(5):
        cases = task.fixed_cases(scale, world_index)
        values = [
            cases.memory,
            cases.questions,
            cases.observers,
            cases.allowed,
            cases.answers,
            cases.statuses,
            cases.boundary_override,
            cases.evidence_first,
            cases.evidence_second,
        ]
        memory, question, observer, allowed, answers, statuses, boundary_override, first, second = [
            value.to(device) for value in values
        ]
        chunk = chunk_size(model, scale)
        for start in range(0, statuses.shape[0], chunk):
            end = min(start + chunk, statuses.shape[0])
            output = model(
                memory[start:end], question[start:end], observer[start:end], allowed[start:end]
            )
            calibrated_status_logits = output["status_logits"].clone()
            calibrated_status_logits[:, 0] += answer_logit_bias
            predicted_status = calibrated_status_logits.argmax(dim=-1)
            forced = boundary_override[start:end]
            predicted_status = torch.where(forced.ge(0), forced, predicted_status)
            predicted_answer = output["answer_logits"].argmax(dim=-1)
            target_status = statuses[start:end]
            target_answer = answers[start:end]
            status_correct += predicted_status.eq(target_status).sum().item()
            status_total += end - start
            for status_id in range(len(STATUS_NAMES)):
                mask = target_status.eq(status_id)
                per_status_correct[status_id] += predicted_status[mask].eq(status_id).sum()
                per_status_total[status_id] += mask.sum()
                per_status_predicted_answer[status_id] += predicted_status[mask].eq(0).sum()

            answer_mask = target_status.eq(0)
            nonanswer_mask = ~answer_mask
            answer_correct += predicted_answer[answer_mask].eq(target_answer[answer_mask]).sum().item()
            answer_total += answer_mask.sum().item()
            end_to_end_correct += (
                predicted_status[answer_mask].eq(0)
                & predicted_answer[answer_mask].eq(target_answer[answer_mask])
            ).sum().item()
            false_refusals += predicted_status[answer_mask].ne(0).sum().item()
            hallucinations += predicted_status[nonanswer_mask].eq(0).sum().item()
            nonanswer_total += nonanswer_mask.sum().item()

            if isinstance(model, ReliableCFormer):
                evidence_mask = first[start:end].ge(0)
                if evidence_mask.any():
                    retrieval_first += output["selected_first"][evidence_mask].eq(
                        first[start:end][evidence_mask]
                    ).sum().item()
                    retrieval_second += output["selected_second"][evidence_mask].eq(
                        second[start:end][evidence_mask]
                    ).sum().item()
                    retrieval_total += evidence_mask.sum().item()

    result = {
        "status_accuracy": status_correct / status_total,
        "answer_head_accuracy": answer_correct / answer_total,
        "end_to_end_answer_accuracy": end_to_end_correct / answer_total,
        "false_refusal_rate": false_refusals / answer_total,
        "hallucination_rate": hallucinations / nonanswer_total,
    }
    for status_id, name in enumerate(STATUS_NAMES):
        result[f"recall_{name}"] = (
            per_status_correct[status_id] / per_status_total[status_id].clamp_min(1)
        ).item()
        if status_id != 0:
            result[f"answer_leak_rate_{name}"] = (
                per_status_predicted_answer[status_id]
                / per_status_total[status_id].clamp_min(1)
            ).item()
    if retrieval_total:
        result["retrieval_first_accuracy"] = retrieval_first / retrieval_total
        result["retrieval_second_accuracy"] = retrieval_second / retrieval_total
    return result


@torch.no_grad()
def benchmark_answer_queries(model, task, scale: int, device: torch.device) -> dict[str, float]:
    model.eval()
    cases = task.fixed_cases(scale, 0)
    memory = cases.memory[:36].to(device)
    question = cases.questions[:36].to(device)
    observer = cases.observers[:36].to(device)
    allowed = cases.allowed[:36].to(device)
    repeats = {128: 20, 512: 8, 2048: 2}[scale]

    if isinstance(model, ReliableCFormer):
        cache = model.encode_world(memory[:1])
        expanded = tuple(value.expand(36, -1, -1) for value in cache)

        def run():
            model.answer_from_cache(expanded, question, observer, allowed)

        run()
        start = time.perf_counter()
        for _ in range(repeats):
            run()
        cache_bytes = sum(value.numel() * value.element_size() for value in cache)
        return {
            "query_36_ms": 1000 * (time.perf_counter() - start) / repeats,
            "cache_fp32_mb": cache_bytes / (1024 * 1024),
        }

    chunk = chunk_size(model, scale)

    def run():
        for start_index in range(0, 36, chunk):
            end = min(start_index + chunk, 36)
            model(
                memory[start_index:end],
                question[start_index:end],
                observer[start_index:end],
                allowed[start_index:end],
            )

    run()
    start = time.perf_counter()
    for _ in range(repeats):
        run()
    return {"query_36_ms": 1000 * (time.perf_counter() - start) / repeats}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = V4Config()
    task = ReliabilityTask()
    models = {
        "dense_transformer": ReliableDenseTransformer(config),
        "evidence_rag": EvidenceRAGTransformer(config),
        "reliable_cformer": ReliableCFormer(config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, name in enumerate(args.models):
        torch.manual_seed(args.seed + index)
        model = models[name]
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
            if missing or unexpected:
                print(f"partial_resume missing={missing} unexpected={unexpected}")
            print(f"resumed={args.resume}")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"\nMODEL={name} parameters={parameters:,}")
        train_model(model, name, task, args, device)
        results = {}
        performance = {}
        for scale in SCALE_FACTS:
            results[scale] = evaluate_scale(model, task, scale, device)
            performance[scale] = benchmark_answer_queries(model, task, scale, device)
            print(f"scale={scale} results={results[scale]}")
            print(f"scale={scale} performance={performance[scale]}")
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": vars(config),
                "results": results,
                "performance": performance,
            },
            args.output_dir / f"{name}.pt",
        )


if __name__ == "__main__":
    main()
