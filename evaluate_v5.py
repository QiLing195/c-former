from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from cformer_v3 import SCALE_FACTS
from cformer_v4 import EvidenceRAGTransformer, ReliableCFormer, ReliableDenseTransformer, V4Config
from cformer_v4.data import STATUS_ANSWER, STATUS_CONFLICT
from cformer_v5 import ConflictResolver, V5WorldSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V5 conflict-aware system")
    parser.add_argument(
        "--cformer",
        type=Path,
        default=Path("artifacts/v4_policy_final/reliable_cformer.pt"),
    )
    parser.add_argument(
        "--dense",
        type=Path,
        default=Path("artifacts/v4_dense_equal_budget/dense_transformer.pt"),
    )
    parser.add_argument("--rag", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v5_results.pt"))
    return parser.parse_args()


def load_model(path: Path, kind: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = V4Config(**checkpoint["config"])
    if kind == "cformer":
        model = ReliableCFormer(config)
    elif kind == "rag":
        model = EvidenceRAGTransformer(config)
    else:
        model = ReliableDenseTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def neural_predictions(model, cases, scale: int):
    indices = [
        index
        for index, case in enumerate(cases)
        if not case.use_conflict_controller and case.forced_status < 0
    ]
    predictions = {}
    chunk = 32 if isinstance(model, ReliableCFormer) else {128: 16, 512: 4, 2048: 1}[scale]
    for start in range(0, len(indices), chunk):
        selected_indices = indices[start : start + chunk]
        selected_cases = [cases[index] for index in selected_indices]
        memory = torch.stack([case.memory for case in selected_cases])
        question = torch.stack([case.question for case in selected_cases])
        observer = torch.tensor([case.observer for case in selected_cases])
        allowed = torch.stack([case.allowed for case in selected_cases])
        output = model(memory, question, observer, allowed)
        statuses = output["status_logits"].argmax(dim=-1)
        answers = output["answer_logits"].argmax(dim=-1)
        for index, status, answer in zip(selected_indices, statuses.tolist(), answers.tolist()):
            predictions[index] = (status, answer, "neural")
    return predictions


@torch.no_grad()
def evaluate_model(model, suite: V5WorldSuite, resolver: ConflictResolver, scale: int):
    cases = [case for world_index in range(5) for case in suite.cases(scale, world_index)]
    predictions = neural_predictions(model, cases, scale)
    for index, case in enumerate(cases):
        if case.forced_status >= 0:
            predictions[index] = (case.forced_status, 0, "policy")
        elif case.use_conflict_controller:
            resolution = resolver.resolve(
                case.memory, case.metadata, case.question, case.query_time
            )
            predictions[index] = (resolution.status, resolution.answer, "conflict_index")

    total = len(cases)
    status_correct = 0
    answer_correct = answer_total = 0
    hallucinations = nonanswer_total = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    category_answer_correct = defaultdict(int)
    predicted_conflict_nonconflict = 0
    nonconflict_special_total = 0

    for index, case in enumerate(cases):
        predicted_status, predicted_answer, _ = predictions[index]
        status_correct += int(predicted_status == case.expected_status)
        category_total[case.category] += 1
        category_correct[case.category] += int(predicted_status == case.expected_status)
        if case.expected_status == STATUS_ANSWER:
            answer_total += 1
            correct = predicted_status == STATUS_ANSWER and predicted_answer == case.expected_answer
            answer_correct += int(correct)
            category_answer_correct[case.category] += int(correct)
        else:
            nonanswer_total += 1
            hallucinations += int(predicted_status == STATUS_ANSWER)
        if case.category in ("time_change", "version_update", "low_confidence_source"):
            nonconflict_special_total += 1
            predicted_conflict_nonconflict += int(predicted_status == STATUS_CONFLICT)

    result = {
        "status_accuracy": status_correct / total,
        "end_to_end_answer_accuracy": answer_correct / answer_total,
        "hallucination_rate": hallucinations / nonanswer_total,
        "false_conflict_rate": predicted_conflict_nonconflict / nonconflict_special_total,
    }
    for category in sorted(category_total):
        result[f"status_{category}"] = category_correct[category] / category_total[category]
        if category in category_answer_correct:
            result[f"answer_{category}"] = (
                category_answer_correct[category] / category_total[category]
            )
    return result


def main() -> None:
    args = parse_args()
    suite = V5WorldSuite()
    resolver = ConflictResolver()
    models = {
        "cformer_v5_system": load_model(args.cformer, "cformer"),
        "dense_v5_system": load_model(args.dense, "dense"),
    }
    if args.rag is not None:
        models["evidence_rag_v5_system"] = load_model(args.rag, "rag")
    results = {}
    for name, model in models.items():
        print(f"\nMODEL={name}")
        results[name] = {}
        for scale in SCALE_FACTS:
            result = evaluate_model(model, suite, resolver, scale)
            results[name][scale] = result
            print(f"scale={scale} result={result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
