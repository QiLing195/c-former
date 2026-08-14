from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

from cformer_v3.data import TASK_NAMES
from cformer_v4 import EvidenceRAGTransformer, ReliableCFormer, V4Config
from cformer_v52 import STRESS_SCALES, V52StressSuite, answer_from_cache_ablation
from cformer_v52.ablation import ABLATION_MODES
from cformer_v5 import ConflictResolver, V5WorldSuite
from evaluate_v5 import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V5.2 stress/OOD and ablations")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cformer", type=Path, required=True)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=5)
    parser.add_argument("--world-offset", type=int, default=50)
    return parser.parse_args()


def load_model(path: Path, kind: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = V4Config(**checkpoint["config"])
    model = ReliableCFormer(config) if kind == "cformer" else EvidenceRAGTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def expanded_cache(model, memory: torch.Tensor, batch: int):
    cache = model.encode_world(memory[None])
    return cache, tuple(value.expand(batch, -1, -1) for value in cache)


@torch.no_grad()
def evaluate_stress_worlds(cformer, rag, suite: V52StressSuite, worlds: int):
    totals = defaultdict(lambda: defaultdict(int))
    performance_samples = defaultdict(lambda: defaultdict(list))
    for scale in STRESS_SCALES:
        for world_index in range(worlds):
            world = suite.world(scale, world_index)
            batch = world.questions.shape[0]
            allowed = torch.ones(batch, scale, dtype=torch.bool)

            start = time.perf_counter()
            cformer_cache, cformer_expanded = expanded_cache(cformer, world.memory, batch)
            performance_samples["cformer"]["encode_ms"].append(
                1000.0 * (time.perf_counter() - start)
            )
            for mode in ABLATION_MODES:
                start = time.perf_counter()
                output = answer_from_cache_ablation(
                    cformer,
                    cformer_expanded,
                    world.questions,
                    world.observers,
                    allowed,
                    mode,
                )
                performance_samples[f"cformer_{mode}"]["query_36_ms"].append(
                    1000.0 * (time.perf_counter() - start)
                )
                predicted_status = output["status_logits"].argmax(dim=-1)
                predicted_answer = output["answer_logits"].argmax(dim=-1)
                key = f"cformer_{mode}"
                totals[key]["total"] += batch
                totals[key]["status_correct"] += predicted_status.eq(0).sum().item()
                totals[key]["answer_head_correct"] += predicted_answer.eq(world.labels).sum().item()
                totals[key]["end_to_end_correct"] += (
                    predicted_status.eq(0) & predicted_answer.eq(world.labels)
                ).sum().item()
                for task_id, task_name in enumerate(TASK_NAMES):
                    mask = world.tasks.eq(task_id)
                    totals[key][f"{task_name}_total"] += mask.sum().item()
                    totals[key][f"{task_name}_correct"] += (
                        predicted_status[mask].eq(0)
                        & predicted_answer[mask].eq(world.labels[mask])
                    ).sum().item()

            start = time.perf_counter()
            rag_cache, rag_expanded = expanded_cache(rag, world.memory, batch)
            performance_samples["rag"]["encode_ms"].append(
                1000.0 * (time.perf_counter() - start)
            )
            start = time.perf_counter()
            rag_output = rag.answer_from_cache(
                rag_expanded, world.questions, world.observers, allowed
            )
            performance_samples["rag"]["query_36_ms"].append(
                1000.0 * (time.perf_counter() - start)
            )
            rag_status = rag_output["status_logits"].argmax(dim=-1)
            rag_answer = rag_output["answer_logits"].argmax(dim=-1)
            totals["rag"]["total"] += batch
            totals["rag"]["status_correct"] += rag_status.eq(0).sum().item()
            totals["rag"]["answer_head_correct"] += rag_answer.eq(world.labels).sum().item()
            totals["rag"]["end_to_end_correct"] += (
                rag_status.eq(0) & rag_answer.eq(world.labels)
            ).sum().item()
            for task_id, task_name in enumerate(TASK_NAMES):
                mask = world.tasks.eq(task_id)
                totals["rag"][f"{task_name}_total"] += mask.sum().item()
                totals["rag"][f"{task_name}_correct"] += (
                    rag_status[mask].eq(0) & rag_answer[mask].eq(world.labels[mask])
                ).sum().item()

            cache_bytes = sum(value.numel() * value.element_size() for value in cformer_cache)
            performance_samples["cformer"]["cache_mb"].append(cache_bytes / 1024**2)
            rag_cache_bytes = sum(value.numel() * value.element_size() for value in rag_cache)
            performance_samples["rag"]["cache_mb"].append(rag_cache_bytes / 1024**2)

        # Flush one scale at a time into a serializable result and reset counters.
        yield scale, totals, performance_samples
        totals = defaultdict(lambda: defaultdict(int))
        performance_samples = defaultdict(lambda: defaultdict(list))


def finalize_metrics(totals):
    result = {}
    for name, counts in totals.items():
        metrics = {
            "status_answer_rate": counts["status_correct"] / counts["total"],
            "false_refusal_rate": 1.0 - counts["status_correct"] / counts["total"],
            "answer_head_accuracy": counts["answer_head_correct"] / counts["total"],
            "end_to_end_accuracy": counts["end_to_end_correct"] / counts["total"],
        }
        for task_name in TASK_NAMES:
            metrics[f"accuracy_{task_name}"] = (
                counts[f"{task_name}_correct"] / counts[f"{task_name}_total"]
            )
        result[name] = metrics
    return result


def finalize_performance(samples):
    result = {}
    for name, metrics in samples.items():
        result[name] = {
            metric: sum(values) / len(values) for metric, values in metrics.items()
        }
    return result


class OffsetV5Suite:
    def __init__(self, offset: int) -> None:
        self.base = V5WorldSuite()
        self.offset = offset

    def cases(self, scale: int, world_index: int):
        return self.base.cases(scale, self.offset + world_index)


def main() -> None:
    args = parse_args()
    cformer = load_model(args.cformer, "cformer")
    rag = load_model(args.rag, "rag")
    suite = V52StressSuite(args.world_offset)
    results = {
        "seed": args.seed,
        "world_offset": args.world_offset,
        "worlds": args.worlds,
        "stress": {},
        "boundary": {},
    }
    for scale, totals, performance in evaluate_stress_worlds(
        cformer, rag, suite, args.worlds
    ):
        results["stress"][scale] = {
            "metrics": finalize_metrics(totals),
            "performance": finalize_performance(performance),
            "effective_facts": 2048,
            "distractor_facts": scale - 2048,
        }
        print(f"scale={scale} metrics={results['stress'][scale]['metrics']}")
        print(f"scale={scale} performance={results['stress'][scale]['performance']}")

    boundary_suite = OffsetV5Suite(args.world_offset)
    resolver = ConflictResolver()
    results["boundary"]["cformer"] = evaluate_model(
        cformer, boundary_suite, resolver, 2048
    )
    results["boundary"]["rag"] = evaluate_model(rag, boundary_suite, resolver, 2048)
    print(f"boundary={results['boundary']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
