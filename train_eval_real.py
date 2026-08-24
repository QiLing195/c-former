from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld, query_variants

ROOT = Path(__file__).resolve().parent
VERIFIER = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)


def split_known(known: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic content-hash split: ~20% holdout never seen in training.

    Same query always lands in the same side regardless of order or seed,
    so the split is reproducible and leak-free.
    """
    train: list[dict] = []
    heldout: list[dict] = []
    for query in known:
        digest = hashlib.md5(query["text"].encode("utf-8")).digest()
        (heldout if digest[0] % 5 == 0 else train).append(query)
    if not heldout:
        heldout.append(train.pop())
    return train, heldout


def build_training_entries(world: AIModelWorld, train_queries: list[dict]) -> list[dict]:
    """One entry per training query with its paraphrase variants and target."""
    entries = []
    for query in train_queries:
        entries.append({
            "variants": query_variants(query["text"], query.get("meta")),
            "target": world.target_label(query["target_id"]),
        })
    return entries


def sample_batch(
    world: AIModelWorld,
    entries: list[dict],
    rng: random.Random,
    batch_size: int,
    hard_k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Draw a batch: random paraphrase per entry + same-series hard negatives.

    Hard-negative column j holds the j-th sibling for every query in the batch,
    matching contrastive_loss's batch-aligned negative-list contract. Sibling
    shortages are filled with random objects.
    """
    chosen = (
        rng.sample(entries, batch_size)
        if len(entries) >= batch_size
        else [rng.choice(entries) for _ in range(batch_size)]
    )
    queries = []
    positives = []
    negatives: list[list] = [[] for _ in range(hard_k)]
    total = len(world.objects)
    for entry in chosen:
        text = rng.choice(entry["variants"])
        tokens, _ = world.encode_query(text)
        queries.append(tokens)
        positives.append(world.objects[entry["target"]])
        siblings = world.series_siblings(entry["target"])
        picks = rng.sample(siblings, min(hard_k, len(siblings)))
        while len(picks) < hard_k:
            picks.append(rng.randrange(total))
        for column, label in enumerate(picks):
            negatives[column].append(world.objects[label])
    return (
        torch.stack(queries),
        world.encode_candidates(positives),
        [world.encode_candidates(column) for column in negatives],
    )


def train(
    model,
    world: AIModelWorld,
    device,
    *,
    entries: list[dict],
    steps: int,
    lr: float,
    batch_size: int,
    hard_k: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device).train()
    losses = []
    started = time.perf_counter()
    for _ in range(steps):
        queries, positives, negatives = sample_batch(
            world, entries, rng, batch_size, hard_k
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.contrastive_loss(
            queries.to(device),
            positives.to(device),
            [column.to(device) for column in negatives],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "initial_loss": statistics.mean(losses[: min(10, len(losses))]),
        "final_loss": statistics.mean(losses[-min(10, len(losses)):]),
    }


@torch.inference_mode()
def _resolve(model, world: AIModelWorld, device, bank: torch.Tensor, text: str):
    tokens, coverage = world.encode_query(text)
    query = model.encode_query(tokens[None].to(device))
    scores = query @ bank.T
    top_scores, top_ids = torch.topk(scores, 2, dim=-1)
    decision = VERIFIER.decide(
        float(top_scores[0, 0]), float(top_scores[0, 1]), float(coverage)
    )
    return int(top_ids[0, 0]), decision


@torch.inference_mode()
def evaluate(model, world: AIModelWorld, device, *, known_train: list[dict], known_heldout: list[dict]) -> dict:
    model.eval()
    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))

    def resolve(text: str):
        return _resolve(model, world, device, bank, text)

    def known_metrics(queries: list[dict]) -> dict:
        hits = sum(
            1 for q in queries if resolve(q["text"])[0] == world.target_label(q["target_id"])
        )
        return {"top1": hits / max(1, len(queries)), "count": len(queries)}

    ambiguous = world.ambiguous_queries()
    unknown = world.unknown_queries()
    return {
        "known_train": known_metrics(known_train),
        "known_heldout": known_metrics(known_heldout),
        "ambiguous_detected": sum(
            1 for q in ambiguous if resolve(q["text"])[1].status == CandidateStatus.AMBIGUOUS
        ) / max(1, len(ambiguous)),
        "unknown_not_supported": sum(
            1 for q in unknown if resolve(q["text"])[1].status != CandidateStatus.SUPPORTED
        ) / max(1, len(unknown)),
        "unknown_rejected": sum(
            1 for q in unknown if resolve(q["text"])[1].status == CandidateStatus.UNKNOWN
        ) / max(1, len(unknown)),
        "counts": {
            "known_train": len(known_train),
            "known_heldout": len(known_heldout),
            "ambiguous": len(ambiguous),
            "unknown": len(unknown),
        },
    }


@torch.inference_mode()
def evaluate_blindset(model, world: AIModelWorld, device, blind: dict) -> dict:
    """Blind-set queries use the main tokenizer; OOV pieces count as UNK and
    lower coverage, which the verifier sees. This measures real robustness."""
    model.eval()
    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))

    def resolve(text: str):
        return _resolve(model, world, device, bank, text)

    queries = blind["queries"]
    by_kind = {kind: [q for q in queries if q["kind"] == kind] for kind in ("known", "ambiguous", "unknown")}
    known_hits = sum(
        1 for q in by_kind["known"]
        if resolve(q["text"])[0] == world.target_label(q["target_id"])
    )
    mean_coverage = statistics.mean(
        world.encode_query(q["text"])[1] for q in queries
    )
    return {
        "known_top1": known_hits / max(1, len(by_kind["known"])),
        "ambiguous_detected": sum(
            1 for q in by_kind["ambiguous"]
            if resolve(q["text"])[1].status == CandidateStatus.AMBIGUOUS
        ) / max(1, len(by_kind["ambiguous"])),
        "unknown_not_supported": sum(
            1 for q in by_kind["unknown"]
            if resolve(q["text"])[1].status != CandidateStatus.SUPPORTED
        ) / max(1, len(by_kind["unknown"])),
        "mean_coverage": mean_coverage,
        "counts": {kind: len(items) for kind, items in by_kind.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--blindset", type=Path, default=ROOT / "data" / "ai_models_blindset.json")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hard-negatives", type=int, default=0,
                        help="same-series hard negatives per query per step; "
                             "ablation showed they degrade ambiguity detection "
                             "and do not improve generalization")
    parser.add_argument("--no-paraphrase", action="store_true",
                        help="disable paraphrase variants (ablation switch)")
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "ai_models_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(args.data)
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=2,
        d_model=64,
        heads=4,
        ffn_dimensions=128,
        output_dimensions=32,
    )
    known_train, known_heldout = split_known(world.known_queries())
    entries = build_training_entries(
        world,
        [
            {**query, "meta": None} if args.no_paraphrase else query
            for query in known_train
        ],
    )
    blind = (
        json.loads(args.blindset.read_text(encoding="utf-8"))
        if args.blindset.exists()
        else None
    )

    per_seed = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        training = train(
            model,
            world,
            device,
            entries=entries,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            hard_k=args.hard_negatives,
            seed=seed,
        )
        metrics = evaluate(
            model, world, device, known_train=known_train, known_heldout=known_heldout
        )
        if blind is not None:
            metrics["blindset"] = evaluate_blindset(model, world, device, blind)
        per_seed[str(seed)] = {**training, **metrics}
        summary = {key: metrics[key] for key in ("known_train", "known_heldout", "blindset") if key in metrics}
        print(json.dumps({"phase": "seed", "seed": seed, **summary}, ensure_ascii=False), flush=True)

    def aggregate(key_path: tuple[str, ...], metric: str | None = None):
        values = []
        for seed in args.seeds:
            node = per_seed[str(seed)]
            for key in key_path:
                node = node[key]
            values.append(node[metric] if metric else node)
        return statistics.mean(values)

    aggregate_result = {
        "known_train_top1": aggregate(("known_train",), "top1"),
        "known_heldout_top1": aggregate(("known_heldout",), "top1"),
        "ambiguous_detected": aggregate((), "ambiguous_detected"),
        "unknown_not_supported": aggregate((), "unknown_not_supported"),
        "unknown_rejected": aggregate((), "unknown_rejected"),
    }
    if blind is not None:
        aggregate_result["blindset_known_top1"] = aggregate(("blindset",), "known_top1")
        aggregate_result["blindset_ambiguous_detected"] = aggregate(("blindset",), "ambiguous_detected")
        aggregate_result["blindset_unknown_not_supported"] = aggregate(("blindset",), "unknown_not_supported")

    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "settings": {key: str(value) for key, value in vars(args).items()},
        "split": {
            "method": "md5(text) mod 5 == 0 -> heldout (~20%), deterministic",
            "known_train": len(known_train),
            "known_heldout": len(known_heldout),
        },
        "note": (
            "真实语料线：known 查询按内容哈希切分为训练/留出集；训练使用同义改写采样 + "
            "同系列硬负例（contrastive_loss 负例列），迫使模型读证据而非背原句。"
            "结果仅验证流水线与给出诚实基线，不代表正式性能。"
        ),
        "per_seed": per_seed,
        "aggregate": aggregate_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate_result}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
