from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld

ROOT = Path(__file__).resolve().parent
VERIFIER = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)


def train(model, world: AIModelWorld, device, *, steps: int, lr: float,
          reject_weight: float = 1.0, margin_weight: float = 1.0,
          margin_target: float = 0.05, reject_target: float = 0.45,
          score_floor: float = 0.50, score_weight: float = 1.5) -> dict:
    """训练 = 已知对比损失 + 未知拒答损失 + 歧义 margin/分数地板损失。

    - known:     query -> 正确对象（对比损失，教"找到它"）
    - unknown:   库外 query 的 top 分数压到 reject_target 以下（教"不知道"）
    - ambiguous: top1-top2 间隔压到 margin_target 以下，同时 top 分数保持
                 >= score_floor（避免低分歧义被 verifier 路由到 UNKNOWN 而非 AMBIGUOUS）
    """
    known = world.known_queries("train")
    known_q = torch.stack([world.encode_query(query["text"])[0] for query in known])
    known_pos = world.encode_candidates(
        [world.objects[world.target_label(query["target_id"])] for query in known]
    )
    ambiguous_q = torch.stack(
        [world.encode_query(query["text"])[0] for query in world.ambiguous_queries("train")]
    )
    unknown_q = torch.stack(
        [world.encode_query(query["text"])[0] for query in world.unknown_queries("train")]
    )
    bank = world.encode_candidates(world.objects)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device).train()
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        # 候选库向量随模型更新，每步重算（对象少，成本低）
        bank_vec = F.normalize(model.encode_candidate(bank.to(device)), dim=-1)

        optimizer.zero_grad(set_to_none=True)
        loss = model.contrastive_loss(known_q.to(device), known_pos.to(device))

        if unknown_q.shape[0]:
            unknown_vec = F.normalize(model.encode_query(unknown_q.to(device)), dim=-1)
            top_unknown = (unknown_vec @ bank_vec.T).max(dim=-1).values
            loss = loss + reject_weight * torch.relu(top_unknown - reject_target).mean()

        if ambiguous_q.shape[0]:
            ambiguous_vec = F.normalize(model.encode_query(ambiguous_q.to(device)), dim=-1)
            scores = ambiguous_vec @ bank_vec.T
            top2 = torch.topk(scores, 2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            loss = loss + margin_weight * torch.relu(margin - margin_target).mean()
            # 分数地板：歧义查询 top 分数保持 >= verifier.minimum_score(0.50)，
            # 否则会被 verifier 路由到 UNKNOWN 而非 AMBIGUOUS（diag_margins 发现的问题）
            loss = loss + score_weight * torch.relu(score_floor - top2[:, 0]).mean()

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
def evaluate(model, world: AIModelWorld, device) -> dict:
    """分别评测 train 分割（样本内参考）与 heldout 分割（泛化真实成绩）。"""
    model.eval()
    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))

    def resolve(text: str):
        tokens, coverage = world.encode_query(text)
        query = model.encode_query(tokens[None].to(device))
        scores = query @ bank.T
        top_scores, top_ids = torch.topk(scores, 2, dim=-1)
        decision = VERIFIER.decide(
            float(top_scores[0, 0]), float(top_scores[0, 1]), float(coverage)
        )
        return int(top_ids[0, 0]), float(top_scores[0, 0]), float(top_scores[0, 1]), decision

    results = {}
    for split in ("train", "heldout"):
        known = world.known_queries(split)
        ambiguous = world.ambiguous_queries(split)
        unknown = world.unknown_queries(split)

        def top1_of(queries):
            return (
                sum(1 for query in queries
                    if resolve(query["text"])[0] == world.target_label(query["target_id"]))
                / len(queries)
            ) if queries else float("nan")

        known_top1 = top1_of(known)
        known_by_subtype = {}
        for query in known:
            subtype = query.get("subtype", "name")
            known_by_subtype.setdefault(subtype, []).append(query)
        known_subtype_top1 = {subtype: top1_of(qs) for subtype, qs in known_by_subtype.items()}

        ambiguous_detected = (
            sum(1 for query in ambiguous
                if resolve(query["text"])[3].status == CandidateStatus.AMBIGUOUS)
            / len(ambiguous)
        ) if ambiguous else float("nan")
        unknown_not_supported = (
            sum(1 for query in unknown
                if resolve(query["text"])[3].status != CandidateStatus.SUPPORTED)
            / len(unknown)
        ) if unknown else float("nan")
        unknown_rejected = (
            sum(1 for query in unknown
                if resolve(query["text"])[3].status == CandidateStatus.UNKNOWN)
            / len(unknown)
        ) if unknown else float("nan")
        results[split] = {
            "known_top1": known_top1,
            "known_subtype_top1": known_subtype_top1,
            "ambiguous_detected": ambiguous_detected,
            "unknown_not_supported": unknown_not_supported,
            "unknown_rejected": unknown_rejected,
            "counts": {"known": len(known), "ambiguous": len(ambiguous), "unknown": len(unknown)},
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--reject-weight", type=float, default=2.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--margin-target", type=float, default=0.05)
    parser.add_argument("--reject-target", type=float, default=0.45)
    parser.add_argument("--score-floor", type=float, default=0.50)
    parser.add_argument("--score-weight", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "ai_models_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(args.data)
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=args.layers,
        d_model=args.d_model,
        heads=4,
        ffn_dimensions=args.ffn,
        output_dimensions=32,
    )

    per_seed = {}
    checkpoint_dir = ROOT / "artifacts" / "real_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        try:
            training = train(model, world, device, steps=args.steps, lr=args.lr,
                             reject_weight=args.reject_weight, margin_weight=args.margin_weight,
                             margin_target=args.margin_target, reject_target=args.reject_target,
                             score_floor=args.score_floor, score_weight=args.score_weight)
            torch.save(model.state_dict(), checkpoint_dir / f"real_seed{seed}.pt")
            metrics = evaluate(model, world, device)
        except Exception as error:  # noqa: BLE001 —— 打印真实错误，避免静默退出
            import traceback
            traceback.print_exc()
            raise SystemExit(f"seed {seed} 失败: {error}")
        per_seed[str(seed)] = {**training, **metrics}
        print(json.dumps({"phase": "seed", "seed": seed,
                          "train": metrics["train"], "heldout": metrics["heldout"]},
                         ensure_ascii=False), flush=True)
        # 释放 GPU 显存，否则多种子串行时 d=256 会在下一个种子 OOM（静默退出）
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def mean_metric(metric: str, split: str) -> float:
        return statistics.mean(
            per_seed[str(seed)][split][metric] for seed in args.seeds
        )

    aggregate = {
        split: {
            metric: mean_metric(metric, split)
            for metric in ("known_top1", "ambiguous_detected", "unknown_not_supported", "unknown_rejected")
        }
        for split in ("train", "heldout")
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "settings": {key: str(value) for key, value in vars(args).items()},
        "note": "P1 边界训练信号 + train/heldout 分割：train 为样本内参考，heldout 为未训查询的泛化成绩",
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
