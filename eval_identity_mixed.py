# -*- coding: utf-8 -*-
"""身份解析端到端评测：确定性精确层（PreciseMatch）+ 神经层混合。

设计（对齐「确定性优先，神经只做检索」）：
- 精确层：全名/别名词法命中 → 直接返回（100%）；
- 神经层：未命中（描述性指代）→ 走冻结 V6.0 编码器检索 + verifier；
- 输出：name/alias/predecessor/latest 各子类准确率 + 整体 identity_top1，
  与纯神经基线（75.8%）对比。

用法：
    D:/conda/envs/cformer-gpu/python.exe eval_identity_mixed.py --checkpoint artifacts/real_checkpoints/real_seed1.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld
from cformer_v63.precise_match import PreciseMatch

ROOT = Path(__file__).resolve().parent
VERIFIER = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = AIModelWorld(args.data)

    # PreciseMatch 需要原始 dict 对象（含 evidence）；从 JSON 重新加载
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    raw_objects = data["objects"]
    precise = PreciseMatch(raw_objects)

    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=args.d_model, heads=4,
        ffn_dimensions=args.ffn, output_dimensions=32,
    )
    model = TokenCFormerResolver(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    bank = model.encode_candidate(world.encode_candidates(world.objects).to(device))

    def neural_resolve(text: str):
        tokens, coverage = world.encode_query(text)
        query = model.encode_query(tokens[None].to(device))
        scores = query @ bank.T
        top_scores, top_ids = torch.topk(scores, 2, dim=-1)
        decision = VERIFIER.decide(
            float(top_scores[0, 0].detach()), float(top_scores[0, 1].detach()), float(coverage)
        )
        return int(top_ids[0, 0]), decision

    def resolve(text: str) -> tuple[int | None, CandidateStatus]:
        """混合解析：精确层优先，神经层兜底。返回 (label 或 None, status)。"""
        hit = precise.hit(text)
        if hit is not None:
            label = next((i for i, o in enumerate(world.objects)
                          if o.object_id == hit.object_id), None)
            return label, CandidateStatus.SUPPORTED
        label, decision = neural_resolve(text)
        return label, decision

    results = {}
    for split in ("train", "heldout"):
        by_subtype: dict[str, list[dict]] = {}
        for q in world.known_queries(split):
            by_subtype.setdefault(q.get("subtype", "name"), []).append(q)

        subtype_report = {}
        for subtype, queries in by_subtype.items():
            total = len(queries)
            ok = sum(1 for q in queries
                     if resolve(q["text"])[0] == world.target_label(q["target_id"]))
            hit_by_precise = sum(1 for q in queries if precise.hit(q["text"]) is not None)
            subtype_report[subtype] = {
                "accuracy": round(ok / total, 4) if total else None,
                "n": total,
                "precise_hit_ratio": round(hit_by_precise / total, 4) if total else None,
            }

        identity_queries = [q for q in world.known_queries(split)
                            if q.get("subtype", "name") in ("name", "alias")]
        identity_ok = sum(1 for q in identity_queries
                          if resolve(q["text"])[0] == world.target_label(q["target_id"]))
        results[split] = {
            "identity_top1": round(identity_ok / len(identity_queries), 4),
            "by_subtype": subtype_report,
        }

    report = {
        "config": {"d_model": args.d_model, "ffn": args.ffn},
        "note": "混合解析：PreciseMatch 精确层（全名/别名命中即返回）+ V6.0 神经层兜底",
        "results": results,
    }
    output = ROOT / "artifacts" / "eval_identity_mixed.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "report": report}, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
