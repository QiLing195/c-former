# -*- coding: utf-8 -*-
"""身份解析失败诊断：加载 d=256 检查点，逐条列出 heldout 失败查询，定位瓶颈。

输出（artifacts/diag_identity_failures.json）：
- 按 subtype（name/alias/predecessor/latest）统计失败率；
- 按对象统计失败次数（找出"总答错"的长尾对象）；
- 逐条失败样本（text / 期望 id / 实际 top1 id / top1 名 / score / margin）；
- 失败查询的文本长度与是否含别名，判断"语义泛化"还是"证据不足"。

用法：
    D:/conda/envs/cformer-gpu/python.exe diag_identity_failures.py --checkpoint artifacts/real_checkpoints/real_seed1.pt
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

ROOT = Path(__file__).resolve().parent
VERIFIER = EvidenceVerifier(minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ai_models_dataset.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=256)  # 与 train_eval_real.py 默认一致
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = AIModelWorld(args.data)
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=args.d_model, heads=4,
        ffn_dimensions=args.ffn, output_dimensions=32,
    )
    model = TokenCFormerResolver(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
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

    failures = []
    by_subtype = Counter()
    by_subtype_fail = Counter()
    by_object_fail = Counter()
    total_known = 0
    for split in ("heldout",):
        for query in world.known_queries(split):
            total_known += 1
            subtype = query.get("subtype", "name")
            by_subtype[subtype] += 1
            top_id, score, second, decision = resolve(query["text"])
            target_id = query["target_id"]
            target_label = world.target_label(target_id)
            if top_id != target_label:
                by_subtype_fail[subtype] += 1
                by_object_fail[target_id] += 1
                failures.append({
                    "split": split, "subtype": subtype,
                    "text": query["text"],
                    "target_id": target_id,
                    "target_name": world.objects[target_label].name,
                    "pred_id": world.objects[top_id].object_id,
                    "pred_name": world.objects[top_id].name,
                    "score": round(score, 4), "second": round(second, 4),
                    "margin": round(score - second, 4),
                    "status": decision.status.value,
                })

    report = {
        "total_known_heldout": total_known,
        "subtype_counts": dict(by_subtype),
        "subtype_fail_counts": dict(by_subtype_fail),
        "subtype_fail_rates": {k: round(by_subtype_fail[k] / by_subtype[k], 4)
                               for k in by_subtype},
        "worst_objects": by_object_fail.most_common(15),
        "n_failures": len(failures),
        "failures": failures,
    }
    output = ROOT / "artifacts" / "diag_identity_failures.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "phase": "done",
        "subtype_fail_rates": report["subtype_fail_rates"],
        "worst_objects": report["worst_objects"],
        "n_failures": len(failures),
    }, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
