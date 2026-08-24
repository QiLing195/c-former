# -*- coding: utf-8 -*-
"""V6.2 观测点端到端评测：身份一致 / 权限零泄漏 / 跨视角召回 / 可追溯 / 单副本。

用法：
    D:\conda\envs\cformer-gpu\python.exe evaluate_observers.py

闸门（V60_TO_V65_ROADMAP §8）：
- 身份解析在合法观测点间保持一致；
- 权限泄漏率 = 0（被掩对象不得以任何 supported 形式暴露）；
- 跨视角证据召回 ≥95%（不同观测点、不同措辞复用同一对象知识）;
- 答案关键断言可追溯证据 ≥95%（supported/access_denied 都必须带理由与对象文档）；
- 对象向量不随观测点数量复制。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_v62 import ObserverFrame, ObserverGate
from cformer_real import AIModelWorld, query_variants
from evaluate_v61c import WorldEncoder, build_chain
from evaluate_v62 import make_reasoner
from train_eval_real import split_known, train

ROOT = Path(__file__).resolve().parent


def build_frames(world: AIModelWorld) -> list[ObserverFrame]:
    companies_by_region: dict[str, set[str]] = {}
    open_companies: set[str] = set()
    closed_companies: set[str] = set()
    raw = json.loads((ROOT / "data" / "ai_models_dataset.json").read_text(encoding="utf-8"))
    for obj in raw["objects"]:
        meta = obj.get("meta") or {}
        company, region = meta.get("company"), meta.get("region")
        if not company:
            continue
        companies_by_region.setdefault(region, set()).add(company)
        if meta.get("open_source"):
            open_companies.add(company)
        else:
            closed_companies.add(company)
    return [
        ObserverFrame("cn-region", allowed_regions=frozenset({"中国"})),
        ObserverFrame("us-region", allowed_regions=frozenset({"美国"})),
        ObserverFrame(
            "open-source-vendors",
            allowed_companies=frozenset(open_companies - closed_companies),
        ),
        ObserverFrame(
            "us-open-only",
            allowed_regions=frozenset({"美国"}),
            allowed_companies=frozenset(open_companies - closed_companies),
        ),
    ]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(ROOT / "data" / "ai_models_dataset.json")
    frames = build_frames(world)

    # object_id -> 公司/区域适配器
    id_meta: dict[str, dict] = {}
    raw_objects = json.loads((ROOT / "data" / "ai_models_dataset.json").read_text(encoding="utf-8"))["objects"]
    for obj in raw_objects:
        id_meta[obj["id"]] = obj.get("meta") or {}
    gate = ObserverGate(
        company_of=lambda oid: id_meta.get(oid, {}).get("company"),
        region_of=lambda oid: id_meta.get(oid, {}).get("region"),
    )

    known_train, heldout = split_known(world.known_queries())
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=64, heads=4,
        ffn_dimensions=128, output_dimensions=32,
    )

    def resolve_all(pipeline, text: str):
        outputs = {"full": pipeline.resolve(text, query_type="known")}
        for frame in frames:
            outputs[frame.observer_id] = pipeline.resolve(text, observer_frame=frame)
        return outputs

    per_seed = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        entries_spec = []
        for query in known_train:
            variants = query_variants(query["text"], query.get("meta"))
            target = world.target_label(query["target_id"])
            entries_spec.append({
                "variants": [variants[i] for i in (0, 2, 3)] if len(variants) >= 5 else [variants[0]],
                "target": target,
            })
        train(model, world, device, entries=entries_spec, steps=400, lr=1e-3,
              batch_size=16, hard_k=0, seed=seed)
        encoder = WorldEncoder(world, model, device)
        store, ledger, index, vectors, pipeline = build_chain(
            world, encoder, minimum_score=0.40, minimum_coverage=0.60, known_margin=0.01,
        )
        pipeline.reasoner = make_reasoner(world)
        pipeline.access_gate = gate

        vectors_before = index.count
        identity_total = identity_consistent = 0
        leak_total = leak_count = 0
        trace_total = trace_ok = 0
        crossview_total = crossview_hit = 0

        for query in world.known_queries():
            target_id = query["target_id"]
            outputs = resolve_all(pipeline, query["text"])
            full_result = outputs["full"]

            # 1) 身份一致性：有权限的观测点必须与无观测点解析到同一对象
            for frame in frames:
                permitted = gate.check(frame, target_id).allowed
                result = outputs[frame.observer_id]
                if permitted:
                    identity_total += 1
                    identity_consistent += int(
                        result.status == CandidateStatus.SUPPORTED
                        and result.object_id == target_id == full_result.object_id
                    )
                else:
                    # 2) 权限零泄漏：受限者不得以任何形式支持该对象
                    leak_total += 1
                    leaked = (
                        result.status == CandidateStatus.SUPPORTED
                        and result.object_id is not None
                    )
                    leak_count += int(leaked)
                # 5) 可追溯：每个结论性回答都带非空理由且对象文档可取回
                if result.status in (CandidateStatus.SUPPORTED, CandidateStatus.ACCESS_DENIED):
                    trace_total += 1
                    doc_visible = result.object_id is None or any(
                        rec.object_id == result.object_id for rec, _ in store.live_records()
                    )
                    trace_ok += int(bool(result.reason) and doc_visible)

            # 3) 跨视角召回：同一对象、不同措辞，在两个"都有权限"的观测点间复用
            variants = query_variants(query["text"], query.get("meta"))
            if len(variants) >= 5:
                permitted = [f for f in frames if gate.check(f, target_id).allowed]
                if len(permitted) >= 2:
                    first, second = permitted[0], permitted[-1]
                    a = pipeline.resolve(variants[0], observer_frame=first)
                    b = pipeline.resolve(variants[4], observer_frame=second)
                    crossview_total += 1
                    crossview_hit += int(
                        a.status == CandidateStatus.SUPPORTED
                        and b.status == CandidateStatus.SUPPORTED
                        and a.object_id == b.object_id == target_id
                    )

        # 留出集同样跑一致性（超级指代路径）
        for query in heldout:
            outputs = resolve_all(pipeline, query["text"])
            for frame in frames:
                if gate.check(frame, query["target_id"]).allowed:
                    result = outputs[frame.observer_id]
                    identity_total += 1
                    identity_consistent += int(
                        result.status == CandidateStatus.SUPPORTED
                        and result.object_id == query["target_id"]
                    )

        per_seed[str(seed)] = {
            "identity_consistency": identity_consistent / max(1, identity_total),
            "permission_leakage": leak_count / max(1, leak_total),
            "cross_view_recall": crossview_hit / max(1, crossview_total),
            "traceability": trace_ok / max(1, trace_total),
            "vector_copies": index.count,
            "vectors_before_resolutions": vectors_before,
            "counts": {"identity": identity_total, "leak_checks": leak_total,
                       "crossview_pairs": crossview_total, "trace_checks": trace_total},
        }
        print(json.dumps({"phase": "seed", "seed": seed,
                          **{k: round(v, 4) for k, v in per_seed[str(seed)].items()
                             if isinstance(v, float)}}, ensure_ascii=False), flush=True)
        store.close()

    aggregate = {
        key: statistics.mean(per_seed[s][key] for s in per_seed)
        for key in ("identity_consistency", "permission_leakage",
                    "cross_view_recall", "traceability")
    }
    gates = {
        "identity_consistency_100pct": aggregate["identity_consistency"] == 1.0,
        "zero_permission_leakage": aggregate["permission_leakage"] == 0.0,
        "cross_view_recall_ge_95": aggregate["cross_view_recall"] >= 0.95,
        "traceability_ge_95": aggregate["traceability"] >= 0.95,
        "single_copy_vectors": all(p["vector_copies"] == p["vectors_before_resolutions"] for p in per_seed.values()),
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "observers": [f.observer_id for f in frames],
        "verifier": {"minimum_score": 0.40, "minimum_coverage": 0.60, "known_margin": 0.01},
        "note": "门控为确定性掩码；身份解析先于观测点注入；向量单副本断言贯穿全程。",
        "gates": gates,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    out = ROOT / "artifacts" / "v62_observer_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "gates": gates, "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
