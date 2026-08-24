import torch

from cformer_v59 import CandidateLedger, CandidateStatus, EvidenceVerifier
from cformer_v61 import IVFConfig, IVFIndex
from cformer_v61c import ObjectRecord, UnifiedObjectStore, UnifiedResolutionPipeline
from cformer_v62 import (
    ObserverFrame,
    ObserverGate,
    WorldReasoner,
    extract_year,
    parse_as_of,
)


def test_parse_as_of_patterns() -> None:
    assert parse_as_of("截至2023年，GPT系列最新模型是什么？") == 2023
    assert parse_as_of("2021年的时候OpenAI最新的是哪个") == 2021
    assert parse_as_of("到2019年为止，最新的模型") == 2019
    assert parse_as_of("Qwen系列最新的模型是哪一个？") is None      # 无年份
    assert parse_as_of("介绍一下Qwen2.5-Coder") is None            # 版本号不是年份
    assert parse_as_of("它于2026年发布") is None                    # 句式不触发


def test_extract_year_ignores_non_year_numbers() -> None:
    assert extract_year("发布于 2018 年") == 2018
    assert extract_year("版本2.5很小") is None


def _temporal_pipeline():
    """S 系列：2020/2022/2024 三成员；神经嵌入故意偏向未来成员。"""
    vectors = torch.eye(3)
    ids = ["s-2020", "s-2022", "s-2024"]
    store = UnifiedObjectStore()
    for label, object_id in enumerate(ids):
        store.upsert_object(
            ObjectRecord(
                object_id=object_id,
                canonical_name=object_id,
                document={"变化": f"它于 {2020 + 2 * label} 年发布"},
                meta={"company": "X", "series": "S", "series_index": label},
            ),
            aliases=[],
        )
    index = IVFIndex(3, IVFConfig(n_centroids=2, n_iter=2))
    index.train(vectors)
    index.add(vectors, [0, 1, 2])
    query = torch.tensor([0.0, 0.0, 1.0])  # 嵌入最贴近 s-2024（未来成员）

    class Encoder:
        def encode_query(self, text):
            return query, 1.0

        def object_id_of(self, label):
            return ids[label]

    reasoner = WorldReasoner(
        series_key_of=lambda label: ("X", "S"),
        evidence_text_of=lambda label: f"它于 {2020 + 2 * label} 年发布",
        series_index_of=lambda label: label,
        series_key_from_text=lambda text: ("series", ("X", "S")),
    )
    pipeline = UnifiedResolutionPipeline(
        store, CandidateLedger(), index, Encoder(), EvidenceVerifier(),
        nprobe=2, top_ann=16, top_rerank=16, reasoner=reasoner,
    )
    return pipeline


def test_as_of_filters_future_members() -> None:
    pipeline = _temporal_pipeline()
    result = pipeline.resolve("截至2021年，S系列最新模型是什么？")
    assert result.status == CandidateStatus.SUPPORTED
    assert result.path == "reasoned"
    assert result.object_id == "s-2020"
    assert "filtered_future" in result.reason


def test_as_of_exact_boundary_is_inclusive() -> None:
    pipeline = _temporal_pipeline()
    result = pipeline.resolve("截至2022年，S系列最新模型是什么？")
    assert result.status == CandidateStatus.SUPPORTED
    assert result.object_id == "s-2022"


def test_temporal_vacuum_never_falls_back_to_neural() -> None:
    """关键泄漏防线：快照内无对象时必须显式 unknown，绝不返回未来成员。"""
    pipeline = _temporal_pipeline()
    result = pipeline.resolve("截至2019年，S系列最新模型是什么？")
    assert result.status == CandidateStatus.UNKNOWN
    assert result.object_id is None
    assert "temporal_no_member_as_of=2019" in result.reason


def test_no_as_of_keeps_latest_behaviour() -> None:
    pipeline = _temporal_pipeline()
    result = pipeline.resolve("S系列最新模型是什么？")
    assert result.status == CandidateStatus.SUPPORTED
    assert result.object_id == "s-2024"


def test_observer_gate_composes_with_temporal() -> None:
    pipeline = _temporal_pipeline()
    gate = ObserverGate(
        company_of=lambda oid: "X",
        region_of=lambda oid: None,
    )
    pipeline.access_gate = gate
    frame = ObserverFrame("denied", allowed_companies=frozenset({"Other"}))
    result = pipeline.resolve("截至2021年，S系列最新模型是什么？", observer_frame=frame)
    # 门控先于 reasoned 支持：被掩对象不得暴露
    assert result.status in (CandidateStatus.ACCESS_DENIED, CandidateStatus.UNKNOWN)
