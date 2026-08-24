import torch

from cformer_v59 import CandidateLedger, CandidateStatus, EvidenceVerifier
from cformer_v61 import IVFConfig, IVFIndex
from cformer_v61c import ObjectRecord, UnifiedObjectStore, UnifiedResolutionPipeline
from cformer_v62 import WorldReasoner, extract_year, parse_direction


def test_parse_direction_and_conflicts() -> None:
    assert parse_direction("Qwen系列最新模型是什么") == "max"
    assert parse_direction("最早那版GPT是哪年发布的") == "min"
    assert parse_direction("初代模型叫什么") == "min"
    assert parse_direction("从最早到最新的演变") is None  # 方向冲突
    assert parse_direction("随便聊聊Kimi") is None


def test_extract_year_patterns() -> None:
    assert extract_year("它于 2026 年发布") == 2026
    assert extract_year("发布于2018年，属于GPT系列") == 2018
    assert extract_year("没有年份信息") is None


def _reasoning_world(neural_favorite: int):
    """三个同系列对象：0=2023, 1=2024, 2=2025。neural_favorite 控制嵌入偏向。"""
    vectors = torch.zeros(3, 8)
    for index in range(3):
        vectors[index, index] = 1.0
    # 把神经空间的"查询方向"拉向指定成员（模拟 dual-encoder 记忆偏好）
    query = torch.zeros(8)
    query[neural_favorite] = 1.0
    ids = ["m-2023", "m-2024", "m-2025"]
    store = UnifiedObjectStore()
    for label, object_id in enumerate(ids):
        store.upsert_object(
            ObjectRecord(
                object_id=object_id,
                canonical_name=object_id,
                document={"变化": f"它于 {2023 + label} 年发布"},
                meta={"company": "X", "series": "S", "series_index": label},
            ),
            aliases=[],
        )
    index = IVFIndex(8, IVFConfig(n_centroids=2, n_iter=2))
    index.train(vectors)
    index.add(vectors, [0, 1, 2])

    class Encoder:
        calls = 0

        def encode_query(self, text):
            Encoder.calls += 1
            return query, 1.0

        def object_id_of(self, label):
            return ids[label]

    reasoner = WorldReasoner(
        series_key_of=lambda label: ("X", "S"),
        evidence_text_of=lambda label: f"它于 {2023 + label} 年发布",
        series_index_of=lambda label: label,
    )
    pipeline = UnifiedResolutionPipeline(
        store, CandidateLedger(), index, Encoder(), EvidenceVerifier(),
        nprobe=2, top_ann=16, top_rerank=16, reasoner=reasoner,
    )
    return pipeline, Encoder


def test_reasoner_overrides_biased_neural_ranking_for_latest() -> None:
    pipeline, encoder = _reasoning_world(neural_favorite=0)  # 神经最偏爱 2023 老成员
    result = pipeline.resolve("S系列最新模型是什么？")
    assert result.status == CandidateStatus.SUPPORTED
    assert result.path == "reasoned"
    assert result.object_id == "m-2025"
    assert "year=2025" in result.reason


def test_reasoner_selects_earliest() -> None:
    pipeline, _ = _reasoning_world(neural_favorite=2)
    result = pipeline.resolve("S系列最早的模型叫什么？")
    assert result.path == "reasoned"
    assert result.object_id == "m-2023"


def test_reasoner_falls_back_when_years_missing() -> None:
    pipeline, _ = _reasoning_world(neural_favorite=0)
    # 抽掉年份线索：evidence_text_of 返回无年份文本 → 回退神经路径（老成员胜出）
    pipeline.reasoner.evidence_text_of = lambda label: "没有年份"
    result = pipeline.resolve("S系列最新模型是什么？")
    assert result.path != "reasoned"
    assert result.object_id == "m-2023"


def test_no_selection_phrase_keeps_old_behaviour() -> None:
    pipeline, _ = _reasoning_world(neural_favorite=1)
    result = pipeline.resolve("S是哪一个模型？")
    # 无方向词：推理块不介入；同分歧义或按神经最高分支持均可，但绝不能 reasoned
    assert result.path != "reasoned"


def test_lexical_anchor_overrides_wrong_neural_top() -> None:
    """神经榜首整个系列都错时，词法锚定把比较拉回正确系列。"""
    pipeline, _ = _reasoning_world(neural_favorite=0)
    # 榜首系列是 ("WrongCo","T")；查询词法命中 ("X","S")
    pipeline.reasoner.series_key_of = lambda label: ("WrongCo", "T") if label == 0 else ("X", "S")
    pipeline.reasoner.series_key_from_text = lambda text: ("series", ("X", "S"))
    result = pipeline.resolve("S系列最新模型是什么？")
    assert result.path == "reasoned"
    assert result.object_id == "m-2025"
    assert "anchor=lexical" in result.reason


def test_single_member_series_resolves_directly() -> None:
    pipeline, _ = _reasoning_world(neural_favorite=0)
    pipeline.reasoner.series_key_of = lambda label: ("X", "S") if label == 2 else ("Other", f"O{label}")
    pipeline.reasoner.series_key_from_text = lambda text: ("series", ("X", "S"))
    result = pipeline.resolve("S系列最新模型是什么？")
    assert result.path == "reasoned"
    assert result.object_id == "m-2025"
    assert "single_member_series" in result.reason


def test_direction_conflict_falls_back() -> None:
    pipeline, _ = _reasoning_world(neural_favorite=2)
    result = pipeline.resolve("从最早到最新的S系列演变")
    assert result.path != "reasoned"
