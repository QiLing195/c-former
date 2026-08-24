import torch

from cformer_v59 import CandidateLedger, CandidateStatus, EvidenceVerifier
from cformer_v61 import IVFConfig, IVFIndex
from cformer_v61c import ObjectRecord, UnifiedObjectStore, UnifiedResolutionPipeline
from cformer_v62 import ObserverFrame, ObserverGate


class Encoder:
    def __init__(self, object_ids, vectors, queries):
        self.object_ids = object_ids
        self.vectors = vectors
        self.queries = queries  # text -> (vector, coverage)
        self.calls = 0

    def encode_query(self, text):
        self.calls += 1
        return self.queries[text]

    def object_id_of(self, label):
        return self.object_ids[label]


def _world_with_gate():
    vectors = torch.eye(3)
    ids = ["obj-openai", "obj-ali", "obj-hf"]
    companies = ["OpenAI", "阿里巴巴", "Hugging Face"]
    regions = ["美国", "中国", "法国"]
    store = UnifiedObjectStore()
    for label, object_id in enumerate(ids):
        store.upsert_object(
            ObjectRecord(
                object_id=object_id,
                canonical_name=object_id,
                document={"变化": f"它于 {2024 + label} 年发布"},
                meta={"company": companies[label], "region": regions[label]},
            ),
            aliases=[],
        )
    index = IVFIndex(3, IVFConfig(n_centroids=2, n_iter=2))
    index.train(vectors)
    index.add(vectors, [0, 1, 2])
    encoder = Encoder(ids, vectors, {
        "查询": (vectors[1], 1.0),           # 命中 obj-ali（中国）
    })
    gate = ObserverGate(
        company_of=lambda object_id: companies[ids.index(object_id)],
        region_of=lambda object_id: regions[ids.index(object_id)],
    )
    pipeline = UnifiedResolutionPipeline(
        store, CandidateLedger(), index, encoder, EvidenceVerifier(),
        nprobe=2, top_ann=16, top_rerank=16, access_gate=gate,
    )
    return store, pipeline


def test_gate_denies_and_never_leaks_object() -> None:
    store, pipeline = _world_with_gate()
    us_only = ObserverFrame("us-observer", allowed_companies=frozenset({"OpenAI"}))
    result = pipeline.resolve("查询", observer_frame=us_only)
    assert result.status == CandidateStatus.ACCESS_DENIED
    assert result.object_id is None          # 不泄漏被拒对象
    assert "company_not_visible" in result.reason

    # 无观测点 / 有权限观测点照常支持
    open_result = pipeline.resolve("查询")
    assert open_result.status == CandidateStatus.SUPPORTED
    cn_observer = ObserverFrame("cn-observer", allowed_regions=frozenset({"中国"}))
    ok = pipeline.resolve("查询", observer_frame=cn_observer)
    assert ok.status == CandidateStatus.SUPPORTED
    assert ok.object_id == "obj-ali"


def test_exact_alias_path_also_gated() -> None:
    store, pipeline = _world_with_gate()
    store.add_verified_alias("秘密模型", "obj-ali")
    us_only = ObserverFrame("us-observer", allowed_companies=frozenset({"OpenAI"}))
    result = pipeline.resolve("秘密模型", observer_frame=us_only)
    assert result.status == CandidateStatus.ACCESS_DENIED
    assert result.path == "exact"
    # 有权限者经同一别名正常命中
    cn = ObserverFrame("cn-observer", allowed_regions=frozenset({"中国"}))
    hit = pipeline.resolve("秘密模型", observer_frame=cn)
    assert hit.status == CandidateStatus.SUPPORTED
    assert hit.object_id == "obj-ali"


def test_identity_consistent_across_permitted_observers() -> None:
    _, pipeline = _world_with_gate()
    frames = [
        None,
        ObserverFrame("a", allowed_companies=frozenset({"阿里巴巴"})),
        ObserverFrame("b", allowed_companies=frozenset({"阿里巴巴", "OpenAI"}),
                      allowed_regions=frozenset({"中国", "美国"})),
    ]
    resolved = [pipeline.resolve("查询", observer_frame=f).object_id for f in frames]
    assert resolved == ["obj-ali"] * 3   # 合法观测点间身份一致


def test_single_copy_principle() -> None:
    """观测点数量不影响向量库规模——掩码不是过滤后的新库。"""
    _, pipeline = _world_with_gate()
    before = pipeline.index.count
    frames = [ObserverFrame(f"o{i}") for i in range(10)]
    for frame in frames:
        pipeline.resolve("查询", observer_frame=frame)
    assert pipeline.index.count == before == 3
