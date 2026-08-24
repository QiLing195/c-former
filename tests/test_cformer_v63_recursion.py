import torch

from cformer_v59 import CandidateLedger, CandidateStatus, EvidenceVerifier
from cformer_v61 import IVFConfig, IVFIndex
from cformer_v61c import ObjectRecord, UnifiedObjectStore, UnifiedResolutionPipeline
from cformer_v63 import MAX_HOPS, MultiHopResolver, TransformationGraph, parse_hop_count
from cformer_real import AIModelObject, AIModelWorld, apply_aliases

DATA = None  # 由 fixture 注入


def _world():
    from pathlib import Path
    return AIModelWorld(Path(__file__).resolve().parents[1] / "data" / "ai_models_dataset.json")


def test_parse_hop_count_numerals() -> None:
    assert parse_hop_count("3") == 3
    assert parse_hop_count("三") == 3
    assert parse_hop_count("两") == 2
    assert parse_hop_count("十") == 10


def test_graph_builds_predecessor_edges_from_real_data() -> None:
    world = _world()
    graph = TransformationGraph(world)
    # GPT 链：gpt-4 的前一代是 gpt-4-turbo？按 builder 顺序 GPT-4 之后是 GPT-4 Turbo
    assert graph.predecessor.get("gpt-4-turbo") == "gpt-4"
    assert graph.predecessor.get("gpt-5-1") == "gpt-5"
    # successor 是反向映射
    assert graph.successor.get("gpt-4") == "gpt-4-turbo"
    # 边数应接近对象数减去各系列首成员
    first_members = sum(1 for obj in world.objects
                        if "前一代" not in obj.evidence[2])
    assert len(graph.predecessor) >= len(world.objects) - first_members - 2


def test_walk_backward_forward_chain_end_and_depth() -> None:
    world = _world()
    graph = TransformationGraph(world)
    back = walk_result = __import__("cformer_v63.recursion", fromlist=["walk"]).walk(
        graph, "gpt-5-4", 3, "backward")
    assert walk_result.status == "ok"
    assert walk_result.object_id == "gpt-5"        # 5.4 → 5.2 → 5.1 → 5
    fwd = __import__("cformer_v63.recursion", fromlist=["walk"]).walk(
        graph, "claude-1", 2, "forward")
    assert fwd.status == "ok"
    end = __import__("cformer_v63.recursion", fromlist=["walk"]).walk(
        graph, "gpt-1", 1, "backward")
    assert end.status == "chain_end"
    deep = __import__("cformer_v63.recursion", fromlist=["walk"]).walk(
        graph, "gpt-5-4", MAX_HOPS + 1, "backward")
    assert deep.status == "depth_limit"


def test_cycle_detection_on_synthetic_cycle() -> None:
    objects = [
        AIModelObject("a", 0, "A模型", ("n", "p", "它在S系列中，前一代是 B模型", "2020年")),
        AIModelObject("b", 1, "B模型", ("n", "p", "它在S系列中，前一代是 A模型", "2021年")),
        AIModelObject("c", 2, "C模型", ("n", "p", "它是该系列早期版本", "2022年")),
    ]

    class FakeWorld:
        pass

    fake = FakeWorld()
    fake.objects = objects
    graph = TransformationGraph(fake)
    result = __import__("cformer_v63.recursion", fromlist=["walk"]).walk(
        graph, "a", 5, "backward")
    assert result.status == "cycle"


def test_multihop_resolver_intents() -> None:
    world = _world()
    resolver = MultiHopResolver(TransformationGraph(world))
    intent = resolver.resolve_intent("Qwen2.5-Coder 的前一代是什么？")
    assert intent.direction == "backward" and intent.hops == 1
    assert intent.start_id == "qwen2-5-coder"

    intent = resolver.resolve_intent("从 Claude Opus 5 往前数三代是哪个模型？")
    assert intent.hops == 3 and intent.direction == "backward"

    intent = resolver.resolve_intent("GPT-1 往后数两代是哪个模型？")
    assert intent.hops == 2 and intent.direction == "forward"

    # 无多跳意图 → no_anchor 路径返回
    result = resolver.run("阿里巴巴登顶国产第一的模型是什么？")
    assert result.status == "no_anchor"


def _pipeline_with_multihop(world):
    graph = TransformationGraph(world)
    device = torch.device("cpu")
    store = UnifiedObjectStore()
    for obj in world.objects:
        evidence = dict(zip(("名称", "属性", "关系", "变化"), obj.evidence))
        store.upsert_object(
            ObjectRecord(object_id=obj.object_id, canonical_name=obj.name,
                         document=evidence, meta={}),
            aliases=[],
        )
    vectors = torch.nn.functional.normalize(torch.randn(len(world.objects), 16), dim=-1)
    index = IVFIndex(16, IVFConfig(n_centroids=8, n_iter=2))
    index.train(vectors)
    index.add(vectors, list(range(len(world.objects))))

    class Encoder:
        def encode_query(self, text):
            return torch.zeros(16), 1.0

        def object_id_of(self, label):
            return world.objects[label].object_id

    return UnifiedResolutionPipeline(
        store, CandidateLedger(), index, Encoder(), EvidenceVerifier(),
        nprobe=8, top_ann=64, top_rerank=16,
        multihop=MultiHopResolver(graph),
    )


def test_pipeline_recursion_path_end_to_end() -> None:
    world = _world()
    pipeline = _pipeline_with_multihop(world)
    result = pipeline.resolve("Qwen2.5-Coder 的前一代是什么？")
    assert result.status == CandidateStatus.SUPPORTED
    assert result.path == "recursion"
    assert result.object_id == "qwen2-5"
    assert "recursion_hops=1" in result.reason


def test_pipeline_recursion_chain_end_is_explicit_unknown() -> None:
    world = _world()
    pipeline = _pipeline_with_multihop(world)
    result = pipeline.resolve("GPT-1 的前一代是什么？")
    assert result.status == CandidateStatus.UNKNOWN
    assert "recursion_chain_end" in result.reason


def test_apply_aliases_rewrites_nicknames() -> None:
    rewritten = apply_aliases(
        "千问系列最新的模型是哪一个？",
        {"阿里": "阿里巴巴"}, {"千问": "Qwen"},
    )
    assert "qwen" in rewritten and "千问" not in rewritten
    rewritten = apply_aliases(
        "阿里家最新开源模型",
        {"阿里": "阿里巴巴"}, {},
    )
    assert "阿里巴巴" in rewritten


def test_series_anchoring_after_alias_rewrite() -> None:
    world = _world()
    lowered = apply_aliases("千问系列最新的模型是哪一个？",
                            world.company_aliases, world.series_aliases)
    lexicon = dict(((key), key) for key in [k for k, _, _ in world.series_lexicon()])
    matched = [key for key, _, series in world.series_lexicon()
               if series in lowered]
    assert ("阿里巴巴", "Qwen") in matched
