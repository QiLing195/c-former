import torch

from cformer_v55 import build_cognitive_worlds
from cformer_v57 import (
    CognitiveTextAdapter,
    ControlledTransformationEngine,
    HashedTextEncoder,
    RouteStatus,
    TextCFormerReranker,
    TextEvidenceRAGReranker,
    TextRetrievalWorld,
    TextTransition,
    normalize_text,
)


def test_unicode_normalization_and_noisy_alias_similarity() -> None:
    assert normalize_text("Ｅ１２３　状态") == "e123 状态"
    encoder = HashedTextEncoder()
    clean = encoder.encode("星体-123456 状态")
    noisy = encoder.encode("星体 - １２３４５６；状态！")
    assert float(clean @ noisy) > 0.35


def test_v55_memory_can_be_rendered_as_text_documents() -> None:
    world = build_cognitive_worlds()[1]
    documents = CognitiveTextAdapter().render(world.memory)
    assert any("Water" in document and "phase=liquid" in document for document in documents)
    assert any("Transformation freeze" in document for document in documents)


def test_text_shortlist_contains_noisy_targets_at_2k() -> None:
    world = TextRetrievalWorld.build(2048, 0)
    _, variants, texts, features, _, _, correct = world.fixed_queries(100)
    _, _, recall, _ = world.text_shortlists(features, correct, query_texts=texts)
    assert recall.float().mean().item() >= 0.95
    for variant in range(4):
        assert recall[variants.eq(variant)].float().mean().item() >= 0.9


def test_text_models_have_equal_parameters_and_expected_shape() -> None:
    cformer = TextCFormerReranker()
    rag = TextEvidenceRAGReranker()
    assert sum(p.numel() for p in cformer.parameters()) == sum(p.numel() for p in rag.parameters())
    world = TextRetrievalWorld.build(2048, 0)
    entities, _, _, query, kinds, observers, correct = world.fixed_queries(3)
    ids, _, _, _ = world.text_shortlists(query, correct)
    logits = cformer(
        world.candidate_features[ids],
        world.candidate_kinds[ids],
        world.candidate_scopes[ids],
        query,
        kinds,
        observers,
    )
    assert logits.shape == (3, 64)


def test_controlled_text_recursion_and_boundaries() -> None:
    edges = (
        TextTransition(1, 1, 2, "cool", "降温 cooling 变换"),
        TextTransition(2, 2, 3, "compress", "压缩 compression 变换"),
        TextTransition(3, 3, 1, "return", "返回 return 变换"),
        TextTransition(4, 10, 11, "secret", "机密 secret 变换", visibility_scope=frozenset({1})),
    )
    engine = ControlledTransformationEngine(edges)
    answer = engine.route(1, ("cooling 降温", "compression 压缩"), observer_scope=1, query_time=1, ingest_cutoff=1)
    assert answer.status == RouteStatus.ANSWER
    assert answer.object_id == 3
    assert answer.evidence_ids == (1, 2)
    cycle = engine.route(1, ("cooling", "compression", "return"), observer_scope=1, query_time=1, ingest_cutoff=1)
    assert cycle.status == RouteStatus.CYCLE
    denied = engine.route(10, ("secret",), observer_scope=2, query_time=1, ingest_cutoff=1)
    assert denied.status == RouteStatus.ACCESS_DENIED
    limited = engine.route(1, ("cooling", "compression"), observer_scope=1, query_time=1, ingest_cutoff=1, max_depth=1)
    assert limited.status == RouteStatus.DEPTH_LIMIT

    changed = False
    def mutate(depth):
        nonlocal changed
        if depth == 1 and not changed:
            changed = True
            engine.touch_version()

    versioned = engine.route(1, ("cooling", "compression"), observer_scope=1, query_time=1, ingest_cutoff=1, on_hop=mutate)
    assert versioned.status == RouteStatus.VERSION_CHANGED
