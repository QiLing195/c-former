import torch

from cformer_v55 import CognitiveStatus, ObserverContext, build_cognitive_worlds
from cformer_v56 import (
    CognitiveCandidateAdapter,
    CognitiveCFormerReranker,
    EvidenceRAGReranker,
    GovernedRetriever,
    NeuralQuery,
    RerankerConfig,
    SyntheticRetrievalWorld,
    sample_training_batch,
)


def test_v55_memory_adapter_and_governed_prefilter() -> None:
    world = build_cognitive_worlds()[1]
    adapter = CognitiveCandidateAdapter()
    candidates = adapter.adapt(world.memory)
    query = NeuralQuery(
        10,
        1,
        adapter.key_for_object(10),
        observer_scope=1,
        query_time=20,
        ingest_cutoff=20,
    )
    result = GovernedRetriever(candidates, top_k=8).prefilter(query)
    assert result.status == CognitiveStatus.ANSWER
    assert 1_000_000_010 in result.correct_candidate_ids


def test_governed_prefilter_preserves_access_boundary() -> None:
    world = build_cognitive_worlds()[2]
    adapter = CognitiveCandidateAdapter()
    candidates = adapter.adapt(world.memory)
    query = NeuralQuery(
        20,
        0,
        adapter.key_for_object(20),
        observer_scope=2,
        query_time=75,
        ingest_cutoff=90,
    )
    result = GovernedRetriever(candidates).prefilter(query)
    assert result.status == CognitiveStatus.ACCESS_DENIED
    assert result.candidates == ()


def test_equal_parameter_rerankers_and_forward_shape() -> None:
    config = RerankerConfig()
    cformer = CognitiveCFormerReranker(config)
    rag = EvidenceRAGReranker(config)
    assert sum(p.numel() for p in cformer.parameters()) == sum(
        p.numel() for p in rag.parameters()
    )
    generator = torch.Generator().manual_seed(1)
    batch = sample_training_batch(3, 16, 8, generator=generator)
    assert cformer(*batch[:-1]).shape == (3, 16)
    assert rag(*batch[:-1]).shape == (3, 16)


def test_indexed_shortlist_is_scale_bounded_and_contains_target() -> None:
    for scale in (2048, 8192, 32768):
        world = SyntheticRetrievalWorld.build(scale, 0)
        entities, _, _, _, correct = world.fixed_queries(20)
        shortlist, labels = world.indexed_shortlists(entities, correct, 64)
        assert shortlist.shape == (20, 64)
        assert shortlist.gather(1, labels[:, None]).squeeze(1).equal(correct)


def test_observer_changes_cformer_query_representation() -> None:
    model = CognitiveCFormerReranker()
    key = torch.randn(1, 8)
    kind = torch.zeros(1, dtype=torch.long)
    first = model.encode_query(key, kind, torch.tensor([1]))
    second = model.encode_query(key, kind, torch.tensor([2]))
    assert not torch.allclose(first, second)
