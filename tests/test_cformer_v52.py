import torch

from cformer_v4 import ReliableCFormer, V4Config
from cformer_v52 import V52StressSuite, answer_from_cache_ablation


def test_stress_suite_distinguishes_effective_and_distractor_facts() -> None:
    suite = V52StressSuite()
    world = suite.world(8192, 0)
    assert world.memory.shape == (8192, 4)
    assert world.effective_facts == 2048
    assert world.distractor_facts == 6144
    assert world.questions.shape[0] == world.labels.shape[0] == 36


def test_full_ablation_path_matches_model_forward() -> None:
    torch.manual_seed(52)
    model = ReliableCFormer(V4Config())
    suite = V52StressSuite()
    world = suite.world(2048, 0)
    question = world.questions[:4]
    observer = world.observers[:4]
    allowed = torch.ones(4, 2048, dtype=torch.bool)
    cache = model.encode_world(world.memory[None])
    expanded = tuple(value.expand(4, -1, -1) for value in cache)
    expected = model.answer_from_cache(expanded, question, observer, allowed)
    actual = answer_from_cache_ablation(
        model, expanded, question, observer, allowed, "full"
    )
    assert torch.allclose(expected["answer_logits"], actual["answer_logits"], atol=1e-6)
    assert torch.allclose(expected["status_logits"], actual["status_logits"], atol=1e-6)


def test_reliability_margin_ignores_equivalent_duplicate_evidence() -> None:
    model = ReliableCFormer(V4Config(reliability_topk=4))
    exact = torch.zeros(1, 4, model.config.d_model)
    exact[:, 2, 0] = 1.0
    exact[:, 3, 1] = 1.0
    scores = [torch.tensor([[10.0, 10.0, 7.0, 2.0]])] * 2
    reliability = model._reliability_features(scores, exact)
    expected_margin = torch.tanh(torch.tensor(3.0 / 10.0))
    assert torch.allclose(reliability[0, 1], expected_margin)
    assert torch.allclose(reliability[0, 3], expected_margin)
