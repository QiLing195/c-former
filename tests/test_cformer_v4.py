import torch

from cformer_v4 import (
    EvidenceRAGTransformer,
    ReliableCFormer,
    ReliableDenseTransformer,
    ReliabilityTask,
    V4Config,
)


def test_v4_batch_contains_all_shapes_and_hard_masks() -> None:
    task = ReliabilityTask()
    batch = task.sample_batch(32, 128, "cpu")
    memory, question, observer, allowed, answers, statuses, override, first, second = batch
    assert memory.shape == (32, 128, 4)
    assert question.shape == (32, 4)
    assert allowed.shape == (32, 128)
    assert answers.shape == statuses.shape == override.shape == first.shape == second.shape == (32,)
    assert (~allowed).any()


def test_v4_models_return_answer_and_status() -> None:
    task = ReliabilityTask()
    batch = task.sample_batch(4, 128, "cpu")
    memory, question, observer, allowed, *_ = batch
    for model in (
        ReliableCFormer(V4Config()),
        EvidenceRAGTransformer(V4Config()),
        ReliableDenseTransformer(V4Config()),
    ):
        output = model(memory, question, observer, allowed)
        assert output["answer_logits"].shape == (4, 80)
        assert output["status_logits"].shape == (4, 5)


def test_hard_mask_prevents_retrieval_of_denied_facts() -> None:
    task = ReliabilityTask()
    model = ReliableCFormer(V4Config())
    memory, question, observer, allowed, *_ = task.sample_batch(8, 128, "cpu")
    allowed[:, 0] = False
    output = model(memory, question, observer, allowed)
    assert output["selected_first"].ne(0).all()
    assert output["selected_second"].ne(0).all()


def test_fixed_cases_cover_all_statuses_at_each_scale() -> None:
    task = ReliabilityTask()
    for scale in (128, 512, 2048):
        cases = task.fixed_cases(scale, 0)
        assert cases.memory.shape == (68, scale, 4)
        assert set(cases.statuses.tolist()) == {0, 1, 2, 3, 4}
