import torch

from cformer_v3 import DenseConcatTransformer, HierarchicalCFormer, ScaleWorldTask, V3Config


def test_scale_batches_and_evidence_are_valid() -> None:
    task = ScaleWorldTask()
    for scale in (128, 512, 2048):
        batch = task.sample_batch(2, scale, "cpu")
        memory, question, observer, labels, tasks, first, second = batch
        assert memory.shape == (2, scale, 4)
        assert question.shape == (2, 3)
        assert observer.shape == labels.shape == tasks.shape == first.shape == second.shape == (2,)
        assert first.max() < scale and second.max() < scale


def test_hierarchical_cached_and_direct_match() -> None:
    task = ScaleWorldTask()
    model = HierarchicalCFormer(V3Config())
    model.eval()
    memory, question, observer, *_ = task.sample_batch(3, 128, "cpu")
    with torch.no_grad():
        direct = model(memory, question, observer)["logits"]
        cached = model.answer_from_cache(model.encode_world(memory), question, observer)["logits"]
    assert torch.allclose(direct, cached)


def test_both_v3_models_return_expected_shape() -> None:
    task = ScaleWorldTask()
    memory, question, observer, *_ = task.sample_batch(2, 128, "cpu")
    config = V3Config(dense_layers=1)
    for model in (DenseConcatTransformer(config), HierarchicalCFormer(config)):
        logits = model(memory, question, observer)["logits"]
        assert logits.shape == (2, 80)
        assert torch.isfinite(logits).all()


def test_fixed_worlds_have_32_queries_and_five_worlds_per_scale() -> None:
    task = ScaleWorldTask()
    for scale in (128, 512, 2048):
        worlds = [task.fixed_world(scale, index) for index in range(5)]
        assert len(worlds) == 5
        assert all(world.memory.shape == (scale, 4) for world in worlds)
        assert all(world.questions.shape[0] == 36 for world in worlds)

