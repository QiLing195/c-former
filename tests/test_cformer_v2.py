import torch

from cformer_v2 import SharedMemoryQA, V2Config, WorldTask, fixed_worlds


def test_v2_batch_and_forward_shapes() -> None:
    task = WorldTask()
    model = SharedMemoryQA(V2Config(d_model=32, n_heads=4, d_ff=64, memory_layers=1, query_layers=1))
    memory, question, observer, labels, tasks = task.sample_batch(5, "cpu")
    logits = model(memory, question, observer)
    assert memory.shape == (5, 14, 4)
    assert question.shape == (5, 3)
    assert logits.shape == (5, 8)
    assert labels.shape == tasks.shape == (5,)


def test_cached_and_uncached_answers_match() -> None:
    task = WorldTask()
    model = SharedMemoryQA(V2Config(dropout=0.0))
    model.eval()
    memory, question, observer, _, _ = task.sample_batch(4, "cpu")
    with torch.no_grad():
        direct = model(memory, question, observer)
        cached = model.answer_from_memory(model.encode_memory(memory), question, observer)
    assert torch.allclose(direct, cached)


def test_memory_fact_order_is_permutation_invariant() -> None:
    task = WorldTask()
    model = SharedMemoryQA(V2Config(dropout=0.0))
    model.eval()
    memory, question, observer, _, _ = task.sample_batch(3, "cpu")
    order = torch.randperm(task.num_facts)
    with torch.no_grad():
        original = model(memory, question, observer)
        permuted = model(memory[:, order], question, observer)
    assert torch.allclose(original, permuted, atol=1e-5)


def test_fixed_world_contains_all_query_types() -> None:
    task = WorldTask()
    memory, questions, observers, labels, tasks = task.fixed_world_tensors(fixed_worlds()[0])
    assert memory.shape == (14, 4)
    assert questions.shape[0] == observers.shape[0] == labels.shape[0] == tasks.shape[0] == 36
    assert set(tasks.tolist()) == {0, 1, 2, 3}

