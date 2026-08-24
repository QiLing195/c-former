import json

from cformer_real import AIModelWorld, MixedTokenizer

DATA = r"E:\deepseek\c-former\data\ai_models_dataset.json"


def _load():
    return AIModelWorld(DATA)


def test_tokenizer_has_no_oov_on_corpus() -> None:
    world = _load()
    texts = []
    for obj in world.objects:
        texts.append(obj.name)
        texts.extend(obj.evidence)
    for query in world.queries:
        texts.append(query["text"])
    for text in texts:
        _, coverage = world.tokenizer.encode(text, world.query_length)
        assert coverage == 1.0, text  # corpus tokens must all be known


def test_tokenizer_mixed_latin_chinese() -> None:
    tokenizer = MixedTokenizer(["GPT-5.2", "通义千问3.7", "DeepSeek-V4-Pro"])
    assert tokenizer.tokenize("GPT-5.2") == ["gpt-5-2"]
    assert tokenizer.tokenize("通义千问3.7") == ["通", "义", "千", "问", "3.7"]
    assert tokenizer.tokenize("DeepSeek-V4-Pro") == ["deepseek-v4-pro"]
    assert "deepseek-v4-pro" in tokenizer.index
    assert "gpt-5-2" in tokenizer.index


def test_loader_shapes() -> None:
    world = _load()
    assert len(world.objects) == 18
    candidates = world.encode_candidates(world.objects)
    assert candidates.shape == (18, 4, world.field_length)
    tokens, coverage = world.encode_query("阿里巴巴的旗舰模型叫什么？")
    assert tokens.shape == (world.query_length,)
    assert 0.0 <= coverage <= 1.0


def test_queries_have_valid_targets() -> None:
    world = _load()
    known = world.known_queries()
    assert len(known) == 15
    for query in known:
        assert 0 <= world.target_label(query["target_id"]) < len(world.objects)
    for query in world.ambiguous_queries() + world.unknown_queries():
        assert query["target_id"] is None


def test_encoding_is_label_invariant() -> None:
    import torch

    from cformer_real import AIModelObject

    world = _load()
    obj = world.objects[0]
    # 改成不同的整数 label、但证据文本相同，编码必须完全一致（编码只依赖证据，不依赖序号）
    relabeled = AIModelObject(obj.object_id, 9999, obj.name, obj.evidence)
    original = world.encode_candidates([obj])
    changed = world.encode_candidates([relabeled])
    assert torch.equal(original, changed)
