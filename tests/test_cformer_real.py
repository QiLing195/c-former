import json
from pathlib import Path

from cformer_real import AIModelWorld, MixedTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ai_models_dataset.json"


def _load():
    return AIModelWorld(DATA)


def _raw():
    return json.loads(DATA.read_text(encoding="utf-8"))


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
    assert tokenizer.tokenize("GPT-5.2") == ["gpt", "5", "2"]
    assert tokenizer.tokenize("通义千问3.7") == ["通", "义", "千", "问", "3", "7"]
    assert "deepseek" in tokenizer.index and "v4" in tokenizer.index


def test_loader_shapes() -> None:
    world = _load()
    raw = _raw()
    assert len(world.objects) == len(raw["objects"])
    assert len(world.objects) >= 100  # 扩充后的真实语料集下限
    candidates = world.encode_candidates(world.objects)
    assert candidates.shape == (len(world.objects), 4, world.field_length)
    tokens, coverage = world.encode_query("阿里巴巴的旗舰模型叫什么？")
    assert tokens.shape == (world.query_length,)
    assert 0.0 <= coverage <= 1.0


def test_queries_have_valid_targets() -> None:
    world = _load()
    raw = _raw()
    known = world.known_queries()
    assert len(known) >= 20
    assert len(known) == sum(1 for q in raw["queries"] if q["kind"] == "known")
    for query in known:
        assert 0 <= world.target_label(query["target_id"]) < len(world.objects)
    for query in world.ambiguous_queries() + world.unknown_queries():
        assert query["target_id"] is None


def test_blindset_targets_valid_and_template_isolated() -> None:
    blind = json.loads(
        (ROOT / "data" / "ai_models_blindset.json").read_text(encoding="utf-8")
    )
    world = _load()
    kinds = {"known", "ambiguous", "unknown"}
    banned = ("系列最新模型是什么", "是哪一个？", "哪个公司发布了")
    for query in blind["queries"]:
        assert query["kind"] in kinds, query
        if query["target_id"] is not None:
            assert 0 <= world.target_label(query["target_id"]) < len(world.objects), query
        for fragment in banned:
            assert fragment not in query["text"], query["text"]


def test_paraphrase_variants_stay_in_vocabulary() -> None:
    from cformer_real import query_variants

    world = _load()
    raw = _raw()
    checked = 0
    for query in raw["queries"]:
        if query.get("meta"):
            variants = query_variants(query["text"], query["meta"])
            assert len(variants) == 5
            assert query["text"] == variants[0]
            for variant in variants:
                _, coverage = world.tokenizer.encode(variant, world.query_length)
                assert coverage == 1.0, variant
                checked += 1
    assert checked >= 100


def test_series_siblings_exclude_self_and_share_meta() -> None:
    world = _load()
    raw = _raw()
    label = world.target_label("qwen3")
    siblings = world.series_siblings(label)
    assert label not in siblings
    assert len(siblings) >= 5
    meta_by_id = {obj["id"]: obj.get("meta", {}) for obj in raw["objects"]}
    target_meta = meta_by_id["qwen3"]
    for sibling in siblings:
        sibling_id = world.objects[sibling].object_id
        assert meta_by_id[sibling_id]["series"] == target_meta["series"]
    # 无 meta 的对象没有兄弟
    hand_written = [obj for obj in raw["objects"] if not obj.get("meta")]
    if hand_written:
        assert world.series_siblings(world.target_label(hand_written[0]["id"])) == []


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
