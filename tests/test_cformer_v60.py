import torch

from cformer_v60 import (
    ChineseAliasWorld,
    ChineseTransformerConfig,
    FlatTransformerResolver,
    MeanPoolMLPResolver,
    TokenCFormerResolver,
)


def test_chinese_world_is_identity_free_and_permutation_unique() -> None:
    world = ChineseAliasWorld(2048)
    objects = world.objects(range(world.scale))
    assert len({obj.values for obj in objects}) == world.scale
    texts = []
    for obj in objects[:40]:
        texts.extend(world.candidate_evidence(obj.values))
        texts.extend(world.query_text(obj.values, variant) for variant in range(6))
    world.assert_identity_free(texts)


def test_family_split_and_label_invariance() -> None:
    world = ChineseAliasWorld(2048)
    train = world.training_objects(100)
    heldout = world.heldout_objects(30)
    assert all(world.family_fold(obj.values) != 4 for obj in train)
    assert all(world.family_fold(obj.values) == 4 for obj in heldout)
    query, _ = world.encode_queries(heldout[:3], [0, 1, 4])
    changed = [type(obj)(999999, obj.values) for obj in heldout[:3]]
    same, _ = world.encode_queries(changed, [0, 1, 4])
    assert torch.equal(query, same)


def test_token_transformer_is_order_sensitive_but_mean_pool_is_not() -> None:
    torch.manual_seed(60)
    world = ChineseAliasWorld(2048)
    first, _ = world.tokenizer.encode("不要赤红需要蔚蓝", world.query_length)
    second, _ = world.tokenizer.encode("不要蔚蓝需要赤红", world.query_length)
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=2,
        d_model=32,
        heads=4,
        ffn_dimensions=64,
        output_dimensions=16,
    )
    mlp = MeanPoolMLPResolver(config).eval()
    transformer = TokenCFormerResolver(config).eval()
    with torch.no_grad():
        assert torch.allclose(
            mlp.encode_query(first[None]), mlp.encode_query(second[None]), atol=1e-6
        )
        assert not torch.allclose(
            transformer.encode_query(first[None]),
            transformer.encode_query(second[None]),
            atol=1e-5,
        )


def test_negation_pairs_have_identical_bags_but_opposite_targets() -> None:
    world = ChineseAliasWorld(2048)
    first = type(world.object_at(0))(-1, (0, 2, 3, 4))
    second = type(first)(-1, (8, 2, 3, 4))
    first_tokens, _ = world.encode_queries([first], [4])
    second_tokens, _ = world.encode_queries([second], [4])
    first_bag = sorted(first_tokens[0][first_tokens[0].ne(0)].tolist())
    second_bag = sorted(second_tokens[0][second_tokens[0].ne(0)].tolist())
    assert first_bag == second_bag
    assert not torch.equal(first_tokens, second_tokens)
    assert world.hard_negative(first, 4).values == second.values


def test_all_resolvers_forward_and_train_with_hard_negatives() -> None:
    world = ChineseAliasWorld(2048)
    objects = world.training_objects(6)
    variants = list(range(6))
    queries, coverage = world.encode_queries(objects, variants)
    positives = world.encode_candidates(objects)
    negatives = world.encode_candidates(
        [world.hard_negative(obj, variant) for obj, variant in zip(objects, variants)]
    )
    assert torch.all(coverage == 1)
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=2,
        d_model=32,
        heads=4,
        ffn_dimensions=64,
        output_dimensions=16,
    )
    for cls in (TokenCFormerResolver, MeanPoolMLPResolver, FlatTransformerResolver):
        model = cls(config)
        assert model.encode_query(queries).shape == (6, 16)
        assert model.encode_candidate(positives).shape == (6, 16)
        assert torch.isfinite(model.contrastive_loss(queries, positives, negatives))


def test_six_layer_default_width_stays_within_v60_parameter_budget() -> None:
    world = ChineseAliasWorld(2048)
    model = TokenCFormerResolver(
        ChineseTransformerConfig(world.tokenizer.size, layers=6)
    )
    assert model.parameter_count() < 15_000_000
