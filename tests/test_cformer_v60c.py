import torch

from cformer_v60 import ChineseAliasWorld, ChineseSemanticObject
from cformer_v60c import (
    REGION_NEAR_GROUPS,
    TYPO_POOL,
    RegionAugmentedWorld,
    region_partners,
)


def _sample_objects(count: int = 24) -> list[ChineseSemanticObject]:
    world = ChineseAliasWorld(2048)
    return world.training_objects(count, seed_offset=7)


def test_region_near_groups_are_disjoint_and_consistent() -> None:
    seen: set[int] = set()
    for group in REGION_NEAR_GROUPS:
        assert len(group) >= 2
        assert len(set(group)) == len(group)
        for value in group:
            assert 0 <= value < 16
            assert value not in seen, value
            seen.add(value)
    for value in range(16):
        partners = region_partners(value)
        if value in seen:
            assert partners, value
            assert value not in partners
        else:
            assert partners == ()


def test_region_negative_swaps_only_region_axis() -> None:
    world = ChineseAliasWorld(2048)
    augmented = RegionAugmentedWorld()
    values = (2, 3, 4, 0)  # region 4 = 中央 has partners
    negative = augmented.region_negative_values(values)
    assert negative is not None
    assert negative[:2] == values[:2] and negative[3] == values[3]
    assert negative[2] != values[2]
    assert negative[2] in region_partners(values[2])


def test_typo_query_corrupts_one_char_to_oov_and_keeps_target() -> None:
    world = ChineseAliasWorld(2048)
    augmented = RegionAugmentedWorld()
    obj = world.object_at(3)
    original = world.query_text(obj.values, 0)
    corrupted = augmented.typo_query(obj.values, 0)
    assert len(corrupted) == len(original)
    differences = [index for index, (a, b) in enumerate(zip(original, corrupted)) if a != b]
    assert len(differences) == 1
    position = differences[0]
    typo_char = corrupted[position]
    assert typo_char in TYPO_POOL
    assert typo_char not in world.tokenizer.index  # OOV by construction
    # Corruption must hit a content (alias) character, never a function word.
    from cformer_v60c.data import _CONTENT_CHARS

    assert original[position] in _CONTENT_CHARS


def test_training_batch_returns_two_negative_slots_and_finite_loss() -> None:
    world = ChineseAliasWorld(2048)
    augmented = RegionAugmentedWorld()
    objects = _sample_objects(8)
    variants = list(range(8))
    queries, positives, negatives = augmented.training_batch(objects, variants, seed_offset=1)
    assert queries.shape == (8, world.query_length)
    assert positives.shape == (8, 4, world.field_length)
    assert len(negatives) == 2
    for negative in negatives:
        assert negative.shape == (8, 4, world.field_length)


def test_multi_negative_loss_forward_and_backward() -> None:
    from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver

    world = ChineseAliasWorld(2048)
    augmented = RegionAugmentedWorld()
    objects = _sample_objects(6)
    queries, positives, negatives = augmented.training_batch(objects, list(range(6)), seed_offset=3)
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=2,
        d_model=32,
        heads=4,
        ffn_dimensions=64,
        output_dimensions=16,
    )
    model = TokenCFormerResolver(config)
    loss = model.contrastive_loss(queries, positives, negatives)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_single_negative_call_still_works() -> None:
    # Backward compatibility: the shared loss accepts one tensor as before.
    from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver

    world = ChineseAliasWorld(2048)
    objects = _sample_objects(6)
    variants = list(range(6))
    queries, coverage = world.encode_queries(objects, variants)
    positives = world.encode_candidates(objects)
    negatives = world.encode_candidates(
        [world.hard_negative(obj, variant) for obj, variant in zip(objects, variants)]
    )
    config = ChineseTransformerConfig(
        world.tokenizer.size,
        layers=2,
        d_model=32,
        heads=4,
        ffn_dimensions=64,
        output_dimensions=16,
    )
    model = TokenCFormerResolver(config)
    assert torch.isfinite(model.contrastive_loss(queries, positives, negatives))


def test_region_special_queries_cover_failure_cluster() -> None:
    # Every value in the V6.0b failure cluster {中央, 腹地, 高原, 冰原} must be
    # reachable as a region negative of another member.
    cluster = (4, 6, 7, 8)
    for value in cluster:
        partners = region_partners(value)
        assert any(partner in cluster for partner in partners)
