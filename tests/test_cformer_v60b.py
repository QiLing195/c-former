import torch

from cformer_v60b import FORBIDDEN_FRAGMENTS, BlindSet


def test_blind_text_has_no_identity_or_template_fragments() -> None:
    blind = BlindSet()
    assert len(blind.queries) == 36
    for query in blind.queries:
        blind.world.assert_identity_free([query.text])
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in query.text, query.text


def test_known_targets_are_holdout_fold_and_exist_at_full_scale() -> None:
    blind = BlindSet()
    assert blind.world.scale == 65536  # every (a,d,r,m) in [0,16)**4 exists
    for query in blind.queries:
        if query.expected in ("known", "disambiguated", "hard"):
            assert query.target is not None
            assert BlindSet._fold(query.target) == 4
            assert all(0 <= value < 16 for value in query.target)


def test_ambiguous_pairs_share_name_axes_only() -> None:
    blind = BlindSet()
    assert len(blind.ambiguous_pairs) == 5
    for first, second in blind.ambiguous_pairs:
        assert first[:2] == second[:2]
        assert first[2:] != second[2:]


def test_known_like_queries_have_sufficient_coverage() -> None:
    # The frozen verifier rejects coverage < 0.6; known/ambiguous/conflict
    # queries must stay above that so rejection is not an artifact of OOV text.
    blind = BlindSet()
    for query in blind.queries:
        if query.expected in ("known", "hard", "ambiguous", "disambiguated", "conflict"):
            assert blind.coverage(query) >= 0.6, query.text


def test_unknown_queries_mix_coverage_and_score_probes() -> None:
    blind = BlindSet()
    low_coverage = [
        query for query in blind.by_expected("unknown") if blind.coverage(query) < 0.6
    ]
    assert len(low_coverage) >= 4  # coverage-based rejection path
    assert any(blind.coverage(query) >= 0.6 for query in blind.by_expected("unknown"))


def test_pattern_generation_is_deterministic_and_distinct() -> None:
    blind_a = BlindSet()
    blind_b = BlindSet()
    assert [q.text for q in blind_a.queries] == [q.text for q in blind_b.queries]
    pattern_texts = {q.text for q in blind_a.by_expected("known")}
    assert len(pattern_texts) == 16  # 12 patterns + 4 bespoke, all distinct


def test_tokenizer_roundtrip_of_blind_text() -> None:
    blind = BlindSet()
    for query in blind.queries[:10]:
        tokens, cov = blind.world.tokenizer.encode(query.text, blind.world.query_length)
        assert tokens.shape == (blind.world.query_length,)
        assert 0.0 <= cov <= 1.0
        assert torch.all(tokens >= 0)
