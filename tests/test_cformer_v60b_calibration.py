from cformer_v59 import (
    RECOMMENDED_MARGINS_V60B,
    CandidateStatus,
    EvidenceVerifier,
)
from cformer_v60b import BlindSet, CalibrationSet


def _fold(values):
    return (values[0] * 3 + values[1] * 5 + values[2] * 7 + values[3] * 11) % 5


def test_calibration_set_is_disjoint_from_blind_targets() -> None:
    calibration = CalibrationSet()
    blind = BlindSet()
    blind_targets = {query.target for query in blind.queries if query.target is not None}
    assert calibration.blind_targets == blind_targets
    for query in calibration.queries:
        if query.target is not None:
            assert query.target not in blind_targets


def test_calibration_set_composition() -> None:
    calibration = CalibrationSet()
    counts = {"known": 0, "typo": 0, "ambiguous": 0, "unknown": 0}
    for query in calibration.queries:
        counts[query.expected] += 1
    assert counts["known"] == 240
    assert counts["typo"] == 24
    assert counts["ambiguous"] == 5
    assert counts["unknown"] >= 6


def test_calibration_known_targets_are_fold4_and_text_is_clean() -> None:
    calibration = CalibrationSet()
    for query in calibration.queries:
        calibration.world.assert_identity_free([query.text])
        if query.expected == "known":
            assert _fold(query.target) == 4


def test_calibrated_verifier_uses_per_type_margin() -> None:
    verifier = EvidenceVerifier(margin_by_type=RECOMMENDED_MARGINS_V60B)
    # margin 0.04: known passes its 0.03 threshold, ambiguous keeps 0.08.
    known = verifier.decide(0.90, 0.86, 1.0, query_type="known")
    ambiguous = verifier.decide(0.90, 0.86, 1.0, query_type="ambiguous")
    assert known.status == CandidateStatus.SUPPORTED
    assert ambiguous.status == CandidateStatus.AMBIGUOUS


def test_verifier_without_query_type_keeps_frozen_behavior() -> None:
    verifier = EvidenceVerifier(margin_by_type=RECOMMENDED_MARGINS_V60B)
    decision = verifier.decide(0.90, 0.86, 1.0)  # no query_type
    assert decision.status == CandidateStatus.AMBIGUOUS  # frozen 0.08 margin


def test_recommended_margins_keep_safety_types_conservative() -> None:
    assert set(RECOMMENDED_MARGINS_V60B) == {"known", "hard", "disambiguated", "ambiguous", "unknown", "conflict"}
    assert RECOMMENDED_MARGINS_V60B["known"] < RECOMMENDED_MARGINS_V60B["ambiguous"]
    assert RECOMMENDED_MARGINS_V60B["unknown"] == 0.08
    assert RECOMMENDED_MARGINS_V60B["conflict"] == 0.08
