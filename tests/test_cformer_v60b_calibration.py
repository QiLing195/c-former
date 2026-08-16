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
