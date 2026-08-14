from pathlib import Path

import numpy as np

from cformer_v58 import (
    AliasCandidateResolver,
    AliasResolutionStatus,
    LayeredAliasStore,
    QuantizedVectorStore,
)


def _records():
    return [
        {
            "object_id": 1,
            "canonical_name": "North Navigator",
            "document": "crimson navigation instrument northern region route planning",
            "aliases": ("北方导航器", "Navigator One"),
        },
        {
            "object_id": 2,
            "canonical_name": "South Medic",
            "document": "crimson medical instrument southern region clinical support",
            "aliases": ("南方医疗器",),
        },
    ]


def test_quantized_vectors_are_single_copy_and_mmap_readable(tmp_path: Path) -> None:
    store = QuantizedVectorStore(tmp_path, dimensions=4)
    vectors = np.asarray([[1.0, 0.5, 0.0, -0.5], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    store.append(vectors)
    assert store.count == 2
    scores = store.score([0, 1], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert scores[0] > scores[1]
    assert (tmp_path / "objects.i8").stat().st_size == 8


def test_exact_alias_and_perspectives_do_not_duplicate_vectors(tmp_path: Path) -> None:
    store = LayeredAliasStore(tmp_path)
    store.add_objects(_records())
    assert store.object_count == 2
    assert store.perspective_count == 8
    assert store.vectors.count == 2
    assert store.lookup_alias("  北方导航器 ") == [1]
    store.close()
    reopened = LayeredAliasStore(tmp_path)
    assert reopened.lookup_alias("Navigator One") == [1]
    assert reopened.vectors.count == 2
    reopened.close()


def test_unknown_expression_is_proposed_but_not_auto_committed(tmp_path: Path) -> None:
    store = LayeredAliasStore(tmp_path)
    store.add_objects(_records())
    resolver = AliasCandidateResolver(store)
    result = resolver.resolve("northern crimson navigation route instrument", source="test")
    assert result.status == AliasResolutionStatus.PROPOSED
    assert result.object_id == 1
    assert store.lookup_alias("northern crimson navigation route instrument") == []
    store.review_candidate(result.candidate_record_id, approve=True)
    verified = resolver.resolve("northern crimson navigation route instrument")
    assert verified.status == AliasResolutionStatus.VERIFIED
    assert verified.object_id == 1


def test_alias_ambiguity_and_versioned_retraction(tmp_path: Path) -> None:
    records = _records()
    records[0]["aliases"] += ("shared name",)
    records[1]["aliases"] += ("shared name",)
    store = LayeredAliasStore(tmp_path)
    store.add_objects(records)
    resolver = AliasCandidateResolver(store)
    ambiguous = resolver.resolve("shared name")
    assert ambiguous.status == AliasResolutionStatus.AMBIGUOUS
    snapshot = store.version
    store.retract_alias("shared name", 2)
    assert store.lookup_alias("shared name") == [1]
    assert sorted(store.lookup_alias("shared name", snapshot)) == [1, 2]


def test_sparse_search_returns_supported_object_without_full_load(tmp_path: Path) -> None:
    store = LayeredAliasStore(tmp_path)
    store.add_objects(_records())
    hits = store.search("northern navigation route planning", limit=2)
    assert hits
    assert hits[0].object_id == 1
    assert store.disk_bytes() > 0
