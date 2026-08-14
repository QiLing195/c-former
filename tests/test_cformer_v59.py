from pathlib import Path

import pytest
import torch

from cformer_v59 import (
    CandidateLedger,
    CandidateStatus,
    DualEncoderConfig,
    EvidenceVerifier,
    OpenAliasWorld,
    SemanticDualEncoder,
)


def test_open_alias_text_has_no_ids_and_family_split_is_disjoint() -> None:
    world = OpenAliasWorld(2048)
    train = world.training_objects(200)
    test = world.heldout_objects(50)
    assert all(world.family_fold(obj.values) != 4 for obj in train)
    assert all(world.family_fold(obj.values) == 4 for obj in test)
    texts = []
    for obj in train[:20] + test[:20]:
        texts.append(world.query_text(obj.values))
        texts.extend(world.candidate_evidence(obj.values))
    world.assert_identity_free(texts)
    assert all("label" not in text and "object_id" not in text for text in texts)


def test_query_encoding_does_not_receive_or_append_identity_key() -> None:
    world = OpenAliasWorld(2048)
    objects = world.heldout_objects(3)
    queries, coverage = world.encode_queries(objects)
    assert queries.shape == (3, world.query_length)
    assert torch.all(coverage == 1)
    changed_labels = [type(obj)(999999, obj.values) for obj in objects]
    same_queries, _ = world.encode_queries(changed_labels)
    assert torch.equal(queries, same_queries)


def test_multi_evidence_dual_encoder_shapes_and_hard_negative_loss() -> None:
    world = OpenAliasWorld(2048)
    objects = world.training_objects(8)
    query, _ = world.encode_queries(objects)
    positive = world.encode_candidates(objects)
    negative = world.encode_candidates([world.hard_negative(obj) for obj in objects])
    model = SemanticDualEncoder(DualEncoderConfig(world.tokenizer.size, d_model=32, embedding_dimensions=24))
    q = model.encode_query(query)
    c = model.encode_candidate(positive)
    loss = model.contrastive_loss(query, positive, negative)
    assert q.shape == c.shape == (8, 24)
    assert torch.isfinite(loss)
    assert model.parameter_count() < 20_000


def test_verifier_separates_supported_ambiguous_and_unknown() -> None:
    verifier = EvidenceVerifier()
    assert verifier.decide(0.8, 0.6, 1.0).status == CandidateStatus.SUPPORTED
    assert verifier.decide(0.8, 0.79, 1.0).status == CandidateStatus.AMBIGUOUS
    assert verifier.decide(0.8, 0.2, 0.1).status == CandidateStatus.UNKNOWN
    assert verifier.decide(0.2, 0.0, 1.0).status == CandidateStatus.UNKNOWN


def test_ambiguous_set_contains_real_live_collisions() -> None:
    world = OpenAliasWorld(2048)
    objects = world.ambiguous_objects(20)
    assert objects
    all_values = [world.object_at(label).values for label in range(world.scale)]
    for obj in objects:
        siblings = [values for values in all_values if values[:3] == obj.values[:3]]
        assert len(siblings) >= 2
        assert len({values[3] for values in siblings}) >= 2


def test_candidate_state_machine_requires_reviewer_and_records_rollback(tmp_path: Path) -> None:
    ledger = CandidateLedger(tmp_path / "candidates.sqlite3")
    candidate = ledger.propose("crimson routing arctic steady", 7)
    ledger.transition(
        candidate,
        CandidateStatus.SUPPORTED,
        actor="model:dual_encoder",
        reason="four_evidence_match",
    )
    with pytest.raises(PermissionError):
        ledger.transition(
            candidate,
            CandidateStatus.VERIFIED,
            actor="model:dual_encoder",
            reason="not_allowed",
        )
    ledger.transition(
        candidate,
        CandidateStatus.VERIFIED,
        actor="reviewer:human",
        reason="evidence_confirmed",
    )
    ledger.transition(
        candidate,
        CandidateStatus.ROLLED_BACK,
        actor="reviewer:human",
        reason="later_conflict",
    )
    assert ledger.status(candidate) == CandidateStatus.ROLLED_BACK
    assert len(ledger.history(candidate)) == 4
    ledger.close()
