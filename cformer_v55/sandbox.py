from __future__ import annotations

from dataclasses import replace

from .engine import CognitiveEngine
from .schema import (
    CognitiveStatus,
    Hypothesis,
    HypothesisStatus,
    ObjectState,
    ObserverContext,
    Value,
)


class HypothesisSandbox:
    """Simulation quarantine: hypotheses cannot mutate memory before verification."""

    def __init__(self, engine: CognitiveEngine) -> None:
        self.engine = engine
        self._hypotheses: dict[int, Hypothesis] = {}
        self._next_id = 1

    def get(self, hypothesis_id: int) -> Hypothesis:
        return self._hypotheses[hypothesis_id]

    def propose(
        self,
        name: str,
        operator: str,
        observer: ObserverContext,
        context: dict[str, Value] | None = None,
    ) -> Hypothesis | CognitiveStatus:
        result = self.engine.simulate(name, operator, observer, context)
        if result.status != CognitiveStatus.ANSWER:
            return result.status
        hypothesis = Hypothesis(
            self._next_id,
            result.transformation.transformation_id,  # type: ignore[union-attr]
            result.source_object_id,  # type: ignore[arg-type]
            result.target_object_id,  # type: ignore[arg-type]
            result.predicted_properties or (),
            result.world_version,
            observer.query_time,
            result.evidence_ids,
        )
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        self._next_id += 1
        return hypothesis

    def verify(self, hypothesis_id: int, observed: ObjectState) -> Hypothesis:
        hypothesis = self.get(hypothesis_id)
        if hypothesis.status != HypothesisStatus.PROPOSED:
            raise ValueError("only proposed hypotheses can be verified")
        actual = dict(observed.properties)
        matched = observed.object_id == hypothesis.target_object_id and all(
            actual.get(field) == expected
            for field, expected in hypothesis.predicted_properties
        )
        updated = replace(
            hypothesis,
            status=HypothesisStatus.VERIFIED if matched else HypothesisStatus.REJECTED,
            reason="observation_matches_prediction" if matched else "observation_rejects_prediction",
        )
        self._hypotheses[hypothesis_id] = updated
        return updated

    def commit(self, hypothesis_id: int, *, event_time: int) -> ObjectState:
        hypothesis = self.get(hypothesis_id)
        if hypothesis.status != HypothesisStatus.VERIFIED:
            raise PermissionError("unverified hypothesis cannot enter canonical memory")
        memory = self.engine.memory
        if memory.version != hypothesis.base_world_version:
            self._hypotheses[hypothesis_id] = replace(
                hypothesis,
                status=HypothesisStatus.BLOCKED,
                reason="world_version_changed_before_commit",
            )
            raise RuntimeError("world version changed before hypothesis commit")
        previous = memory.state_candidates(
            hypothesis.target_object_id, hypothesis.base_world_version
        )
        active_previous = [
            state
            for state in previous
            if state.valid_from <= event_time < state.valid_to
        ]
        supersedes = max(active_previous, key=lambda state: state.version).state_id if active_previous else None
        state = ObjectState(
            state_id=memory.next_state_id(),
            object_id=hypothesis.target_object_id,
            properties=hypothesis.predicted_properties,
            event_time=event_time,
            ingest_time=event_time,
            valid_from=event_time,
            version=1 + max((item.version for item in previous), default=0),
            supersedes=supersedes,
            coordinate_frame=(
                0 if dict(hypothesis.predicted_properties).get("position") is not None else None
            ),
        )
        memory.add_states([state])
        self._hypotheses[hypothesis_id] = replace(
            hypothesis,
            status=HypothesisStatus.COMMITTED,
            reason="verified_hypothesis_committed_as_new_version",
        )
        return state
