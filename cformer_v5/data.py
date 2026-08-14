from __future__ import annotations

from dataclasses import dataclass

import torch

from cformer_v3.data import (
    FACT_AT,
    LOCATION_BASE,
    NUM_LOCATIONS,
    NUM_PEOPLE,
    PAD,
    PERSON_BASE,
    Q_PERSON_LOCATION,
    ScaleWorldTask,
)
from cformer_v4.data import (
    MODE_AUTHORIZED,
    MODE_EPISTEMIC,
    MODE_GLOBAL,
    Q_UNKNOWN,
    STATUS_ACCESS_DENIED,
    STATUS_ANSWER,
    STATUS_CONFLICT,
    STATUS_OBSERVER_UNKNOWN,
    STATUS_UNKNOWN,
)

from .conflict import FactMetadata


@dataclass
class V5Case:
    category: str
    memory: torch.Tensor
    metadata: FactMetadata
    question: torch.Tensor
    observer: int
    allowed: torch.Tensor
    expected_status: int
    expected_answer: int
    query_time: int
    use_conflict_controller: bool
    forced_status: int = -1


class V5WorldSuite:
    """Create 120 reliability cases from one fixed structured world."""

    def __init__(self) -> None:
        self.base = ScaleWorldTask()

    @staticmethod
    def _at_slots(memory: torch.Tensor) -> torch.Tensor:
        return memory[:, 0].eq(FACT_AT).nonzero(as_tuple=False).flatten()

    @staticmethod
    def _different_location(location_token: int, offset: int) -> int:
        return LOCATION_BASE + (location_token - LOCATION_BASE + offset) % NUM_LOCATIONS

    def _base_case(
        self,
        memory: torch.Tensor,
        question: torch.Tensor,
        observer: int,
        status: int,
        answer: int,
        category: str,
    ) -> V5Case:
        return V5Case(
            category,
            memory.clone(),
            FactMetadata.defaults(memory.shape[0]),
            question.clone(),
            observer,
            torch.ones(memory.shape[0], dtype=torch.bool),
            status,
            answer,
            75,
            False,
        )

    def _temporal_case(
        self,
        base_memory: torch.Tensor,
        at_slot: int,
        category: str,
        kind: str,
        index: int,
    ) -> V5Case:
        memory = base_memory.clone()
        metadata = FactMetadata.defaults(memory.shape[0])
        target = memory[at_slot, 1].item()
        old_location = memory[at_slot, 2].item()
        extra_slot = memory.shape[0] - 1 - (index % 8)
        if extra_slot == at_slot:
            extra_slot = memory.shape[0] - 10
        new_location = self._different_location(old_location, 1 + index % 3)
        memory[extra_slot] = torch.tensor([FACT_AT, target, new_location, 0])
        metadata.source[at_slot] = 1
        metadata.source[extra_slot] = 2
        expected_status = STATUS_ANSWER
        expected_answer = 0

        if kind == "conflict":
            expected_status = STATUS_CONFLICT
        elif kind == "time_change":
            metadata.valid_to[at_slot] = 49
            metadata.valid_from[extra_slot] = 50
            expected_answer = NUM_PEOPLE + new_location - LOCATION_BASE
        elif kind == "version_update":
            metadata.version[extra_slot] = 2
            metadata.supersedes[extra_slot] = at_slot
            expected_answer = NUM_PEOPLE + new_location - LOCATION_BASE
        elif kind == "low_confidence":
            metadata.confidence[extra_slot] = 0.2
            expected_answer = NUM_PEOPLE + old_location - LOCATION_BASE
        else:
            raise ValueError(kind)

        return V5Case(
            category=category,
            memory=memory,
            metadata=metadata,
            question=torch.tensor([Q_PERSON_LOCATION, target, MODE_GLOBAL, 0]),
            observer=PERSON_BASE,
            allowed=torch.ones(memory.shape[0], dtype=torch.bool),
            expected_status=expected_status,
            expected_answer=expected_answer,
            query_time=75,
            use_conflict_controller=True,
        )

    def cases(self, num_facts: int, world_index: int) -> list[V5Case]:
        world = self.base.fixed_world(num_facts, world_index)
        cases: list[V5Case] = []

        # 30 ordinary neural questions.
        for index in range(30):
            q = torch.tensor(
                [world.questions[index, 0], world.questions[index, 1], MODE_GLOBAL, 0]
            )
            cases.append(
                self._base_case(
                    world.memory,
                    q,
                    world.observers[index].item(),
                    STATUS_ANSWER,
                    world.labels[index].item(),
                    "normal_answer",
                )
            )

        at_slots = self._at_slots(world.memory)
        for index in range(20):
            cases.append(
                self._temporal_case(
                    world.memory, at_slots[index % len(at_slots)].item(), "true_conflict", "conflict", index
                )
            )
        for index in range(15):
            cases.append(
                self._temporal_case(
                    world.memory, at_slots[(index + 7) % len(at_slots)].item(), "time_change", "time_change", index
                )
            )
        for index in range(10):
            cases.append(
                self._temporal_case(
                    world.memory, at_slots[(index + 13) % len(at_slots)].item(), "version_update", "version_update", index
                )
            )
        for index in range(10):
            cases.append(
                self._temporal_case(
                    world.memory, at_slots[(index + 19) % len(at_slots)].item(), "low_confidence_source", "low_confidence", index
                )
            )

        # Ten missing facts plus five ambiguous questions use the neural UNKNOWN path.
        for index in range(15):
            target = world.memory[at_slots[index % len(at_slots)], 1].item()
            category = "unknown" if index < 10 else "ambiguous"
            cases.append(
                self._base_case(
                    world.memory,
                    torch.tensor([Q_UNKNOWN, target, MODE_GLOBAL, 0]),
                    PERSON_BASE,
                    STATUS_UNKNOWN,
                    0,
                    category,
                )
            )

        # Trusted policy engine, not the neural model, owns these two boundaries.
        for index in range(10):
            target_slot = at_slots[(index + 23) % len(at_slots)].item()
            target = world.memory[target_slot, 1].item()
            case = self._base_case(
                world.memory,
                torch.tensor([Q_PERSON_LOCATION, target, MODE_EPISTEMIC, 0]),
                PERSON_BASE,
                STATUS_OBSERVER_UNKNOWN,
                0,
                "observer_unknown",
            )
            case.allowed[target_slot] = False
            case.forced_status = STATUS_OBSERVER_UNKNOWN
            cases.append(case)
        for index in range(10):
            target_slot = at_slots[(index + 31) % len(at_slots)].item()
            target = world.memory[target_slot, 1].item()
            case = self._base_case(
                world.memory,
                torch.tensor([Q_PERSON_LOCATION, target, MODE_AUTHORIZED, 0]),
                PERSON_BASE,
                STATUS_ACCESS_DENIED,
                0,
                "access_denied",
            )
            case.allowed[target_slot] = False
            case.forced_status = STATUS_ACCESS_DENIED
            cases.append(case)

        assert len(cases) == 120
        return cases
