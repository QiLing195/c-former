from __future__ import annotations

from dataclasses import dataclass

import torch

from cformer_v3.data import (
    FACT_AT,
    LOCATION_BASE,
    NUM_LOCATIONS,
    PAD,
    PERSON_BASE,
    Q_PERSON_LOCATION,
    SCALE_FACTS,
    ScaleWorldTask,
)


STATUS_ANSWER = 0
STATUS_UNKNOWN = 1
STATUS_CONFLICT = 2
STATUS_OBSERVER_UNKNOWN = 3
STATUS_ACCESS_DENIED = 4
STATUS_NAMES = ("answer", "unknown", "conflict", "observer_unknown", "access_denied")

MODE_GLOBAL = 120
MODE_EPISTEMIC = 121
MODE_AUTHORIZED = 122
Q_UNKNOWN = 123
TOKEN_VOCAB_SIZE = 124


@dataclass(frozen=True)
class ReliabilityCases:
    memory: torch.Tensor
    questions: torch.Tensor
    observers: torch.Tensor
    allowed: torch.Tensor
    answers: torch.Tensor
    statuses: torch.Tensor
    boundary_override: torch.Tensor
    evidence_first: torch.Tensor
    evidence_second: torch.Tensor


class ReliabilityTask:
    """Add unanswerable, conflicting, epistemic and permission boundaries to V3 worlds."""

    question_fields = 4

    def __init__(self) -> None:
        self.base = ScaleWorldTask()

    @staticmethod
    def _first_at_slot(memory_row: torch.Tensor) -> int:
        return memory_row[:, 0].eq(FACT_AT).nonzero(as_tuple=False)[0, 0].item()

    @staticmethod
    def _different_location(token: int, offset: int = 1) -> int:
        value = token - LOCATION_BASE
        return LOCATION_BASE + (value + offset) % NUM_LOCATIONS

    def sample_batch(self, batch_size: int, num_facts: int, device: torch.device | str):
        memory, base_question, observer, answers, _, first, second = self.base.sample_batch(
            batch_size, num_facts, device
        )
        device = memory.device
        questions = torch.zeros(batch_size, self.question_fields, dtype=torch.long, device=device)
        questions[:, :2] = base_question[:, :2]
        questions[:, 2] = MODE_GLOBAL
        allowed = torch.ones(batch_size, num_facts, dtype=torch.bool, device=device)
        statuses = torch.zeros(batch_size, dtype=torch.long, device=device)
        boundary_override = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # Four positive slots and four boundary classes provide a 50:50 prior.
        case_types = torch.randint(8, (batch_size,), device=device)
        for row in range(batch_size):
            case = case_types[row].item()
            if case <= 3:
                questions[row, 2] = (
                    MODE_GLOBAL,
                    MODE_GLOBAL,
                    MODE_EPISTEMIC,
                    MODE_AUTHORIZED,
                )[case]
                continue

            at_slot = self._first_at_slot(memory[row])
            target_person = memory[row, at_slot, 1].item()
            if case == 4:
                statuses[row] = STATUS_UNKNOWN
                questions[row] = torch.tensor(
                    [Q_UNKNOWN, target_person, MODE_GLOBAL, PAD], device=device
                )
                first[row] = second[row] = -1
                answers[row] = 0
            elif case == 5:
                statuses[row] = STATUS_CONFLICT
                questions[row] = torch.tensor(
                    [Q_PERSON_LOCATION, target_person, MODE_GLOBAL, PAD], device=device
                )
                original_location = memory[row, at_slot, 2].item()
                slot_a, slot_b = num_facts - 2, num_facts - 1
                memory[row, slot_a] = torch.tensor(
                    [FACT_AT, target_person, self._different_location(original_location, 1), PAD],
                    device=device,
                )
                memory[row, slot_b] = torch.tensor(
                    [FACT_AT, target_person, self._different_location(original_location, 2), PAD],
                    device=device,
                )
                first[row], second[row] = slot_a, slot_b
                answers[row] = 0
            elif case == 6:
                statuses[row] = STATUS_OBSERVER_UNKNOWN
                boundary_override[row] = STATUS_OBSERVER_UNKNOWN
                questions[row] = torch.tensor(
                    [Q_PERSON_LOCATION, target_person, MODE_EPISTEMIC, PAD], device=device
                )
                allowed[row, at_slot] = False
                first[row] = second[row] = -1
                answers[row] = 0
            else:
                statuses[row] = STATUS_ACCESS_DENIED
                boundary_override[row] = STATUS_ACCESS_DENIED
                questions[row] = torch.tensor(
                    [Q_PERSON_LOCATION, target_person, MODE_AUTHORIZED, PAD], device=device
                )
                allowed[row, at_slot] = False
                first[row] = second[row] = -1
                answers[row] = 0
        return (
            memory,
            questions,
            observer,
            allowed,
            answers,
            statuses,
            boundary_override,
            first,
            second,
        )

    def fixed_cases(self, num_facts: int, world_index: int) -> ReliabilityCases:
        world = self.base.fixed_world(num_facts, world_index)
        base_count = world.questions.shape[0]
        boundary_count = 8
        total = base_count + 4 * boundary_count
        memory = world.memory[None].repeat(total, 1, 1)
        questions = torch.zeros(total, self.question_fields, dtype=torch.long)
        questions[:base_count, :2] = world.questions[:, :2]
        questions[:base_count, 2] = MODE_GLOBAL
        # Positive examples in non-global modes prevent shortcutting on the mode token.
        questions[:6, 2] = MODE_EPISTEMIC
        questions[6:12, 2] = MODE_AUTHORIZED
        observers = torch.cat(
            (world.observers, world.observers[:boundary_count].repeat(4)), dim=0
        )
        allowed = torch.ones(total, num_facts, dtype=torch.bool)
        answers = torch.zeros(total, dtype=torch.long)
        answers[:base_count] = world.labels
        statuses = torch.zeros(total, dtype=torch.long)
        boundary_override = torch.full((total,), -1, dtype=torch.long)
        first = torch.full((total,), -1, dtype=torch.long)
        second = torch.full((total,), -1, dtype=torch.long)
        first[:base_count] = world.evidence_first
        second[:base_count] = world.evidence_second

        cursor = base_count
        # UNKNOWN
        for index in range(boundary_count):
            target_slot = self._first_at_slot(memory[cursor + index])
            target = memory[cursor + index, target_slot, 1].item()
            questions[cursor + index] = torch.tensor([Q_UNKNOWN, target, MODE_GLOBAL, PAD])
            statuses[cursor + index] = STATUS_UNKNOWN
        cursor += boundary_count

        # CONFLICT: two explicit contradictory facts for the same subject.
        for index in range(boundary_count):
            row = cursor + index
            target_slot = self._first_at_slot(memory[row])
            target = memory[row, target_slot, 1].item()
            location = memory[row, target_slot, 2].item()
            slot_a, slot_b = num_facts - 2, num_facts - 1
            memory[row, slot_a] = torch.tensor(
                [FACT_AT, target, self._different_location(location, 1), PAD]
            )
            memory[row, slot_b] = torch.tensor(
                [FACT_AT, target, self._different_location(location, 2), PAD]
            )
            questions[row] = torch.tensor([Q_PERSON_LOCATION, target, MODE_GLOBAL, PAD])
            statuses[row] = STATUS_CONFLICT
            first[row], second[row] = slot_a, slot_b
        cursor += boundary_count

        # OBSERVER_UNKNOWN
        for index in range(boundary_count):
            row = cursor + index
            target_slot = self._first_at_slot(memory[row])
            target = memory[row, target_slot, 1].item()
            questions[row] = torch.tensor([Q_PERSON_LOCATION, target, MODE_EPISTEMIC, PAD])
            statuses[row] = STATUS_OBSERVER_UNKNOWN
            boundary_override[row] = STATUS_OBSERVER_UNKNOWN
            allowed[row, target_slot] = False
        cursor += boundary_count

        # ACCESS_DENIED
        for index in range(boundary_count):
            row = cursor + index
            target_slot = self._first_at_slot(memory[row])
            target = memory[row, target_slot, 1].item()
            questions[row] = torch.tensor([Q_PERSON_LOCATION, target, MODE_AUTHORIZED, PAD])
            statuses[row] = STATUS_ACCESS_DENIED
            boundary_override[row] = STATUS_ACCESS_DENIED
            allowed[row, target_slot] = False

        return ReliabilityCases(
            memory,
            questions,
            observers,
            allowed,
            answers,
            statuses,
            boundary_override,
            first,
            second,
        )
