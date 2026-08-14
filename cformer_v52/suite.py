from __future__ import annotations

from dataclasses import dataclass

import torch

from cformer_v3.data import (
    FACT_AT,
    FACT_BELIEF,
    FACT_HAS,
    NUM_OBJECTS,
    NUM_PEOPLE,
    OBJECT_BASE,
    PERSON_BASE,
    ScaleWorldTask,
)


BASE_WORLD_FACTS = 2048
STRESS_SCALES = (2048, 8192, 32768)


@dataclass(frozen=True)
class StressWorld:
    name: str
    memory: torch.Tensor
    questions: torch.Tensor
    observers: torch.Tensor
    labels: torch.Tensor
    tasks: torch.Tensor
    effective_facts: int
    distractor_facts: int


class V52StressSuite:
    """Novel fixed worlds enlarged with deterministic exact and near-key distractors."""

    def __init__(self, world_offset: int = 50) -> None:
        self.base = ScaleWorldTask()
        self.world_offset = world_offset

    @staticmethod
    def _protected_keys(world) -> tuple[set[int], set[int], set[tuple[int, int]]]:
        protected_people: set[int] = set()
        protected_objects: set[int] = set()
        protected_beliefs: set[tuple[int, int]] = set()
        for question, observer in zip(world.questions.tolist(), world.observers.tolist()):
            question_type, target = question[:2]
            if question_type == 5:  # Q_BELIEF_OBJECT_LOCATION
                protected_beliefs.add((observer, target))
            elif question_type in (4, 6):  # true object location / object holder
                protected_objects.add(target)
            elif question_type == 7:  # person location
                protected_people.add(target)

        # A true-location query also depends on the holder's person-location fact.
        for row in world.memory.tolist():
            if row[0] == FACT_HAS and row[2] in protected_objects:
                protected_people.add(row[1])
        return protected_people, protected_objects, protected_beliefs

    @staticmethod
    def _first_unprotected(start: int, count: int, protected: set[int]) -> int:
        for offset in range(count):
            candidate = start + offset
            if candidate not in protected:
                return candidate
        raise ValueError("all candidate keys are protected")

    def _distractors(self, world, count: int, generator: torch.Generator) -> torch.Tensor:
        memory = world.memory
        selected = torch.randint(memory.shape[0], (count,), generator=generator)
        distractors = memory[selected].clone()
        # Half are exact duplicates (benign evidence multiplicity); half are near-key
        # distractors that share relation/value fields but cannot alter a queried key.
        mutate = torch.arange(count).remainder(2).eq(1)
        protected_people, protected_objects, protected_beliefs = self._protected_keys(world)
        safe_person = self._first_unprotected(PERSON_BASE, NUM_PEOPLE, protected_people)
        safe_object = self._first_unprotected(OBJECT_BASE, NUM_OBJECTS, protected_objects)

        at_rows = mutate & distractors[:, 0].eq(FACT_AT)
        has_rows = mutate & distractors[:, 0].eq(FACT_HAS)
        belief_rows = mutate & distractors[:, 0].eq(FACT_BELIEF)
        distractors[at_rows, 1] = safe_person
        distractors[has_rows, 2] = safe_object

        belief_indices = belief_rows.nonzero(as_tuple=False).flatten().tolist()
        for index in belief_indices:
            observer = distractors[index, 1].item()
            obj = distractors[index, 2].item()
            candidate = PERSON_BASE + (observer - PERSON_BASE + 1) % NUM_PEOPLE
            for _ in range(NUM_PEOPLE):
                if (candidate, obj) not in protected_beliefs:
                    break
                candidate = PERSON_BASE + (candidate - PERSON_BASE + 1) % NUM_PEOPLE
            distractors[index, 1] = candidate
        return distractors

    def world(self, scale: int, world_index: int) -> StressWorld:
        if scale not in STRESS_SCALES:
            raise ValueError(f"unsupported V5.2 stress scale {scale}")
        source_index = self.world_offset + world_index
        base = self.base.fixed_world(BASE_WORLD_FACTS, source_index)
        if scale == BASE_WORLD_FACTS:
            memory = base.memory.clone()
        else:
            generator = torch.Generator().manual_seed(52_000_000 + scale * 100 + source_index)
            extras = self._distractors(base, scale - BASE_WORLD_FACTS, generator)
            memory = torch.cat((base.memory, extras), dim=0)
            permutation = torch.randperm(scale, generator=generator)
            memory = memory[permutation]
        return StressWorld(
            name=f"V52-S{scale}-W{world_index + 1}",
            memory=memory,
            questions=base.questions.clone(),
            observers=base.observers.clone(),
            labels=base.labels.clone(),
            tasks=base.tasks.clone(),
            effective_facts=BASE_WORLD_FACTS,
            distractor_facts=scale - BASE_WORLD_FACTS,
        )
