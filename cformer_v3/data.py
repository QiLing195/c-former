from __future__ import annotations

from dataclasses import dataclass

import torch


PAD = 0
FACT_AT = 1
FACT_HAS = 2
FACT_BELIEF = 3
Q_TRUE_OBJECT_LOCATION = 4
Q_BELIEF_OBJECT_LOCATION = 5
Q_OBJECT_HOLDER = 6
Q_PERSON_LOCATION = 7

NUM_PEOPLE = 64
NUM_OBJECTS = 32
NUM_LOCATIONS = 16
PERSON_BASE = 8
OBJECT_BASE = PERSON_BASE + NUM_PEOPLE
LOCATION_BASE = OBJECT_BASE + NUM_OBJECTS
TOKEN_VOCAB_SIZE = LOCATION_BASE + NUM_LOCATIONS
ANSWER_CLASSES = NUM_PEOPLE + NUM_LOCATIONS
BASE_FACTS = NUM_PEOPLE + NUM_OBJECTS
MAX_BELIEF_FACTS = NUM_PEOPLE * NUM_OBJECTS
SCALE_FACTS = (128, 512, 2048)
TASK_NAMES = ("true_location_2hop", "observer_belief", "object_holder", "outside_person_location")


@dataclass(frozen=True)
class FixedScaleWorld:
    name: str
    num_facts: int
    memory: torch.Tensor
    questions: torch.Tensor
    observers: torch.Tensor
    labels: torch.Tensor
    tasks: torch.Tensor
    evidence_first: torch.Tensor
    evidence_second: torch.Tensor


class ScaleWorldTask:
    fact_fields = 4
    question_fields = 3
    answer_classes = ANSWER_CLASSES

    def _make_canonical_memory(
        self,
        person_locations: torch.Tensor,
        object_holders: torch.Tensor,
        belief_pairs: torch.Tensor,
        belief_locations: torch.Tensor,
    ) -> torch.Tensor:
        batch = person_locations.shape[0]
        belief_count = belief_pairs.shape[0]
        device = person_locations.device
        memory = torch.zeros(
            batch, BASE_FACTS + belief_count, self.fact_fields, dtype=torch.long, device=device
        )
        people = torch.arange(NUM_PEOPLE, device=device)
        objects = torch.arange(NUM_OBJECTS, device=device)

        memory[:, :NUM_PEOPLE, 0] = FACT_AT
        memory[:, :NUM_PEOPLE, 1] = PERSON_BASE + people
        memory[:, :NUM_PEOPLE, 2] = LOCATION_BASE + person_locations

        memory[:, NUM_PEOPLE:BASE_FACTS, 0] = FACT_HAS
        memory[:, NUM_PEOPLE:BASE_FACTS, 1] = PERSON_BASE + object_holders
        memory[:, NUM_PEOPLE:BASE_FACTS, 2] = OBJECT_BASE + objects

        memory[:, BASE_FACTS:, 0] = FACT_BELIEF
        memory[:, BASE_FACTS:, 1] = PERSON_BASE + belief_pairs[:, 0]
        memory[:, BASE_FACTS:, 2] = OBJECT_BASE + belief_pairs[:, 1]
        memory[:, BASE_FACTS:, 3] = LOCATION_BASE + belief_locations
        return memory

    @staticmethod
    def _all_belief_pairs(device: torch.device) -> torch.Tensor:
        people = torch.arange(NUM_PEOPLE, device=device)
        objects = torch.arange(NUM_OBJECTS, device=device)
        return torch.cartesian_prod(people, objects)

    def sample_batch(
        self, batch_size: int, num_facts: int, device: torch.device | str
    ) -> tuple[torch.Tensor, ...]:
        if num_facts not in SCALE_FACTS:
            raise ValueError(f"unsupported scale {num_facts}")
        device = torch.device(device)
        belief_count = num_facts - BASE_FACTS
        pair_order = torch.randperm(MAX_BELIEF_FACTS, device=device)[:belief_count]
        belief_pairs = self._all_belief_pairs(device)[pair_order]

        person_locations = torch.randint(NUM_LOCATIONS, (batch_size, NUM_PEOPLE), device=device)
        object_holders = torch.randint(NUM_PEOPLE, (batch_size, NUM_OBJECTS), device=device)
        belief_locations = torch.randint(NUM_LOCATIONS, (batch_size, belief_count), device=device)
        canonical = self._make_canonical_memory(
            person_locations, object_holders, belief_pairs, belief_locations
        )

        tasks = torch.randint(4, (batch_size,), device=device)
        random_observers = torch.randint(NUM_PEOPLE, (batch_size,), device=device)
        random_objects = torch.randint(NUM_OBJECTS, (batch_size,), device=device)
        belief_slots = torch.randint(belief_count, (batch_size,), device=device)
        belief_observers = belief_pairs[belief_slots, 0]
        belief_objects = belief_pairs[belief_slots, 1]
        observers = torch.where(tasks.eq(1), belief_observers, random_observers)
        objects = torch.where(tasks.eq(1), belief_objects, random_objects)
        targets = (observers + 1 + torch.randint(NUM_PEOPLE - 1, (batch_size,), device=device)) % NUM_PEOPLE

        question = torch.zeros(batch_size, self.question_fields, dtype=torch.long, device=device)
        question[:, 0] = Q_TRUE_OBJECT_LOCATION + tasks
        question[:, 1] = torch.where(tasks.eq(3), PERSON_BASE + targets, OBJECT_BASE + objects)
        rows = torch.arange(batch_size, device=device)
        holders = object_holders[rows, objects]
        true_locations = NUM_PEOPLE + person_locations[rows, holders]
        belief_answers = NUM_PEOPLE + belief_locations[rows, belief_slots]
        holder_answers = holders
        person_answers = NUM_PEOPLE + person_locations[rows, targets]
        labels = torch.stack(
            (true_locations, belief_answers, holder_answers, person_answers), dim=1
        )[rows, tasks]

        first_evidence = torch.where(
            tasks.eq(0) | tasks.eq(2),
            NUM_PEOPLE + objects,
            torch.where(tasks.eq(1), BASE_FACTS + belief_slots, targets),
        )
        second_evidence = torch.where(tasks.eq(0), holders, first_evidence)

        permutation = torch.randperm(num_facts, device=device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(num_facts, device=device)
        memory = canonical[:, permutation]
        first_evidence = inverse[first_evidence]
        second_evidence = inverse[second_evidence]
        return (
            memory,
            question,
            PERSON_BASE + observers,
            labels,
            tasks,
            first_evidence,
            second_evidence,
        )

    def fixed_world(self, num_facts: int, world_index: int) -> FixedScaleWorld:
        generator = torch.Generator().manual_seed(10_000 * num_facts + world_index)
        belief_count = num_facts - BASE_FACTS
        pair_order = torch.randperm(MAX_BELIEF_FACTS, generator=generator)[:belief_count]
        belief_pairs = self._all_belief_pairs(torch.device("cpu"))[pair_order]
        locations = torch.randint(NUM_LOCATIONS, (1, NUM_PEOPLE), generator=generator)
        holders = torch.randint(NUM_PEOPLE, (1, NUM_OBJECTS), generator=generator)
        belief_locations = torch.randint(NUM_LOCATIONS, (1, belief_count), generator=generator)
        canonical = self._make_canonical_memory(locations, holders, belief_pairs, belief_locations)

        core_pair_slots = torch.linspace(0, belief_count - 1, 8).long()
        core_pairs = belief_pairs[core_pair_slots]
        questions: list[list[int]] = []
        observers: list[int] = []
        labels: list[int] = []
        tasks: list[int] = []
        first: list[int] = []
        second: list[int] = []

        # Eight observer-sensitive questions, using pairs spread across the memory.
        for slot, pair in zip(core_pair_slots.tolist(), core_pairs.tolist()):
            observer, obj = pair
            questions.append([Q_BELIEF_OBJECT_LOCATION, OBJECT_BASE + obj, PAD])
            observers.append(PERSON_BASE + observer)
            labels.append(NUM_PEOPLE + belief_locations[0, slot].item())
            tasks.append(1)
            first.append(BASE_FACTS + slot)
            second.append(BASE_FACTS + slot)

        # Eight questions for each objective object task.
        query_objects = torch.randperm(NUM_OBJECTS, generator=generator)[:8]
        query_observers = torch.randperm(NUM_PEOPLE, generator=generator)[:8]
        for obj, observer in zip(query_objects.tolist(), query_observers.tolist()):
            holder = holders[0, obj].item()
            questions.append([Q_TRUE_OBJECT_LOCATION, OBJECT_BASE + obj, PAD])
            observers.append(PERSON_BASE + observer)
            labels.append(NUM_PEOPLE + locations[0, holder].item())
            tasks.append(0)
            first.append(NUM_PEOPLE + obj)
            second.append(holder)

            questions.append([Q_OBJECT_HOLDER, OBJECT_BASE + obj, PAD])
            observers.append(PERSON_BASE + observer)
            labels.append(holder)
            tasks.append(2)
            first.append(NUM_PEOPLE + obj)
            second.append(NUM_PEOPLE + obj)

        # Twelve questions that explicitly target someone other than the observer.
        for index in range(12):
            observer = query_observers[index % 8].item()
            target = (observer + 1 + index) % NUM_PEOPLE
            questions.append([Q_PERSON_LOCATION, PERSON_BASE + target, PAD])
            observers.append(PERSON_BASE + observer)
            labels.append(NUM_PEOPLE + locations[0, target].item())
            tasks.append(3)
            first.append(target)
            second.append(target)

        permutation = torch.randperm(num_facts, generator=generator)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(num_facts)
        return FixedScaleWorld(
            name=f"S{num_facts}-W{world_index + 1}",
            num_facts=num_facts,
            memory=canonical[0, permutation],
            questions=torch.tensor(questions),
            observers=torch.tensor(observers),
            labels=torch.tensor(labels),
            tasks=torch.tensor(tasks),
            evidence_first=inverse[torch.tensor(first)],
            evidence_second=inverse[torch.tensor(second)],
        )

