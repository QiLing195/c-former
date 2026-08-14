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
PERSON_BASE = 8
OBJECT_BASE = 12
LOCATION_BASE = 14
TOKEN_VOCAB_SIZE = 18

TASK_NAMES = ("true_location_2hop", "observer_belief", "object_holder", "outside_person_location")


@dataclass(frozen=True)
class FixedWorld:
    name: str
    person_locations: tuple[int, int, int, int]
    object_holders: tuple[int, int]
    beliefs: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


def fixed_worlds() -> list[FixedWorld]:
    return [
        FixedWorld("Orion", (0, 1, 2, 3), (1, 2), ((0, 2), (1, 1), (3, 2), (0, 3))),
        FixedWorld("Lyra", (2, 0, 3, 1), (3, 0), ((2, 1), (0, 3), (1, 0), (3, 2))),
        FixedWorld("Cygnus", (1, 3, 0, 2), (2, 1), ((3, 1), (2, 3), (0, 0), (1, 2))),
        FixedWorld("Draco", (3, 2, 1, 0), (0, 3), ((3, 0), (1, 2), (2, 1), (0, 3))),
    ]


class WorldTask:
    """Structured worlds where all objective and observer-relative facts coexist."""

    num_people = 4
    num_objects = 2
    num_locations = 4
    num_facts = 14
    fact_fields = 4
    question_fields = 3
    answer_classes = 8  # people 0..3 and locations 4..7

    def _facts_from_state(
        self,
        person_locations: torch.Tensor,
        object_holders: torch.Tensor,
        beliefs: torch.Tensor,
        shuffle: bool,
    ) -> torch.Tensor:
        batch = person_locations.shape[0]
        device = person_locations.device
        facts = torch.zeros(batch, self.num_facts, self.fact_fields, dtype=torch.long, device=device)

        people = torch.arange(self.num_people, device=device)
        objects = torch.arange(self.num_objects, device=device)

        facts[:, :4, 0] = FACT_AT
        facts[:, :4, 1] = PERSON_BASE + people
        facts[:, :4, 2] = LOCATION_BASE + person_locations

        facts[:, 4:6, 0] = FACT_HAS
        facts[:, 4:6, 1] = PERSON_BASE + object_holders
        facts[:, 4:6, 2] = OBJECT_BASE + objects

        facts[:, 6:, 0] = FACT_BELIEF
        facts[:, 6:, 1] = PERSON_BASE + people[:, None].expand(-1, self.num_objects).reshape(-1)
        facts[:, 6:, 2] = OBJECT_BASE + objects.repeat(self.num_people)
        facts[:, 6:, 3] = LOCATION_BASE + beliefs.reshape(batch, -1)

        if shuffle:
            order = torch.rand(batch, self.num_facts, device=device).argsort(dim=1)
            facts = facts.gather(1, order[:, :, None].expand(-1, -1, self.fact_fields))
        return facts

    def sample_batch(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        person_locations = torch.randint(self.num_locations, (batch_size, self.num_people), device=device)
        object_holders = torch.randint(self.num_people, (batch_size, self.num_objects), device=device)
        beliefs = torch.randint(
            self.num_locations, (batch_size, self.num_people, self.num_objects), device=device
        )
        memory = self._facts_from_state(person_locations, object_holders, beliefs, shuffle=True)

        tasks = torch.randint(4, (batch_size,), device=device)
        observers = torch.randint(self.num_people, (batch_size,), device=device)
        objects = torch.randint(self.num_objects, (batch_size,), device=device)
        # Always choose another person for the outside-information question.
        targets = (observers + 1 + torch.randint(3, (batch_size,), device=device)) % self.num_people

        question = torch.zeros(batch_size, self.question_fields, dtype=torch.long, device=device)
        question[:, 0] = Q_TRUE_OBJECT_LOCATION + tasks
        question[:, 1] = torch.where(tasks.eq(3), PERSON_BASE + targets, OBJECT_BASE + objects)

        rows = torch.arange(batch_size, device=device)
        true_locations = person_locations[rows, object_holders[rows, objects]] + self.num_people
        belief_locations = beliefs[rows, observers, objects] + self.num_people
        holder_answers = object_holders[rows, objects]
        person_location_answers = person_locations[rows, targets] + self.num_people
        candidates = torch.stack(
            (true_locations, belief_locations, holder_answers, person_location_answers), dim=1
        )
        labels = candidates[rows, tasks]
        observer_tokens = PERSON_BASE + observers
        return memory, question, observer_tokens, labels, tasks

    def fixed_world_tensors(self, world: FixedWorld) -> tuple[torch.Tensor, ...]:
        locations = torch.tensor([world.person_locations])
        holders = torch.tensor([world.object_holders])
        beliefs = torch.tensor([world.beliefs])
        memory = self._facts_from_state(locations, holders, beliefs, shuffle=False).squeeze(0)

        questions: list[list[int]] = []
        observers: list[int] = []
        labels: list[int] = []
        tasks: list[int] = []
        for observer in range(self.num_people):
            for obj in range(self.num_objects):
                holder = world.object_holders[obj]
                questions.append([Q_TRUE_OBJECT_LOCATION, OBJECT_BASE + obj, PAD])
                observers.append(PERSON_BASE + observer)
                labels.append(self.num_people + world.person_locations[holder])
                tasks.append(0)

                questions.append([Q_BELIEF_OBJECT_LOCATION, OBJECT_BASE + obj, PAD])
                observers.append(PERSON_BASE + observer)
                labels.append(self.num_people + world.beliefs[observer][obj])
                tasks.append(1)

                questions.append([Q_OBJECT_HOLDER, OBJECT_BASE + obj, PAD])
                observers.append(PERSON_BASE + observer)
                labels.append(holder)
                tasks.append(2)

            for target in range(self.num_people):
                if target == observer:
                    continue
                questions.append([Q_PERSON_LOCATION, PERSON_BASE + target, PAD])
                observers.append(PERSON_BASE + observer)
                labels.append(self.num_people + world.person_locations[target])
                tasks.append(3)

        return (
            memory,
            torch.tensor(questions),
            torch.tensor(observers),
            torch.tensor(labels),
            torch.tensor(tasks),
        )

