from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


V56_SCALES = (2048, 8192, 32768)


def target_scope(
    kind: torch.Tensor, observer: torch.Tensor, semantic_key: torch.Tensor
) -> torch.Tensor:
    """View mapping depends jointly on object state, query kind and observer."""
    inverse = kind.bool() ^ semantic_key[..., 0].ge(0)
    return torch.where(inverse, 5 - observer, observer)


def sample_training_batch(
    batch_size: int,
    shortlist_size: int,
    key_dimensions: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
):
    query_keys = F.normalize(
        torch.randn(batch_size, key_dimensions, generator=generator, device=device),
        dim=-1,
    )
    query_kinds = torch.randint(
        2, (batch_size,), generator=generator, device=device
    )
    observers = torch.randint(
        1, 5, (batch_size,), generator=generator, device=device
    )
    candidate_keys = F.normalize(
        torch.randn(
            batch_size,
            shortlist_size,
            key_dimensions,
            generator=generator,
            device=device,
        ),
        dim=-1,
    )
    candidate_kinds = torch.randint(
        2,
        (batch_size, shortlist_size),
        generator=generator,
        device=device,
    )
    candidate_scopes = torch.randint(
        1,
        5,
        (batch_size, shortlist_size),
        generator=generator,
        device=device,
    )
    candidate_keys[:, :4] = query_keys[:, None]
    candidate_kinds[:, :4] = query_kinds[:, None]
    candidate_scopes[:, :4] = torch.arange(1, 5, device=device)[None]
    correct = target_scope(query_kinds, observers, query_keys) - 1
    permutation = torch.rand(
        batch_size, shortlist_size, generator=generator, device=device
    ).argsort(dim=-1)
    gather_key = permutation.unsqueeze(-1).expand(-1, -1, key_dimensions)
    candidate_keys = candidate_keys.gather(1, gather_key)
    candidate_kinds = candidate_kinds.gather(1, permutation)
    candidate_scopes = candidate_scopes.gather(1, permutation)
    labels = permutation.eq(correct[:, None]).long().argmax(dim=-1)
    return (
        candidate_keys,
        candidate_kinds,
        candidate_scopes,
        query_keys,
        query_kinds,
        observers,
        labels,
    )


@dataclass(frozen=True)
class SyntheticRetrievalWorld:
    scale: int
    world_index: int
    candidate_keys: torch.Tensor
    candidate_kinds: torch.Tensor
    candidate_scopes: torch.Tensor
    entity_keys: torch.Tensor
    entity_kinds: torch.Tensor

    @classmethod
    def build(
        cls, scale: int, world_index: int, key_dimensions: int = 8
    ) -> "SyntheticRetrievalWorld":
        if scale not in V56_SCALES or scale % 4:
            raise ValueError(f"unsupported V5.6 scale {scale}")
        generator = torch.Generator().manual_seed(56_000 + scale + world_index)
        entities = scale // 4
        entity_keys = F.normalize(
            torch.randn(entities, key_dimensions, generator=generator), dim=-1
        )
        entity_kinds = torch.randint(2, (entities,), generator=generator)
        return cls(
            scale,
            world_index,
            entity_keys.repeat_interleave(4, dim=0),
            entity_kinds.repeat_interleave(4),
            torch.arange(1, 5).repeat(entities),
            entity_keys,
            entity_kinds,
        )

    @property
    def compact_cache_bytes(self) -> int:
        tensors = (
            self.candidate_keys,
            self.candidate_kinds,
            self.candidate_scopes,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def fixed_queries(self, count: int = 200):
        generator = torch.Generator().manual_seed(
            57_000 + self.scale + self.world_index
        )
        entities = torch.randperm(
            self.entity_keys.shape[0], generator=generator
        )[:count]
        observers = torch.randint(1, 5, (count,), generator=generator)
        kinds = self.entity_kinds[entities]
        correct_scopes = target_scope(kinds, observers, self.entity_keys[entities])
        correct_ids = entities * 4 + correct_scopes - 1
        return entities, self.entity_keys[entities], kinds, observers, correct_ids

    def indexed_shortlists(
        self,
        entities: torch.Tensor,
        correct_ids: torch.Tensor,
        shortlist_size: int = 64,
    ):
        """Exact object shard plus deterministic hard negatives.

        V5.5 resolves object identity before neural ranking, so the structured index
        can always insert all four views of that object.  Scale changes the number
        of independent objects, not the neural candidate budget.
        """
        rows: list[torch.Tensor] = []
        labels: list[int] = []
        for row, (entity, correct_id) in enumerate(zip(entities.tolist(), correct_ids.tolist())):
            generator = torch.Generator().manual_seed(
                58_000 + self.scale + self.world_index * 1_000 + row
            )
            target = torch.arange(entity * 4, entity * 4 + 4)
            pool = torch.cat(
                (
                    torch.arange(0, entity * 4),
                    torch.arange(entity * 4 + 4, self.scale),
                )
            )
            negative = pool[
                torch.randperm(pool.shape[0], generator=generator)[
                    : shortlist_size - 4
                ]
            ]
            ids = torch.cat((target, negative))
            ids = ids[torch.randperm(shortlist_size, generator=generator)]
            rows.append(ids)
            labels.append(ids.eq(correct_id).nonzero(as_tuple=False)[0, 0].item())
        return torch.stack(rows), torch.tensor(labels)
