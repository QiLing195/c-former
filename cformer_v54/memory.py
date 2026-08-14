from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .schema import Fact


@dataclass
class _Record:
    fact: Fact
    created_version: int
    removed_version: int | None = None


class VersionedMemory:
    """Append-oriented fact store with indexed and historical snapshot access."""

    def __init__(self) -> None:
        self._records: dict[int, _Record] = {}
        self._index: dict[tuple[int, str], list[int]] = defaultdict(list)
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def fact_count(self) -> int:
        return len(self._records)

    @property
    def index_entries(self) -> int:
        return sum(len(values) for values in self._index.values())

    def add_many(self, facts: list[Fact]) -> int:
        ids = [fact.fact_id for fact in facts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate fact id in batch")
        if any(fact_id in self._records for fact_id in ids):
            raise ValueError("fact id already exists")
        self._version += 1
        for fact in facts:
            self._records[fact.fact_id] = _Record(fact, self._version)
            self._index[(fact.subject, fact.relation)].append(fact.fact_id)
        return self._version

    def retract(self, fact_id: int) -> int:
        record = self._records[fact_id]
        if record.removed_version is not None:
            raise ValueError("fact already retracted")
        self._version += 1
        record.removed_version = self._version
        return self._version

    def candidates(
        self,
        subject: int,
        relation: str,
        world_version: int,
        *,
        indexed: bool = True,
    ) -> list[Fact]:
        if not 0 <= world_version <= self._version:
            raise ValueError(f"invalid world version {world_version}")
        if indexed:
            records = (self._records[fact_id] for fact_id in self._index.get((subject, relation), ()))
        else:
            records = self._records.values()
        return [
            record.fact
            for record in records
            if record.fact.subject == subject
            and record.fact.relation == relation
            and record.created_version <= world_version
            and (record.removed_version is None or record.removed_version > world_version)
        ]
