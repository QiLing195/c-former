from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, TypeVar

from .schema import ObjectState, SemanticObject, Transformation


T = TypeVar("T")


@dataclass
class _Record(Generic[T]):
    value: T
    created_version: int
    removed_version: int | None = None


class CognitiveMemory:
    """Append-oriented object/dynamics store with historical snapshots.

    Identity is explicit and auditable in V5.5.  The store deliberately does not
    claim that aliases are automatically discovered semantic attractors.
    """

    def __init__(self) -> None:
        self._version = 0
        self._objects: dict[int, _Record[SemanticObject]] = {}
        self._states: dict[int, _Record[ObjectState]] = {}
        self._transformations: dict[int, _Record[Transformation]] = {}
        self._name_index: dict[str, list[int]] = defaultdict(list)
        self._state_index: dict[int, list[int]] = defaultdict(list)
        self._transformation_index: dict[tuple[int, str], list[int]] = defaultdict(list)

    @property
    def version(self) -> int:
        return self._version

    @property
    def object_count(self) -> int:
        return len(self._objects)

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def transformation_count(self) -> int:
        return len(self._transformations)

    def _next_version(self) -> int:
        self._version += 1
        return self._version

    @staticmethod
    def _check_new(values, records, attr: str) -> None:
        ids = [getattr(value, attr) for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {attr} in batch")
        if any(value_id in records for value_id in ids):
            raise ValueError(f"{attr} already exists")

    def add_objects(self, objects: list[SemanticObject]) -> int:
        self._check_new(objects, self._objects, "object_id")
        version = self._next_version()
        for obj in objects:
            self._objects[obj.object_id] = _Record(obj, version)
            for name in obj.names:
                self._name_index[name.casefold()].append(obj.object_id)
        return version

    def add_states(self, states: list[ObjectState]) -> int:
        self._check_new(states, self._states, "state_id")
        if any(state.object_id not in self._objects for state in states):
            raise ValueError("state references unknown object")
        version = self._next_version()
        for state in states:
            self._states[state.state_id] = _Record(state, version)
            self._state_index[state.object_id].append(state.state_id)
        return version

    def add_transformations(self, transformations: list[Transformation]) -> int:
        self._check_new(
            transformations, self._transformations, "transformation_id"
        )
        known = self._objects
        for transformation in transformations:
            if transformation.output_object_id not in known or any(
                object_id not in known for object_id in transformation.input_object_ids
            ):
                raise ValueError("transformation references unknown object")
        version = self._next_version()
        for transformation in transformations:
            self._transformations[transformation.transformation_id] = _Record(
                transformation, version
            )
            for object_id in transformation.input_object_ids:
                self._transformation_index[(object_id, transformation.operator)].append(
                    transformation.transformation_id
                )
        return version

    def _active(self, record: _Record[T], version: int) -> bool:
        return record.created_version <= version and (
            record.removed_version is None or record.removed_version > version
        )

    def _validate_version(self, version: int) -> None:
        if not 0 <= version <= self._version:
            raise ValueError(f"invalid world version {version}")

    def object_candidates(
        self, name: str, query_time: int, version: int
    ) -> list[SemanticObject]:
        self._validate_version(version)
        ids = self._name_index.get(name.casefold(), ())
        candidates = [
            self._objects[object_id].value
            for object_id in ids
            if self._active(self._objects[object_id], version)
            and self._objects[object_id].value.valid_from <= query_time
            < self._objects[object_id].value.valid_to
        ]
        exact = [
            obj for obj in candidates if obj.canonical_name.casefold() == name.casefold()
        ]
        return exact or candidates

    def get_object(self, object_id: int, version: int) -> SemanticObject | None:
        self._validate_version(version)
        record = self._objects.get(object_id)
        return record.value if record is not None and self._active(record, version) else None

    def state_candidates(self, object_id: int, version: int) -> list[ObjectState]:
        self._validate_version(version)
        return [
            self._states[state_id].value
            for state_id in self._state_index.get(object_id, ())
            if self._active(self._states[state_id], version)
        ]

    def transformation_candidates(
        self, object_id: int, operator: str, version: int
    ) -> list[Transformation]:
        self._validate_version(version)
        return [
            self._transformations[transformation_id].value
            for transformation_id in self._transformation_index.get(
                (object_id, operator), ()
            )
            if self._active(self._transformations[transformation_id], version)
        ]

    def next_state_id(self) -> int:
        return max(self._states, default=0) + 1

    def snapshot_objects(self, version: int | None = None) -> tuple[SemanticObject, ...]:
        snapshot = self._version if version is None else version
        self._validate_version(snapshot)
        return tuple(
            record.value
            for record in self._objects.values()
            if self._active(record, snapshot)
        )

    def snapshot_states(self, version: int | None = None) -> tuple[ObjectState, ...]:
        snapshot = self._version if version is None else version
        self._validate_version(snapshot)
        return tuple(
            record.value
            for record in self._states.values()
            if self._active(record, snapshot)
        )

    def snapshot_transformations(
        self, version: int | None = None
    ) -> tuple[Transformation, ...]:
        snapshot = self._version if version is None else version
        self._validate_version(snapshot)
        return tuple(
            record.value
            for record in self._transformations.values()
            if self._active(record, snapshot)
        )
