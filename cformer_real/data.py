from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import MixedTokenizer

FIELDS = ("名称", "属性", "关系", "变化")


@dataclass(frozen=True)
class AIModelObject:
    object_id: str
    label: int
    name: str
    evidence: tuple[str, str, str, str]


class AIModelWorld:
    """Loads the AI-model four-evidence dataset and encodes it to tensors.

    Reuses the V6.0 TokenCFormerResolver verbatim: candidates are
    (batch, 4, field_length) and queries are (batch, query_length).
    """

    field_length = 48
    query_length = 48

    def __init__(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        objects = data["objects"]
        self.queries = data["queries"]

        texts: list[str] = []
        for obj in objects:
            texts.append(obj["name"])
            texts.extend(obj["evidence"][field] for field in FIELDS)
        for query in self.queries:
            texts.append(query["text"])

        self.tokenizer = MixedTokenizer(texts)
        self.objects = [
            AIModelObject(
                object_id=obj["id"],
                label=index,
                name=obj["name"],
                evidence=tuple(obj["evidence"][field] for field in FIELDS),
            )
            for index, obj in enumerate(objects)
        ]
        self._label_by_id = {obj.object_id: obj.label for obj in self.objects}
        metas = [obj.get("meta") or {} for obj in objects]
        groups: dict[tuple, list[int]] = {}
        for index, meta in enumerate(metas):
            key = (meta.get("company"), meta.get("series"))
            if key != (None, None):
                groups.setdefault(key, []).append(index)
        self._siblings_by_label: dict[int, list[int]] = {
            label: [item for item in members if item != label]
            for members in groups.values()
            for label in members
        }

    def series_siblings(self, label: int) -> list[int]:
        """Labels sharing the same (company, series), excluding the object itself.

        Returns [] for objects without meta; used to sample hard negatives so
        training must discriminate near-identical evidence instead of memorizing.
        """
        return self._siblings_by_label.get(label, [])

    def encode_candidates(self, objects: list[AIModelObject]) -> torch.Tensor:
        rows = []
        for obj in objects:
            fields = [
                self.tokenizer.encode(obj.evidence[index], self.field_length)[0]
                for index in range(len(FIELDS))
            ]
            rows.append(torch.stack(fields))
        if not rows:
            return torch.empty(0, len(FIELDS), self.field_length, dtype=torch.long)
        return torch.stack(rows)

    def encode_query(self, text: str) -> tuple[torch.Tensor, float]:
        return self.tokenizer.encode(text, self.query_length)

    def target_label(self, target_id: str | None) -> int:
        return self._label_by_id[target_id] if target_id else -1

    def known_queries(self) -> list[dict]:
        return [query for query in self.queries if query["kind"] == "known"]

    def ambiguous_queries(self) -> list[dict]:
        return [query for query in self.queries if query["kind"] == "ambiguous"]

    def unknown_queries(self) -> list[dict]:
        return [query for query in self.queries if query["kind"] == "unknown"]
