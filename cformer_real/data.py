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
        self.company_aliases: dict[str, str] = data.get("company_aliases", {})
        self.series_aliases: dict[str, str] = data.get("series_aliases", {})

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
        self._metas = metas
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

    def series_key_of(self, label: int) -> tuple | None:
        meta = self._metas[label]
        if not meta:
            return None
        return (meta.get("company"), meta.get("series"))

    def series_index_of(self, label: int) -> int:
        return int(self._metas[label].get("series_index", -1))

    def series_lexicon(self) -> list[tuple[tuple, str, str]]:
        """Deduplicated [(key, company_lower, series_lower)] for lexical anchoring."""
        seen: dict[tuple, tuple[str, str]] = {}
        for meta in self._metas:
            if not meta:
                continue
            key = (meta.get("company"), meta.get("series"))
            if key not in seen:
                seen[key] = (str(meta.get("company")).lower(), str(meta.get("series")).lower())
        return [(key, company, series) for key, (company, series) in seen.items()]

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


def apply_aliases(text: str, company_aliases: dict[str, str],
                  series_aliases: dict[str, str]) -> str:
    """指代消解：把别称改写为规范名（长别名优先，避免子串误替换）。"""
    out = " ".join(text.split()).lower()
    for mapping in (series_aliases, company_aliases):
        for alias in sorted(mapping, key=len, reverse=True):
            canonical = mapping[alias].lower()
            needle = alias.lower()
            if needle in out:
                out = out.replace(needle, f" {canonical} ")
    return out
