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
    domain: str = ""


class AIModelWorld:
    """Loads one or more four-evidence datasets (合并 tokenizer，支持跨域评测).

    Reuses the V6.0 TokenCFormerResolver verbatim: candidates are
    (batch, 4, field_length) and queries are (batch, query_length).
    传入多个路径时：共享 tokenizer（两域词表都覆盖），对象/查询带 domain 标签。
    """

    field_length = 48
    query_length = 48

    def __init__(self, *paths: str | Path) -> None:
        if not paths:
            raise ValueError("at least one dataset path is required")
        self.queries: list[dict] = []
        objects_raw: list[tuple[dict, str]] = []
        texts: list[str] = []
        for path in paths:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            domain = str(data.get("meta", {}).get("dataset", "unknown"))
            for obj in data["objects"]:
                objects_raw.append((obj, domain))
                texts.append(obj["name"])
                texts.extend(obj["evidence"][field] for field in FIELDS)
            dataset_queries = []
            for query in data["queries"]:
                query = dict(query)
                query["domain"] = domain
                dataset_queries.append(query)
                texts.append(query["text"])
            self.queries.extend(dataset_queries)

        self.tokenizer = MixedTokenizer(texts)
        self.objects = [
            AIModelObject(
                object_id=obj["id"],
                label=index,
                name=obj["name"],
                evidence=tuple(obj["evidence"][field] for field in FIELDS),
                domain=domain,
            )
            for index, (obj, domain) in enumerate(objects_raw)
        ]
        self._label_by_id = {obj.object_id: obj.label for obj in self.objects}

    def objects_by_domain(self, domain: str) -> list[AIModelObject]:
        return [obj for obj in self.objects if obj.domain == domain]

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

    def _by_kind(self, kind: str, split: str | None, domain: str | None) -> list[dict]:
        queries = self.queries
        if split is not None:
            queries = [query for query in queries if query.get("split", "train") == split]
        if domain is not None:
            queries = [query for query in queries if query.get("domain") == domain]
        return [query for query in queries if query["kind"] == kind]

    def known_queries(self, split: str | None = None, domain: str | None = None) -> list[dict]:
        return self._by_kind("known", split, domain)

    def ambiguous_queries(self, split: str | None = None, domain: str | None = None) -> list[dict]:
        return self._by_kind("ambiguous", split, domain)

    def unknown_queries(self, split: str | None = None, domain: str | None = None) -> list[dict]:
        return self._by_kind("unknown", split, domain)
