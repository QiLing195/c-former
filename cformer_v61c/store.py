# -*- coding: utf-8 -*-
"""V6.1c UnifiedObjectStore: exact-alias B-tree equivalent + FTS shard source.

SQLite-backed object records with versioned lifecycle (tombstone removal),
verified/proposed alias states and an FTS5 trigram document index used only to
generate candidate shards when the exact alias misses. Neural vectors are NOT
stored here: they live once in the V6.1 IVF index (single-copy principle).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


def normalize_surface(text: str) -> str:
    """Canonical surface form: whitespace-collapsed and case-folded."""
    return " ".join(text.split()).lower()


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    canonical_name: str
    document: dict = field(default_factory=dict)  # 四证据文本
    meta: dict = field(default_factory=dict)
    version: int = 0                              # 写入版本（证据引用用）


class UnifiedObjectStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS objects(
                object_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                document TEXT NOT NULL,
                meta TEXT NOT NULL,
                version INTEGER NOT NULL,
                removed INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS aliases(
                alias TEXT PRIMARY KEY,
                object_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_version INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS aliases_by_object ON aliases(object_id);
            """
        )
        self._setup_fts()

    def _setup_fts(self) -> None:
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING "
                "fts5(object_id UNINDEXED, text, tokenize='trigram')"
            )
            self.fts_mode = "trigram"
        except sqlite3.OperationalError:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS docs_plain("
                "object_id TEXT NOT NULL, text TEXT NOT NULL)"
            )
            self.fts_mode = "like"

    @property
    def version(self) -> int:
        row = self.connection.execute("SELECT max(version) FROM objects").fetchone()
        return int(row[0] or 0)

    # -- write path -----------------------------------------------------------

    def upsert_object(
        self,
        record: ObjectRecord,
        aliases: list[str],
    ) -> int:
        """Insert or refresh an object with its verified aliases."""
        version = self.version + 1
        self.connection.execute(
            "INSERT INTO objects(object_id, canonical_name, document, meta, version, removed) "
            "VALUES(?, ?, ?, ?, ?, 0) "
            "ON CONFLICT(object_id) DO UPDATE SET canonical_name=?, document=?, "
            "meta=?, version=?, removed=0",
            (
                record.object_id,
                record.canonical_name,
                json.dumps(record.document, ensure_ascii=False),
                json.dumps(record.meta, ensure_ascii=False),
                version,
                record.canonical_name,
                json.dumps(record.document, ensure_ascii=False),
                json.dumps(record.meta, ensure_ascii=False),
                version,
            ),
        )
        surfaces = {normalize_surface(a) for a in aliases + [record.canonical_name]}
        surfaces.discard("")
        for surface in surfaces:
            self.connection.execute(
                "INSERT INTO aliases(alias, object_id, status, created_version) "
                "VALUES(?, ?, 'verified', ?) "
                "ON CONFLICT(alias) DO UPDATE SET object_id=?, status='verified', "
                "created_version=?",
                (surface, record.object_id, version, record.object_id, version),
            )
        self._index_document(record.object_id, record.canonical_name, record.document)
        self.connection.commit()
        return version

    def _index_document(self, object_id: str, name: str, document: dict) -> None:
        text = name + "\n" + "\n".join(str(v) for v in document.values())
        if self.fts_mode == "trigram":
            self.connection.execute("DELETE FROM docs WHERE object_id=?", (object_id,))
            self.connection.execute(
                "INSERT INTO docs(object_id, text) VALUES(?, ?)", (object_id, text)
            )
        else:
            self.connection.execute(
                "DELETE FROM docs_plain WHERE object_id=?", (object_id,)
            )
            self.connection.execute(
                "INSERT INTO docs_plain(object_id, text) VALUES(?, ?)",
                (object_id, text),
            )

    def remove_object(self, object_id: str) -> int:
        """Tombstone removal: object rows stay for audit but go invisible."""
        version = self.version + 1
        self.connection.execute(
            "UPDATE objects SET removed=1, version=? WHERE object_id=?",
            (version, object_id),
        )
        self.connection.execute("DELETE FROM aliases WHERE object_id=?", (object_id,))
        if self.fts_mode == "trigram":
            self.connection.execute("DELETE FROM docs WHERE object_id=?", (object_id,))
        else:
            self.connection.execute(
                "DELETE FROM docs_plain WHERE object_id=?", (object_id,)
            )
        self.connection.commit()
        return version

    def add_verified_alias(self, alias: str, object_id: str) -> int:
        """Review-flow entry: only call AFTER the ledger shows reviewer approval."""
        version = self.version + 1
        surface = normalize_surface(alias)
        row = self.connection.execute(
            "SELECT removed FROM objects WHERE object_id=?", (object_id,)
        ).fetchone()
        if row is None or row[0]:
            raise KeyError(f"unknown or removed object: {object_id}")
        self.connection.execute(
            "INSERT INTO aliases(alias, object_id, status, created_version) "
            "VALUES(?, ?, 'verified', ?) "
            "ON CONFLICT(alias) DO UPDATE SET object_id=?, status='verified'",
            (surface, object_id, version, object_id),
        )
        self.connection.commit()
        return version

    # -- read path ------------------------------------------------------------

    def exact_lookup(self, surface_form: str) -> str | None:
        """Exact alias hit for a LIVE object, else None."""
        row = self.connection.execute(
            "SELECT a.object_id FROM aliases a JOIN objects o USING(object_id) "
            "WHERE a.alias=? AND a.status='verified' AND o.removed=0",
            (normalize_surface(surface_form),),
        ).fetchone()
        return row[0] if row else None

    def fts_candidates(self, text: str, limit: int = 64,
                       match_query: str | None = None) -> list[str]:
        """Candidate shards from the document index; best-effort, never authoritative.

        match_query 允许调用方传自定义 MATCH 表达式（如 OR 关键词），默认整句短语。
        """
        if self.fts_mode == "trigram":
            query = match_query or '"' + text.replace('"', '""') + '"'
            try:
                rows = self.connection.execute(
                    "SELECT object_id FROM docs WHERE docs MATCH ? LIMIT ?",
                    (query, limit),
                ).fetchall()
                return [r[0] for r in rows]
            except sqlite3.OperationalError:
                return []
        like = f"%{text}%"
        rows = self.connection.execute(
            "SELECT object_id FROM docs_plain WHERE text LIKE ? LIMIT ?",
            (like, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def live_records(self) -> list[tuple[ObjectRecord, list[str]]]:
        rows = self.connection.execute(
            "SELECT object_id, canonical_name, document, meta, version "
            "FROM objects WHERE removed=0"
        ).fetchall()
        out = []
        for object_id, name, document, meta, version in rows:
            alias_rows = self.connection.execute(
                "SELECT alias FROM aliases WHERE object_id=? AND status='verified'",
                (object_id,),
            ).fetchall()
            out.append((
                ObjectRecord(object_id, name, json.loads(document),
                             json.loads(meta), int(version)),
                [a for (a,) in alias_rows],
            ))
        return out

    def close(self) -> None:
        self.connection.close()
