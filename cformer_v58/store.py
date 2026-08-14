from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cformer_v57 import HashedTextEncoder, normalize_text


_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]")


class SparseTermEncoder:
    """Bounded sparse terms for an on-disk FTS posting index."""

    def __init__(self, max_terms: int = 48) -> None:
        self.max_terms = max_terms

    @staticmethod
    def _term(feature: str) -> str:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).hexdigest()
        return f"t{digest}"

    def terms(self, text: str) -> tuple[str, ...]:
        normalized = normalize_text(text)
        compact = re.sub(r"[^a-z0-9\u3400-\u9fff]", "", normalized)
        features: list[str] = []
        features.extend(f"w:{token}" for token in _TOKEN.findall(normalized))
        for size in (2, 3):
            features.extend(
                f"c{size}:{compact[index:index + size]}"
                for index in range(max(0, len(compact) - size + 1))
            )
        seen: set[str] = set()
        result: list[str] = []
        for feature in features:
            term = self._term(feature)
            if term not in seen:
                seen.add(term)
                result.append(term)
            if len(result) >= self.max_terms:
                break
        return tuple(result)

    def document(self, text: str) -> str:
        return " ".join(self.terms(text))

    def query(self, text: str) -> str:
        normalized = normalize_text(text)
        specific_tokens = [
            token
            for token in _TOKEN.findall(normalized)
            if len(token) >= 4 and any(character.isdigit() for character in token)
        ]
        if len(specific_tokens) >= 2:
            # Attribute/code-like tokens have much smaller posting lists. Their
            # intersection selects a shard before generic lexical expansion.
            return " AND ".join(
                self._term(f"w:{token}") for token in dict.fromkeys(specific_tokens)
            )
        return " OR ".join(self.terms(text))


class QuantizedVectorStore:
    """Append-only int8 object vectors with per-row scale and mmap reads."""

    def __init__(self, directory: Path, dimensions: int = 256) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.vector_path = self.directory / "objects.i8"
        self.scale_path = self.directory / "objects.scale.f32"

    @property
    def count(self) -> int:
        if not self.scale_path.exists():
            return 0
        return self.scale_path.stat().st_size // np.dtype(np.float32).itemsize

    def append(self, vectors: np.ndarray) -> tuple[int, int]:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimensions:
            raise ValueError("invalid vector matrix")
        start = self.count
        scales = np.maximum(np.abs(values).max(axis=1), 1e-8) / 127.0
        quantized = np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8)
        with self.vector_path.open("ab") as vector_file:
            quantized.tofile(vector_file)
        with self.scale_path.open("ab") as scale_file:
            scales.astype(np.float32).tofile(scale_file)
        return start, start + values.shape[0]

    def score(self, rows: list[int], query: np.ndarray) -> np.ndarray:
        if not rows:
            return np.empty(0, dtype=np.float32)
        count = self.count
        vectors = np.memmap(
            self.vector_path, dtype=np.int8, mode="r", shape=(count, self.dimensions)
        )
        scales = np.memmap(self.scale_path, dtype=np.float32, mode="r", shape=(count,))
        indices = np.asarray(rows, dtype=np.int64)
        restored = np.asarray(vectors[indices], dtype=np.float32) * np.asarray(
            scales[indices], dtype=np.float32
        )[:, None]
        return restored @ np.asarray(query, dtype=np.float32)


@dataclass(frozen=True)
class SearchHit:
    object_id: int
    score: float
    row_index: int
    lexical_score: float = 0.0
    vector_score: float = 0.0


class LayeredAliasStore:
    """Disk-first exact aliases + FTS candidates + quantized object vectors."""

    def __init__(self, directory: Path, *, dimensions: int = 256) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.directory / "aliases.sqlite3"
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.term_encoder = SparseTermEncoder()
        self.text_encoder = HashedTextEncoder(dimensions)
        self.vectors = QuantizedVectorStore(self.directory / "vectors", dimensions)
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS objects(
                object_id INTEGER PRIMARY KEY,
                row_index INTEGER UNIQUE NOT NULL,
                canonical_name TEXT NOT NULL,
                document TEXT NOT NULL,
                created_version INTEGER NOT NULL,
                removed_version INTEGER
            );
            CREATE TABLE IF NOT EXISTS aliases(
                normalized_alias TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_version INTEGER NOT NULL,
                removed_version INTEGER,
                PRIMARY KEY(normalized_alias, object_id, created_version)
            );
            CREATE INDEX IF NOT EXISTS alias_lookup
                ON aliases(normalized_alias, created_version, removed_version);
            CREATE TABLE IF NOT EXISTS perspectives(
                object_id INTEGER NOT NULL,
                scope INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(object_id, scope)
            );
            CREATE TABLE IF NOT EXISTS alias_candidates(
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                surface_form TEXT NOT NULL,
                normalized_form TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                score REAL NOT NULL,
                margin REAL NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                created_version INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS lexical_index USING fts5(
                terms, content='', tokenize='ascii'
            );
            """
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES('world_version', '0')"
        )
        self.connection.commit()

    @property
    def version(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='world_version'"
        ).fetchone()
        return int(row[0])

    def _next_version(self) -> int:
        version = self.version + 1
        self.connection.execute(
            "UPDATE metadata SET value=? WHERE key='world_version'", (str(version),)
        )
        return version

    @property
    def object_count(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM objects").fetchone()[0])

    @property
    def alias_count(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM aliases").fetchone()[0])

    @property
    def perspective_count(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM perspectives").fetchone()[0])

    def add_objects(self, records: list[dict]) -> int:
        if not records:
            return self.version
        vectors = np.stack(
            [self.text_encoder.encode(record["document"]).numpy() for record in records]
        )
        start, _ = self.vectors.append(vectors)
        version = self._next_version()
        for offset, record in enumerate(records):
            object_id = int(record["object_id"])
            row_index = start + offset
            self.connection.execute(
                "INSERT INTO objects VALUES(?, ?, ?, ?, ?, NULL)",
                (
                    object_id,
                    row_index,
                    record["canonical_name"],
                    record["document"],
                    version,
                ),
            )
            self.connection.execute(
                "INSERT INTO lexical_index(rowid, terms) VALUES(?, ?)",
                (object_id, self.term_encoder.document(record["document"])),
            )
            for alias in record.get("aliases", ()):
                self.connection.execute(
                    "INSERT INTO aliases VALUES(?, ?, ?, ?, ?, NULL)",
                    (normalize_text(alias), object_id, "ingest", 1.0, version),
                )
            for scope in record.get("perspectives", (1, 2, 3, 4)):
                self.connection.execute(
                    "INSERT INTO perspectives VALUES(?, ?, ?)",
                    (object_id, int(scope), json.dumps({"scope": int(scope)})),
                )
        self.connection.commit()
        return version

    def lookup_alias(self, surface_form: str, version: int | None = None) -> list[int]:
        snapshot = self.version if version is None else version
        rows = self.connection.execute(
            """
            SELECT object_id FROM aliases
            WHERE normalized_alias=? AND created_version<=?
              AND (removed_version IS NULL OR removed_version>?)
            ORDER BY confidence DESC, object_id
            """,
            (normalize_text(surface_form), snapshot, snapshot),
        ).fetchall()
        return [int(row[0]) for row in rows]

    def search(self, text: str, limit: int = 64) -> list[SearchHit]:
        query = self.term_encoder.query(text)
        if not query:
            return []
        rows = self.connection.execute(
            """
            SELECT rowid, bm25(lexical_index) FROM lexical_index
            WHERE lexical_index MATCH ?
            ORDER BY bm25(lexical_index)
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        object_ids = [int(row[0]) for row in rows]
        if not object_ids:
            return []
        placeholders = ",".join("?" for _ in object_ids)
        mapping = {
            int(object_id): int(row_index)
            for object_id, row_index in self.connection.execute(
                f"SELECT object_id, row_index FROM objects WHERE object_id IN ({placeholders})",
                object_ids,
            )
        }
        ordered = [object_id for object_id in object_ids if object_id in mapping]
        row_indices = [mapping[object_id] for object_id in ordered]
        query_vector = self.text_encoder.encode(text).numpy()
        vector_scores = self.vectors.score(row_indices, query_vector)
        lexical_raw = np.asarray([-float(row[1]) for row in rows], dtype=np.float32)
        lexical_ordered = np.asarray(
            [lexical_raw[object_ids.index(object_id)] for object_id in ordered],
            dtype=np.float32,
        )
        span = float(lexical_ordered.max() - lexical_ordered.min())
        lexical_scores = (
            (lexical_ordered - lexical_ordered.min()) / span
            if span > 1e-8
            else np.ones_like(lexical_ordered)
        )
        scores = 0.7 * lexical_scores + 0.3 * vector_scores
        hits = [
            SearchHit(
                object_id,
                float(score),
                row_index,
                float(lexical_score),
                float(vector_score),
            )
            for object_id, score, row_index, lexical_score, vector_score in zip(
                ordered, scores, row_indices, lexical_scores, vector_scores
            )
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def propose_alias(
        self, surface_form: str, object_id: int, score: float, margin: float, source: str
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO alias_candidates(
                surface_form, normalized_form, object_id, score, margin,
                source, status, created_version
            ) VALUES(?, ?, ?, ?, ?, ?, 'proposed', ?)
            """,
            (
                surface_form,
                normalize_text(surface_form),
                object_id,
                score,
                margin,
                source,
                self.version,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def review_candidate(self, candidate_id: int, *, approve: bool) -> int:
        row = self.connection.execute(
            "SELECT normalized_form, object_id, source, status FROM alias_candidates WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is None or row[3] != "proposed":
            raise ValueError("candidate is not reviewable")
        version = self._next_version()
        status = "verified" if approve else "rejected"
        self.connection.execute(
            "UPDATE alias_candidates SET status=? WHERE candidate_id=?",
            (status, candidate_id),
        )
        if approve:
            self.connection.execute(
                "INSERT INTO aliases VALUES(?, ?, ?, ?, ?, NULL)",
                (row[0], row[1], f"review:{row[2]}", 1.0, version),
            )
        self.connection.commit()
        return version

    def retract_alias(self, surface_form: str, object_id: int) -> int:
        version = self._next_version()
        self.connection.execute(
            """
            UPDATE aliases SET removed_version=?
            WHERE normalized_alias=? AND object_id=? AND removed_version IS NULL
            """,
            (version, normalize_text(surface_form), object_id),
        )
        self.connection.commit()
        return version

    def disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def close(self) -> None:
        self.connection.close()
