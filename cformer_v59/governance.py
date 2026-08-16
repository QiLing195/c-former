from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import sqlite3
from pathlib import Path


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
    VERIFIED = "verified"
    ROLLED_BACK = "rolled_back"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VerificationDecision:
    status: CandidateStatus
    score: float
    margin: float
    reason: str


class EvidenceVerifier:
    """Conservative boundary: model evidence can support, never verify, an alias.

    ``margin_by_type`` enables per-type thresholds (V6.0b calibration finding):
    complete-knowledge queries may use a lower margin to recover coverage,
    while ambiguous/unknown/conflict queries keep a conservative margin so a
    single global relaxation cannot leak false support. Existing callers that
    omit ``query_type`` keep the frozen single-margin behaviour unchanged.
    """

    def __init__(
        self,
        *,
        minimum_score: float = 0.50,
        minimum_margin: float = 0.08,
        minimum_coverage: float = 0.60,
        margin_by_type: dict[str, float] | None = None,
    ) -> None:
        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin
        self.minimum_coverage = minimum_coverage
        self.margin_by_type = dict(margin_by_type or {})

    def decide(
        self,
        score: float,
        runner_up: float,
        coverage: float,
        query_type: str | None = None,
    ) -> VerificationDecision:
        margin = score - runner_up
        if coverage < self.minimum_coverage:
            return VerificationDecision(CandidateStatus.UNKNOWN, score, margin, "insufficient_semantic_coverage")
        if score < self.minimum_score:
            return VerificationDecision(CandidateStatus.UNKNOWN, score, margin, "score_below_support_threshold")
        required_margin = (
            self.margin_by_type.get(query_type, self.minimum_margin)
            if query_type is not None
            else self.minimum_margin
        )
        if margin < required_margin:
            return VerificationDecision(CandidateStatus.AMBIGUOUS, score, margin, "candidate_margin_too_small")
        return VerificationDecision(CandidateStatus.SUPPORTED, score, margin, "multi_evidence_support_requires_review")


# V6.0b calibration recommendation (see V60B_CALIBRATION_REPORT.md): complete
# known/hard/disambiguated queries recover coverage at a lower margin, while
# ambiguous/unknown/conflict keep the frozen 0.08 so they are never auto-supported.
RECOMMENDED_MARGINS_V60B = {
    "known": 0.03,
    "hard": 0.03,
    "disambiguated": 0.03,
    "ambiguous": 0.08,
    "unknown": 0.08,
    "conflict": 0.08,
}


class CandidateLedger:
    """Versioned candidate state machine with explicit review and rollback."""

    _ALLOWED = {
        CandidateStatus.PROPOSED: {CandidateStatus.SUPPORTED, CandidateStatus.AMBIGUOUS, CandidateStatus.REJECTED},
        CandidateStatus.SUPPORTED: {CandidateStatus.VERIFIED, CandidateStatus.REJECTED},
        CandidateStatus.AMBIGUOUS: {CandidateStatus.SUPPORTED, CandidateStatus.REJECTED},
        CandidateStatus.VERIFIED: {CandidateStatus.ROLLED_BACK},
        CandidateStatus.REJECTED: set(),
        CandidateStatus.ROLLED_BACK: set(),
    }

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates(
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                surface_form TEXT NOT NULL,
                object_label INTEGER NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidate_history(
                candidate_id INTEGER NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason TEXT NOT NULL,
                version INTEGER NOT NULL
            );
            """
        )

    @property
    def version(self) -> int:
        row = self.connection.execute("SELECT max(version) FROM candidate_history").fetchone()
        return int(row[0] or 0)

    def propose(self, surface_form: str, object_label: int, *, actor: str = "model") -> int:
        version = self.version + 1
        cursor = self.connection.execute(
            "INSERT INTO candidates(surface_form, object_label, status, version) VALUES(?, ?, ?, ?)",
            (surface_form, object_label, CandidateStatus.PROPOSED.value, version),
        )
        candidate_id = int(cursor.lastrowid)
        self.connection.execute(
            "INSERT INTO candidate_history VALUES(?, NULL, ?, ?, ?, ?)",
            (candidate_id, CandidateStatus.PROPOSED.value, actor, "new_candidate", version),
        )
        self.connection.commit()
        return candidate_id

    def status(self, candidate_id: int) -> CandidateStatus:
        row = self.connection.execute(
            "SELECT status FROM candidates WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return CandidateStatus(row[0])

    def transition(
        self,
        candidate_id: int,
        target: CandidateStatus,
        *,
        actor: str,
        reason: str,
    ) -> int:
        current = self.status(candidate_id)
        if target not in self._ALLOWED[current]:
            raise ValueError(f"invalid candidate transition: {current.value} -> {target.value}")
        if target == CandidateStatus.VERIFIED and not actor.startswith("reviewer:"):
            raise PermissionError("only an explicit reviewer can verify an alias")
        version = self.version + 1
        self.connection.execute(
            "UPDATE candidates SET status=?, version=? WHERE candidate_id=?",
            (target.value, version, candidate_id),
        )
        self.connection.execute(
            "INSERT INTO candidate_history VALUES(?, ?, ?, ?, ?, ?)",
            (candidate_id, current.value, target.value, actor, reason, version),
        )
        self.connection.commit()
        return version

    def history(self, candidate_id: int) -> list[tuple]:
        return self.connection.execute(
            "SELECT from_status, to_status, actor, reason, version FROM candidate_history WHERE candidate_id=? ORDER BY version",
            (candidate_id,),
        ).fetchall()

    def close(self) -> None:
        self.connection.close()
