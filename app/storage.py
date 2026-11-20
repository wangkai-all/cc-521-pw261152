"""SQLite-backed storage.

Design guarantees:
* calibration rows are append-only; activation flips status rows but never
  mutates a stored config JSON;
* runs are immutable once inserted (the API never UPDATEs a run row);
* each run pins the exact calibration version id *and* its sha256 checksum,
  so historical scores stay explainable even after new versions activate;
* backfill/re-score is idempotent per (version_id, dataset, batch_id).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .domain.models import (
    BatchScoreResult,
    BatchSubmission,
    CalibrationConfig,
    VersionDiff,
    VersionStatus,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_versions (
    version_id    TEXT PRIMARY KEY,
    config_json   TEXT NOT NULL,
    config_checksum TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('draft','active','archived')),
    created_at    TEXT NOT NULL,
    activated_at  TEXT,
    created_by    TEXT NOT NULL,
    notes         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    version_id      TEXT NOT NULL REFERENCES calibration_versions(version_id),
    config_checksum TEXT NOT NULL,
    dataset         TEXT NOT NULL,
    batch_id        TEXT NOT NULL,
    observed_at     TEXT NOT NULL,
    backfill        INTEGER NOT NULL DEFAULT 0,
    rescore_of_run_id TEXT,
    submission_json TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    state           TEXT NOT NULL,
    overall_score   REAL,
    gate_passed     INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (version_id, dataset, batch_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    version_id  TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    batch_id    TEXT NOT NULL,
    alert_type  TEXT NOT NULL,
    severity    TEXT NOT NULL,
    rule_id     TEXT,
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_lookup ON runs(dataset, batch_id);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj: Any) -> str:
    """Stable serialization used both for checksums and DB storage."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def checksum_config(config: CalibrationConfig) -> str:
    # mode="json" renders datetimes as ISO strings, keeping the hash stable
    # across processes and identical to the stored model_dump_json() semantics.
    return hashlib.sha256(canonical_json(config.model_dump(mode="json")).encode("utf-8")).hexdigest()


class Storage:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ #
    # Calibration versions
    # ------------------------------------------------------------------ #
    def create_version(
        self, version_id: str, config: CalibrationConfig, created_by: str
    ) -> dict[str, Any]:
        config_json = config.model_dump_json()
        with self.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO calibration_versions "
                    "(version_id, config_json, config_checksum, status, created_at, created_by, notes) "
                    "VALUES (?, ?, ?, 'draft', ?, ?, ?)",
                    (
                        version_id,
                        config_json,
                        checksum_config(config),
                        utc_now_iso(),
                        created_by,
                        config.notes,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"version_id {version_id!r} already exists") from exc
        return self.get_version(version_id)

    def activate_version(self, version_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM calibration_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown version {version_id!r}")
            if row["status"] == VersionStatus.archived.value:
                raise ValueError(f"version {version_id!r} is archived and cannot be activated")
            # exactly one active version at a time; activation never edits configs
            conn.execute(
                "UPDATE calibration_versions SET status = 'archived', "
                "activated_at = COALESCE(activated_at, ?) WHERE status = 'active'",
                (utc_now_iso(),),
            )
            conn.execute(
                "UPDATE calibration_versions SET status = 'active', activated_at = ? "
                "WHERE version_id = ?",
                (utc_now_iso(), version_id),
            )
        return self.get_version(version_id)

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM calibration_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown version {version_id!r}")
        return self._version_row(row)

    def get_active_version(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM calibration_versions WHERE status = 'active' LIMIT 1"
            ).fetchone()
        return self._version_row(row) if row else None

    def list_versions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM calibration_versions ORDER BY created_at DESC, version_id DESC"
            ).fetchall()
        return [self._version_row(r) for r in rows]

    @staticmethod
    def _version_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "version_id": row["version_id"],
            "config": CalibrationConfig.model_validate_json(row["config_json"]),
            "config_json": row["config_json"],
            "config_checksum": row["config_checksum"],
            "status": VersionStatus(row["status"]),
            "created_at": row["created_at"],
            "activated_at": row["activated_at"],
            "created_by": row["created_by"],
            "notes": row["notes"],
        }

    def diff_versions(self, from_id: str, to_id: str) -> VersionDiff:
        a = self.get_version(from_id)["config"]
        b = self.get_version(to_id)["config"]
        ra, rb = a.rule_index(), b.rule_index()
        ids = sorted(set(ra) | set(rb))
        added = sorted(set(rb) - set(ra))
        removed = sorted(set(ra) - set(rb))

        weights: dict[str, dict[str, float | None]] = {}
        thresholds: dict[str, dict[str, float | None]] = {}
        min_samples: dict[str, dict[str, int | None]] = {}
        for rid in ids:
            weights[rid] = {
                "from": ra[rid].weight if rid in ra else None,
                "to": rb[rid].weight if rid in rb else None,
            }
            thresholds[rid] = {
                "from": ra[rid].pass_threshold if rid in ra else None,
                "to": rb[rid].pass_threshold if rid in rb else None,
            }
            min_samples[rid] = {
                "from": ra[rid].min_samples if rid in ra else None,
                "to": rb[rid].min_samples if rid in rb else None,
            }

        def _windows(cfg: CalibrationConfig) -> list[str]:
            return sorted(w.window_id for w in cfg.exemption_windows)

        def _groups(cfg: CalibrationConfig) -> list[list[str]]:
            return [list(g) for g in cfg.conflict_groups]

        return VersionDiff(
            from_version=from_id,
            to_version=to_id,
            weights=weights,
            thresholds=thresholds,
            min_samples=min_samples,
            min_batch_samples={"from": a.min_batch_samples, "to": b.min_batch_samples},
            exemption_windows={"from": _windows(a), "to": _windows(b)},
            conflict_groups={"from": _groups(a), "to": _groups(b)},
            gate_policy={"from": a.gate_policy.model_dump(), "to": b.gate_policy.model_dump()},
            added_rules=added,
            removed_rules=removed,
        )

    # ------------------------------------------------------------------ #
    # Runs (immutable)
    # ------------------------------------------------------------------ #
    def insert_run(
        self,
        version: dict[str, Any],
        submission: BatchSubmission,
        result: BatchScoreResult,
    ) -> dict[str, Any]:
        run_id = "run_" + uuid.uuid4().hex[:12]
        with self.connect() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO runs (run_id, version_id, config_checksum, dataset, batch_id, "
                    "observed_at, backfill, rescore_of_run_id, submission_json, result_json, "
                    "state, overall_score, gate_passed, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        version["version_id"],
                        version["config_checksum"],
                        submission.dataset,
                        submission.batch_id,
                        submission.observed_at.isoformat(),
                        int(submission.backfill),
                        submission.rescore_of_run_id,
                        submission.model_dump_json(),
                        result.model_dump_json(),
                        result.state.value,
                        result.overall_score,
                        int(result.gate.passed),
                        utc_now_iso(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"run already exists for version={version['version_id']} "
                    f"dataset={submission.dataset!r} batch={submission.batch_id!r}; "
                    "backfill is idempotent per calibration version"
                ) from exc
            for alert in result.alerts:
                conn.execute(
                    "INSERT INTO alerts (run_id, version_id, dataset, batch_id, alert_type, "
                    "severity, rule_id, message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        version["version_id"],
                        submission.dataset,
                        submission.batch_id,
                        alert.alert_type,
                        alert.severity,
                        alert.rule_id,
                        alert.message,
                        utc_now_iso(),
                    ),
                )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id!r}")
        return self._run_row(row)

    def find_run(self, version_id: str, dataset: str, batch_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE version_id = ? AND dataset = ? AND batch_id = ?",
                (version_id, dataset, batch_id),
            ).fetchone()
        return self._run_row(row) if row else None

    def list_runs(self, dataset: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        q = "SELECT * FROM runs"
        params: list[Any] = []
        if dataset:
            q += " WHERE dataset = ?"
            params.append(dataset)
        q += " ORDER BY observed_at DESC, created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(q, params).fetchall()
        return [self._run_row(r) for r in rows]

    def list_batch_runs(self, dataset: str, batch_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE dataset = ? AND batch_id = ? "
                "ORDER BY created_at ASC",
                (dataset, batch_id),
            ).fetchall()
        return [self._run_row(r) for r in rows]

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "version_id": row["version_id"],
            "config_checksum": row["config_checksum"],
            "dataset": row["dataset"],
            "batch_id": row["batch_id"],
            "observed_at": row["observed_at"],
            "backfill": bool(row["backfill"]),
            "rescore_of_run_id": row["rescore_of_run_id"],
            "submission": json.loads(row["submission_json"]),
            "result": BatchScoreResult.model_validate_json(row["result_json"]),
            "state": row["state"],
            "overall_score": row["overall_score"],
            "gate_passed": bool(row["gate_passed"]),
            "created_at": row["created_at"],
        }

    def list_alerts(self, run_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        q = "SELECT * FROM alerts"
        params: list[Any] = []
        if run_id:
            q += " WHERE run_id = ?"
            params.append(run_id)
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]
