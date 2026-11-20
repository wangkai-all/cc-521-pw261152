"""FastAPI application: versioned calibration API, immutable run records,
alerts and release-gate views."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .domain.models import (
    BatchScoreResult,
    BatchSubmission,
    CalibrationConfig,
    VersionCreate,
    VersionStatus,
)
from .domain.scoring import score_batch
from .storage import Storage

DB_PATH = os.environ.get("DQ_CALIBRATION_DB", str(Path(__file__).parent / "data" / "dq.db"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Data Quality Score Calibration", version="1.0.0")
storage = Storage(DB_PATH)


def _get_version_or_404(version_id: str) -> dict:
    try:
        return storage.get_version(version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "db": DB_PATH}


# --------------------------------------------------------------------------- #
# Calibration versions
# --------------------------------------------------------------------------- #
@app.post("/api/versions", status_code=201)
def create_version(payload: VersionCreate) -> dict:
    try:
        row = storage.create_version(payload.version_id, payload.config, payload.created_by)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _serialize_version(row)


@app.post("/api/versions/{version_id}/activate")
def activate_version(version_id: str) -> dict:
    try:
        row = storage.activate_version(version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _serialize_version(row)


@app.get("/api/versions")
def list_versions() -> list[dict]:
    return [_serialize_version(r) for r in storage.list_versions()]


@app.get("/api/versions/active")
def get_active_version() -> dict:
    row = storage.get_active_version()
    if row is None:
        raise HTTPException(status_code=404, detail="no active calibration version")
    return _serialize_version(row)


@app.get("/api/versions/{version_id}")
def get_version(version_id: str) -> dict:
    return _serialize_version(_get_version_or_404(version_id))


@app.get("/api/versions/{from_id}/diff/{to_id}")
def diff_versions(from_id: str, to_id: str) -> dict:
    try:
        return storage.diff_versions(from_id, to_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _serialize_version(row: dict) -> dict:
    return {
        "version_id": row["version_id"],
        "status": row["status"].value,
        "config": row["config"].model_dump(),
        "config_checksum": row["config_checksum"],
        "created_at": row["created_at"],
        "activated_at": row["activated_at"],
        "created_by": row["created_by"],
        "notes": row["notes"],
    }


# --------------------------------------------------------------------------- #
# Runs / scoring / backfill
# --------------------------------------------------------------------------- #
@app.post("/api/runs", status_code=201)
def submit_run(
    payload: BatchSubmission,
    version_id: str | None = Query(default=None, description="calibration version; defaults to active"),
) -> dict:
    if version_id:
        version = _get_version_or_404(version_id)
    else:
        version = storage.get_active_version()
        if version is None:
            raise HTTPException(status_code=409, detail="no active calibration version")

    # idempotency: a given (version, dataset, batch) is scored at most once;
    # backfill of an already-seen pair returns the stored, immutable run.
    existing = storage.find_run(version["version_id"], payload.dataset, payload.batch_id)
    if existing is not None:
        return _serialize_run(existing, reused=True)

    config: CalibrationConfig = version["config"]
    try:
        result: BatchScoreResult = score_batch(config, payload)
    except ValueError as exc:
        # malformed batch (unknown rule, duplicate samples/observations)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row = storage.insert_run(version, payload, result)
    return _serialize_run(row, reused=False)


@app.get("/api/runs")
def list_runs(dataset: str | None = None, limit: int = Query(default=100, le=500)) -> list[dict]:
    return [_serialize_run(r) for r in storage.list_runs(dataset=dataset, limit=limit)]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        row = storage.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _serialize_run(row)


@app.get("/api/datasets/{dataset}/batches/{batch_id}/runs")
def get_batch_runs(dataset: str, batch_id: str) -> list[dict]:
    return [_serialize_run(r) for r in storage.list_batch_runs(dataset, batch_id)]


@app.get("/api/alerts")
def list_alerts(run_id: str | None = None, limit: int = Query(default=200, le=1000)) -> list[dict]:
    return storage.list_alerts(run_id=run_id, limit=limit)


def _serialize_run(row: dict, reused: bool | None = None) -> dict:
    return {
        "run_id": row["run_id"],
        "version_id": row["version_id"],
        "config_checksum": row["config_checksum"],
        "dataset": row["dataset"],
        "batch_id": row["batch_id"],
        "observed_at": row["observed_at"],
        "backfill": row["backfill"],
        "rescore_of_run_id": row["rescore_of_run_id"],
        "state": row["state"],
        "overall_score": row["overall_score"],
        "gate_passed": row["gate_passed"],
        "created_at": row["created_at"],
        "result": row["result"].model_dump(),
        "reused_existing": reused,
    }


# --------------------------------------------------------------------------- #
# Demo seed (idempotent): exercises every run state incl. empty & backfill
# --------------------------------------------------------------------------- #
@app.post("/api/demo/seed", status_code=201)
def seed_demo() -> dict:
    from .demo import seed_all

    return seed_all(storage)


# --------------------------------------------------------------------------- #
# Static UI
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/index.html")


app.mount("/ui", StaticFiles(directory=str(STATIC_DIR)), name="ui")
