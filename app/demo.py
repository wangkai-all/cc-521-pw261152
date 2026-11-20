"""Idempotent demo data.

Seeds two calibration versions (v1 -> v2, exercising weight/threshold/window
versioning) and batches that hit every engine state: scored pass, scored gate
failure, rule conflict, insufficient samples, empty batch, fully-exempt batch
and backfill (both same-version idempotency and cross-version re-score).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .domain.models import (
    BatchSubmission,
    CalibrationConfig,
    ConflictTolerance,
    ExemptionWindow,
    GatePolicy,
    RuleConfig,
    RuleObservation,
    SampleInput,
    VersionCreate,
)
from .domain.scoring import score_batch
from .storage import Storage

DATASET = "orders_dq"
T0 = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


def _obs(rule_id: str, passed: bool, ts: datetime, **evidence) -> RuleObservation:
    return RuleObservation(rule_id=rule_id, passed=passed, observed_at=ts, evidence=evidence)


def _sample(sid: str, ts: datetime, verdicts: dict[str, bool], **meta) -> SampleInput:
    return SampleInput(
        sample_id=sid,
        observed_at=ts,
        observations=tuple(
            _obs(rid, ok, ts, row=i, value=sid) for i, (rid, ok) in enumerate(verdicts.items())
        ),
        metadata=meta,
    )


def _v1() -> CalibrationConfig:
    return CalibrationConfig(
        min_batch_samples=3,
        rules=(
            RuleConfig(rule_id="null_check", name="Null rate check", weight=0.4,
                       pass_threshold=0.9, min_samples=0),
            RuleConfig(rule_id="uniqueness", name="Uniqueness check", weight=0.35,
                       pass_threshold=0.95, min_samples=2),
            RuleConfig(rule_id="completeness", name="Completeness check", weight=0.25,
                       pass_threshold=0.8, min_samples=2),
        ),
        conflict_groups=(("null_check", "completeness"),),
        conflict_tolerance=(ConflictTolerance(group_id="g0", max_conflict_rate=0.3),),
        exemption_windows=(
            ExemptionWindow(
                window_id="w1", rule_id="null_check", dataset=DATASET,
                start_at=T0, end_at=T0 + timedelta(days=2),
                reason="upstream schema migration in progress",
            ),
            ExemptionWindow(
                window_id="w1u", rule_id="uniqueness", dataset=DATASET,
                start_at=T0, end_at=T0 + timedelta(days=2),
                reason="upstream schema migration in progress",
            ),
            ExemptionWindow(
                window_id="w1c", rule_id="completeness", dataset=DATASET,
                start_at=T0, end_at=T0 + timedelta(days=2),
                reason="upstream schema migration in progress",
            ),
        ),
        gate_policy=GatePolicy(
            min_overall_score=0.8,
            block_on_rule_failure=True,
            block_on_alert=True,
            alert_rule_ids=("null_check",),
            alert_severity="critical",
        ),
        notes="initial calibration",
    )


def _v2() -> CalibrationConfig:
    return CalibrationConfig(
        min_batch_samples=4,
        rules=(
            RuleConfig(rule_id="null_check", name="Null rate check", weight=0.5,
                       pass_threshold=0.95, min_samples=3),
            RuleConfig(rule_id="uniqueness", name="Uniqueness check", weight=0.3,
                       pass_threshold=0.95, min_samples=2),
            RuleConfig(rule_id="completeness", name="Completeness check", weight=0.2,
                       pass_threshold=0.85, min_samples=2),
        ),
        conflict_groups=(("null_check", "completeness"),),
        conflict_tolerance=(ConflictTolerance(group_id="g0", max_conflict_rate=0.3),),
        exemption_windows=(
            ExemptionWindow(
                window_id="w1", rule_id="null_check", dataset=DATASET,
                start_at=T0, end_at=T0 + timedelta(days=3),
                reason="upstream schema migration in progress (extended)",
            ),
            ExemptionWindow(
                window_id="w1u", rule_id="uniqueness", dataset=DATASET,
                start_at=T0, end_at=T0 + timedelta(days=3),
                reason="upstream schema migration in progress (extended)",
            ),
            ExemptionWindow(
                window_id="w1c", rule_id="completeness", dataset=DATASET,
                start_at=T0, end_at=T0 + timedelta(days=3),
                reason="upstream schema migration in progress (extended)",
            ),
        ),
        gate_policy=GatePolicy(
            min_overall_score=0.85,
            block_on_rule_failure=True,
            block_on_alert=True,
            alert_rule_ids=("null_check",),
            alert_severity="critical",
        ),
        notes="tightened null/completeness thresholds after incident review",
    )


def _batches() -> list[BatchSubmission]:
    # b1: healthy scored run, gate passes
    b1 = BatchSubmission(
        dataset=DATASET, batch_id="b1-healthy", observed_at=T0 + timedelta(days=4),
        samples=tuple(
            _sample(f"s{i}", T0 + timedelta(days=4, hours=i),
                    {"null_check": True, "uniqueness": True, "completeness": True})
            for i in range(5)
        ),
    )

    # b2: one null failure; null_check (0.8) < its 0.9 threshold -> weighted
    # score still computed, rule-failure alert fires, gate blocks.
    # completeness stays passing on every sample so the conflict group stays
    # silent (no disagreeing shared sample).
    b2 = BatchSubmission(
        dataset=DATASET, batch_id="b2-rule-fail", observed_at=T0 + timedelta(days=5),
        samples=(
            _sample("s1", T0 + timedelta(days=5), {"null_check": False, "uniqueness": True, "completeness": True}),
            *(
                _sample(f"s{i}", T0 + timedelta(days=5, hours=i),
                        {"null_check": True, "uniqueness": True, "completeness": True})
                for i in range(2, 6)
            ),
        ),
    )

    # b3: null vs completeness disagree on 3/4 shared samples -> rule_conflict
    b3 = BatchSubmission(
        dataset=DATASET, batch_id="b3-conflict", observed_at=T0 + timedelta(days=6),
        samples=tuple(
            _sample(
                f"s{i}", T0 + timedelta(days=6, hours=i),
                {"null_check": i % 2 == 0, "uniqueness": True, "completeness": i % 2 == 1},
            )
            for i in range(4)
        ),
    )

    # b4: only 2 samples, min_batch_samples=3 -> insufficient
    b4 = BatchSubmission(
        dataset=DATASET, batch_id="b4-small", observed_at=T0 + timedelta(days=7),
        samples=tuple(
            _sample(f"s{i}", T0 + timedelta(days=7, hours=i),
                    {"null_check": True, "uniqueness": True, "completeness": True})
            for i in range(2)
        ),
    )

    # b5: empty batch -> empty state, no score, blocked gate
    b5 = BatchSubmission(
        dataset=DATASET, batch_id="b5-empty", observed_at=T0 + timedelta(days=8), samples=(),
    )

    # b6: every observation inside exemption window w1 -> zero evaluated
    # evidence -> empty state (no fake score), with exemption reasons kept.
    # null_check is configured with min_samples=0 so the absence is an
    # explicit, tolerated exemption rather than an insufficient-evidence flag.
    exempt_ts = T0 + timedelta(hours=6)
    b6 = BatchSubmission(
        dataset=DATASET, batch_id="b6-all-exempt", observed_at=exempt_ts,
        samples=tuple(
            _sample(f"s{i}", exempt_ts + timedelta(minutes=i),
                    {"null_check": True, "uniqueness": True, "completeness": True})
            for i in range(4)
        ),
    )

    return [b1, b2, b3, b4, b5, b6]


def seed_all(storage: Storage) -> dict:
    created: dict[str, list[str]] = {"versions": [], "runs": [], "reused": []}

    if storage.get_active_version() is None:
        v1_row = storage.create_version("v1.0", _v1(), "demo")
        storage.activate_version("v1.0")
        created["versions"].append(v1_row["version_id"])
        if not any(v["version_id"] == "v2.0" for v in storage.list_versions()):
            storage.create_version("v2.0", _v2(), "demo")
            created["versions"].append("v2.0")

    v1 = storage.get_version("v1.0")
    for submission in _batches():
        existing = storage.find_run("v1.0", submission.dataset, submission.batch_id)
        if existing:
            created["reused"].append(existing["run_id"])
            continue
        result = score_batch(v1["config"], submission)
        row = storage.insert_run(v1, submission, result)
        created["runs"].append(row["run_id"])

    # Backfill A: same (version, dataset, batch) -> idempotent, stored run reused
    healthy = next(b for b in _batches() if b.batch_id == "b1-healthy")
    backfilled = BatchSubmission(
        dataset=healthy.dataset, batch_id=healthy.batch_id,
        observed_at=healthy.observed_at, samples=healthy.samples, backfill=True,
    )
    existing = storage.find_run("v1.0", backfilled.dataset, backfilled.batch_id)
    created["reused"].append(existing["run_id"])

    # Backfill B: re-score b2 under v2.0 -> NEW immutable run pinned to v2,
    # while the original v1 run (and its gate result) stay untouched.
    v2 = storage.get_version("v2.0")
    b2 = next(b for b in _batches() if b.batch_id == "b2-rule-fail")
    rescored = BatchSubmission(
        dataset=b2.dataset, batch_id=b2.batch_id, observed_at=b2.observed_at,
        samples=b2.samples, backfill=True,
        rescore_of_run_id=storage.list_batch_runs(b2.dataset, b2.batch_id)[0]["run_id"],
    )
    if not storage.find_run("v2.0", rescored.dataset, rescored.batch_id):
        result = score_batch(v2["config"], rescored)
        row = storage.insert_run(v2, rescored, result)
        created["runs"].append(row["run_id"])

    created["versions_seeded"] = [v["version_id"] for v in storage.list_versions()]
    return created
