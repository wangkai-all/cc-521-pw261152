"""Pydantic models: calibration configs (versioned) and scoring I/O.

All datetimes are normalized to timezone-aware UTC at parse time so that
exemption-window comparison and checksum hashing are deterministic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Calibration configuration (the versioned artifact)
# --------------------------------------------------------------------------- #
class ExemptionWindow(BaseModel):
    """Time interval (inclusive start, exclusive end) during which a rule is
    exempted for a dataset / rule combination."""

    model_config = ConfigDict(frozen=True)

    window_id: str = Field(..., min_length=1)
    rule_id: str = Field(..., min_length=1)
    dataset: str = Field(..., min_length=1)
    start_at: datetime
    end_at: datetime
    reason: str = ""

    @field_validator("start_at", "end_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @model_validator(mode="after")
    def _check_range(self) -> "ExemptionWindow":
        if self.end_at <= self.start_at:
            raise ValueError(
                f"exemption window {self.window_id}: end_at must be after start_at"
            )
        return self


class RuleConfig(BaseModel):
    """One quality rule with its calibration weight and per-rule thresholds."""

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    pass_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="rule score below this is a rule failure"
    )
    min_samples: int = Field(
        ..., ge=0, description="rule-level minimum evaluated sample count"
    )
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "RuleConfig":
        if not (0.0 < self.weight <= 1000.0):
            raise ValueError(f"rule {self.rule_id}: weight out of sane range")
        return self


class ConflictTolerance(BaseModel):
    """Acceptable disagreement rate inside one static conflict group."""

    model_config = ConfigDict(frozen=True)

    group_id: str = Field(..., min_length=1)
    max_conflict_rate: float = Field(..., ge=0.0, le=1.0)


class GatePolicy(BaseModel):
    """Release-gate policy embedded in a calibration version."""

    model_config = ConfigDict(frozen=True)

    min_overall_score: float = Field(..., ge=0.0, le=1.0)
    block_on_rule_failure: bool = True
    block_on_alert: bool = True
    alert_rule_ids: tuple[str, ...] = ()
    alert_severity: Literal["info", "warning", "critical"] = "critical"


class CalibrationConfig(BaseModel):
    """The full, versioned calibration artifact."""

    model_config = ConfigDict(frozen=True)

    min_batch_samples: int = Field(..., ge=0)
    rules: tuple[RuleConfig, ...] = Field(..., min_length=1)
    conflict_groups: tuple[tuple[str, ...], ...] = ()
    conflict_tolerance: tuple[ConflictTolerance, ...] = ()
    exemption_windows: tuple[ExemptionWindow, ...] = ()
    gate_policy: GatePolicy
    notes: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "CalibrationConfig":
        errs: list[str] = []
        ids = [r.rule_id for r in self.rules]
        if len(set(ids)) != len(ids):
            errs.append("duplicate rule_id in rules")

        id_set = set(ids)
        for group in self.conflict_groups:
            if len(group) < 2:
                errs.append(f"conflict group {group!r} needs at least 2 rules")
            if len(set(group)) != len(group):
                errs.append(f"conflict group {group!r} has duplicate rules")
            unknown = [r for r in group if r not in id_set]
            if unknown:
                errs.append(f"conflict group references unknown rules: {unknown}")

        group_ids = {f"g{i}" for i in range(len(self.conflict_groups))}
        tol_ids = {t.group_id for t in self.conflict_tolerance}
        # tolerance may also reference a declared group by its index id g0/g1...
        for t in self.conflict_tolerance:
            if t.group_id not in group_ids:
                errs.append(
                    f"conflict_tolerance references unknown group_id {t.group_id!r} "
                    f"(groups are addressed as g0, g1, ...)"
                )

        seen_windows: set[str] = set()
        for w in self.exemption_windows:
            if w.window_id in seen_windows:
                errs.append(f"duplicate exemption window id {w.window_id}")
            seen_windows.add(w.window_id)
            if w.rule_id not in id_set:
                errs.append(
                    f"exemption window {w.window_id} references unknown rule {w.rule_id}"
                )

        unknown_alert = [r for r in self.gate_policy.alert_rule_ids if r not in id_set]
        if unknown_alert:
            errs.append(
                f"gate_policy.alert_rule_ids references unknown rules: {unknown_alert}"
            )

        if errs:
            raise ValueError("; ".join(errs))
        return self

    def rule_index(self) -> dict[str, RuleConfig]:
        return {r.rule_id: r for r in self.rules}

    def total_weight(self) -> float:
        return sum(r.weight for r in self.rules)


class VersionStatus(str, Enum):
    draft = "draft"
    active = "active"
    archived = "archived"


class VersionCreate(BaseModel):
    version_id: str = Field(..., min_length=1, pattern=r"^[A-Za-z0-9._-]{1,64}$")
    config: CalibrationConfig
    created_by: str = "anonymous"


# --------------------------------------------------------------------------- #
# Scoring inputs
# --------------------------------------------------------------------------- #
class RuleObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(..., min_length=1)
    passed: bool
    observed_at: datetime | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        return _ensure_utc(v)


class SampleInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(..., min_length=1)
    observed_at: datetime
    observations: tuple[RuleObservation, ...] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


class BatchSubmission(BaseModel):
    dataset: str = Field(..., min_length=1)
    batch_id: str = Field(..., min_length=1)
    observed_at: datetime
    samples: tuple[SampleInput, ...] = ()
    backfill: bool = False
    rescore_of_run_id: str | None = None

    @field_validator("observed_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @model_validator(mode="after")
    def _check(self) -> "BatchSubmission":
        if self.rescore_of_run_id and not self.backfill:
            raise ValueError("rescore_of_run_id requires backfill=true")
        return self


# --------------------------------------------------------------------------- #
# Scoring outputs (also persisted as the run evidence snapshot)
# --------------------------------------------------------------------------- #
class RunState(str, Enum):
    scored = "scored"
    empty = "empty"
    insufficient_samples = "insufficient_samples"
    rule_conflict = "rule_conflict"


class ExemptMark(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_id: str
    reason: str


class SampleResultView(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    observed_at: datetime
    rule_id: str
    passed: bool
    exempt: bool
    exemption: ExemptMark | None
    evaluated: bool
    evidence: dict[str, Any]
    metadata: dict[str, Any]


class RuleResultView(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    name: str
    weight: float
    normalized_weight: float
    pass_threshold: float
    min_samples: int
    evaluated_samples: int
    passed_samples: int
    failed_samples: int
    exempted_samples: int
    rule_score: float | None
    passed_threshold: bool | None
    insufficient: bool
    in_conflict_group: bool


class ConflictDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    group_id: str
    rules: tuple[str, ...]
    shared_samples: int
    conflicting_samples: int
    conflict_rate: float
    max_conflict_rate: float
    exceeded: bool


class GateResultView(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    reasons: tuple[str, ...]
    min_overall_score: float
    block_on_rule_failure: bool
    block_on_alert: bool
    alert_rule_ids: tuple[str, ...]
    alert_severity: str


class AlertView(BaseModel):
    model_config = ConfigDict(frozen=True)

    alert_type: Literal["rule_failure", "gate_score", "insufficient", "conflict", "empty"]
    severity: Literal["info", "warning", "critical"]
    rule_id: str | None
    message: str


class BatchScoreResult(BaseModel):
    state: RunState
    state_reason: str
    overall_score: float | None
    rules: tuple[RuleResultView, ...]
    sample_results: tuple[SampleResultView, ...]
    conflicts: tuple[ConflictDetail, ...]
    gate: GateResultView
    alerts: tuple[AlertView, ...]
    total_samples: int
    evaluated_samples: int
    exempted_samples: int


# --------------------------------------------------------------------------- #
# Version diff view
# --------------------------------------------------------------------------- #
class VersionDiff(BaseModel):
    from_version: str
    to_version: str
    weights: dict[str, dict[str, float | None]]
    thresholds: dict[str, dict[str, float | None]]
    min_samples: dict[str, dict[str, int | None]]
    min_batch_samples: dict[str, int]
    exemption_windows: dict[str, list[str]]
    conflict_groups: dict[str, list[list[str]] | None]
    gate_policy: dict[str, Any]
    added_rules: list[str]
    removed_rules: list[str]
