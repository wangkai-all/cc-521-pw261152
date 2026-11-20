"""Pure scoring engine.

No I/O here: given a frozen calibration config and a batch of samples it
produces a :class:`BatchScoreResult`. The engine distinguishes four run
states and only emits a numeric overall score in the ``scored`` state;
in every other state ``overall_score`` is ``None`` so callers can never
report a falsely precise number.
"""
from __future__ import annotations

from collections import defaultdict

from .models import (
    AlertView,
    BatchScoreResult,
    BatchSubmission,
    CalibrationConfig,
    ConflictDetail,
    ExemptMark,
    GateResultView,
    RuleResultView,
    RunState,
    SampleInput,
    SampleResultView,
)

_ROUND = 4


def _round(x: float) -> float:
    return round(x, _ROUND)


def _matching_exemption(config: CalibrationConfig, dataset: str, rule_id: str, when) -> ExemptMark | None:
    for w in config.exemption_windows:
        if w.rule_id == rule_id and w.dataset == dataset and w.start_at <= when < w.end_at:
            return ExemptMark(window_id=w.window_id, reason=w.reason)
    return None


def _validate_inputs(config: CalibrationConfig, submission: BatchSubmission) -> None:
    known = set(config.rule_index())
    seen_samples: set[str] = set()
    for sample in submission.samples:
        if sample.sample_id in seen_samples:
            raise ValueError(f"duplicate sample_id in batch: {sample.sample_id}")
        seen_samples.add(sample.sample_id)
        seen_rules: set[str] = set()
        for obs in sample.observations:
            if obs.rule_id not in known:
                raise ValueError(f"unknown rule_id {obs.rule_id!r} (not in calibration config)")
            if obs.rule_id in seen_rules:
                raise ValueError(
                    f"sample {sample.sample_id}: duplicate observation for rule {obs.rule_id}"
                )
            seen_rules.add(obs.rule_id)


def _build_sample_views(
    config: CalibrationConfig, submission: BatchSubmission
) -> list[SampleResultView]:
    views: list[SampleResultView] = []
    for sample in submission.samples:
        for obs in sample.observations:
            when = obs.observed_at or sample.observed_at
            exemption = _matching_exemption(config, submission.dataset, obs.rule_id, when)
            views.append(
                SampleResultView(
                    sample_id=sample.sample_id,
                    observed_at=when,
                    rule_id=obs.rule_id,
                    passed=obs.passed,
                    exempt=exemption is not None,
                    exemption=exemption,
                    evaluated=exemption is None,
                    evidence=obs.evidence,
                    metadata=sample.metadata,
                )
            )
    return views


def _aggregate_rules(
    config: CalibrationConfig, views: list[SampleResultView]
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {
        r.rule_id: {"evaluated": 0, "passed": 0, "failed": 0, "exempted": 0}
        for r in config.rules
    }
    for v in views:
        bucket = stats[v.rule_id]
        if v.exempt:
            bucket["exempted"] += 1
        else:
            bucket["evaluated"] += 1
            if v.passed:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
    return stats


def _evaluate_conflicts(
    config: CalibrationConfig, submission: BatchSubmission
) -> list[ConflictDetail]:
    tolerance = {f"g{i}": t.max_conflict_rate for i, t in enumerate(config.conflict_tolerance)}
    details: list[ConflictDetail] = []

    for idx, group in enumerate(config.conflict_groups):
        group_id = f"g{idx}"
        group_set = set(group)
        shared = conflicting = 0
        for sample in submission.samples:
            verdicts: list[bool] = []
            for obs in sample.observations:
                if obs.rule_id not in group_set:
                    continue
                when = obs.observed_at or sample.observed_at
                if _matching_exemption(config, submission.dataset, obs.rule_id, when):
                    continue  # exempted observations carry no verdict
                verdicts.append(obs.passed)
            if len(verdicts) >= 2:
                shared += 1
                if any(verdicts) and not all(verdicts):
                    conflicting += 1
        rate = conflicting / shared if shared else 0.0
        max_rate = tolerance.get(group_id, 0.0)
        details.append(
            ConflictDetail(
                group_id=group_id,
                rules=tuple(group),
                shared_samples=shared,
                conflicting_samples=conflicting,
                conflict_rate=_round(rate),
                max_conflict_rate=max_rate,
                exceeded=shared > 0 and rate > max_rate,
            )
        )
    return details


def _rule_views(
    config: CalibrationConfig,
    stats: dict[str, dict[str, int]],
    scored: bool,
    conflict_rule_ids: set[str],
    raw_scores: dict[str, float | None],
) -> tuple[RuleResultView, ...]:
    total_weight = config.total_weight()
    out: list[RuleResultView] = []
    for rule in config.rules:
        bucket = stats[rule.rule_id]
        evaluated = bucket["evaluated"]
        # min_samples=0 explicitly allows zero evaluated samples (e.g. a rule
        # covered by an exemption window); its weight is then renormalized
        # away instead of being scored as a zero.
        insufficient = evaluated < rule.min_samples
        raw = raw_scores[rule.rule_id]
        if scored and raw is not None:
            rule_score = _round(raw)
            # threshold compared at full precision, not on the rounded display value
            passed_threshold = raw >= rule.pass_threshold
        else:
            rule_score = None
            passed_threshold = None
        out.append(
            RuleResultView(
                rule_id=rule.rule_id,
                name=rule.name,
                weight=rule.weight,
                normalized_weight=_round(rule.weight / total_weight),
                pass_threshold=rule.pass_threshold,
                min_samples=rule.min_samples,
                evaluated_samples=evaluated,
                passed_samples=bucket["passed"],
                failed_samples=bucket["failed"],
                exempted_samples=bucket["exempted"],
                rule_score=rule_score,
                passed_threshold=passed_threshold,
                insufficient=insufficient,
                in_conflict_group=rule.rule_id in conflict_rule_ids,
            )
        )
    return tuple(out)


def _raw_rule_scores(
    config: CalibrationConfig, stats: dict[str, dict[str, int]]
) -> dict[str, float | None]:
    """Full-precision per-rule scores used for overall aggregation; only the
    final displayed numbers are rounded, so rounding never compounds."""
    raw: dict[str, float | None] = {}
    for rule in config.rules:
        evaluated = stats[rule.rule_id]["evaluated"]
        if evaluated >= rule.min_samples and evaluated > 0:
            raw[rule.rule_id] = stats[rule.rule_id]["passed"] / evaluated
        else:
            raw[rule.rule_id] = None
    return raw


def _gate_and_alerts(
    config: CalibrationConfig,
    state: RunState,
    state_reason: str,
    rule_views: tuple[RuleResultView, ...],
    conflicts: tuple[ConflictDetail, ...],
    overall_score: float | None,
) -> tuple[GateResultView, tuple[AlertView, ...]]:
    gp = config.gate_policy
    reasons: list[str] = []
    alerts: list[AlertView] = []
    alert_rules = set(gp.alert_rule_ids)

    if state is not RunState.scored:
        # Indeterminate runs are never allowed through the gate: we cannot
        # prove quality, so we block instead of guessing.
        reasons.append(f"run state {state.value}: {state_reason}")
        if state is RunState.rule_conflict:
            for c in conflicts:
                if c.exceeded:
                    msg = (
                        f"conflict group {c.group_id} {list(c.rules)}: "
                        f"{c.conflicting_samples}/{c.shared_samples} shared samples disagree "
                        f"(rate {c.conflict_rate} > tolerance {c.max_conflict_rate})"
                    )
                    alerts.append(AlertView(alert_type="conflict", severity="critical", rule_id=None, message=msg))
        if state is RunState.insufficient_samples:
            for rv in rule_views:
                if rv.insufficient:
                    msg = (
                        f"rule {rv.rule_id}: only {rv.evaluated_samples} evaluated samples, "
                        f"needs >= {rv.min_samples}"
                    )
                    alerts.append(AlertView(alert_type="insufficient", severity="warning", rule_id=rv.rule_id, message=msg))
        if state is RunState.empty:
            alerts.append(
                AlertView(alert_type="empty", severity="warning", rule_id=None, message=state_reason)
            )
    else:
        assert overall_score is not None
        if overall_score < gp.min_overall_score:
            reasons.append(
                f"overall score {overall_score:.4f} below gate minimum {gp.min_overall_score:.4f}"
            )
            alerts.append(
                AlertView(
                    alert_type="gate_score",
                    severity="critical",
                    rule_id=None,
                    message=reasons[-1],
                )
            )
        for rv in rule_views:
            if rv.passed_threshold is False:
                sev = gp.alert_severity if rv.rule_id in alert_rules else "warning"
                msg = (
                    f"rule {rv.rule_id} score {rv.rule_score:.4f} "
                    f"below threshold {rv.pass_threshold:.4f}"
                )
                alerts.append(AlertView(alert_type="rule_failure", severity=sev, rule_id=rv.rule_id, message=msg))
                if gp.block_on_rule_failure:
                    reasons.append(msg)
            if gp.block_on_alert and rv.rule_id in alert_rules and rv.passed_threshold is False:
                reasons.append(f"alert rule {rv.rule_id} failed (block_on_alert)")

    gate = GateResultView(
        passed=not reasons,
        reasons=tuple(reasons),
        min_overall_score=gp.min_overall_score,
        block_on_rule_failure=gp.block_on_rule_failure,
        block_on_alert=gp.block_on_alert,
        alert_rule_ids=gp.alert_rule_ids,
        alert_severity=gp.alert_severity,
    )
    return gate, tuple(alerts)


def score_batch(config: CalibrationConfig, submission: BatchSubmission) -> BatchScoreResult:
    """Evaluate one batch against one immutable calibration config.

    State precedence: empty -> rule_conflict -> insufficient_samples -> scored.
    Only ``scored`` carries an ``overall_score``; every other state yields
    ``None`` plus explicit reasons and a blocking gate.
    """
    _validate_inputs(config, submission)
    views = _build_sample_views(config, submission)
    stats = _aggregate_rules(config, views)

    total_samples = len(submission.samples)
    evaluated_samples = sum(b["evaluated"] for b in stats.values())
    exempted_samples = sum(b["exempted"] for b in stats.values())
    conflicts = tuple(_evaluate_conflicts(config, submission))

    # ---- state 1: empty -------------------------------------------------- #
    if total_samples == 0:
        state, state_reason = RunState.empty, "batch contains no samples"
    elif evaluated_samples == 0:
        state, state_reason = RunState.empty, (
            f"all {total_samples} samples are covered by exemption windows; "
            "no evaluated evidence exists"
        )
    else:
        # ---- state 2: rule conflict ------------------------------------- #
        exceeded = [c for c in conflicts if c.exceeded]
        if exceeded:
            state = RunState.rule_conflict
            state_reason = "; ".join(
                f"group {c.group_id} conflict rate {c.conflict_rate} > {c.max_conflict_rate}"
                for c in exceeded
            )
        else:
            # ---- state 3: insufficient evidence -------------------------- #
            missing_rules = [
                r.rule_id
                for r in config.rules
                if stats[r.rule_id]["evaluated"] < r.min_samples
            ]
            batch_short = total_samples < config.min_batch_samples
            if missing_rules or batch_short:
                state = RunState.insufficient_samples
                parts = []
                if batch_short:
                    parts.append(
                        f"batch size {total_samples} < min_batch_samples {config.min_batch_samples}"
                    )
                if missing_rules:
                    parts.append(f"rules below sample minimum: {missing_rules}")
                state_reason = "; ".join(parts)
            else:
                state = RunState.scored
                state_reason = "all rules evaluated with sufficient samples"

    conflict_rule_ids = {rid for c in conflicts for rid in c.rules}
    scored = state is RunState.scored
    raw_scores = _raw_rule_scores(config, stats)
    rule_views = _rule_views(config, stats, scored, conflict_rule_ids, raw_scores)

    overall_score = None
    if scored:
        # Rules without evaluated samples (exempted, explicitly allowed via
        # min_samples=0) must not count as zero points — that would punish
        # missing evidence as failure. Renormalize weights over the rules
        # that actually have scores, all at full precision.
        weighted = sum(
            rule.weight * raw_scores[rule.rule_id]
            for rule in config.rules
            if raw_scores[rule.rule_id] is not None
        )
        active_weight = sum(
            rule.weight
            for rule in config.rules
            if raw_scores[rule.rule_id] is not None
        )
        overall_score = _round(weighted / active_weight)

    gate, alerts = _gate_and_alerts(
        config, state, state_reason, rule_views, conflicts, overall_score
    )

    return BatchScoreResult(
        state=state,
        state_reason=state_reason,
        overall_score=overall_score,
        rules=rule_views,
        sample_results=tuple(views),
        conflicts=conflicts,
        gate=gate,
        alerts=alerts,
        total_samples=total_samples,
        evaluated_samples=evaluated_samples,
        exempted_samples=exempted_samples,
    )
