"""Unit tests for the pure scoring engine: all four states, exemption
windows, conflict tolerance, weighted math and the no-false-precision rule."""
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.models import (
    BatchSubmission,
    CalibrationConfig,
    ConflictTolerance,
    ExemptionWindow,
    GatePolicy,
    RuleConfig,
    RuleObservation,
    RunState,
    SampleInput,
)
from app.domain.scoring import score_batch

T0 = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


def make_config(**overrides) -> CalibrationConfig:
    defaults = dict(
        min_batch_samples=2,
        rules=(
            RuleConfig(rule_id="r1", name="R1", weight=0.5, pass_threshold=0.8, min_samples=1),
            RuleConfig(rule_id="r2", name="R2", weight=0.5, pass_threshold=0.8, min_samples=1),
        ),
        conflict_groups=(),
        conflict_tolerance=(),
        exemption_windows=(),
        gate_policy=GatePolicy(min_overall_score=0.8),
    )
    defaults.update(overrides)
    return CalibrationConfig(**defaults)


def obs(rule_id, passed, ts=T0, **evidence):
    return RuleObservation(rule_id=rule_id, passed=passed, observed_at=ts, evidence=evidence)


def sample(sid, verdicts, ts=T0):
    return SampleInput(
        sample_id=sid, observed_at=ts,
        observations=tuple(obs(rid, ok, ts) for rid, ok in verdicts.items()),
        metadata={"row": sid},
    )


def batch(samples, dataset="ds", batch_id="b1", observed_at=T0):
    return BatchSubmission(dataset=dataset, batch_id=batch_id, observed_at=observed_at, samples=samples)


# --------------------------------------------------------------------------- #
# happy path math
# --------------------------------------------------------------------------- #
def test_weighted_score_and_breakdown():
    cfg = make_config()
    sub = batch([
        sample("s1", {"r1": True, "r2": True}),
        sample("s2", {"r1": False, "r2": True}),
        sample("s3", {"r1": True, "r2": True}),
    ])
    result = score_batch(cfg, sub)

    assert result.state is RunState.scored
    # r1 = 2/3, r2 = 3/3, equal weights -> 5/6
    assert result.overall_score == round(5 / 6, 4)
    by_id = {r.rule_id: r for r in result.rules}
    assert by_id["r1"].rule_score == round(2 / 3, 4)
    assert by_id["r2"].rule_score == 1.0
    assert by_id["r1"].normalized_weight == 0.5
    # sample-level evidence preserved
    assert len(result.sample_results) == 6
    s1r1 = next(v for v in result.sample_results if v.sample_id == "s1" and v.rule_id == "r1")
    assert s1r1.passed and s1r1.evaluated and not s1r1.exempt
    assert s1r1.metadata == {"row": "s1"}
    # r1 (0.667) below its 0.8 threshold -> rule failure alert, gate blocks
    assert by_id["r1"].passed_threshold is False
    assert result.gate.passed is False
    assert any(a.alert_type == "rule_failure" and a.rule_id == "r1" for a in result.alerts)


def test_gate_pass_when_all_above_thresholds():
    cfg = make_config()
    sub = batch([sample("s1", {"r1": True, "r2": True}), sample("s2", {"r1": True, "r2": True})])
    result = score_batch(cfg, sub)
    assert result.state is RunState.scored
    assert result.overall_score == 1.0
    assert result.gate.passed is True
    assert result.alerts == ()


def test_weight_normalization_is_relative_not_absolute():
    cfg = make_config(min_batch_samples=1, rules=(
        RuleConfig(rule_id="r1", name="R1", weight=3.0, pass_threshold=0.5, min_samples=1),
        RuleConfig(rule_id="r2", name="R2", weight=1.0, pass_threshold=0.5, min_samples=1),
    ))
    sub = batch([sample("s1", {"r1": True, "r2": False})])
    result = score_batch(cfg, sub)
    assert result.state is RunState.scored
    # 0.75*1 + 0.25*0
    assert result.overall_score == 0.75
    weights = {r.rule_id: r.normalized_weight for r in result.rules}
    assert weights == {"r1": 0.75, "r2": 0.25}


# --------------------------------------------------------------------------- #
# empty batches — no false precision
# --------------------------------------------------------------------------- #
def test_empty_batch_has_no_score_and_blocks_gate():
    cfg = make_config()
    result = score_batch(cfg, batch(()))
    assert result.state is RunState.empty
    assert result.overall_score is None
    assert all(r.rule_score is None for r in result.rules)
    assert result.gate.passed is False
    assert any(a.alert_type == "empty" for a in result.alerts)
    assert result.sample_results == ()


def test_fully_exempt_batch_is_empty_state_not_a_fake_score():
    window = ExemptionWindow(
        window_id="w1", rule_id="r1", dataset="ds",
        start_at=T0 - timedelta(hours=1), end_at=T0 + timedelta(hours=1),
        reason="migration",
    )
    # r1 tolerates zero evaluated samples (min_samples=0); the run is still
    # scored honestly off r2's evidence rather than inventing r1 points.
    cfg = make_config(rules=(
        RuleConfig(rule_id="r1", name="R1", weight=0.5, pass_threshold=0.8, min_samples=0),
        RuleConfig(rule_id="r2", name="R2", weight=0.5, pass_threshold=0.8, min_samples=1),
    ), exemption_windows=(window,))
    sub = batch([sample("s1", {"r1": True, "r2": True}), sample("s2", {"r1": True, "r2": False})])
    result = score_batch(cfg, sub)
    assert result.state is RunState.scored
    views_r1 = [v for v in result.sample_results if v.rule_id == "r1"]
    assert all(v.exempt and v.exemption.window_id == "w1" for v in views_r1)
    by_id = {r.rule_id: r for r in result.rules}
    assert by_id["r1"].evaluated_samples == 0 and by_id["r1"].rule_score is None
    assert by_id["r2"].rule_score == 0.5
    # r1's weight is renormalized away; overall score equals r2's score
    assert result.overall_score == 0.5

    # with the default min_samples>=1 rule, an exempted rule means the run is
    # indeterminate (insufficient) rather than silently scored
    result_default = score_batch(make_config(exemption_windows=(window,)), sub)
    assert result_default.state is RunState.insufficient_samples
    assert result_default.overall_score is None

    # exempting every rule -> zero evaluated evidence -> empty, no score
    cfg2 = make_config(exemption_windows=(
        window,
        ExemptionWindow(window_id="w2", rule_id="r2", dataset="ds",
                        start_at=T0 - timedelta(hours=1), end_at=T0 + timedelta(hours=1)),
    ))
    result2 = score_batch(cfg2, sub)
    assert result2.state is RunState.empty
    assert result2.overall_score is None
    assert "exemption" in result2.state_reason


def test_exemption_window_boundaries_half_open():
    cfg = make_config(exemption_windows=(
        ExemptionWindow(window_id="w1", rule_id="r1", dataset="ds",
                        start_at=T0, end_at=T0 + timedelta(hours=1)),
    ))
    at_start = batch([sample("s1", {"r1": True, "r2": True}, ts=T0)])
    at_end = batch([sample("s1", {"r1": True, "r2": True}, ts=T0 + timedelta(hours=1))])
    assert score_batch(cfg, at_start).sample_results[0].exempt is True
    assert score_batch(cfg, at_end).sample_results[0].exempt is False


def test_exemption_uses_observation_timestamp_over_sample_timestamp():
    cfg = make_config(exemption_windows=(
        ExemptionWindow(window_id="w1", rule_id="r1", dataset="ds",
                        start_at=T0 + timedelta(hours=2), end_at=T0 + timedelta(hours=4)),
    ))
    late_obs = obs("r1", True, ts=T0 + timedelta(hours=3))
    sub = batch([SampleInput(sample_id="s1", observed_at=T0, observations=(late_obs, obs("r2", True, T0)))])
    r1_view = next(v for v in score_batch(cfg, sub).sample_results if v.rule_id == "r1")
    assert r1_view.exempt is True


# --------------------------------------------------------------------------- #
# insufficient samples — no false precision
# --------------------------------------------------------------------------- #
def test_insufficient_rule_samples_blocks_and_withholds_score():
    cfg = make_config(rules=(
        RuleConfig(rule_id="r1", name="R1", weight=0.5, pass_threshold=0.8, min_samples=5),
        RuleConfig(rule_id="r2", name="R2", weight=0.5, pass_threshold=0.8, min_samples=1),
    ))
    sub = batch([sample("s1", {"r1": True, "r2": True}), sample("s2", {"r1": True, "r2": True})])
    result = score_batch(cfg, sub)
    assert result.state is RunState.insufficient_samples
    assert result.overall_score is None
    by_id = {r.rule_id: r for r in result.rules}
    assert by_id["r1"].insufficient and by_id["r1"].rule_score is None
    assert by_id["r2"].rule_score is None  # whole run is indeterminate
    assert result.gate.passed is False
    assert any(a.alert_type == "insufficient" and a.rule_id == "r1" for a in result.alerts)


def test_insufficient_batch_size():
    cfg = make_config(min_batch_samples=10)
    sub = batch([sample("s1", {"r1": True, "r2": True})])
    result = score_batch(cfg, sub)
    assert result.state is RunState.insufficient_samples
    assert "min_batch_samples" in result.state_reason
    assert result.overall_score is None


# --------------------------------------------------------------------------- #
# rule conflicts — no false precision
# --------------------------------------------------------------------------- #
def _conflict_config(tolerance=0.1):
    return make_config(
        min_batch_samples=1,
        conflict_groups=(("r1", "r2"),),
        conflict_tolerance=(ConflictTolerance(group_id="g0", max_conflict_rate=tolerance),),
    )


def test_rule_conflict_withholds_score_and_blocks():
    cfg = _conflict_config()
    sub = batch([
        sample("s1", {"r1": True, "r2": False}),   # disagree
        sample("s2", {"r1": True, "r2": True}),
        sample("s3", {"r1": False, "r2": True}),   # disagree -> 2/3 > 0.1
    ])
    result = score_batch(cfg, sub)
    assert result.state is RunState.rule_conflict
    assert result.overall_score is None
    detail = result.conflicts[0]
    assert detail.shared_samples == 3 and detail.conflicting_samples == 2
    assert detail.exceeded is True
    assert result.gate.passed is False
    assert any(a.alert_type == "conflict" for a in result.alerts)


def test_conflict_within_tolerance_still_scores():
    cfg = _conflict_config(tolerance=0.5)
    sub = batch([
        sample("s1", {"r1": True, "r2": False}),  # 1/3 conflict <= 0.5
        sample("s2", {"r1": True, "r2": True}),
        sample("s3", {"r1": True, "r2": True}),
    ])
    result = score_batch(cfg, sub)
    assert result.state is RunState.scored
    assert result.overall_score is not None
    assert result.conflicts[0].exceeded is False


def test_conflict_ignores_exempted_verdicts():
    cfg = make_config(
        min_batch_samples=1,
        conflict_groups=(("r1", "r2"),),
        conflict_tolerance=(ConflictTolerance(group_id="g0", max_conflict_rate=0.0),),
        exemption_windows=(
            ExemptionWindow(window_id="w1", rule_id="r2", dataset="ds",
                            start_at=T0 - timedelta(hours=1), end_at=T0 + timedelta(hours=1)),
        ),
    )
    # r2 verdicts are all exempted -> no shared verdicts -> no conflict
    sub = batch([sample("s1", {"r1": True, "r2": False}), sample("s2", {"r1": False, "r2": True})])
    result = score_batch(cfg, sub)
    assert result.conflicts[0].shared_samples == 0
    assert result.state is not RunState.rule_conflict


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def test_unknown_rule_rejected():
    with pytest.raises(ValueError, match="unknown rule_id"):
        score_batch(make_config(), batch([sample("s1", {"r1": True, "r3": True})]))


def test_duplicate_sample_and_duplicate_observation_rejected():
    with pytest.raises(ValueError, match="duplicate sample_id"):
        score_batch(make_config(), batch([
            sample("s1", {"r1": True, "r2": True}),
            sample("s1", {"r1": False, "r2": False}),
        ]))
    bad = BatchSubmission(
        dataset="ds", batch_id="b", observed_at=T0,
        samples=(SampleInput(sample_id="s1", observed_at=T0, observations=(
            obs("r1", True), obs("r1", False),
        )),),
    )
    with pytest.raises(ValueError, match="duplicate observation"):
        score_batch(make_config(), bad)


def test_invalid_config_rejected():
    with pytest.raises(Exception):
        CalibrationConfig(
            min_batch_samples=1,
            rules=(RuleConfig(rule_id="r1", name="R1", weight=0.5, pass_threshold=0.8, min_samples=1),),
            conflict_groups=(("r1", "r9"),),  # r9 unknown
            gate_policy=GatePolicy(),
        )
    with pytest.raises(Exception):
        ExemptionWindow(window_id="w", rule_id="r1", dataset="ds",
                        start_at=T0 + timedelta(hours=2), end_at=T0)  # reversed range
