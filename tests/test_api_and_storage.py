"""Storage + API end-to-end tests: version lifecycle, immutable history
pinned to versions/checksums, empty batch, backfill idempotency, re-score."""
from datetime import datetime, timezone

import pytest
from fastapi.encoders import jsonable_encoder

from app.domain.models import (
    BatchSubmission,
    CalibrationConfig,
    GatePolicy,
    RuleConfig,
    VersionCreate,
)
from app.domain.scoring import score_batch
from app.storage import checksum_config

T0 = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)


def payload(obj):
    """httpx's json= needs plain primitives, not pydantic models."""
    return jsonable_encoder(obj)


def config(threshold=0.8, min_batch=1):
    return CalibrationConfig(
        min_batch_samples=min_batch,
        rules=(
            RuleConfig(rule_id="r1", name="R1", weight=0.6, pass_threshold=threshold, min_samples=1),
            RuleConfig(rule_id="r2", name="R2", weight=0.4, pass_threshold=threshold, min_samples=1),
        ),
        gate_policy=GatePolicy(min_overall_score=0.8, alert_rule_ids=("r1",)),
    )


def submission(samples=(), batch_id="b1", backfill=False, rescore=None):
    return BatchSubmission(
        dataset="ds", batch_id=batch_id, observed_at=T0, samples=samples,
        backfill=backfill, rescore_of_run_id=rescore,
    )


def healthy_samples(n=3, fail_r1=0):
    from tests.test_scoring_engine import sample as _sample  # reuse builder
    out = []
    for i in range(n):
        out.append(_sample(f"s{i}", {"r1": i >= fail_r1, "r2": True}))
    return out


# --------------------------------------------------------------------------- #
# storage-level
# --------------------------------------------------------------------------- #
def test_version_lifecycle_and_checksum_uniqueness(tmp_db):
    c1, c2 = config(0.8), config(0.9)
    assert checksum_config(c1) != checksum_config(c2)
    tmp_db.create_version("v1", c1, "tester")
    tmp_db.activate_version("v1")
    assert tmp_db.get_active_version()["version_id"] == "v1"
    assert tmp_db.get_version("v1")["config_checksum"] == checksum_config(c1)

    tmp_db.create_version("v2", c2, "tester")
    tmp_db.activate_version("v2")
    # old version archived, never mutated
    assert tmp_db.get_version("v1")["status"].value == "archived"
    assert tmp_db.get_active_version()["version_id"] == "v2"
    assert tmp_db.get_version("v1")["config_checksum"] == checksum_config(c1)

    with pytest.raises(ValueError):
        tmp_db.create_version("v2", c2, "tester")  # duplicate id
    with pytest.raises(ValueError):
        tmp_db.activate_version("v1")  # archived cannot re-activate


def test_runs_are_immutable_and_pin_version(tmp_db):
    c1, c2 = config(0.8), config(0.95)
    v1 = tmp_db.create_version("v1", c1, "t")
    tmp_db.activate_version("v1")
    tmp_db.create_version("v2", c2, "t")
    samples = tuple(healthy_samples(4, fail_r1=1))

    run1 = tmp_db.insert_run(v1, submission(samples), score_batch(c1, submission(samples)))
    score_v1 = run1["overall_score"]

    # activate v2: the stored v1 run keeps its score and pinned checksum
    v2 = tmp_db.activate_version("v2")
    run1_reloaded = tmp_db.get_run(run1["run_id"])
    assert run1_reloaded["version_id"] == "v1"
    assert run1_reloaded["config_checksum"] == checksum_config(c1)
    assert run1_reloaded["overall_score"] == score_v1

    # re-scoring the same batch under v2 creates a separate immutable run
    rescore = submission(samples, backfill=True, rescore=run1["run_id"])
    run2 = tmp_db.insert_run(v2, rescore, score_batch(c2, rescore))
    assert run2["run_id"] != run1["run_id"]
    assert run2["version_id"] == "v2"
    assert run2["rescore_of_run_id"] == run1["run_id"]
    assert len(tmp_db.list_batch_runs("ds", "b1")) == 2
    # original still untouched
    assert tmp_db.get_run(run1["run_id"])["overall_score"] == score_v1


def test_backfill_idempotent_per_version(tmp_db):
    v1 = tmp_db.create_version("v1", config(), "t")
    sub = submission(tuple(healthy_samples(2)))
    tmp_db.insert_run(v1, sub, score_batch(v1["config"], sub))
    with pytest.raises(ValueError, match="idempotent"):
        tmp_db.insert_run(v1, sub, score_batch(v1["config"], sub))
    assert tmp_db.find_run("v1", "ds", "b1") is not None


def test_alerts_persisted_with_run(tmp_db):
    v1 = tmp_db.create_version("v1", config(), "t")
    sub = submission(())  # empty -> alert
    run = tmp_db.insert_run(v1, sub, score_batch(v1["config"], sub))
    alerts = tmp_db.list_alerts(run["run_id"])
    assert any(a["alert_type"] == "empty" for a in alerts)
    assert all(a["version_id"] == "v1" for a in alerts)


# --------------------------------------------------------------------------- #
# API-level
# --------------------------------------------------------------------------- #
def _create_and_activate(client, vid, cfg):
    r = client.post(
        "/api/versions",
        json=payload(VersionCreate(version_id=vid, config=cfg, created_by="tester")),
    )
    assert r.status_code == 201, r.text
    r = client.post(f"/api/versions/{vid}/activate")
    assert r.status_code == 200, r.text


def test_full_api_flow_empty_and_backfill(client):
    _create_and_activate(client, "v1", config())

    # 1) empty batch via API -> empty state, no score, blocked gate
    r = client.post("/api/runs", json=payload(submission(())))
    assert r.status_code == 201
    body = r.json()
    assert body["state"] == "empty"
    assert body["overall_score"] is None
    assert body["gate_passed"] is False
    assert body["reused_existing"] is False
    run_id = body["run_id"]

    # 2) healthy batch -> scored, gate passes
    r = client.post("/api/runs", json=payload(submission(tuple(healthy_samples(3)), batch_id="b2")))
    assert r.status_code == 201
    healthy = r.json()
    assert healthy["state"] == "scored"
    assert healthy["overall_score"] == 1.0
    assert healthy["gate_passed"] is True
    # evidence snapshot is fully stored and retrievable
    detail = client.get(f"/api/runs/{healthy['run_id']}").json()
    assert len(detail["result"]["sample_results"]) == 6
    assert detail["result"]["rules"][0]["normalized_weight"] == 0.6

    # 3) backfill same (version, dataset, batch) -> idempotent, same run id
    again = client.post(
        "/api/runs",
        json=payload(submission(tuple(healthy_samples(3)), batch_id="b2", backfill=True)),
    )
    assert again.status_code == 201
    assert again.json()["run_id"] == healthy["run_id"]
    assert again.json()["reused_existing"] is True

    # 4) alerts endpoint shows the empty-batch alert
    alerts = client.get("/api/alerts").json()
    assert any(a["run_id"] == run_id and a["alert_type"] == "empty" for a in alerts)


def test_history_retained_across_version_change(client):
    _create_and_activate(client, "v1", config(threshold=0.5))
    samples = tuple(healthy_samples(4, fail_r1=1))  # r1 = 0.75
    first = client.post("/api/runs", json=payload(submission(samples))).json()
    assert first["state"] == "scored"
    first_score = first["overall_score"]
    assert first["gate_passed"] is True  # 0.75 >= 0.5 threshold

    # tighten calibration; gate on the new version blocks the same data
    _create_and_activate(client, "v2", config(threshold=0.95))
    rescored = client.post(
        "/api/runs?version_id=v2",
        json=payload(submission(samples, backfill=True, rescore=first["run_id"])),
    ).json()
    assert rescored["run_id"] != first["run_id"]
    assert rescored["version_id"] == "v2"
    assert rescored["gate_passed"] is False

    # the historical v1 run remains exactly as originally scored
    history = client.get(f"/api/runs/{first['run_id']}").json()
    assert history["version_id"] == "v1"
    assert history["overall_score"] == first_score
    assert history["gate_passed"] is True

    # batch view shows both runs, oldest first
    both = client.get("/api/datasets/ds/batches/b1/runs").json()
    assert [r["version_id"] for r in both] == ["v1", "v2"]

    # gate view reflects per-version decisions, no retroactive recomputation
    gates = {r["run_id"]: r["gate_passed"] for r in client.get("/api/runs").json()}
    assert gates[first["run_id"]] is True
    assert gates[rescored["run_id"]] is False


def test_submit_without_active_version_conflicts(client):
    r = client.post("/api/runs", json=payload(submission(())))
    assert r.status_code == 409


def test_unknown_rule_rejected_at_api(client):
    _create_and_activate(client, "v1", config())
    bad = submission(({
        "sample_id": "x", "observed_at": T0.isoformat(),
        "observations": [{"rule_id": "ghost", "passed": True, "evidence": {}}],
        "metadata": {},
    },), batch_id="bad")
    r = client.post("/api/runs", json=payload(bad))
    assert r.status_code == 422


def test_version_diff_endpoint(client):
    _create_and_activate(client, "v1", config(0.8))
    client.post(
        "/api/versions",
        json=payload(VersionCreate(version_id="v2", config=config(0.95), created_by="tester")),
    )
    d = client.get("/api/versions/v1/diff/v2").json()
    assert d["thresholds"]["r1"] == {"from": 0.8, "to": 0.95}
    assert d["weights"]["r1"]["from"] == 0.6
    assert d["gate_policy"]["to"]["min_overall_score"] == 0.8


def test_demo_seed_covers_all_states_and_is_idempotent(client):
    out1 = client.post("/api/demo/seed").json()
    out2 = client.post("/api/demo/seed").json()
    states = {r["state"] for r in client.get("/api/runs?limit=500").json()}
    assert {"scored", "empty", "insufficient_samples", "rule_conflict"} <= states
    # second seed creates nothing new
    assert out2["runs"] == []
    # active version exists and UI is served
    assert client.get("/api/versions/active").status_code == 200
    page = client.get("/ui/index.html")
    assert page.status_code == 200 and "数据质量评分校准台" in page.text
    # root serves the same UI (via the static mount's index fallback)
    root = client.get("/")
    assert root.status_code == 200 and "数据质量评分校准台" in root.text
