"""Shared fixtures: isolated temp DB + FastAPI TestClient."""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("DQ_CALIBRATION_DB", str(db))
    # import lazily so the env var is picked up at module import
    from app import main

    main.storage = main.Storage(str(db))
    main.DB_PATH = str(db)
    return main.storage


@pytest.fixture()
def client(tmp_db):
    from app.main import app

    with TestClient(app) as c:
        yield c
