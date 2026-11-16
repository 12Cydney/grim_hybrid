"""
Tests for the "Models Reached" status + library version checks.
"""

import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from server.silver_bullet_server import (
    ModelRegistry, Orchestrator, load_model_file, runtime_versions,
    trained_versions, version_verdict, format_status_table,
)
from trainer.train_model import _save, library_versions

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _tiny_pack():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    sc = StandardScaler().fit(X)
    return LogisticRegression().fit(sc.transform(X), y), sc


def test_version_verdict():
    rt = {"sklearn": "1.9.0", "xgboost": "3.2.0"}
    assert version_verdict({"sklearn": "1.9.0", "xgboost": "3.2.0"}, rt) == "MATCH"
    assert version_verdict({"sklearn": "1.8.0", "xgboost": "3.2.0"}, rt) == "MISMATCH"
    assert version_verdict({"sklearn": "1.9.0", "xgboost": None}, rt) == "MATCH"
    assert version_verdict({"sklearn": None, "xgboost": None}, rt) == "UNKNOWN"


def test_shipped_models_reached_and_match_pinned_versions(config):
    reg = ModelRegistry(config, verbose=False)
    assert len(reg.status) == len(config["symbols"]) + 1
    for r in reg.status:
        assert r["status"] == "REACHED", r
        assert r["trained_sklearn"], r        # detected from un-stamped pickle bytes
        assert r["trained_xgboost"], r        # detected from booster config
    # requirements.txt pins the training versions -> must match here
    assert reg.all_versions_match, format_status_table(reg.status)


def test_missing_and_failed_reported(tmp_path, config):
    shutil.copy(MODELS / "xauusd_model.pkl", tmp_path / "xauusd_model.pkl")
    (tmp_path / "eurusd_model.pkl").write_bytes(b"not a pickle")
    reg = ModelRegistry(config, models_dir=tmp_path, verbose=False)
    by = {r["name"]: r for r in reg.status}
    assert by["XAUUSD"]["status"] == "REACHED"
    assert by["EURUSD"]["status"] == "FAILED" and by["EURUSD"]["error"]
    assert by["GBPUSD"]["status"] == "MISSING"
    assert by["FALLBACK"]["status"] == "MISSING"
    assert "XAUUSD" in reg.models and "EURUSD" not in reg.models


def test_trainer_stamps_versions(tmp_path):
    model, sc = _tiny_pack()
    _save(model, sc, {"label": "T"}, ["a", "b", "c"], tmp_path, "t_model.pkl")
    pack = joblib.load(tmp_path / "t_model.pkl")
    assert pack["versions"] == library_versions()
    assert trained_versions(pack)["sklearn"] == runtime_versions()["sklearn"]


def test_stamped_mismatch_flagged(tmp_path):
    model, sc = _tiny_pack()
    p = tmp_path / "x_model.pkl"
    joblib.dump({"model": model, "scaler": sc, "features": [],
                 "versions": {"sklearn": "0.0.1", "xgboost": None}}, p)
    _, row = load_model_file(p, "X")
    assert row["status"] == "REACHED" and row["version_check"] == "MISMATCH"
    assert "pip install scikit-learn==0.0.1" in format_status_table([row])


def test_ping_backwards_compatible_and_extended(config):
    ping = Orchestrator(config).handle_ping()
    for k in ("status", "server", "symbol_models", "model_count", "has_fallback", "feature_count"):
        assert k in ping
    assert ping["models_reached"] == ping["models_expected"] == len(ping["model_status"])
    assert ping["versions_ok"] is True


def test_check_models_script_exit_code():
    from scripts.check_models import main
    assert main([]) == 0
