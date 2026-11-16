"""
trainer.train_model
===================
Read CSV (from the collector) and train per-symbol models + one global
fallback.  Feature schema comes from
``strategy_engine.feature_builder.FEATURE_ORDER`` so the trainer and
the live server are guaranteed to agree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategy_engine.feature_builder import FEATURE_ORDER     # noqa: E402
from strategy_engine.utils import load_config                 # noqa: E402


log = logging.getLogger("trainer")


try:
    import xgboost as xgb   # type: ignore
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def library_versions() -> dict:
    """Versions stamped into every saved model pack."""
    import sklearn
    return {
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__ if HAS_XGB else None,
        "python": sys.version.split()[0],
    }


def _make_model():
    if HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42,
        )
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, min_samples_split=10, min_samples_leaf=5, random_state=42,
    )


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    available = [f for f in FEATURE_ORDER if f in df.columns]
    if len(available) != len(FEATURE_ORDER):
        missing = [f for f in FEATURE_ORDER if f not in df.columns]
        log.warning("Missing %d features in CSV: %s", len(missing), missing[:5])
    X = df[available].copy().fillna(0.0)
    y = df["outcome"].astype(int).copy()
    return X, y, available


def pick_threshold(y_true, proba, lo: float = 0.50, hi: float = 0.95, default: float = 0.65) -> float:
    """Confidence threshold that maximises F1 on the VALIDATION split.

    Falls back to ``default`` when no threshold yields a positive prediction.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    best_t, best_f1 = default, -1.0
    for t in np.round(np.arange(lo, hi + 1e-9, 0.01), 2):
        pred = (proba >= t).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def _train_and_eval(X, y, features, label: str, default_threshold: float = 0.65) -> tuple[object, StandardScaler, dict]:
    log.info("--- Training %s (%d samples, %d features) ---", label, len(X), len(features))
    if len(X) < 40 or y.nunique() < 2:
        log.warning("%s: not enough data / single class - skipping", label)
        return None, None, None

    # Chronological split: 64% train / 16% validation (threshold) / 20% test
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=0.2, shuffle=False)
    if y_tr.nunique() < 2:
        log.warning("%s: single class in training split - skipping", label)
        return None, None, None
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    model = _make_model()
    model.fit(X_tr_s, y_tr)

    threshold = pick_threshold(y_va, model.predict_proba(X_va_s)[:, 1], default=default_threshold)
    y_proba = model.predict_proba(X_te_s)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "val_samples": int(len(X_va)),
        "label": label,
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_te, y_pred, zero_division=0)),
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "features": features,
        "trained_at": datetime.now().isoformat(),
        "algorithm": "XGBoost" if HAS_XGB else "GradientBoosting",
    }

    log.info("%s | thr=%.2f acc=%.3f prec=%.3f rec=%.3f f1=%.3f", label, threshold,
             metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"])
    return model, scaler, metrics


def _save(model, scaler, metrics: dict, features: list[str], out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / filename
    versions = library_versions()
    metrics = {**metrics, "versions": versions}
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": features,
        "metrics": metrics,
        "versions": versions,   # read by the server's "Models Reached" check
        "threshold": metrics.get("threshold"),   # validation-picked confidence threshold
        "is_trained": True,
    }, pkl_path)
    metrics_path = pkl_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Saved %s (+ %s)", pkl_path.name, metrics_path.name)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None, help="Training CSV (defaults to config path)")
    ap.add_argument("--models-dir", default=None)
    args = ap.parse_args()

    config = load_config()
    csv_path = Path(args.csv) if args.csv else _ROOT / config["paths"]["training_dir"] / config["paths"]["training_csv"]
    if not csv_path.exists():
        log.error("Training CSV not found: %s", csv_path)
        log.error("Run: python collector/collect_training_data.py --synthetic (or with MT5)")
        sys.exit(1)

    models_dir = Path(args.models_dir) if args.models_dir else _ROOT / config["models"]["base_dir"]
    default_thr = float(config["models"]["min_confidence"])
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    if "symbol" not in df.columns:
        log.error("CSV missing 'symbol' column")
        sys.exit(1)

    # Per-symbol models
    trained_syms = []
    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym]
        if len(sub) < 40:
            log.info("%s: only %d samples, skipping per-symbol model", sym, len(sub))
            continue
        X, y, feats = _prepare(sub)
        model, scaler, metrics = _train_and_eval(X, y, feats, sym, default_thr)
        if model is None:
            continue
        _save(model, scaler, metrics, feats, models_dir, f"{sym.lower()}_model.pkl")
        trained_syms.append(sym)

    # Global fallback model (all data)
    X, y, feats = _prepare(df)
    model, scaler, metrics = _train_and_eval(X, y, feats, "FALLBACK", default_thr)
    if model is not None:
        _save(model, scaler, metrics, feats, models_dir, config["models"]["fallback_file"])

    log.info("Training complete. Per-symbol: %s", trained_syms)


if __name__ == "__main__":
    main()
