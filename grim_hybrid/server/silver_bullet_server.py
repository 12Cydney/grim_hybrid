"""
server.silver_bullet_server
===========================

Thin TCP orchestrator.  ZERO trading logic - all decisions go through
``strategy_engine.StrategyEngine``.

Protocol (unchanged from v3.1 so the MT5 EA needs no changes):
  PING                      -> {status, server, symbol_models, model_count, ...}
  PRICE_DATA_MTF            -> full decision JSON (action, confidence, ...)
"""

from __future__ import annotations

import json
import os
import socket
import socketserver
import sys
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Make project root importable when run as a script
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategy_engine import StrategyEngine, MarketData         # noqa: E402
from strategy_engine.utils import load_config                  # noqa: E402
from risk.risk_engine import RiskEngine, AccountState          # noqa: E402


# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------
def runtime_versions() -> dict[str, str | None]:
    """Library versions installed on THIS machine."""
    out: dict[str, str | None] = {"sklearn": None, "xgboost": None}
    try:
        import sklearn
        out["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import xgboost
        out["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return out


_SK_VER_RE = re.compile(rb"_sklearn_version\x94?\x8c.([0-9][0-9A-Za-z.\-+]*)")


def trained_versions(pack: dict, path: Path | None = None) -> dict[str, str | None]:
    """Library versions a model pack was trained with.

    New packs carry an explicit ``versions`` stamp (trainer >= v1.1).
    Older packs are inspected: sklearn's version is embedded in the
    pickle bytes, xgboost's in the booster config.
    """
    stamp = pack.get("versions") if isinstance(pack, dict) else None
    if isinstance(stamp, dict) and stamp:
        return {"sklearn": stamp.get("sklearn"), "xgboost": stamp.get("xgboost")}
    out: dict[str, str | None] = {"sklearn": None, "xgboost": None}
    if path is not None and path.exists():
        m = _SK_VER_RE.search(path.read_bytes())
        if m:
            out["sklearn"] = m.group(1).decode("ascii", errors="replace")
    model = pack.get("model") if isinstance(pack, dict) else None
    if model is not None and hasattr(model, "get_booster"):
        try:
            ver = json.loads(model.get_booster().save_config()).get("version")
            if ver:
                out["xgboost"] = ".".join(str(v) for v in ver)
        except Exception:
            pass
    return out


def version_verdict(trained: dict, runtime: dict) -> str:
    """MATCH when every known trained version equals the runtime one."""
    known = [k for k, v in trained.items() if v]
    if not known:
        return "UNKNOWN"
    for k in known:
        if trained[k] != runtime.get(k):
            return "MISMATCH"
    return "MATCH"


def load_model_file(path: Path, name: str) -> tuple[dict | None, dict]:
    """Load one model pack and return (pack, status_row)."""
    rt = runtime_versions()
    row: dict[str, Any] = {
        "name": name, "file": path.name, "status": "MISSING",
        "trained_sklearn": None, "trained_xgboost": None,
        "runtime_sklearn": rt["sklearn"], "runtime_xgboost": rt["xgboost"],
        "version_check": "N/A", "error": "",
    }
    if not path.exists():
        return None, row
    try:
        with warnings.catch_warnings():
            # we report version drift ourselves, in one readable table
            warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.simplefilter("ignore", InconsistentVersionWarning)
            except ImportError:
                pass
            pack = joblib.load(path)
        if not isinstance(pack, dict) or "model" not in pack or "scaler" not in pack:
            raise ValueError("not a Grim Hybrid model pack (missing model/scaler)")
    except Exception as e:
        row["status"] = "FAILED"
        row["error"] = f"{type(e).__name__}: {e}"
        return None, row
    tv = trained_versions(pack, path)
    row["trained_sklearn"] = tv["sklearn"]
    row["trained_xgboost"] = tv["xgboost"]
    row["version_check"] = version_verdict(tv, rt)
    row["status"] = "REACHED"
    return pack, row


def format_status_table(rows: list[dict]) -> str:
    hdr = f"  {'MODEL':<9} {'STATUS':<8} {'SKLEARN trained/runtime':<24} {'XGBOOST trained/runtime':<24} CHECK"
    lines = [hdr, "  " + "-" * (len(hdr) - 2)]
    for r in rows:
        sk = f"{r['trained_sklearn'] or '-'} / {r['runtime_sklearn'] or '-'}"
        xg = f"{r['trained_xgboost'] or '-'} / {r['runtime_xgboost'] or '-'}"
        lines.append(f"  {r['name']:<9} {r['status']:<8} {sk:<24} {xg:<24} {r['version_check']}")
        if r.get("error"):
            lines.append(f"            error: {r['error']}")
    reached = sum(1 for r in rows if r["status"] == "REACHED")
    lines.append(f"  Models reached: {reached}/{len(rows)}")
    bad = [r for r in rows if r["version_check"] == "MISMATCH"]
    if bad:
        want = {k: v for r in bad for k, v in
                (("scikit-learn", r["trained_sklearn"]), ("xgboost", r["trained_xgboost"])) if v}
        pins = " ".join(f"{k}=={v}" for k, v in want.items())
        lines.append("")
        lines.append("  !!! VERSION MISMATCH - predictions from these models may be INVALID: "
                     + ", ".join(r["name"] for r in bad))
        lines.append(f"  !!! Fix: pip install {pins}   (or retrain: python grim_hybrid/trainer/train_model.py)")
    return "\n".join(lines)


class ModelRegistry:
    """Loads symbol-specific + fallback models from ``models/``."""

    def __init__(self, config: dict, models_dir: Path | None = None, verbose: bool = True):
        self.config = config
        self.models_dir = Path(models_dir) if models_dir else Path(_ROOT / config["models"]["base_dir"])
        self.models: dict[str, dict] = {}
        self.fallback: dict | None = None
        self.status: list[dict] = []
        self._load(verbose)

    def _load(self, verbose: bool = True):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for base in self.config["symbols"].keys():
            pack, row = load_model_file(self.models_dir / f"{base.lower()}_model.pkl", base.upper())
            if pack is not None:
                self.models[base.upper()] = pack
            self.status.append(row)

        fb_path = self.models_dir / self.config["models"]["fallback_file"]
        self.fallback, row = load_model_file(fb_path, "FALLBACK")
        self.status.append(row)

        if verbose:
            print("=" * 60)
            print("MODELS REACHED")
            print("=" * 60)
            print(format_status_table(self.status))
            print("=" * 60)
            if not self.models and self.fallback is None:
                print("WARNING: No models loaded - engine will emit HOLDs for missing model.")

    @property
    def all_versions_match(self) -> bool:
        return not any(r["version_check"] == "MISMATCH" for r in self.status)

    def resolve(self, symbol: str) -> tuple[dict | None, str]:
        s = (symbol or "").upper()
        if s in self.models:
            return self.models[s], f"Symbol-specific ({s})"
        # Alias matching
        for base, spec in self.config["symbols"].items():
            aliases = [a.upper() for a in spec.get("aliases", [])]
            if s in aliases or any(a in s or s in a for a in aliases):
                b = base.upper()
                if b in self.models:
                    return self.models[b], f"Symbol-specific ({b})"
        if self.fallback:
            return self.fallback, "Fallback (global)"
        return None, "None (no model)"


# ---------------------------------------------------------------------
# Prediction glue
# ---------------------------------------------------------------------
def _predict(model_pack: dict, features: dict) -> tuple[int, float]:
    """Vectorise ``features`` in the schema the model was trained on."""
    model = model_pack["model"]
    scaler = model_pack["scaler"]
    schema = model_pack.get("features", list(features.keys()))
    vec = [features.get(f, 0.0) for f in schema]
    X = np.array([vec], dtype=float)
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = model.predict_proba(Xs)[0]
    # binary classifier - index 1 = "TP"
    confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
    return pred, confidence


def decision_threshold(config: dict, pack: dict | None) -> float:
    """Validation-picked threshold stored in the model pack (trainer >= v1.1),
    else ``models.min_confidence`` from config."""
    mconf = config["models"]
    if pack is not None and mconf.get("use_model_threshold", True):
        t = pack.get("threshold")
        if isinstance(t, (int, float)) and 0.0 < float(t) < 1.0:
            return float(t)
    return float(mconf["min_confidence"])


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.engine = StrategyEngine(config)
        self.registry = ModelRegistry(config)
        self.risk = RiskEngine(config)
        self.account = AccountState()

    def handle_ping(self) -> dict:
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "server": "Grim Hybrid Silver Bullet Server v1.0",
            "symbol_models": list(self.registry.models.keys()),
            "model_count": len(self.registry.models),
            "has_fallback": self.registry.fallback is not None,
            "feature_count": len(self.engine.feature_names()),
            # "Models Reached" (additive - EA ignores unknown keys)
            "models_reached": sum(1 for r in self.registry.status if r["status"] == "REACHED"),
            "models_expected": len(self.registry.status),
            "versions_ok": self.registry.all_versions_match,
            "model_status": self.registry.status,
        }

    def handle_price_data(self, req: dict) -> dict:
        symbol = req.get("symbol", "UNKNOWN")
        md = MarketData(
            symbol=symbol,
            data_1h=req.get("data_1h", []),
            data_15m=req.get("data_15m", []),
            data_1m=req.get("data_1m", []),
            spread=float(req.get("spread", 0.0)),
            timestamp=req.get("timestamp"),
        )
        setup = self.engine.analyze(md)

        # ---- No setup: HOLD immediately, don't call model ----
        if not setup.valid:
            return self._build_response(setup, symbol, action="HOLD", confidence=0.0,
                                        model_type="N/A", method="ENGINE",
                                        hold_reason=setup.hold_reason or "No valid setup",
                                        prediction=0)

        # ---- Valid setup -> features -> AI ----
        features = self.engine.build_features(setup)
        pack, model_type = self.registry.resolve(symbol)

        if pack is None:
            # No model available - use score-based fallback so system still works
            score = setup.score.overall
            pred = 1 if score >= 0.65 else 0
            confidence = score
            method = "SCORE_ONLY"
        else:
            try:
                pred, confidence = _predict(pack, features)
                method = "ML"
                # Auto-fallback to global model if confidence is very low
                if (
                    self.config["models"].get("auto_fallback_to_global", True)
                    and self.registry.fallback is not None
                    and pack is not self.registry.fallback
                    and confidence < float(self.config["models"]["low_confidence_threshold"])
                ):
                    p2, c2 = _predict(self.registry.fallback, features)
                    if c2 > confidence:
                        pred, confidence, method = p2, c2, "ML_FALLBACK"
                        model_type = "Fallback (auto)"
                        pack = self.registry.fallback
            except Exception as e:
                print(f"[predict] {symbol}: {e}")
                pred = 1 if setup.score.overall >= 0.65 else 0
                confidence = setup.score.overall
                method = "SCORE_ONLY"

        # ---- Decision ----
        min_conf = decision_threshold(self.config, pack if method.startswith("ML") else None)
        if pred == 1 and confidence >= min_conf:
            action = "BUY" if setup.direction == 1 else ("SELL" if setup.direction == -1 else "HOLD")
            hold_reason = "" if action != "HOLD" else "No direction"
        else:
            action = "HOLD"
            if confidence < min_conf:
                hold_reason = f"Confidence {confidence:.1%} below {min_conf:.0%}"
            else:
                hold_reason = "Model predicts SL"

        setup.ml_prediction = pred
        setup.ml_confidence = confidence
        setup.ml_model_type = model_type

        return self._build_response(setup, symbol, action=action, confidence=confidence,
                                    model_type=model_type, method=method,
                                    hold_reason=hold_reason, prediction=pred)

    # -----------------------------------------------------------------
    def _build_response(self, setup, symbol, *, action: str, confidence: float,
                        model_type: str, method: str, hold_reason: str, prediction: int) -> dict:
        sl = setup.stoploss
        tp = setup.takeprofit
        session_name = setup.session.name if setup.session else "OFF"
        liq_type = setup.liquidity.liquidity_type if setup.liquidity else "NONE"

        signal = 1 if action == "BUY" else (-1 if action == "SELL" else 0)
        tp_prob = confidence if prediction == 1 else 1 - confidence
        sl_prob = 1.0 - tp_prob

        # SL distance in pips (protocol backwards compat with the EA)
        pip = setup.pip_value or 0.0001
        sl_dist_pips = (sl.distance / pip) if (sl and sl.distance > 0) else 0.0

        response = {
            "status": "success",
            "symbol": symbol,
            "signal": signal,
            "action": action,
            "confidence": float(confidence),
            "tp_probability": float(tp_prob),
            "sl_probability": float(sl_prob),
            "session": session_name,
            "liquidity_type": liq_type,
            "sweep_confirmed": bool(setup.ltf_confirmed),
            "method": method,
            "model_type": model_type,
            "hold_reason": hold_reason,
            "sl_distance": float(sl_dist_pips),
            "direction": int(setup.direction),
            "htf_bias": int(setup.structure_1h.bos_direction if setup.structure_1h else 0),
            "model_version": "v1.0-grim-hybrid",
            "timestamp": datetime.now().isoformat(),
            # Rich engine context (new in v1)
            "setup_score": float(setup.score.overall if setup.score else 0.0),
            "setup_grade": setup.score.grade if setup.score else "F",
            "setup_description": setup.score.description if setup.score else "",
            "sl_kind": sl.kind if sl else "NONE",
            "tp_kind": tp.kind if tp else "NONE",
            "sl_price": float(sl.price if sl else 0.0),
            "tp_price": float(tp.price if tp else 0.0),
            "rr_planned": float(tp.rr if tp else 0.0),
            "volatility_regime": setup.vol_15m.regime if setup.vol_15m else "UNKNOWN",
        }

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} | {action} "
            f"| conf={confidence:.1%} | score={response['setup_score']:.2f} "
            f"| {method} | {model_type} | {hold_reason}"
        )
        return response


# ---------------------------------------------------------------------
# TCP handler
# ---------------------------------------------------------------------
ORCH: Orchestrator | None = None
CFG: dict | None = None


class EAHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(float(CFG["server"]["socket_timeout"]))
        try:
            buf = bytearray()
            while True:
                chunk = self.request.recv(int(CFG["server"]["recv_buf"]))
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in buf:
                    break
                if len(buf) > 10 * 1024 * 1024:
                    return
            if not buf:
                return
            line = buf.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
            if not line:
                return
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                err = json.dumps({"status": "error", "message": f"bad json: {e}"}) + "\n"
                try:
                    self.request.sendall(err.encode("utf-8"))
                except Exception:
                    pass
                return

            msg_type = req.get("type", "")
            if msg_type == "PING":
                resp = ORCH.handle_ping()
            elif msg_type == "PRICE_DATA_MTF":
                resp = ORCH.handle_price_data(req)
            else:
                resp = {"status": "error", "message": f"Unknown type: {msg_type}"}

            self.request.sendall((json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8"))
        except socket.timeout:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{self.client_address[0]}] handler error: {e}")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global ORCH, CFG
    CFG = load_config()

    print("=" * 60)
    print("GRIM HYBRID SILVER BULLET SERVER v1.0")
    print("=" * 60)

    ORCH = Orchestrator(CFG)

    host = CFG["server"]["host"]
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(CFG["server"]["port"])
    print(f"TCP listening on: {host}:{port}")
    print(f"Feature count:    {len(ORCH.engine.feature_names())}")
    print(f"Symbol models:    {list(ORCH.registry.models.keys()) or 'none'}")
    print(f"Fallback model:   {'yes' if ORCH.registry.fallback else 'no'}")
    print("=" * 60)

    with ThreadedTCPServer((host, port), EAHandler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()


if __name__ == "__main__":
    main()
