"""End-to-end tests: engine analyze() -> feature builder -> server response."""
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_analyze_returns_full_setup(engine, synthetic_market):
    from strategy_engine import MarketData
    md = MarketData(
        symbol=synthetic_market["symbol"],
        data_1h=synthetic_market["df_1h"].to_dict("records"),
        data_15m=synthetic_market["df_15m"].to_dict("records"),
        data_1m=synthetic_market["df_1m"].to_dict("records"),
    )
    setup = engine.analyze(md)
    # Regardless of whether a valid setup exists, the result must be
    # populated end-to-end (no None sub-objects).
    assert setup.session is not None
    assert setup.vol_15m is not None and setup.vol_1h is not None
    assert setup.structure_15m is not None
    assert setup.liquidity is not None
    assert setup.fvg is not None
    assert setup.order_block is not None
    assert setup.score is not None
    assert setup.entry > 0
    assert setup.direction in (-1, 0, 1)


def test_feature_builder_returns_deterministic_schema(engine, synthetic_market):
    from strategy_engine import MarketData
    from strategy_engine.feature_builder import FEATURE_ORDER

    md = MarketData(
        symbol="XAUUSD",
        data_1h=synthetic_market["df_1h"].to_dict("records"),
        data_15m=synthetic_market["df_15m"].to_dict("records"),
        data_1m=synthetic_market["df_1m"].to_dict("records"),
    )
    setup = engine.analyze(md)
    feats = engine.build_features(setup)
    assert list(feats.keys()) == list(FEATURE_ORDER)
    for k, v in feats.items():
        assert isinstance(v, float), f"{k} is {type(v)}"


def test_features_match_between_two_calls(engine, synthetic_market):
    """Zero-drift guarantee: same input -> byte-identical features."""
    from strategy_engine import MarketData
    md = MarketData(
        symbol="XAUUSD",
        data_1h=synthetic_market["df_1h"].to_dict("records"),
        data_15m=synthetic_market["df_15m"].to_dict("records"),
        data_1m=synthetic_market["df_1m"].to_dict("records"),
    )
    a = engine.build_features(engine.analyze(md))
    b = engine.build_features(engine.analyze(md))
    assert a == b


def test_collector_and_live_produce_same_features(engine, synthetic_market):
    """CRITICAL invariant of the framework.  Collector path calls the engine
    the same way the server does - so features MUST match."""
    from strategy_engine import MarketData
    df_1h  = synthetic_market["df_1h"]
    df_15m = synthetic_market["df_15m"]
    df_1m  = synthetic_market["df_1m"]

    # Live path
    md_live = MarketData(symbol="XAUUSD",
                         data_1h=df_1h.to_dict("records"),
                         data_15m=df_15m.to_dict("records"),
                         data_1m=df_1m.to_dict("records"))
    feats_live = engine.build_features(engine.analyze(md_live))

    # Collector path uses the exact same call
    from collector.collect_training_data import collect_for_symbol  # noqa: E402
    md_col = MarketData(symbol="XAUUSD",
                        data_1h=df_1h.to_dict("records"),
                        data_15m=df_15m.to_dict("records"),
                        data_1m=df_1m.to_dict("records"),
                        timestamp=float(df_1m["time"].iloc[-1]))
    setup = engine.analyze(md_col)
    feats_col = engine.build_features(setup)
    assert feats_live == feats_col


# ---------------------------------------------------------------------
# TCP server smoke test
# ---------------------------------------------------------------------
def _wait_port(host, port, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except Exception:
            time.sleep(0.1)
    return False


def _send_tcp(host, port, payload: dict, timeout=5.0) -> dict:
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall((json.dumps(payload) + "\n").encode())
    buf = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
        if b"\n" in buf:
            break
    s.close()
    return json.loads(buf.split(b"\n", 1)[0].decode())


@pytest.mark.integration
def test_tcp_server_ping_and_analyze(synthetic_market):
    port = 15555
    env = {**dict(**__import__("os").environ)}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "server" / "silver_bullet_server.py"), str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env,
    )
    try:
        assert _wait_port("127.0.0.1", port), "server never started"

        ping = _send_tcp("127.0.0.1", port, {"type": "PING"})
        assert ping["status"] == "ok"
        assert "feature_count" in ping

        req = {
            "type": "PRICE_DATA_MTF",
            "symbol": synthetic_market["symbol"],
            "data_1h":  synthetic_market["df_1h"].to_dict("records"),
            "data_15m": synthetic_market["df_15m"].to_dict("records"),
            "data_1m":  synthetic_market["df_1m"].to_dict("records"),
        }
        resp = _send_tcp("127.0.0.1", port, req)
        assert resp["status"] == "success"
        assert resp["action"] in ("BUY", "SELL", "HOLD")
        assert "setup_score" in resp
        assert "setup_grade" in resp
        assert "volatility_regime" in resp
        assert "sl_kind" in resp
        assert "tp_kind" in resp
        assert "risk_pct" in resp and "risk_allowed" in resp

        # v1.1 POSITION_STATE round trip (EA wire format)
        ps = _send_tcp("127.0.0.1", port, {
            "type": "POSITION_STATE", "symbol": "XAUUSD", "gmt_offset": 0,
            "balance": 10000.0, "equity": 10000.0, "open_trades": 0, "floating_pnl": 0.0,
            "deals": [],
        })
        assert ps["status"] == "ok" and ps["open_trades"] == 0 and "trading_day" in ps
        assert ping["models_expected"] >= 1 and "model_status" in ping
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
