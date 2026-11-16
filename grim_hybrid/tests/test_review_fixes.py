"""
Phase-2 regression tests - one section per review fix.
"""

import numpy as np
import pandas as pd

from collector.collect_training_data import closed_slices, TF_1H, TF_15M, TF_1M


def _bars(start, n, tf):
    t = np.arange(n) * tf + start
    return pd.DataFrame({"time": t.astype(float), "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": 1.0, "tick_volume": 1})


# ---------------------------------------------------------------- fix #1
def test_closed_slices_never_include_forming_bar():
    t0 = 1_705_000_000 - (1_705_000_000 % 3600)
    df_1h, df_15m, df_1m = _bars(t0, 200, TF_1H), _bars(t0, 800, TF_15M), _bars(t0, 12000, TF_1M)
    # decision at 10:15 (mid-hour) -> the 10:00 1h bar is still forming
    decision = t0 + 100 * TF_1H + TF_15M
    s1h, s15, s1m = closed_slices(df_1h, df_15m, df_1m, decision)
    for s, tf in ((s1h, TF_1H), (s15, TF_15M), (s1m, TF_1M)):
        assert (s["time"] + tf <= decision).all()
    # last visible bar is the most recent CLOSED one on each timeframe
    assert s1h["time"].iloc[-1] == t0 + 99 * TF_1H
    assert s15["time"].iloc[-1] == decision - TF_15M
    assert s1m["time"].iloc[-1] == decision - TF_1M


def test_ea_sends_closed_bars_only():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "mt5" / "GrimHybridEA_ZMQ.mq5").read_text()
    assert "CopyRates(g_symbol, tf, 1, count, rates)" in src
    assert "CopyRates(g_symbol, tf, 0, count, rates)" not in src


def test_synthetic_resample_keeps_unix_seconds():
    from collector.collect_training_data import synthetic_ohlc, _resample_to_15m, _resample_to_1h
    df_1m = synthetic_ohlc(bars=600, tf_seconds=60, seed=1)
    for df in (_resample_to_15m(df_1m), _resample_to_1h(df_1m)):
        # same order of magnitude as the 1m source (seconds, not us/ns)
        assert abs(df["time"].iloc[-1] - df_1m["time"].iloc[-1]) < 4000


# ---------------------------------------------------------------- fix #2
def _sl_inputs():
    from strategy_engine.liquidity import LiquidityResult
    from strategy_engine.market_structure import StructureResult
    from strategy_engine.order_blocks import OrderBlockResult
    # long after sweeping 99.0 -> signal bar low 98.5, structure low 99.2
    liq = LiquidityResult(swept=True, sweep_direction=1, ssl_level=99.0, bsl_level=102.0)
    st = StructureResult()
    st.recent_low, st.recent_high = 99.2, 101.8
    return liq, st, OrderBlockResult()


def test_sl_never_inside_sweep(config):
    from strategy_engine import stoploss
    liq, st, ob = _sl_inputs()
    res = stoploss.evaluate(entry=100.0, direction=1, atr_value=1.0, liquidity=liq, structure=st,
                            order_block=ob, config=config, signal_high=100.2, signal_low=98.5)
    assert res.price < 98.5, res
    assert all(c.price < 98.5 for c in res.candidates)
    assert res.kind == "SWEEP"


def test_sl_short_mirror(config):
    from strategy_engine import stoploss
    from strategy_engine.liquidity import LiquidityResult
    from strategy_engine.market_structure import StructureResult
    from strategy_engine.order_blocks import OrderBlockResult
    liq = LiquidityResult(swept=True, sweep_direction=-1, ssl_level=98.0, bsl_level=101.0)
    st = StructureResult(); st.recent_low, st.recent_high = 98.2, 100.8
    res = stoploss.evaluate(entry=100.0, direction=-1, atr_value=1.0, liquidity=liq, structure=st,
                            order_block=OrderBlockResult(), config=config,
                            signal_high=101.5, signal_low=99.8)
    assert res.price > 101.5


def test_sl_without_signal_bar_is_unchanged(config):
    """Callers that don't pass the signal bar keep the old behaviour."""
    from strategy_engine import stoploss
    liq, st, ob = _sl_inputs()
    res = stoploss.evaluate(entry=100.0, direction=1, atr_value=1.0, liquidity=liq, structure=st,
                            order_block=ob, config=config)
    assert res.kind == "STRUCTURE"


# ---------------------------------------------------------------- fix #3
def _ob_frame(retrace_low):
    """60 quiet bars, a bearish OB candle, a bullish displacement, then 3 bars."""
    rows = [dict(open=100.0, high=100.3, low=99.7, close=100.1) for _ in range(60)]
    rows.append(dict(open=100.5, high=100.6, low=99.4, close=99.5))        # OB: top 100.5
    rows.append(dict(open=99.5, high=102.5, low=99.45, close=102.4))      # displacement
    rows += [dict(open=102.4, high=102.8, low=retrace_low, close=102.6) for _ in range(3)]
    df = pd.DataFrame(rows)
    df["time"] = np.arange(len(df)) * 900.0
    df["tick_volume"] = 100
    return df


def test_ob_not_mitigated_by_its_own_displacement_candle(config):
    from strategy_engine import order_blocks
    res = order_blocks.detect(_ob_frame(retrace_low=102.0), atr_value=1.0, config=config)
    assert res.present and res.direction == 1 and res.top == 100.5
    assert res.mitigated is False


def test_ob_mitigated_when_price_returns(config):
    from strategy_engine import order_blocks
    res = order_blocks.detect(_ob_frame(retrace_low=100.4), atr_value=1.0, config=config)
    assert res.present and res.direction == 1 and res.mitigated is True


# ---------------------------------------------------------------- fix #4
def _sweep_frame(last_close):
    rows = [dict(open=100.0, high=100.5, low=99.5, close=100.0) for _ in range(25)]
    rows.append(dict(open=99.8, high=99.9, low=99.0, close=last_close))   # sweeps 99.5 low
    df = pd.DataFrame(rows); df["time"] = np.arange(len(df)) * 900.0; df["tick_volume"] = 1
    return df


def test_sweep_reclaim_flag(config):
    from strategy_engine import liquidity
    yes = liquidity.detect_sweep(_sweep_frame(99.7), atr_value=1.0, config=config)
    no = liquidity.detect_sweep(_sweep_frame(99.2), atr_value=1.0, config=config)
    assert yes.swept and yes.sweep_direction == 1 and yes.reclaimed
    assert no.swept and no.sweep_direction == 1 and not no.reclaimed


def test_direction_requires_reclaim_and_respects_htf(config, synthetic_market, monkeypatch):
    """Unreclaimed sweeps and counter-HTF fades never produce a direction."""
    import copy
    from strategy_engine import StrategyEngine, MarketData
    from strategy_engine import engine as eng_mod
    from strategy_engine.liquidity import LiquidityResult
    from strategy_engine.market_structure import StructureResult

    def run(cfg, liq, s15, s1h):
        monkeypatch.setattr(eng_mod.liquidity_mod, "detect_sweep", lambda *a, **k: liq)
        real = eng_mod.ms_mod.analyze
        monkeypatch.setattr(eng_mod.ms_mod, "analyze",
                            lambda df, c, level: s15 if level == "MTF" else (s1h if level == "HTF" else real(df, c, level=level)))
        m = synthetic_market
        return StrategyEngine(cfg).analyze(MarketData("XAUUSD", m["df_1h"].to_dict("records"),
                                                      m["df_15m"].to_dict("records"), m["df_1m"].to_dict("records")))

    flat = StructureResult()
    unreclaimed = LiquidityResult(swept=True, sweep_direction=1, reclaimed=False)
    assert run(config, unreclaimed, flat, flat).direction == 0
    assert run(config, LiquidityResult(swept=True, sweep_direction=1, reclaimed=True), flat, flat).direction == 1

    broke_low = StructureResult(broke_low=True)
    bear_htf = StructureResult(bos_direction=-1)
    assert run(config, LiquidityResult(), broke_low, bear_htf).direction == 0
    assert run(config, LiquidityResult(), broke_low, flat).direction == 1

    off = copy.deepcopy(config)
    off["direction"] = {"require_sweep_reclaim": False, "block_counter_htf_bias": False}
    assert run(off, unreclaimed, flat, flat).direction == 1
    assert run(off, LiquidityResult(), broke_low, bear_htf).direction == 1


# ---------------------------------------------------------------- fix #7 (labels)
def _path(*bars):
    return pd.DataFrame([dict(low=l, high=h) for l, h in bars])


def test_same_bar_sl_and_tp_is_skipped_not_sl():
    from collector.collect_training_data import _simulate_outcome, AMBIGUOUS
    # long, SL 99, TP 102: one wide bar touches both
    assert _simulate_outcome(_path((98.9, 102.1)), 100, 99, 102, 1) == AMBIGUOUS
    assert _simulate_outcome(_path((99.5, 101), (98.9, 100)), 100, 99, 102, 1) == 0
    assert _simulate_outcome(_path((99.5, 101), (100, 102.1)), 100, 99, 102, 1) == 1
    assert _simulate_outcome(_path((97.9, 101.1)), 100, 101, 98, -1) == AMBIGUOUS


def test_half_spread_costs_the_trade():
    from collector.collect_training_data import _simulate_outcome
    # long TP 102 touched by exactly 102.0 high: TP without spread, not with 0.1 half-spread
    p = _path((99.5, 102.0))
    assert _simulate_outcome(p, 100, 99, 102, 1) == 1
    assert _simulate_outcome(p, 100, 99, 102, 1, half_spread=0.1) == -1
    # short SL 101 not touched by 100.95 high - but is once ask (+spread) is used
    p = _path((99.5, 100.95))
    assert _simulate_outcome(p, 100, 101, 98, -1) == -1
    assert _simulate_outcome(p, 100, 101, 98, -1, half_spread=0.1) == 0


def test_stride_reduces_overlapping_samples(engine, config, synthetic_market):
    from collector.collect_training_data import collect_for_symbol
    m = synthetic_market
    a = collect_for_symbol(engine, "XAUUSD", "XAUUSD", m["df_1h"], m["df_15m"], m["df_1m"], config,
                           future_bars_1m=240, stride=1)
    b = collect_for_symbol(engine, "XAUUSD", "XAUUSD", m["df_1h"], m["df_15m"], m["df_1m"], config,
                           future_bars_1m=240, stride=3)
    assert len(a) > 0 and len(b) < len(a)
    ts = sorted(s["timestamp"] for s in b)
    assert all(t2 - t1 >= 3 * 900 for t1, t2 in zip(ts, ts[1:]))


# ---------------------------------------------------------------- fix #8 (sweep)
def test_1m_displacement_uses_1m_atr(engine, synthetic_market, monkeypatch):
    from strategy_engine import MarketData
    from strategy_engine import engine as eng_mod
    seen = []
    real = eng_mod.displacement_mod.calculate
    monkeypatch.setattr(eng_mod.displacement_mod, "calculate",
                        lambda df, atr, cfg: (seen.append((len(df), atr)), real(df, atr, cfg))[1])
    m = synthetic_market
    r = engine.analyze(MarketData("XAUUSD", m["df_1h"].to_dict("records"),
                                  m["df_15m"].to_dict("records"), m["df_1m"].to_dict("records")))
    atr_1m_used = [a for n, a in seen if n == len(m["df_1m"])][0]
    assert 0 < atr_1m_used < r.atr_15m


def test_daily_reset_uses_ny_date(config):
    from datetime import datetime, timezone
    from risk.risk_engine import RiskEngine, AccountState
    re_ = RiskEngine(config)
    # 2024-01-15 23:30 NY == 2024-01-16 04:30 UTC: still the 15th in NY
    late_ny = datetime(2024, 1, 16, 4, 30, tzinfo=timezone.utc)
    assert re_.trading_day(late_ny) == "2024-01-15"
    st = AccountState(daily_trades=5, last_reset="2024-01-15")
    re_.guardrails(st, now=late_ny)
    assert st.daily_trades == 5                      # no UTC-midnight reset
    re_.guardrails(st, now=datetime(2024, 1, 16, 5, 1, tzinfo=timezone.utc))  # 00:01 NY
    assert st.daily_trades == 0 and st.last_reset == "2024-01-16"


def test_trainer_threshold_from_validation_and_server_uses_it(tmp_path, config):
    import joblib
    from trainer.train_model import pick_threshold, _train_and_eval, _save
    from server.silver_bullet_server import decision_threshold
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    p = np.array([.1, .52, .58, .62, .7, .9, .61, .8])
    assert pick_threshold(y, p) == 0.62                  # tightest cut keeping all positives
    assert pick_threshold(np.zeros(4), np.full(4, .3)) == 0.65   # nothing predicted -> default

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    yy = pd.Series((X["a"] + rng.normal(scale=.5, size=300) > 0).astype(int))
    model, sc, metrics = _train_and_eval(X, yy, list("abcd"), "T")
    assert 0.5 <= metrics["threshold"] <= 0.95 and metrics["val_samples"] > 0
    _save(model, sc, metrics, list("abcd"), tmp_path, "t_model.pkl")
    pack = joblib.load(tmp_path / "t_model.pkl")
    assert decision_threshold(config, pack) == metrics["threshold"]
    assert decision_threshold(config, {"model": 1}) == config["models"]["min_confidence"]
    assert decision_threshold(config, None) == config["models"]["min_confidence"]


def test_trainer_has_no_deprecated_label_encoder_flag():
    from trainer.train_model import _make_model
    assert "use_label_encoder" not in _make_model().get_params() or \
        _make_model().get_params().get("use_label_encoder") is None


def test_ea_bool_parser_matches_server_ping_wire_format(config):
    import json, re
    from pathlib import Path
    from server.silver_bullet_server import Orchestrator
    src = (Path(__file__).resolve().parents[1] / "mt5" / "GrimHybridEA_ZMQ.mq5").read_text()
    assert 'GetJSONBool(response, "has_fallback")' in src
    assert re.search(r"input int\s+InpEntryCooldownMin", src)
    assert re.search(r"input int\s+InpMaxSpread\s*=\s*300;", src)
    wire = json.dumps(Orchestrator(config).handle_ping(), separators=(",", ":"))
    # EA's GetJSONBool looks for `"has_fallback":` followed by literal true/false
    assert re.search(r'"has_fallback":(true|false)', wire)
