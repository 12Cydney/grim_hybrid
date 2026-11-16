"""
Entry-filter tightening (killzone, 1m MSS, POI retrace, score floor) and the
broker-clock fix they depend on.
"""

import copy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from strategy_engine.sessions import SessionResolver, broker_to_utc, broker_offset_seconds


# ---------------------------------------------------------------- broker clock
def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc).timestamp()


def test_ny_plus_7_is_gmt2_winter_gmt3_summer(config):
    winter = _utc(2024, 1, 15, 15, 0)        # 10:00 NY (EST)
    summer = _utc(2024, 7, 15, 14, 0)        # 10:00 NY (EDT)
    assert broker_offset_seconds(winter + 2 * 3600, config) == 2 * 3600
    assert broker_offset_seconds(summer + 3 * 3600, config) == 3 * 3600
    assert broker_to_utc(winter + 2 * 3600, config) == winter
    assert broker_to_utc(summer + 3 * 3600, config) == summer


def test_ea_gmt_offset_wins_and_other_modes(config):
    ts = _utc(2024, 1, 15, 15, 0)
    assert broker_to_utc(ts + 5400, config, gmt_offset=5400) == ts
    fixed = {**config, "broker_clock": {"mode": "fixed", "fixed_offset_hours": 2}}
    assert broker_to_utc(ts + 7200, fixed) == ts
    assert broker_to_utc(ts, {**config, "broker_clock": {"mode": "utc"}}) == ts


def test_broker_time_now_lands_in_silver_bullet(config):
    """10:15 NY on a GMT+2 broker = 17:15 server time. Before the fix the
    server read that as 17:15 UTC (12:15 NY) - outside the AM Silver Bullet."""
    sr = SessionResolver(config)
    broker_ts = _utc(2024, 1, 15, 15, 15) + 2 * 3600
    assert sr.resolve(broker_ts).name != "NY_AM_SB"                    # old behaviour
    assert sr.resolve(broker_to_utc(broker_ts, config)).name == "NY_AM_SB"


def test_server_converts_and_warns_on_clock_mismatch(config, tmp_path, capsys):
    from server.silver_bullet_server import Orchestrator
    orch = Orchestrator(config, logs_dir=tmp_path)
    ts = _utc(2024, 1, 15, 15, 15) + 2 * 3600
    assert orch._utc_timestamp({"timestamp": ts, "gmt_offset": 7200}) == ts - 7200
    capsys.readouterr()
    orch2 = Orchestrator(config, logs_dir=tmp_path)
    orch2._utc_timestamp({"timestamp": ts, "gmt_offset": 0})
    assert "BROKER CLOCK MISMATCH" in capsys.readouterr().out


# ---------------------------------------------------------------- helpers
def _setup(config, ts, direction=1, **kw):
    from strategy_engine.engine import SetupResult
    r = SetupResult(direction=direction, entry=100.0, atr_15m=1.0, timestamp=ts)
    r.session = SessionResolver(config).resolve(ts)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def _engine(config, **filters):
    from strategy_engine import StrategyEngine
    cfg = copy.deepcopy(config)
    cfg["entry_filters"] = {**{"require_killzone": False, "require_ltf_confirmation": False,
                               "require_poi": False, "require_poi_retrace": False,
                               "structure_break_only": "allow"}, **filters}
    return StrategyEngine(cfg)


NY_AM_SB = _utc(2024, 1, 15, 15, 30)     # 10:30 NY
ASIAN = _utc(2024, 1, 16, 1, 0)          # 20:00 NY
LONDON_KZ = _utc(2024, 1, 15, 8, 0)      # 03:00 NY


# ---------------------------------------------------------------- 1) killzone
@pytest.mark.parametrize("ts,ok", [(NY_AM_SB, True), (LONDON_KZ, True), (ASIAN, False)])
def test_killzone_required(config, ts, ok):
    eng = _engine(config, require_killzone=True, killzones=["LONDON_KZ", "NY_AM_SB", "NY_PM_SB"])
    name, reason = eng._entry_filters(_setup(config, ts))
    assert (name == "") is ok
    if not ok:
        assert name == "killzone" and reason.startswith("Outside killzone (ASIAN)")


def test_killzone_switch_off(config):
    eng = _engine(config, require_killzone=False)
    assert eng._entry_filters(_setup(config, ASIAN)) == ("", "")


# ---------------------------------------------------------------- 2) 1m MSS
def _m1(closes, wick=0.05):
    """1m frame from a close path; open = previous close."""
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    up = c >= o   # full wick in the candle's direction, half on the other side -> no tied extremes
    df = pd.DataFrame({"open": o, "close": c,
                       "high": np.maximum(o, c) + np.where(up, wick, wick / 2),
                       "low": np.minimum(o, c) - np.where(up, wick / 2, wick)})
    df["time"] = np.arange(len(df)) * 60.0
    df["tick_volume"] = 1
    return df


# drop into a low, lower high at ~100.0, final low, then closes back above that high
_BULL_MSS = [101, 100.6, 100.2, 99.8, 99.4, 99.6, 99.9, 100.0, 99.7, 99.3, 99.0, 98.8,
             99.0, 99.4, 99.8, 100.3, 100.5]


def test_mss_detected_long_and_mirror_short():
    from strategy_engine.engine import StrategyEngine as E
    assert E._ltf_mss(_m1(_BULL_MSS), 1) is True
    assert E._ltf_mss(_m1([200 - x for x in _BULL_MSS]), -1) is True


def test_no_mss_without_close_through_swing():
    from strategy_engine.engine import StrategyEngine as E
    path = _BULL_MSS[:-3] + [99.6, 99.8, 99.9]        # rallies but never closes above 100.0+wick
    assert E._ltf_mss(_m1(path), 1) is False


def test_mss_too_old_is_rejected():
    from strategy_engine.engine import StrategyEngine as E
    path = _BULL_MSS + [100.6] * 15                    # break happened >10 bars ago
    assert E._ltf_mss(_m1(path), 1, window=10) is False
    assert E._ltf_mss(_m1(path), 1, window=30) is True


def test_ltf_filter_blocks_unconfirmed(config):
    eng = _engine(config, require_ltf_confirmation=True, ltf_confirmation="mss")
    assert eng._entry_filters(_setup(config, NY_AM_SB, ltf_confirmed=False))[0] == "ltf"
    assert eng._entry_filters(_setup(config, NY_AM_SB, ltf_confirmed=True)) == ("", "")


# ---------------------------------------------------------------- 3) POI + retrace
def _m15_with_bull_fvg(close_through=False):
    rows = [dict(open=100.0, high=100.3, low=99.7, close=100.0) for _ in range(30)]
    rows.append(dict(open=100.0, high=100.4, low=99.8, close=100.3))      # c1 high 100.4
    rows.append(dict(open=100.3, high=101.9, low=100.2, close=101.8))     # impulse
    rows.append(dict(open=101.8, high=102.3, low=101.0, close=102.1))     # c3 low 101.0 -> gap 100.4-101.0
    rows.append(dict(open=102.1, high=102.2, low=101.5, close=100.0 if close_through else 101.6))
    df = pd.DataFrame(rows); df["time"] = np.arange(len(df)) * 900.0; df["tick_volume"] = 1
    return df


def _poi_setup(config, df15, direction=1):
    from strategy_engine import fvg as fvg_mod, order_blocks as ob_mod
    r = _setup(config, NY_AM_SB, direction=direction)
    r.fvg = fvg_mod.detect(df15, 1.0, config)
    r.order_block = ob_mod.OrderBlockResult()
    return r


def test_aligned_fvg_found_and_closed_through_rejected(config):
    eng = _engine(config)
    assert eng._find_poi(_m15_with_bull_fvg(), _poi_setup(config, _m15_with_bull_fvg())) == ("FVG", 101.0, 100.4)
    broken = _m15_with_bull_fvg(close_through=True)
    assert eng._find_poi(broken, _poi_setup(config, broken))[0] == ""
    # wrong direction -> no POI
    assert eng._find_poi(_m15_with_bull_fvg(), _poi_setup(config, _m15_with_bull_fvg(), -1))[0] == ""


def test_retrace_tap_rules():
    from strategy_engine.engine import StrategyEngine as E
    tapped = _m1([101.6, 101.3, 101.1, 100.9, 101.2, 101.5])        # dips into 100.4-101.0
    floated = _m1([101.9, 101.7, 101.6, 101.5, 101.7, 101.8])       # never reaches the gap
    through = _m1([101.5, 101.0, 100.5, 100.1, 99.9, 100.0])        # closes below the zone
    assert E._poi_tapped(tapped, 1, 101.0, 100.4, 15, 0.0) is True
    assert E._poi_tapped(floated, 1, 101.0, 100.4, 15, 0.0) is False
    assert E._poi_tapped(through, 1, 101.0, 100.4, 15, 0.0) is False


def test_poi_filters_block_in_order(config):
    eng = _engine(config, require_poi=True, require_poi_retrace=True)
    assert eng._entry_filters(_setup(config, NY_AM_SB))[0] == "poi"
    assert eng._entry_filters(_setup(config, NY_AM_SB, poi_kind="FVG", poi_tapped=False)) == \
        ("poi_retrace", "No retrace into FVG")
    assert eng._entry_filters(_setup(config, NY_AM_SB, poi_kind="OB", poi_tapped=True)) == ("", "")


# ---------------------------------------------------------------- 4) floor + structure fades
def test_config_floor_is_055(config):
    assert config["min_setup_score"] == 0.55


def test_structure_only_block_mode(config):
    eng = _engine(config, structure_break_only="block")
    assert eng._entry_filters(_setup(config, NY_AM_SB, trigger="STRUCTURE"))[0] == "structure_only"
    assert eng._entry_filters(_setup(config, NY_AM_SB, trigger="SWEEP")) == ("", "")


def test_structure_only_penalized_one_grade(config, synthetic_market, monkeypatch):
    """Same market, forced into a structure-only fade: penalize mode costs exactly the penalty."""
    from strategy_engine import MarketData
    from strategy_engine import engine as eng_mod
    from strategy_engine.liquidity import LiquidityResult
    from strategy_engine.market_structure import StructureResult
    monkeypatch.setattr(eng_mod.liquidity_mod, "detect_sweep", lambda *a, **k: LiquidityResult())
    real = eng_mod.ms_mod.analyze
    monkeypatch.setattr(eng_mod.ms_mod, "analyze", lambda df, c, level: StructureResult(broke_low=True)
                        if level == "MTF" else (StructureResult() if level == "HTF" else real(df, c, level=level)))
    m = synthetic_market
    md = MarketData("XAUUSD", m["df_1h"].to_dict("records"), m["df_15m"].to_dict("records"),
                    m["df_1m"].to_dict("records"))
    allow = _engine(config, structure_break_only="allow").analyze(md)
    pen = _engine(config, structure_break_only="penalize", structure_break_penalty=0.10).analyze(md)
    assert allow.trigger == pen.trigger == "STRUCTURE" and allow.direction == 1
    assert pen.score.overall == pytest.approx(max(0.0, allow.score.overall - 0.10), abs=1e-4)
    assert "structure fade" in pen.score.description


def test_decision_log_new_columns_never_mix_with_old_file(tmp_path):
    from server.decision_log import DecisionLog
    now = datetime(2024, 1, 16, 15, 0, tzinfo=timezone.utc)
    (tmp_path / "decisions_20240116.csv").write_text("logged_at,symbol,action\nx,y,z\n")
    p = DecisionLog(tmp_path).append({"symbol": "XAUUSD", "trigger": "SWEEP"}, now=now)
    assert p.name == "decisions_20240116_2.csv"
    assert DecisionLog(tmp_path).append({"symbol": "XAUUSD"}, now=now).name == "decisions_20240116_2.csv"


# ---------------------------------------------------------------- recent sweep (SB sequence)
def _recent_sweep_frame(after_closes):
    rows = [dict(open=100.0, high=100.5, low=99.5, close=100.0) for _ in range(25)]
    rows.append(dict(open=99.8, high=99.9, low=99.0, close=99.7))         # sweep + reclaim of 99.5
    rows += [dict(open=c, high=c + 0.2, low=c - 0.2, close=c) for c in after_closes]
    df = pd.DataFrame(rows); df["time"] = np.arange(len(df)) * 900.0; df["tick_volume"] = 1
    return df


def test_sweep_found_on_recent_bar_and_must_stay_reclaimed(config):
    from strategy_engine import liquidity
    cfg = copy.deepcopy(config); cfg["thresholds"]["liquidity_sweep"]["recent_bars"] = 4
    held = liquidity.detect_sweep(_recent_sweep_frame([99.9, 100.1]), 1.0, cfg)
    assert held.swept and held.sweep_direction == 1 and held.bars_ago == 2 and held.reclaimed
    lost = liquidity.detect_sweep(_recent_sweep_frame([99.9, 99.3]), 1.0, cfg)  # closed back below
    assert lost.swept and not lost.reclaimed
    cfg["thresholds"]["liquidity_sweep"]["recent_bars"] = 1                       # legacy behaviour
    assert not liquidity.detect_sweep(_recent_sweep_frame([99.9, 100.1]), 1.0, cfg).swept


def test_sl_sits_beyond_the_whole_sweep_when_sweep_is_older(config, monkeypatch):
    """SL uses the extreme since the sweep bar, not just the latest bar's low."""
    from strategy_engine import StrategyEngine, MarketData
    df15 = _recent_sweep_frame([99.9, 100.1, 100.2])
    df1h = df15.iloc[::4].reset_index(drop=True)
    df1 = _m1(list(np.linspace(100.4, 100.2, 30)))
    cfg = copy.deepcopy(config); cfg["entry_filters"] = {}
    cfg["thresholds"]["liquidity_sweep"]["recent_bars"] = 4
    r = StrategyEngine(cfg).analyze(MarketData("XAUUSD", df1h, df15, df1))
    assert r.direction == 1 and r.trigger == "SWEEP" and r.liquidity.bars_ago == 3
    assert r.stoploss.price < 99.0


def test_collector_killzone_prefilter_gives_identical_samples(config, synthetic_market, monkeypatch):
    """Skipping out-of-killzone bars before analysis must not change the result."""
    from strategy_engine import StrategyEngine
    from collector.collect_training_data import collect_for_symbol
    cfg = copy.deepcopy(config); cfg["min_setup_score"] = 0.0
    cfg["entry_filters"] = {"require_killzone": True, "killzones": ["NY", "LONDON", "ASIAN", "NY_AM_SB",
                                                                   "NY_PM_SB", "LONDON_KZ"]}
    m = synthetic_market
    args = (m["df_1h"], m["df_15m"], m["df_1m"], cfg)
    fast = collect_for_symbol(StrategyEngine(cfg), "XAUUSD", "XAUUSD", *args, future_bars_1m=240,
                              stride=1, clock_mode="utc")
    eng = StrategyEngine(cfg)
    monkeypatch.setattr(eng, "killzone_blocks", lambda ts: False)          # no prefilter
    slow = collect_for_symbol(eng, "XAUUSD", "XAUUSD", *args, future_bars_1m=240, stride=1, clock_mode="utc")
    assert [s["timestamp"] for s in fast] == [s["timestamp"] for s in slow]
