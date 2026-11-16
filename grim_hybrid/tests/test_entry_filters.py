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
