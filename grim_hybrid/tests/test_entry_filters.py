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
