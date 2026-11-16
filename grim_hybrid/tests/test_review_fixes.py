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
