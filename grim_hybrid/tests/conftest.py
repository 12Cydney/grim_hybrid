"""
Shared pytest fixtures & synthetic data helpers.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from strategy_engine.utils import load_config
from strategy_engine import StrategyEngine


@pytest.fixture(scope="session")
def config():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def engine(config):
    return StrategyEngine(config)


def _random_walk(bars, tf_seconds, base_price=2000.0, seed=42, impulse_ratio=60):
    rng = np.random.default_rng(seed)
    # Anchor to a known NY silver-bullet slot: 2024-01-15 (Mon) 14:30 UTC ~= 09:30 NY
    end = 1_705_329_600  # 2024-01-15 14:00 UTC
    start = end - bars * tf_seconds
    times = np.array([start + i * tf_seconds for i in range(bars)], dtype=float)
    steps = rng.normal(0, base_price * 0.0008, size=bars)
    impulse_idx = rng.choice(bars, size=max(1, bars // impulse_ratio), replace=False)
    steps[impulse_idx] *= 8.0
    closes = base_price + np.cumsum(steps)
    opens = np.concatenate([[base_price], closes[:-1]])
    hn = np.abs(rng.normal(0, base_price * 0.0005, size=bars))
    ln = np.abs(rng.normal(0, base_price * 0.0005, size=bars))
    highs = np.maximum(opens, closes) + hn
    lows = np.minimum(opens, closes) - ln
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": rng.integers(50, 500, size=bars),
    })


@pytest.fixture
def synthetic_market():
    df_1m = _random_walk(bars=60 * 24 * 5, tf_seconds=60, base_price=2000.0, seed=7)

    def resample(rule):
        d = df_1m.copy()
        d["dt"] = pd.to_datetime(d["time"], unit="s")
        d = d.set_index("dt")
        r = pd.DataFrame({
            "open":  d["open"].resample(rule).first(),
            "high":  d["high"].resample(rule).max(),
            "low":   d["low"].resample(rule).min(),
            "close": d["close"].resample(rule).last(),
            "tick_volume": d["tick_volume"].resample(rule).sum(),
        }).dropna()
        r["time"] = (r.index.view("int64") // 10**9) if str(r.index.dtype) == "datetime64[ns]" else r.index.view("int64")
        return r.reset_index(drop=True)[["time", "open", "high", "low", "close", "tick_volume"]]

    return {
        "symbol": "XAUUSD",
        "df_1h":  resample("1h"),
        "df_15m": resample("15min"),
        "df_1m":  df_1m,
    }
