"""
strategy_engine.volatility
==========================
ATR / average-candle / spread / market-speed calculations.  Everything
that turns a "pip threshold" into a volatility-relative threshold lives
here.  No trading logic - just numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from strategy_engine.utils import safe_div


@dataclass
class VolatilityProfile:
    atr: float               # current ATR (absolute price units)
    avg_atr: float           # long-run mean of ATR
    atr_ratio: float         # atr / avg_atr, 1.0 == "normal"
    avg_candle: float        # mean candle range over the same lookback
    speed: float             # bars-per-move momentum proxy in [0, ~2]
    regime: str              # 'LOW', 'NORMAL', 'HIGH', 'EXTREME'


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> tuple[float, float]:
    """Return (current_atr, long_run_avg_atr).  0-safe."""
    if len(df) < max(period + 1, 2):
        return 0.0, 0.0
    tr = _true_range(df)
    rolling = tr.rolling(period).mean()
    current = float(rolling.iloc[-1]) if not np.isnan(rolling.iloc[-1]) else 0.0
    avg = float(rolling.mean()) if len(rolling.dropna()) > 0 else 0.0
    return current, avg


def average_candle_range(df: pd.DataFrame, lookback: int = 20) -> float:
    if df.empty:
        return 0.0
    window = df.tail(lookback)
    return float((window["high"] - window["low"]).mean())


def market_speed(df: pd.DataFrame, lookback: int = 20) -> float:
    """Momentum proxy: net directional move / summed candle range.

    Returns a value roughly in [0, 2].  >1 = strong trend, <0.3 = chop.
    """
    if len(df) < 2:
        return 0.0
    window = df.tail(lookback)
    net = float(window["close"].iloc[-1] - window["open"].iloc[0])
    total_range = float((window["high"] - window["low"]).sum())
    return safe_div(abs(net) * lookback, total_range, 0.0)


def profile(df: pd.DataFrame, config: Mapping) -> VolatilityProfile:
    """Full volatility profile used by other modules for dynamic thresholds."""
    tconf = config["thresholds"]["volatility"]
    period = int(tconf["atr_period"])
    lookback = int(tconf["speed_lookback"])

    cur, avg = atr(df, period)
    ratio = safe_div(cur, avg, 1.0) if avg > 0 else 1.0
    avg_c = average_candle_range(df, lookback)
    spd = market_speed(df, lookback)

    if ratio < 0.7:
        regime = "LOW"
    elif ratio < 1.3:
        regime = "NORMAL"
    elif ratio < 1.8:
        regime = "HIGH"
    else:
        regime = "EXTREME"

    return VolatilityProfile(cur, avg, ratio, avg_c, spd, regime)


# ---------------------------------------------------------------------
# Dynamic threshold helpers - called by every module that needs a pip
# equivalent.  Never return "pips" - return absolute price distances.
# ---------------------------------------------------------------------
def dynamic_sweep_depth(atr_value: float, config: Mapping) -> float:
    """Minimum sweep depth (price units) = atr_factor * ATR."""
    factor = float(config["thresholds"]["liquidity_sweep"]["atr_factor"])
    return factor * atr_value


def dynamic_fvg_min_size(atr_value: float, config: Mapping) -> float:
    factor = float(config["thresholds"]["fvg"]["min_size_atr_factor"])
    return factor * atr_value


def dynamic_ob_min_body(atr_value: float, config: Mapping) -> float:
    factor = float(config["thresholds"]["order_block"]["min_body_atr_factor"])
    return factor * atr_value


def volatility_score(vp: VolatilityProfile) -> float:
    """0-1 sub-score.  We reward NORMAL/HIGH, penalise LOW and EXTREME."""
    if vp.regime == "NORMAL":
        return 1.0
    if vp.regime == "HIGH":
        return 0.85
    if vp.regime == "LOW":
        return 0.35
    return 0.25  # EXTREME - unpredictable
