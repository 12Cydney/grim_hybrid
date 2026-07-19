"""
strategy_engine.liquidity
=========================
Buy-side / Sell-side liquidity pools, equal highs/lows, and volatility-
driven sweep detection.  Threshold is derived from ATR (config.thresholds
.liquidity_sweep.atr_factor) - NO hardcoded pip values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from strategy_engine.utils import safe_div


@dataclass
class LiquidityResult:
    swept: bool = False
    sweep_depth: float = 0.0        # absolute price units (not pips)
    sweep_direction: int = 0        # +1 = swept lows -> bullish setup
    liquidity_type: str = "NONE"    # 'BUYSIDE' | 'SELLSIDE' | 'NONE'
    strength: float = 0.0           # 0-1 relative to ATR
    equal_highs: bool = False
    equal_lows: bool = False
    bsl_level: float = 0.0          # buy-side liquidity (recent high)
    ssl_level: float = 0.0          # sell-side liquidity (recent low)


def detect_equal_levels(df: pd.DataFrame, atr_value: float, tolerance_factor: float, lookback: int = 20) -> tuple[bool, bool]:
    if len(df) < lookback:
        return False, False
    window = df.tail(lookback)
    tol = tolerance_factor * atr_value if atr_value > 0 else 0.0
    highs = window["high"].values
    lows = window["low"].values

    top = highs.max()
    bottom = lows.min()

    # count how many highs/lows are within `tol` of the extreme
    near_top = ((highs >= top - tol) & (highs <= top + tol)).sum()
    near_bot = ((lows >= bottom - tol) & (lows <= bottom + tol)).sum()
    return bool(near_top >= 2), bool(near_bot >= 2)


def detect_sweep(df: pd.DataFrame, atr_value: float, config: Mapping) -> LiquidityResult:
    """Detect a liquidity sweep in the last bar of ``df``.

    A sweep = current bar prints beyond the recent high/low by at least
    ``atr_factor * ATR`` (dynamic threshold).
    """
    conf = config["thresholds"]["liquidity_sweep"]
    lookback = int(conf["lookback"])
    factor = float(conf["atr_factor"])
    eq_tol = float(conf["equal_hl_tolerance_atr"])

    res = LiquidityResult()
    if len(df) < lookback + 1:
        return res

    current = df.iloc[-1]
    recent = df.iloc[-(lookback + 1):-1]
    rh = float(recent["high"].max())
    rl = float(recent["low"].min())
    res.bsl_level = rh
    res.ssl_level = rl

    min_depth = factor * atr_value if atr_value > 0 else 0.0

    # --- Sweep below recent lows -> bullish setup (direction = +1)
    swept_low_depth = 0.0
    if current["low"] < rl:
        swept_low_depth = float(rl - current["low"])
        if swept_low_depth >= min_depth:
            res.swept = True
            res.sweep_depth = swept_low_depth
            res.sweep_direction = 1
            res.liquidity_type = "SELLSIDE"

    # --- Sweep above recent highs -> bearish setup (direction = -1)
    if current["high"] > rh:
        swept_high_depth = float(current["high"] - rh)
        if swept_high_depth >= min_depth:
            if not res.swept or swept_high_depth > swept_low_depth:
                res.swept = True
                res.sweep_depth = swept_high_depth
                res.sweep_direction = -1
                res.liquidity_type = "BUYSIDE"

    if res.swept and atr_value > 0:
        res.strength = min(1.0, safe_div(res.sweep_depth, atr_value * 2.0, 0.0))

    eh, el = detect_equal_levels(df, atr_value, eq_tol, lookback=lookback)
    res.equal_highs = eh
    res.equal_lows = el
    return res


def liquidity_score(res: LiquidityResult) -> float:
    """0-1 score.  Sweep + equal-level liquidity pool = maximum."""
    if not res.swept:
        return 0.0
    s = 0.6 + 0.4 * res.strength  # baseline for any sweep
    if (res.sweep_direction == 1 and res.equal_lows) or (res.sweep_direction == -1 and res.equal_highs):
        s = min(1.0, s + 0.15)
    return float(min(1.0, s))
