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
    reclaimed: bool = False         # sweep bar CLOSED back inside the swept level (and stayed inside)
    bars_ago: int = 0               # 0 = the sweep is on the latest closed bar


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


def _sweep_at(df: pd.DataFrame, i: int, lookback: int, min_depth: float) -> LiquidityResult | None:
    """Sweep printed by bar ``i`` against the ``lookback`` bars before it."""
    if i < lookback:
        return None
    bar = df.iloc[i]
    window = df.iloc[i - lookback:i]
    rh, rl = float(window["high"].max()), float(window["low"].min())
    res = LiquidityResult(bsl_level=rh, ssl_level=rl)
    later_closes = df["close"].iloc[i:]                 # sweep bar close .. latest close

    low_depth = float(rl - bar["low"]) if bar["low"] < rl else 0.0
    high_depth = float(bar["high"] - rh) if bar["high"] > rh else 0.0
    if low_depth >= min_depth and low_depth > 0 and low_depth >= high_depth:
        res.swept, res.sweep_depth, res.sweep_direction, res.liquidity_type = True, low_depth, 1, "SELLSIDE"
        # reclaimed = closed back above the swept low AND has stayed above it since
        res.reclaimed = bool((later_closes > rl).all())
    elif high_depth >= min_depth and high_depth > 0:
        res.swept, res.sweep_depth, res.sweep_direction, res.liquidity_type = True, high_depth, -1, "BUYSIDE"
        res.reclaimed = bool((later_closes < rh).all())
    else:
        return None
    return res


def detect_sweep(df: pd.DataFrame, atr_value: float, config: Mapping) -> LiquidityResult:
    """Detect a liquidity sweep on one of the last ``recent_bars`` bars.

    A sweep = a bar prints beyond the preceding ``lookback`` bars' high/low
    by at least ``atr_factor * ATR`` (dynamic threshold).  The most recent
    sweep wins.  ``reclaimed`` = the sweep bar closed back inside the swept
    level and every close since has stayed inside (a true stop-run); an
    unreclaimed sweep is a breakout, NOT a reversal signal.

    ``recent_bars`` > 1 lets the Silver Bullet sequence (sweep -> 1m shift
    -> retrace into the FVG) play out over the following bars.
    """
    conf = config["thresholds"]["liquidity_sweep"]
    lookback = int(conf["lookback"])
    factor = float(conf["atr_factor"])
    eq_tol = float(conf["equal_hl_tolerance_atr"])
    recent_bars = max(1, int(conf.get("recent_bars", 1)))

    res = LiquidityResult()
    if len(df) < lookback + 1:
        return res

    # default levels = relative to the current bar (as before)
    cur_window = df.iloc[-(lookback + 1):-1]
    res.bsl_level = float(cur_window["high"].max())
    res.ssl_level = float(cur_window["low"].min())

    min_depth = factor * atr_value if atr_value > 0 else 0.0
    n = len(df)
    for k in range(recent_bars):                       # newest bar first
        hit = _sweep_at(df, n - 1 - k, lookback, min_depth)
        if hit is not None:
            hit.bars_ago = k
            res = hit
            break

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
