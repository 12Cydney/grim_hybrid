"""
strategy_engine.fvg
===================
Fair Value Gap detection.  We port the legacy 3-candle FVG rule and add
a size / age / fill-percent evaluation, then filter by ATR-relative
minimum size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from strategy_engine.utils import safe_div


@dataclass
class FVGResult:
    present: bool = False
    direction: int = 0           # +1 bullish, -1 bearish
    size: float = 0.0            # absolute price units
    size_atr: float = 0.0        # size / ATR
    age: int = 0                 # bars since it formed
    fill_pct: float = 0.0        # 0-1, how much of it has been filled
    upper: float = 0.0
    lower: float = 0.0
    all_gaps: list = field(default_factory=list)  # [(idx, dir, size, upper, lower), ...]


def _detect_last_3(df: pd.DataFrame) -> tuple[bool, int, float, float, float]:
    """Legacy 3-candle FVG on the *last* 3 candles."""
    if len(df) < 3:
        return False, 0, 0.0, 0.0, 0.0
    c1, c3 = df.iloc[-3], df.iloc[-1]
    if c1["high"] < c3["low"]:
        return True, 1, float(c3["low"] - c1["high"]), float(c3["low"]), float(c1["high"])
    if c1["low"] > c3["high"]:
        return True, -1, float(c1["low"] - c3["high"]), float(c1["low"]), float(c3["high"])
    return False, 0, 0.0, 0.0, 0.0


def _scan_recent(df: pd.DataFrame, max_bars: int) -> list[tuple[int, int, float, float, float]]:
    """Scan the last ``max_bars`` bars for FVGs.  Returns most-recent first."""
    n = len(df)
    if n < 3:
        return []
    out: list[tuple[int, int, float, float, float]] = []
    start = max(2, n - max_bars)
    for i in range(start, n):
        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]
        if c1["high"] < c3["low"]:
            out.append((i, 1, float(c3["low"] - c1["high"]), float(c3["low"]), float(c1["high"])))
        elif c1["low"] > c3["high"]:
            out.append((i, -1, float(c1["low"] - c3["high"]), float(c1["low"]), float(c3["high"])))
    return list(reversed(out))


def detect(df: pd.DataFrame, atr_value: float, config: Mapping) -> FVGResult:
    conf = config["thresholds"]["fvg"]
    min_size = float(conf["min_size_atr_factor"]) * atr_value if atr_value > 0 else 0.0
    max_age = int(conf["max_age_bars"])

    res = FVGResult()
    all_gaps = _scan_recent(df, max_age + 3)
    res.all_gaps = all_gaps

    if not all_gaps:
        return res

    n = len(df)
    # pick the most recent gap that clears the size filter
    for (idx, direction, size, upper, lower) in all_gaps:
        if size < min_size:
            continue
        res.present = True
        res.direction = direction
        res.size = size
        res.size_atr = safe_div(size, atr_value, 0.0)
        res.age = n - 1 - idx
        res.upper = upper
        res.lower = lower

        # fill %: how much of the FVG has been eaten by wicks after formation
        if direction == 1:
            after = df.iloc[idx + 1:] if idx + 1 < n else df.iloc[0:0]
            min_low = float(after["low"].min()) if len(after) else lower
            filled = max(0.0, upper - min_low)
        else:
            after = df.iloc[idx + 1:] if idx + 1 < n else df.iloc[0:0]
            max_high = float(after["high"].max()) if len(after) else upper
            filled = max(0.0, max_high - lower)
        res.fill_pct = float(min(1.0, safe_div(filled, size, 0.0)))
        break

    return res


def fvg_score(res: FVGResult) -> float:
    if not res.present:
        return 0.0
    # size factor: bigger gaps = better (capped at 3x ATR)
    size_component = min(1.0, res.size_atr / 3.0)
    # age factor: fresher gaps are stronger
    age_component = max(0.0, 1.0 - res.age / 40.0)
    # fill factor: mostly-unfilled gaps are strongest
    fill_component = max(0.0, 1.0 - res.fill_pct)
    return float(min(1.0, 0.4 * size_component + 0.3 * age_component + 0.3 * fill_component))
