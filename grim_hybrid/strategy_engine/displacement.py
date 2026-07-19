"""
strategy_engine.displacement
============================
Impulsive-move / imbalance detection.  A displacement candle is one
whose range is >= ``min_atr_multiple * ATR`` and whose body fills at
least ``min_body_ratio`` of that range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from strategy_engine.utils import safe_div


@dataclass
class DisplacementResult:
    present: bool = False
    direction: int = 0          # +1 bullish, -1 bearish
    body_ratio: float = 0.0     # 0-1
    range_atr: float = 0.0      # candle range / ATR
    strength: float = 0.0       # 0-1


def calculate(df: pd.DataFrame, atr_value: float, config: Mapping) -> DisplacementResult:
    conf = config["thresholds"]["displacement"]
    period = int(conf["period"])
    min_body = float(conf["min_body_ratio"])
    min_mult = float(conf["min_atr_multiple"])

    res = DisplacementResult()
    if len(df) < period or atr_value <= 0:
        return res

    window = df.tail(period)
    total_range = float((window["high"] - window["low"]).sum())
    total_body = float((window["close"] - window["open"]).abs().sum())
    body_ratio = safe_div(total_body, total_range, 0.0)

    last = df.iloc[-1]
    last_range = float(last["high"] - last["low"])
    range_atr = safe_div(last_range, atr_value, 0.0)

    res.body_ratio = body_ratio
    res.range_atr = range_atr
    res.direction = 1 if last["close"] > last["open"] else (-1 if last["close"] < last["open"] else 0)

    if body_ratio >= min_body and range_atr >= min_mult:
        res.present = True
        # strength = how far above the minimums we are
        res.strength = min(
            1.0,
            0.5 * (body_ratio / max(min_body, 1e-6)) * (range_atr / max(min_mult, 1e-6)),
        )
    else:
        res.strength = min(1.0, 0.5 * body_ratio + 0.5 * min(range_atr / max(min_mult, 1e-6), 1.0))

    return res


def displacement_score(res: DisplacementResult) -> float:
    return float(max(0.0, min(1.0, res.strength)))
