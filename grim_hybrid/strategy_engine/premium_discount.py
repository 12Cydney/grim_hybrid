"""
strategy_engine.premium_discount
================================
Premium/Discount/Equilibrium position within a swing range + OTE zone
(0.618-0.786 fib retracement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class PremiumDiscountResult:
    position: float = 0.0        # -1 (deep discount) .. +1 (deep premium)
    zone: str = "EQUILIBRIUM"    # 'PREMIUM' | 'DISCOUNT' | 'EQUILIBRIUM'
    range_high: float = 0.0
    range_low: float = 0.0
    midpoint: float = 0.0
    in_ote_long: bool = False    # in 0.618-0.786 discount for longs
    in_ote_short: bool = False   # in 0.618-0.786 premium for shorts


def analyze(df: pd.DataFrame, config: Mapping) -> PremiumDiscountResult:
    conf = config["thresholds"]["premium_discount"]
    lookback = int(conf["lookback"])
    ote_lo = float(conf["ote_low"])
    ote_hi = float(conf["ote_high"])

    res = PremiumDiscountResult()
    if len(df) == 0:
        return res

    lb = min(lookback, len(df))
    window = df.tail(lb)
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    price = float(df["close"].iloc[-1])
    rng = hi - lo
    if rng <= 0:
        return res

    mid = (hi + lo) / 2.0
    pos = float(np.clip((price - mid) / (rng / 2), -1.0, 1.0))
    res.position = pos
    res.range_high = hi
    res.range_low = lo
    res.midpoint = mid

    if pos > 0.2:
        res.zone = "PREMIUM"
    elif pos < -0.2:
        res.zone = "DISCOUNT"
    else:
        res.zone = "EQUILIBRIUM"

    # OTE long: retrace INTO discount range - 0.618 to 0.786 measured from swing high
    ote_long_hi = hi - ote_lo * rng    # 0.618 down from high
    ote_long_lo = hi - ote_hi * rng    # 0.786 down from high
    res.in_ote_long = bool(ote_long_lo <= price <= ote_long_hi)

    ote_short_lo = lo + ote_lo * rng
    ote_short_hi = lo + ote_hi * rng
    res.in_ote_short = bool(ote_short_lo <= price <= ote_short_hi)
    return res


def premium_discount_score(res: PremiumDiscountResult, direction: int) -> float:
    """Score based on how well the price zone aligns with intended direction."""
    if direction == 1:
        if res.in_ote_long:
            return 1.0
        if res.zone == "DISCOUNT":
            return 0.8
        if res.zone == "EQUILIBRIUM":
            return 0.5
        return 0.2
    if direction == -1:
        if res.in_ote_short:
            return 1.0
        if res.zone == "PREMIUM":
            return 0.8
        if res.zone == "EQUILIBRIUM":
            return 0.5
        return 0.2
    return 0.4
