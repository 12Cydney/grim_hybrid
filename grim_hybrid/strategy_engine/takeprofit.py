"""
strategy_engine.takeprofit
==========================
Dynamic TP evaluator.  Considers:
  - Opposite liquidity pool (buy-side vs sell-side)
  - Session extremes (PDH/PDL, PWH/PWL) [approx via lookback]
  - ADR-based target
  - R:R floor from config.risk.min_rr

Picks the closest target that satisfies min R:R.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import pandas as pd

from strategy_engine.liquidity import LiquidityResult
from strategy_engine.premium_discount import PremiumDiscountResult
from strategy_engine.utils import safe_div


@dataclass
class TakeProfitCandidate:
    kind: str            # 'LIQUIDITY' | 'PDH' | 'PDL' | 'RANGE' | 'ADR' | 'RR_FLOOR'
    price: float
    distance: float
    rr: float            # reward-to-risk


@dataclass
class TakeProfitResult:
    chosen: Optional[TakeProfitCandidate] = None
    candidates: list = field(default_factory=list)
    kind: str = "RR_FLOOR"
    price: float = 0.0
    distance: float = 0.0
    rr: float = 0.0


def _adr(df: pd.DataFrame, days: int = 5) -> float:
    """Approximate Average Daily Range from H1 data (24 bars per day)."""
    if df is None or len(df) < 24:
        return 0.0
    bars = 24 * days
    window = df.tail(bars)
    day_groups = [window.iloc[i:i + 24] for i in range(0, len(window), 24)]
    ranges = [float(g["high"].max() - g["low"].min()) for g in day_groups if len(g) >= 4]
    return sum(ranges) / len(ranges) if ranges else 0.0


def evaluate(
    entry: float,
    direction: int,
    sl_distance: float,
    df_1h: pd.DataFrame,
    liquidity: LiquidityResult,
    premium_discount: PremiumDiscountResult,
    config: Mapping,
    atr_value: float = 0.0,
) -> TakeProfitResult:
    res = TakeProfitResult()
    if direction not in (-1, 1) or entry <= 0 or sl_distance <= 0:
        return res

    min_rr = float(config["risk"]["min_rr"])

    def add(kind: str, price: float):
        if price <= 0:
            return
        d = (price - entry) if direction == 1 else (entry - price)
        if d <= 0:
            return
        rr = safe_div(d, sl_distance, 0.0)
        res.candidates.append(TakeProfitCandidate(kind, price, d, rr))

    # 1) Opposite liquidity pool
    if direction == 1 and liquidity.bsl_level > 0:
        add("LIQUIDITY", liquidity.bsl_level)
    elif direction == -1 and liquidity.ssl_level > 0:
        add("LIQUIDITY", liquidity.ssl_level)

    # 2) Range extremes as PDH/PDL proxy (from premium/discount analysis)
    if direction == 1 and premium_discount.range_high > 0:
        add("RANGE_HIGH", premium_discount.range_high)
    elif direction == -1 and premium_discount.range_low > 0:
        add("RANGE_LOW", premium_discount.range_low)

    # 3) PDH / PDL (last 24 H1 bars = previous day)
    if df_1h is not None and len(df_1h) >= 24:
        pd_hi = float(df_1h["high"].tail(24).max())
        pd_lo = float(df_1h["low"].tail(24).min())
        if direction == 1:
            add("PDH", pd_hi)
        else:
            add("PDL", pd_lo)

    # 4) ADR-based target
    adr = _adr(df_1h) if df_1h is not None else 0.0
    if adr > 0:
        if direction == 1:
            add("ADR", entry + 0.5 * adr)
        else:
            add("ADR", entry - 0.5 * adr)

    # Filter candidates by min R:R and pick the CLOSEST valid one (highest
    # probability of getting hit)
    valid = [c for c in res.candidates if c.rr >= min_rr]
    if valid:
        chosen = min(valid, key=lambda c: c.distance)
    else:
        # Fallback: force the R:R floor
        dist = sl_distance * min_rr
        price = entry + dist if direction == 1 else entry - dist
        chosen = TakeProfitCandidate("RR_FLOOR", price, dist, min_rr)
        res.candidates.append(chosen)

    res.chosen = chosen
    res.kind = chosen.kind
    res.price = chosen.price
    res.distance = chosen.distance
    res.rr = chosen.rr
    return res
