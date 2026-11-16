"""
strategy_engine.stoploss
========================
Multi-location SL evaluator.  Considers:
  - Behind swept liquidity (the invalidation point)
  - Behind the last swing high/low
  - Behind the mitigated order block edge
  - ATR-scaled floor / ceiling from config.risk

Returns candidate distances (price units, not pips) and picks the
"safest logical" one that is inside the ATR bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from strategy_engine.liquidity import LiquidityResult
from strategy_engine.market_structure import StructureResult
from strategy_engine.order_blocks import OrderBlockResult
from strategy_engine.utils import safe_div


@dataclass
class StopLossCandidate:
    kind: str            # 'LIQUIDITY' | 'STRUCTURE' | 'ORDER_BLOCK' | 'SWEEP' | 'ATR'
    price: float
    distance: float      # absolute price units from entry


@dataclass
class StopLossResult:
    chosen: Optional[StopLossCandidate] = None
    candidates: list = field(default_factory=list)
    kind: str = "ATR"
    price: float = 0.0
    distance: float = 0.0
    atr_multiple: float = 0.0


def _clamp_by_atr(distance: float, atr_value: float, config: Mapping) -> float:
    r = config["risk"]
    lo = float(r["sl_min_atr"]) * atr_value
    hi = float(r["sl_max_atr"]) * atr_value
    return max(lo, min(hi, distance))


def evaluate(
    entry: float,
    direction: int,
    atr_value: float,
    liquidity: LiquidityResult,
    structure: StructureResult,
    order_block: OrderBlockResult,
    config: Mapping,
    buffer_atr_frac: float = 0.1,
    signal_high: float = 0.0,
    signal_low: float = 0.0,
) -> StopLossResult:
    """Pick the safest logical SL.  ``direction`` is +1 (long) / -1 (short).

    ``signal_high`` / ``signal_low`` = extremes of the signal (sweep) bar.
    A candidate the signal bar already traded through would sit INSIDE the
    sweep - it's discarded, and "behind the signal bar" is added instead.
    """
    res = StopLossResult()
    if direction not in (-1, 1) or entry <= 0 or atr_value <= 0:
        return res

    buffer = buffer_atr_frac * atr_value

    def add(kind: str, price: float):
        if price <= 0:
            return
        d = (entry - price) if direction == 1 else (price - entry)
        if d <= 0:
            return
        # already violated by the signal bar -> would be inside the sweep
        if direction == 1 and signal_low > 0 and price >= signal_low:
            return
        if direction == -1 and signal_high > 0 and price <= signal_high:
            return
        res.candidates.append(StopLossCandidate(kind, price, d))

    if direction == 1:
        # Behind swept sell-side liquidity
        if liquidity.swept and liquidity.sweep_direction == 1 and liquidity.ssl_level > 0:
            add("LIQUIDITY", liquidity.ssl_level - buffer)
        elif liquidity.ssl_level > 0:
            add("LIQUIDITY", liquidity.ssl_level - buffer)

        # Behind last swing low
        if structure.recent_low > 0:
            add("STRUCTURE", structure.recent_low - buffer)

        # Behind bullish OB bottom
        if order_block.present and order_block.direction == 1 and order_block.bottom > 0:
            add("ORDER_BLOCK", order_block.bottom - buffer)

        # Behind the signal bar's low (always beyond the sweep)
        if signal_low > 0:
            add("SWEEP", signal_low - buffer)

        # ATR fallback (below entry)
        add("ATR", entry - 1.5 * atr_value)
    else:  # short
        if liquidity.swept and liquidity.sweep_direction == -1 and liquidity.bsl_level > 0:
            add("LIQUIDITY", liquidity.bsl_level + buffer)
        elif liquidity.bsl_level > 0:
            add("LIQUIDITY", liquidity.bsl_level + buffer)

        if structure.recent_high > 0:
            add("STRUCTURE", structure.recent_high + buffer)

        if order_block.present and order_block.direction == -1 and order_block.top > 0:
            add("ORDER_BLOCK", order_block.top + buffer)

        if signal_high > 0:
            add("SWEEP", signal_high + buffer)

        add("ATR", entry + 1.5 * atr_value)

    if not res.candidates:
        return res

    # Choose the tightest candidate that respects ATR bounds (safest = tightest
    # valid invalidation).
    r = config["risk"]
    lo = float(r["sl_min_atr"]) * atr_value
    hi = float(r["sl_max_atr"]) * atr_value

    valid = [c for c in res.candidates if lo <= c.distance <= hi]
    if valid:
        chosen = min(valid, key=lambda c: c.distance)
    else:
        # None valid -> take widest but clamp
        widest = max(res.candidates, key=lambda c: c.distance)
        clamped = _clamp_by_atr(widest.distance, atr_value, config)
        chosen = StopLossCandidate(widest.kind + "_CLAMPED", entry - clamped if direction == 1 else entry + clamped, clamped)

    res.chosen = chosen
    res.kind = chosen.kind
    res.price = chosen.price
    res.distance = chosen.distance
    res.atr_multiple = safe_div(chosen.distance, atr_value, 0.0)
    return res
