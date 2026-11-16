"""
strategy_engine.order_blocks
============================
Detect bullish / bearish order blocks - the last opposite-color candle
before a displacement move that broke structure.  Quality scored by
body size (vs ATR), age, and mitigation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from strategy_engine.utils import safe_div


@dataclass
class OrderBlockResult:
    present: bool = False
    direction: int = 0        # +1 bullish OB (long entry zone), -1 bearish
    top: float = 0.0
    bottom: float = 0.0
    age: int = 0
    body_atr: float = 0.0     # ob body / ATR
    mitigated: bool = False   # price returned into the OB after formation
    quality: float = 0.0      # 0-1


def _candle_body_range(c) -> tuple[float, float, float]:
    body = float(abs(c["close"] - c["open"]))
    top = float(max(c["open"], c["close"]))
    bot = float(min(c["open"], c["close"]))
    return body, top, bot


def detect(df: pd.DataFrame, atr_value: float, config: Mapping) -> OrderBlockResult:
    conf = config["thresholds"]["order_block"]
    lookback = int(conf["lookback"])
    min_body_atr = float(conf["min_body_atr_factor"])
    max_age = int(conf["max_age_bars"])

    res = OrderBlockResult()
    if len(df) < lookback + 3 or atr_value <= 0:
        return res

    n = len(df)
    scan_start = max(2, n - max_age - 3)
    best: OrderBlockResult | None = None
    best_score = -1.0

    for i in range(scan_start, n - 2):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]
        nxt2 = df.iloc[i + 2]

        cur_bull = cur["close"] > cur["open"]
        cur_bear = cur["close"] < cur["open"]

        # Bullish OB: last down candle before a strong up move
        if cur_bear and (nxt["close"] > cur["high"] or nxt2["close"] > cur["high"]):
            body, top, bot = _candle_body_range(cur)
            body_atr = safe_div(body, atr_value, 0.0)
            if body_atr < min_body_atr:
                continue
            age = n - 1 - i
            # i+1 is the displacement candle itself - its low always sits back
            # inside the OB, so mitigation is measured from i+2 onward.
            after = df.iloc[i + 2:]
            mitigated = bool(len(after) and after["low"].min() <= top)
            quality = min(1.0, 0.4 * min(body_atr, 2.0) + 0.3 * (1 - age / max_age) + (0.0 if mitigated else 0.3))
            if quality > best_score:
                best_score = quality
                best = OrderBlockResult(True, 1, top, bot, age, body_atr, mitigated, quality)

        elif cur_bull and (nxt["close"] < cur["low"] or nxt2["close"] < cur["low"]):
            body, top, bot = _candle_body_range(cur)
            body_atr = safe_div(body, atr_value, 0.0)
            if body_atr < min_body_atr:
                continue
            age = n - 1 - i
            after = df.iloc[i + 2:]
            mitigated = bool(len(after) and after["high"].max() >= bot)
            quality = min(1.0, 0.4 * min(body_atr, 2.0) + 0.3 * (1 - age / max_age) + (0.0 if mitigated else 0.3))
            if quality > best_score:
                best_score = quality
                best = OrderBlockResult(True, -1, top, bot, age, body_atr, mitigated, quality)

    return best or res


def order_block_score(res: OrderBlockResult) -> float:
    return float(res.quality if res.present else 0.0)
