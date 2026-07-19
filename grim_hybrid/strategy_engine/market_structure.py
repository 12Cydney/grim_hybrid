"""
strategy_engine.market_structure
================================
Swings, Break of Structure (BOS), Market Structure Shift (MSS) and
Change of Character (CHOCH).  Ported 1:1 from the legacy SMCDetector
then extended with proper swing detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class Swing:
    idx: int
    price: float
    kind: str    # 'HIGH' or 'LOW'


@dataclass
class StructureResult:
    bos_direction: int = 0            # +1 bullish break, -1 bearish break, 0 none
    bos_age: int = 0                  # bars since the break
    mss: int = 0                      # +1/-1 shift, 0 none (opposite to prior BOS)
    choch: int = 0                    # change of character: prior swing flipped
    broke_high: bool = False
    broke_low: bool = False
    high_break_size: float = 0.0
    low_break_size: float = 0.0
    recent_high: float = 0.0
    recent_low: float = 0.0
    swings_high: list = field(default_factory=list)
    swings_low: list = field(default_factory=list)


def _find_swings(df: pd.DataFrame, left: int, right: int) -> tuple[list[Swing], list[Swing]]:
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    sh: list[Swing] = []
    sl: list[Swing] = []
    for i in range(left, n - right):
        wh = highs[i - left:i + right + 1]
        wl = lows[i - left:i + right + 1]
        if highs[i] == wh.max() and (wh == highs[i]).sum() == 1:
            sh.append(Swing(i, float(highs[i]), "HIGH"))
        if lows[i] == wl.min() and (wl == lows[i]).sum() == 1:
            sl.append(Swing(i, float(lows[i]), "LOW"))
    return sh, sl


def detect_bos_legacy(df: pd.DataFrame, lookback: int = 10) -> tuple[int, int]:
    """1:1 port of legacy SMCDetector.detect_bos."""
    if len(df) < lookback + 1:
        return 0, 0
    highs = df["high"].values
    lows = df["low"].values
    for i in range(len(df) - lookback, len(df)):
        if i < lookback:
            continue
        rh = highs[i - lookback:i].max()
        if highs[i] > rh:
            return 1, len(df) - i
        rl = lows[i - lookback:i].min()
        if lows[i] < rl:
            return -1, len(df) - i
    return 0, 0


def detect_swing_break(df: pd.DataFrame, lookback: int = 20) -> tuple[bool, bool, float, float, float, float]:
    """1:1 port of legacy detect_swing_break.  Returns
    (broke_high, broke_low, high_break_size, low_break_size, recent_high, recent_low)."""
    if len(df) < lookback + 1:
        return False, False, 0.0, 0.0, 0.0, 0.0
    current = df.iloc[-1]
    recent = df.iloc[-(lookback + 1):-1]
    rh = float(recent["high"].max())
    rl = float(recent["low"].min())
    bh = bool(current["high"] > rh)
    bl = bool(current["low"] < rl)
    hbs = float(current["high"] - rh) if bh else 0.0
    lbs = float(rl - current["low"]) if bl else 0.0
    return bh, bl, hbs, lbs, rh, rl


def analyze(df: pd.DataFrame, config: Mapping, level: str = "MTF") -> StructureResult:
    """High-level structure analysis for HTF/MTF/LTF.

    ``level`` selects the lookback to use from config.market_structure.
    """
    ms = config["thresholds"]["market_structure"]
    if level == "HTF":
        lb = int(ms["htf_lookback"])
    elif level == "LTF":
        lb = int(ms["ltf_lookback"])
    else:
        lb = int(ms["mtf_lookback"])
    left = int(ms["swing_left"])
    right = int(ms["swing_right"])

    res = StructureResult()
    if len(df) < lb + 1:
        return res

    bos_dir, bos_age = detect_bos_legacy(df, lookback=lb)
    res.bos_direction = bos_dir
    res.bos_age = bos_age

    bh, bl, hbs, lbs, rh, rl = detect_swing_break(df, lookback=lb)
    res.broke_high = bh
    res.broke_low = bl
    res.high_break_size = hbs
    res.low_break_size = lbs
    res.recent_high = rh
    res.recent_low = rl

    try:
        sh, sl = _find_swings(df, left, right)
        res.swings_high = [(s.idx, s.price) for s in sh[-5:]]
        res.swings_low  = [(s.idx, s.price) for s in sl[-5:]]

        # MSS: current BOS opposite to the prior confirmed swing extreme
        if bos_dir == 1 and sh:
            res.mss = 1 if df["high"].iloc[-1] > max(p for _, p in res.swings_high) else 0
        elif bos_dir == -1 and sl:
            res.mss = -1 if df["low"].iloc[-1] < min(p for _, p in res.swings_low) else 0

        # CHOCH: last two same-side swings flipped
        if len(sh) >= 2 and sh[-1].price < sh[-2].price and bos_dir == -1:
            res.choch = -1
        elif len(sl) >= 2 and sl[-1].price > sl[-2].price and bos_dir == 1:
            res.choch = 1
    except Exception:
        pass

    return res
