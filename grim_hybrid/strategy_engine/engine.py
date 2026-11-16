"""
strategy_engine.engine
======================

The public entrypoint for the whole system.

Usage
-----
    from strategy_engine import StrategyEngine, MarketData

    engine = StrategyEngine(config)                # config is dict from utils.load_config
    setup  = engine.analyze(MarketData(
        symbol="XAUUSD",
        data_1h=[{...}], data_15m=[{...}], data_1m=[{...}],
    ))
    if setup.valid:
        feats = engine.build_features(setup)       # ML feature dict

Every downstream module (server, collector, trainer, backtester) MUST
call these two methods.  No trading logic outside this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from strategy_engine import (
    displacement as displacement_mod,
    fvg as fvg_mod,
    liquidity as liquidity_mod,
    market_structure as ms_mod,
    order_blocks as ob_mod,
    premium_discount as pd_mod,
    sessions as sess_mod,
    stoploss as sl_mod,
    takeprofit as tp_mod,
    trade_score as score_mod,
    volatility as vol_mod,
)
from strategy_engine.feature_builder import build_features as _build_features, feature_names
from strategy_engine.utils import pip_value_for, to_ohlc_df, safe_div, unix_seconds


@dataclass
class MarketData:
    """Input container passed to ``engine.analyze()``."""
    symbol: str
    data_1h: Any
    data_15m: Any
    data_1m: Any
    spread: float = 0.0
    timestamp: Optional[float] = None    # unix seconds; else uses last 1m bar time


@dataclass
class SetupResult:
    """Everything the engine knows about the current market moment."""
    # Context
    symbol: str = ""
    valid: bool = False              # engine says there IS a setup to score
    direction: int = 0               # +1 long, -1 short, 0 none
    entry: float = 0.0
    timestamp: float = 0.0
    pip_value: float = 0.0
    hold_reason: str = ""

    # Volatility (both timeframes)
    atr_15m: float = 0.0
    atr_1h: float = 0.0
    vol_15m: Any = None
    vol_1h: Any = None

    # Session
    session: Any = None

    # Structure
    structure_1h: Any = None
    structure_15m: Any = None
    structure_1m: Any = None

    # Detections
    liquidity: Any = None
    fvg: Any = None
    order_block: Any = None
    displacement_1h: Any = None
    displacement_15m: Any = None
    displacement_1m: Any = None
    premium_discount: Any = None

    # LTF confirmation
    ltf_confirmed: bool = False
    ltf_reversal_strength: float = 0.0

    # Entry-filter context (config.entry_filters)
    trigger: str = "NONE"            # 'SWEEP' | 'STRUCTURE' | 'NONE' - what set the direction
    in_killzone: bool = False
    filter_failed: str = ""          # first entry filter that blocked the setup ('' = passed)
    poi_kind: str = ""               # 'FVG' | 'OB' | '' - aligned 15m point of interest
    poi_top: float = 0.0
    poi_bottom: float = 0.0
    poi_tapped: bool = False         # 1m price retraced into the POI without closing through it

    # Scoring
    score: Any = None

    # Volume
    volume_ratio: float = 1.0

    # SL/TP
    stoploss: Any = None
    takeprofit: Any = None

    # ML hints (populated by server AFTER model call, not by engine)
    ml_prediction: Optional[int] = None
    ml_confidence: Optional[float] = None
    ml_model_type: Optional[str] = None


class StrategyEngine:
    """The single source of truth for trading logic."""

    def __init__(self, config: dict):
        self.config = config
        self.session_resolver = sess_mod.SessionResolver(config)

    # -----------------------------------------------------------------
    # LTF confirmation - kept in engine.py (small, ports legacy check)
    # -----------------------------------------------------------------
    def _ltf_confirmation(self, df_1m: pd.DataFrame, direction: int) -> tuple[bool, float]:
        conf = self.config["thresholds"]["reversal"]
        count = int(conf["ltf_confirm_count"])
        min_needed = int(conf["min_reversal_candles"])
        if len(df_1m) < count or direction == 0:
            return False, 0.0
        candles = df_1m.iloc[-count:]
        bull = int((candles["close"] > candles["open"]).sum())
        bear = int((candles["close"] < candles["open"]).sum())
        if direction == 1:
            confirmed = bull >= min_needed
            strength = bull / count
        else:
            confirmed = bear >= min_needed
            strength = bear / count
        return bool(confirmed), float(strength)

    @staticmethod
    def _ltf_mss(df_1m: pd.DataFrame, direction: int, strength: int = 2, window: int = 10) -> bool:
        """1m market-structure shift in ``direction``.

        Long: a confirmed 1m swing high (``strength`` bars each side) is broken
        by a 1m CLOSE within the last ``window`` bars, and the lowest low since
        that swing printed BEFORE the break (the shift comes off the low).
        Short is the mirror image.
        """
        n = len(df_1m)
        if direction == 0 or n < 2 * strength + 2:
            return False
        hi = df_1m["high"].to_numpy(float)
        lo = df_1m["low"].to_numpy(float)
        cl = df_1m["close"].to_numpy(float)
        k = strength
        swings = []
        for j in range(k, n - k):
            seg = hi[j - k:j + k + 1] if direction == 1 else lo[j - k:j + k + 1]
            v = seg[k]
            if (direction == 1 and v == seg.max() and (seg == v).sum() == 1) or \
               (direction == -1 and v == seg.min() and (seg == v).sum() == 1):
                swings.append(j)
        for j in reversed(swings[-3:]):          # newest swings first
            level = hi[j] if direction == 1 else lo[j]
            after = cl[j + 1:]
            hits = np.nonzero(after > level)[0] if direction == 1 else np.nonzero(after < level)[0]
            if not len(hits):
                continue
            m = j + 1 + int(hits[0])             # first close through the swing
            if m < n - window:
                continue                         # shift is too old
            leg = lo[j:m + 1] if direction == 1 else hi[j:m + 1]
            extreme = j + int(np.argmin(leg) if direction == 1 else np.argmax(leg))
            if extreme < m:
                return True
        return False

    def _find_poi(self, df_15m: pd.DataFrame, r: SetupResult) -> tuple[str, float, float]:
        """Most recent 15m FVG or OB aligned with ``r.direction`` that price
        has not CLOSED through since it formed.  Returns (kind, top, bottom)."""
        d = r.direction
        if d == 0 or len(df_15m) < 3:
            return "", 0.0, 0.0
        n = len(df_15m)
        closes = df_15m["close"].to_numpy(float)
        th = self.config["thresholds"]
        min_fvg = float(th["fvg"]["min_size_atr_factor"]) * r.atr_15m
        max_fvg_age = int(th["fvg"]["max_age_bars"])

        def intact(idx: int, top: float, bot: float) -> bool:
            after = closes[idx + 1:]
            return not len(after) or (after.min() >= bot if d == 1 else after.max() <= top)

        cands: list[tuple[int, str, float, float]] = []
        for (idx, gdir, size, upper, lower) in (r.fvg.all_gaps if r.fvg else []):
            if gdir == d and size >= min_fvg and n - 1 - idx <= max_fvg_age and intact(idx, upper, lower):
                cands.append((idx, "FVG", upper, lower))
                break                                   # gaps are most-recent first
        ob = r.order_block
        if ob is not None and ob.present and ob.direction == d:
            idx = n - 1 - int(ob.age)
            if intact(idx, ob.top, ob.bottom):
                cands.append((idx, "OB", ob.top, ob.bottom))
        if not cands:
            return "", 0.0, 0.0
        _, kind, top, bot = max(cands, key=lambda c: c[0])   # most recent zone
        return kind, float(top), float(bot)

    @staticmethod
    def _poi_tapped(df_1m: pd.DataFrame, direction: int, top: float, bottom: float,
                    touch_bars: int, tol: float) -> bool:
        """Long: a recent 1m low reached into [bottom, top] and the last 1m close
        is still above the zone bottom.  Short mirrors it."""
        if direction == 0 or top <= 0 or len(df_1m) == 0:
            return False
        recent = df_1m.tail(max(1, touch_bars))
        last = float(df_1m["close"].iloc[-1])
        if direction == 1:
            return float(recent["low"].min()) <= top + tol and last >= bottom - tol
        return float(recent["high"].max()) >= bottom - tol and last <= top + tol

    def _volume_ratio(self, df_1m: pd.DataFrame) -> float:
        if "tick_volume" not in df_1m.columns or len(df_1m) < 20:
            return 1.0
        recent = float(df_1m["tick_volume"].tail(20).mean())
        cur = float(df_1m["tick_volume"].iloc[-1])
        return safe_div(cur, recent, 1.0)

    # -----------------------------------------------------------------
    # Entry filters (config.entry_filters) - each one switchable
    # -----------------------------------------------------------------
    def _entry_filters(self, r: SetupResult) -> tuple[str, str]:
        """Return (filter_name, hold_reason) of the first failing filter, or ('', '')."""
        ef = self.config.get("entry_filters", {}) or {}
        killzones = [str(k).upper() for k in ef.get("killzones", ["LONDON_KZ", "NY_AM_SB", "NY_PM_SB"])]
        r.in_killzone = bool(r.session and r.session.name in killzones)

        if ef.get("require_killzone", False) and not r.in_killzone:
            return "killzone", f"Outside killzone ({r.session.name if r.session else 'OFF'})"
        if ef.get("require_ltf_confirmation", False) and not r.ltf_confirmed:
            mode = str(ef.get("ltf_confirmation", "candles")).lower()
            return "ltf", "No 1m market-structure shift" if mode == "mss" else "No 1m confirmation"
        if ef.get("require_poi", False) and not r.poi_kind:
            return "poi", "No aligned FVG/OB"
        if ef.get("require_poi_retrace", False) and r.poi_kind and not r.poi_tapped:
            return "poi_retrace", f"No retrace into {r.poi_kind}"
        return "", ""

    # -----------------------------------------------------------------
    # Public entrypoint
    # -----------------------------------------------------------------
    def analyze(self, md: MarketData) -> SetupResult:
        """Run the full ICT / Silver Bullet analysis and return a SetupResult."""
        result = SetupResult(symbol=md.symbol)
        result.pip_value = pip_value_for(md.symbol, self.config)

        try:
            df_1h = to_ohlc_df(md.data_1h)
            df_15m = to_ohlc_df(md.data_15m)
            df_1m = to_ohlc_df(md.data_1m)
        except Exception as e:
            result.hold_reason = f"Bad market data: {e}"
            return result

        if df_1m.empty or df_15m.empty or df_1h.empty:
            result.hold_reason = "Insufficient bars"
            return result

        # Timestamp: prefer explicit, else last 1m bar
        result.timestamp = unix_seconds(md.timestamp) if md.timestamp else float(df_1m["time"].iloc[-1])
        result.entry = float(df_1m["close"].iloc[-1])

        # 1) Session
        result.session = self.session_resolver.resolve(result.timestamp)

        # 2) Volatility profiles
        result.vol_15m = vol_mod.profile(df_15m, self.config)
        result.vol_1h = vol_mod.profile(df_1h, self.config)
        result.atr_15m = result.vol_15m.atr
        result.atr_1h = result.vol_1h.atr

        # 3) Market structure (HTF, MTF, LTF)
        result.structure_1h = ms_mod.analyze(df_1h, self.config, level="HTF")
        result.structure_15m = ms_mod.analyze(df_15m, self.config, level="MTF")
        result.structure_1m = ms_mod.analyze(df_1m, self.config, level="LTF")

        # 4) Liquidity
        result.liquidity = liquidity_mod.detect_sweep(df_15m, result.atr_15m, self.config)

        # 5) FVG / OB / Displacement / PD
        result.fvg = fvg_mod.detect(df_15m, result.atr_15m, self.config)
        result.order_block = ob_mod.detect(df_15m, result.atr_15m, self.config)
        result.displacement_15m = displacement_mod.calculate(df_15m, result.atr_15m, self.config)
        result.displacement_1h = displacement_mod.calculate(df_1h, result.atr_1h, self.config)
        # 1m displacement is measured against 1m ATR (15m ATR made every 1m candle look tiny)
        atr_1m, _ = vol_mod.atr(df_1m, int(self.config["thresholds"]["volatility"]["atr_period"]))
        result.displacement_1m = displacement_mod.calculate(df_1m, atr_1m, self.config)
        result.premium_discount = pd_mod.analyze(df_1h, self.config)

        # 6) Direction inference (same rule set as legacy, but volatility-based sweep)
        broke_high = result.structure_15m.broke_high
        broke_low = result.structure_15m.broke_low
        swept = result.liquidity.swept
        sd = result.liquidity.sweep_direction

        dconf = self.config.get("direction", {}) or {}
        need_reclaim = bool(dconf.get("require_sweep_reclaim", True))
        block_counter_htf = bool(dconf.get("block_counter_htf_bias", True))
        htf = result.structure_1h.bos_direction

        # A sweep is directional only once price reclaims the swept level;
        # an unreclaimed sweep is a breakout, not a reversal.
        sweep_long = swept and sd == 1 and (result.liquidity.reclaimed or not need_reclaim)
        sweep_short = swept and sd == -1 and (result.liquidity.reclaimed or not need_reclaim)
        # Structure breaks are faded only when HTF bias isn't hard against the fade.
        fade_long = broke_low and not (block_counter_htf and htf == -1)
        fade_short = broke_high and not (block_counter_htf and htf == 1)

        direction = 0
        if fade_long or sweep_long:
            direction = 1
        elif fade_short or sweep_short:
            direction = -1

        result.direction = direction

        # 7) LTF confirmation
        confirmed, strength = self._ltf_confirmation(df_1m, direction)
        ef = self.config.get("entry_filters", {}) or {}
        if str(ef.get("ltf_confirmation", "candles")).lower() == "mss":
            confirmed = self._ltf_mss(df_1m, direction, int(ef.get("ltf_swing_strength", 2)),
                                      int(ef.get("ltf_mss_window", 10)))
        result.ltf_confirmed = confirmed
        result.ltf_reversal_strength = strength

        # 7b) Point of interest (aligned FVG / OB) + 1m retrace into it
        result.poi_kind, result.poi_top, result.poi_bottom = self._find_poi(df_15m, result)
        if result.poi_kind:
            result.poi_tapped = self._poi_tapped(
                df_1m, direction, result.poi_top, result.poi_bottom,
                int(ef.get("poi_touch_bars", 15)), float(ef.get("poi_tolerance_atr", 0.10)) * result.atr_15m)

        # 8) Volume
        result.volume_ratio = self._volume_ratio(df_1m)

        # 9) Sub-scores → overall setup grade
        components = {
            "bias": self._bias_score(result),
            "liquidity": liquidity_mod.liquidity_score(result.liquidity),
            "displacement": displacement_mod.displacement_score(result.displacement_15m),
            "fvg": self._fvg_score_aligned(result),
            "order_block": self._ob_score_aligned(result),
            "session": 1.0 if result.session.silver_bullet else (0.7 if result.session.active else 0.2),
            "volatility": vol_mod.volatility_score(result.vol_15m),
            "premium_discount": pd_mod.premium_discount_score(result.premium_discount, direction),
            "ltf_confirmation": 1.0 if confirmed else (0.5 * strength),
        }
        result.score = score_mod.combine(components, self.config)

        # 10) Setup validity gate: trigger -> direction -> entry filters -> score floor
        result.trigger = ("SWEEP" if (sweep_long and direction == 1) or (sweep_short and direction == -1)
                          else ("STRUCTURE" if direction != 0 else "NONE"))
        min_score = float(self.config["min_setup_score"])
        has_setup = bool(swept or broke_high or broke_low)
        if not has_setup:
            result.hold_reason = "No structure/liquidity trigger"
        elif direction == 0:
            result.hold_reason = "No clear direction"
        else:
            result.filter_failed, result.hold_reason = self._entry_filters(result)
            if not result.hold_reason and result.score.overall < min_score:
                result.hold_reason = f"Score {result.score.overall:.2f} below floor {min_score:.2f}"
        result.valid = not result.hold_reason

        # 11) SL / TP (always compute if we have a direction, useful for logging)
        if direction != 0 and result.atr_15m > 0:
            result.stoploss = sl_mod.evaluate(
                entry=result.entry,
                direction=direction,
                atr_value=result.atr_15m,
                liquidity=result.liquidity,
                structure=result.structure_15m,
                order_block=result.order_block,
                config=self.config,
                signal_high=float(df_15m["high"].iloc[-1]),
                signal_low=float(df_15m["low"].iloc[-1]),
            )
            result.takeprofit = tp_mod.evaluate(
                entry=result.entry,
                direction=direction,
                sl_distance=result.stoploss.distance if result.stoploss.chosen else result.atr_15m,
                df_1h=df_1h,
                liquidity=result.liquidity,
                premium_discount=result.premium_discount,
                config=self.config,
                atr_value=result.atr_15m,
            )
        else:
            # Placeholders so feature_builder never blows up
            result.stoploss = sl_mod.StopLossResult()
            result.takeprofit = tp_mod.TakeProfitResult()

        return result

    # -----------------------------------------------------------------
    # ML features (single source of truth)
    # -----------------------------------------------------------------
    def build_features(self, setup: SetupResult) -> dict[str, float]:
        return _build_features(setup, self.config)

    @staticmethod
    def feature_names() -> list[str]:
        return feature_names()

    # -----------------------------------------------------------------
    # Sub-scores that need cross-module info live here
    # -----------------------------------------------------------------
    def _bias_score(self, r: SetupResult) -> float:
        htf = r.structure_1h.bos_direction
        mtf = r.structure_15m.bos_direction
        if htf == 0:
            return 0.3
        if r.direction == htf:
            return 1.0 if mtf in (0, htf) else 0.7
        if mtf == htf:
            return 0.5
        return 0.2

    def _fvg_score_aligned(self, r: SetupResult) -> float:
        base = fvg_mod.fvg_score(r.fvg)
        if not r.fvg.present or r.direction == 0:
            return base
        # Bonus for direction alignment
        return min(1.0, base + 0.15) if r.fvg.direction == r.direction else max(0.0, base - 0.2)

    def _ob_score_aligned(self, r: SetupResult) -> float:
        base = ob_mod.order_block_score(r.order_block)
        if not r.order_block.present or r.direction == 0:
            return base
        return min(1.0, base + 0.15) if r.order_block.direction == r.direction else max(0.0, base - 0.2)
