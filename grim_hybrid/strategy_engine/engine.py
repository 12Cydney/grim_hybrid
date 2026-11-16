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

    def _volume_ratio(self, df_1m: pd.DataFrame) -> float:
        if "tick_volume" not in df_1m.columns or len(df_1m) < 20:
            return 1.0
        recent = float(df_1m["tick_volume"].tail(20).mean())
        cur = float(df_1m["tick_volume"].iloc[-1])
        return safe_div(cur, recent, 1.0)

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
        result.displacement_1m = displacement_mod.calculate(df_1m, result.atr_15m, self.config)
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
        result.ltf_confirmed = confirmed
        result.ltf_reversal_strength = strength

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

        # 10) Setup validity gate
        min_score = float(self.config["min_setup_score"])
        has_setup = bool(swept or broke_high or broke_low)
        if not has_setup:
            result.hold_reason = "No structure/liquidity trigger"
            result.valid = False
        elif direction == 0:
            result.hold_reason = "No clear direction"
            result.valid = False
        elif result.score.overall < min_score:
            result.hold_reason = f"Score {result.score.overall:.2f} below floor {min_score:.2f}"
            result.valid = False
        else:
            result.valid = True
            result.hold_reason = ""

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
