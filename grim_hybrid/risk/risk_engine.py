"""
risk.risk_engine
================
Position sizing, exposure caps, breakeven and trailing.  Consumes AI's
suggested risk % and applies the guardrails from ``config.risk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional

from strategy_engine.utils import clamp, pip_value_for, safe_div


@dataclass
class RiskDecision:
    lot_size: float = 0.0
    risk_pct: float = 0.0
    risk_amount: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_distance: float = 0.0
    tp_distance: float = 0.0
    rr: float = 0.0
    breakeven_at_r: float = 1.0
    trailing_start_r: float = 1.5
    trailing_step_atr: float = 0.5
    allow_trade: bool = True
    reason: str = ""


@dataclass
class AccountState:
    balance: float = 10_000.0
    equity: float = 10_000.0
    open_trades: int = 0
    daily_trades: int = 0
    daily_pnl_pct: float = 0.0
    last_reset: str = ""    # YYYY-MM-DD (NY) for daily counters


class RiskEngine:
    def __init__(self, config: Mapping):
        self.config = config
        self.rconf = config["risk"]

    # -----------------------------------------------------------------
    def _reset_if_new_day(self, state: AccountState) -> None:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if state.last_reset != today:
            state.last_reset = today
            state.daily_trades = 0
            state.daily_pnl_pct = 0.0

    # -----------------------------------------------------------------
    def guardrails(self, state: AccountState) -> tuple[bool, str]:
        """Pre-trade checks against daily limits / exposure."""
        self._reset_if_new_day(state)
        if state.open_trades >= int(self.rconf["max_concurrent_trades"]):
            return False, f"Max concurrent trades ({self.rconf['max_concurrent_trades']}) reached"
        if state.daily_trades >= int(self.rconf["max_daily_trades"]):
            return False, f"Max daily trades ({self.rconf['max_daily_trades']}) reached"
        if state.daily_pnl_pct <= -float(self.rconf["max_daily_loss_pct"]):
            return False, f"Daily loss cap ({self.rconf['max_daily_loss_pct']}%) hit"
        return True, ""

    # -----------------------------------------------------------------
    def size(
        self,
        symbol: str,
        entry: float,
        sl_price: float,
        tp_price: float,
        confidence: float,
        state: AccountState,
        ai_suggested_risk_pct: Optional[float] = None,
        contract_size: float = 100_000.0,   # 1 lot = 100k units (FX default)
    ) -> RiskDecision:
        """Return a fully-populated RiskDecision.

        We deliberately don't try to be a broker.  ``contract_size`` is a
        reasonable default; the EA re-derives lots from tick value on its
        end (this method still returns a sane approximation).
        """
        dec = RiskDecision(sl_price=sl_price, tp_price=tp_price)

        allow, why = self.guardrails(state)
        if not allow:
            dec.allow_trade = False
            dec.reason = why
            return dec

        # Distances
        sl_dist = abs(entry - sl_price)
        tp_dist = abs(tp_price - entry)
        dec.sl_distance = sl_dist
        dec.tp_distance = tp_dist
        dec.rr = safe_div(tp_dist, sl_dist, 0.0)

        # Risk % - base * confidence scaling, clamped, override by AI suggestion
        base = float(self.rconf["base_risk_pct"])
        risk_pct = base * clamp(confidence, 0.5, 1.0)
        if ai_suggested_risk_pct is not None and ai_suggested_risk_pct > 0:
            risk_pct = float(ai_suggested_risk_pct)
        risk_pct = clamp(risk_pct, float(self.rconf["min_risk_pct"]), float(self.rconf["max_risk_pct"]))
        dec.risk_pct = risk_pct
        dec.risk_amount = state.balance * risk_pct / 100.0

        # Lot sizing (approximate). tick_value ~= pip_value * contract_size / price
        # For simplicity: money_at_risk = lots * sl_dist_price * contract_size / entry
        # -> lots = money_at_risk * entry / (sl_dist_price * contract_size)
        if sl_dist > 0 and entry > 0:
            lots = dec.risk_amount * entry / (sl_dist * contract_size)
        else:
            lots = 0.0
        dec.lot_size = round(max(lots, 0.01), 2)

        dec.breakeven_at_r = float(self.rconf["breakeven_r"])
        dec.trailing_start_r = float(self.rconf["trailing_start_r"])
        dec.trailing_step_atr = float(self.rconf["trailing_step_atr"])
        dec.allow_trade = True
        return dec

    # -----------------------------------------------------------------
    def should_move_to_be(self, current_r: float, dec: RiskDecision) -> bool:
        return current_r >= dec.breakeven_at_r

    def new_trailing_sl(self, entry: float, current_price: float, direction: int, atr: float, dec: RiskDecision) -> float:
        step = dec.trailing_step_atr * atr
        if direction == 1:
            return max(entry, current_price - step)
        return min(entry, current_price + step)
