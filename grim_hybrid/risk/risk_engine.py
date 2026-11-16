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

import pytz

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
        self.tz = pytz.timezone(config.get("sessions", {}).get("timezone", "America/New_York"))

    # -----------------------------------------------------------------
    def trading_day(self, now: Optional[datetime] = None) -> str:
        """Trading date in the session timezone (NY), not UTC - a UTC reset
        lands at 19:00/20:00 NY, mid-way through the Asian session."""
        now = now or datetime.now(tz=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(self.tz).strftime("%Y-%m-%d")

    def _reset_if_new_day(self, state: AccountState, now: Optional[datetime] = None) -> None:
        today = self.trading_day(now)
        if state.last_reset != today:
            state.last_reset = today
            state.daily_trades = 0
            state.daily_pnl_pct = 0.0

    # -----------------------------------------------------------------
    def guardrails(self, state: AccountState, now: Optional[datetime] = None) -> tuple[bool, str]:
        """Pre-trade checks against daily limits / exposure."""
        self._reset_if_new_day(state, now)
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


# ---------------------------------------------------------------------
# Live position book (fed by the EA's POSITION_STATE message)
# ---------------------------------------------------------------------
class PositionBook:
    """Aggregates per-symbol EA reports into one ``AccountState``.

    Each EA (one per chart/symbol) reports its open trades, floating P&L
    and its recent deals every timer tick.  Deals are de-duplicated by
    ticket, so re-sending is idempotent, and bucketed by the NY trading
    day.  The book is persisted so a server restart keeps the daily caps.
    """

    def __init__(self, risk: RiskEngine, path: Optional["Path"] = None):
        import threading
        from pathlib import Path
        self.risk = risk
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.symbols: dict[str, dict] = {}     # symbol -> latest snapshot
        self.deals: dict[str, dict] = {}       # ticket -> {symbol, time_utc, entry, pnl}
        self.load()

    # -- persistence --------------------------------------------------
    def load(self) -> None:
        import json
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.symbols = dict(data.get("symbols", {}))
            self.deals = {str(k): v for k, v in data.get("deals", {}).items()}
        except Exception as e:   # corrupt file -> start clean, never crash the server
            print(f"[risk] could not load {self.path}: {e}")

    def save(self, now: Optional[datetime] = None) -> None:
        import json
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            st = self.state(now)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"symbols": self.symbols, "deals": self.deals,
                                       "account": st.__dict__}, indent=2))
            tmp.replace(self.path)
        except Exception as e:
            print(f"[risk] could not save {self.path}: {e}")

    # -- updates -------------------------------------------------------
    def update(self, report: Mapping, now: Optional[datetime] = None) -> AccountState:
        now = now or datetime.now(tz=timezone.utc)
        sym = str(report.get("symbol", "UNKNOWN")).upper()
        offset = float(report.get("gmt_offset", 0) or 0)   # broker server time - GMT, seconds
        with self.lock:
            self.symbols[sym] = {
                "balance": float(report.get("balance", 0) or 0),
                "equity": float(report.get("equity", 0) or 0),
                "open_trades": int(report.get("open_trades", 0) or 0),
                "floating_pnl": float(report.get("floating_pnl", 0) or 0),
                "updated_utc": now.timestamp(),
            }
            for d in report.get("deals", []) or []:
                try:
                    self.deals[str(int(d["ticket"]))] = {
                        "symbol": sym,
                        "time_utc": float(d["time"]) - offset,
                        "entry": str(d.get("entry", "OTHER")).upper(),
                        "pnl": float(d.get("pnl", 0) or 0),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
            # keep the book small: drop deals older than 3 days
            cutoff = now.timestamp() - 3 * 86400
            self.deals = {k: v for k, v in self.deals.items() if v["time_utc"] >= cutoff}
            self.save(now)
            return self.state(now)

    # -- aggregation ---------------------------------------------------
    def state(self, now: Optional[datetime] = None) -> AccountState:
        now = now or datetime.now(tz=timezone.utc)
        day = self.risk.trading_day(now)
        today = [d for d in self.deals.values()
                 if self.risk.trading_day(datetime.fromtimestamp(d["time_utc"], tz=timezone.utc)) == day]
        realized = sum(d["pnl"] for d in today)
        floating = sum(s["floating_pnl"] for s in self.symbols.values())
        snaps = sorted(self.symbols.values(), key=lambda s: s["updated_utc"])
        balance = snaps[-1]["balance"] if snaps and snaps[-1]["balance"] > 0 else AccountState.balance
        equity = snaps[-1]["equity"] if snaps and snaps[-1]["equity"] > 0 else balance
        start_balance = balance - realized
        return AccountState(
            balance=balance,
            equity=equity,
            open_trades=sum(s["open_trades"] for s in self.symbols.values()),
            daily_trades=sum(1 for d in today if d["entry"] == "IN"),
            daily_pnl_pct=safe_div(realized + floating, start_balance, 0.0) * 100.0,
            last_reset=day,
        )
