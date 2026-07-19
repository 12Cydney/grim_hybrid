"""Risk engine tests."""
from risk.risk_engine import RiskEngine, AccountState


def test_position_sizing_respects_min_max(config):
    r = RiskEngine(config)
    state = AccountState(balance=10_000)
    dec = r.size("EURUSD", entry=1.10, sl_price=1.095, tp_price=1.11,
                 confidence=0.8, state=state)
    assert dec.allow_trade is True
    assert dec.risk_pct >= config["risk"]["min_risk_pct"]
    assert dec.risk_pct <= config["risk"]["max_risk_pct"]
    assert dec.rr > 0
    assert dec.lot_size > 0


def test_ai_suggested_risk_pct_wins_within_bounds(config):
    r = RiskEngine(config)
    state = AccountState(balance=10_000)
    dec = r.size("XAUUSD", entry=2000, sl_price=1990, tp_price=2020,
                 confidence=0.9, state=state, ai_suggested_risk_pct=1.5)
    assert dec.risk_pct == 1.5


def test_daily_trade_cap_blocks(config):
    from datetime import datetime, timezone
    r = RiskEngine(config)
    state = AccountState(balance=10_000)
    state.last_reset = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    state.daily_trades = int(config["risk"]["max_daily_trades"])
    dec = r.size("XAUUSD", entry=2000, sl_price=1990, tp_price=2020,
                 confidence=0.8, state=state)
    assert dec.allow_trade is False
    assert "daily trades" in dec.reason.lower()


def test_concurrent_trade_cap_blocks(config):
    r = RiskEngine(config)
    state = AccountState(balance=10_000)
    state.open_trades = int(config["risk"]["max_concurrent_trades"])
    dec = r.size("XAUUSD", entry=2000, sl_price=1990, tp_price=2020,
                 confidence=0.8, state=state)
    assert dec.allow_trade is False


def test_breakeven_trigger(config):
    r = RiskEngine(config)
    state = AccountState()
    dec = r.size("XAUUSD", entry=2000, sl_price=1990, tp_price=2020,
                 confidence=0.8, state=state)
    assert r.should_move_to_be(current_r=1.1, dec=dec) is True
    assert r.should_move_to_be(current_r=0.5, dec=dec) is False
