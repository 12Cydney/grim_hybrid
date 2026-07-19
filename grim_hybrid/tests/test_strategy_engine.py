"""Unit tests for individual strategy_engine modules."""
import pandas as pd
import pytest

from strategy_engine import volatility, liquidity, fvg, displacement, order_blocks
from strategy_engine import market_structure, premium_discount, sessions, trade_score
from strategy_engine.utils import to_ohlc_df, pip_value_for


def test_config_loads_all_symbols(config):
    assert "XAUUSD" in config["symbols"]
    assert "NAS100" in config["symbols"]
    assert pip_value_for("XAUUSD", config) == 0.01
    assert pip_value_for("US30", config) == 1.0
    assert pip_value_for("EURUSD", config) == 0.0001


def test_to_ohlc_df_normalises(config):
    raw = [{"time": 1_700_000_000, "open": 100, "high": 101, "low": 99, "close": 100.5}]
    df = to_ohlc_df(raw)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "tick_volume"]
    assert df["time"].dtype == float


def test_volatility_profile(config, synthetic_market):
    vp = volatility.profile(synthetic_market["df_15m"], config)
    assert vp.atr > 0
    assert vp.avg_atr > 0
    assert vp.regime in ("LOW", "NORMAL", "HIGH", "EXTREME")
    assert 0 <= volatility.volatility_score(vp) <= 1


def test_dynamic_thresholds_are_atr_relative(config):
    atr = 2.5
    depth = volatility.dynamic_sweep_depth(atr, config)
    assert depth == config["thresholds"]["liquidity_sweep"]["atr_factor"] * atr
    fvg_size = volatility.dynamic_fvg_min_size(atr, config)
    assert fvg_size == config["thresholds"]["fvg"]["min_size_atr_factor"] * atr


def test_liquidity_detects_sweep_when_price_takes_prior_low(config, synthetic_market):
    df = synthetic_market["df_15m"].copy()
    # Force a sweep: push last bar's low below prior 20-bar min by 3*ATR
    atr = volatility.profile(df, config).atr
    prev_low = df["low"].iloc[-21:-1].min()
    df.loc[df.index[-1], "low"] = prev_low - 3 * atr
    df.loc[df.index[-1], "close"] = prev_low + 0.5 * atr
    res = liquidity.detect_sweep(df, atr, config)
    assert res.swept is True
    assert res.sweep_direction == 1
    assert res.sweep_depth > 0
    assert 0 < liquidity.liquidity_score(res) <= 1


def test_liquidity_no_sweep_on_quiet_bar(config, synthetic_market):
    df = synthetic_market["df_15m"].copy()
    atr = volatility.profile(df, config).atr
    # Make last bar sit inside prior range
    mid = (df["high"].iloc[-21:-1].max() + df["low"].iloc[-21:-1].min()) / 2
    df.loc[df.index[-1], "open"] = mid
    df.loc[df.index[-1], "close"] = mid + atr * 0.05
    df.loc[df.index[-1], "high"] = mid + atr * 0.1
    df.loc[df.index[-1], "low"] = mid - atr * 0.1
    res = liquidity.detect_sweep(df, atr, config)
    assert res.swept is False


def test_fvg_detects_gap(config):
    # Craft a 3-candle bullish FVG: c1.high < c3.low
    rows = []
    base = 2000.0
    t0 = 1_700_000_000
    for i, (o, h, l, c) in enumerate([
        (base, base + 1, base - 1, base + 0.5),        # c1
        (base + 1, base + 5, base, base + 4),          # c2 impulse
        (base + 4, base + 6, base + 3, base + 5),      # c3 low=3 > c1.high=1 -> gap
    ]):
        rows.append({"time": t0 + i * 900, "open": o, "high": h, "low": l, "close": c, "tick_volume": 100})
    df = pd.DataFrame(rows)
    res = fvg.detect(df, atr_value=0.5, config=config)
    assert res.present is True
    assert res.direction == 1
    assert res.size > 0


def test_displacement_flags_big_body(config):
    rows = []
    atr = 1.0
    base = 100.0
    for i in range(15):
        rows.append({"time": 1_700_000_000 + i * 900, "open": base, "high": base + 0.2,
                     "low": base - 0.2, "close": base + 0.05, "tick_volume": 100})
        base += 0.05
    # Final displacement candle: big body, big range
    rows.append({"time": rows[-1]["time"] + 900, "open": base, "high": base + 3.5,
                 "low": base - 0.1, "close": base + 3.4, "tick_volume": 500})
    df = pd.DataFrame(rows)
    res = displacement.calculate(df, atr_value=atr, config=config)
    assert res.direction == 1
    assert res.range_atr >= 1.2


def test_market_structure_detects_break(config, synthetic_market):
    res = market_structure.analyze(synthetic_market["df_15m"], config, level="MTF")
    assert res.recent_high > 0 and res.recent_low > 0
    assert res.bos_direction in (-1, 0, 1)


def test_sessions_returns_valid_names(config):
    r = sessions.SessionResolver(config)
    # 2024-01-15 14:30 UTC ~= 09:30 NY (Monday) - should be in NY broad session
    info = r.resolve(1_705_329_000)
    assert info.name in ("NY", "NY_AM_SB", "LONDON", "LONDON_KZ", "OFF")
    assert info.weekday == 0


def test_sessions_weekend(config):
    r = sessions.SessionResolver(config)
    # 2024-01-14 is a Sunday
    info = r.resolve(1_705_190_400)
    assert info.name == "WEEKEND"
    assert info.active is False


def test_premium_discount_bounds(config, synthetic_market):
    res = premium_discount.analyze(synthetic_market["df_1h"], config)
    assert -1.0 <= res.position <= 1.0
    assert res.zone in ("PREMIUM", "DISCOUNT", "EQUILIBRIUM")


def test_trade_score_combine(config):
    ts = trade_score.combine({
        "bias": 1.0, "liquidity": 1.0, "displacement": 1.0, "fvg": 1.0,
        "order_block": 1.0, "session": 1.0, "volatility": 1.0,
        "premium_discount": 1.0, "ltf_confirmation": 1.0,
    }, config)
    assert ts.overall == pytest.approx(1.0)
    assert ts.grade == "A+"

    ts_zero = trade_score.combine({}, config)
    assert ts_zero.overall == 0.0
    assert ts_zero.grade == "F"
