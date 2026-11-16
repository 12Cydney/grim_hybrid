"""
collector.collect_training_data
===============================

Streams historical bars through ``StrategyEngine`` (SAME logic as live)
and saves valid setups + their outcomes to CSV.

Because we call the engine and feature_builder here, the training CSV
schema is *guaranteed* to match the live prediction schema.

MT5 mode (default): pulls bars from MetaTrader 5.
Synthetic mode: use ``--synthetic`` for a quick smoke run without MT5.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategy_engine import StrategyEngine, MarketData          # noqa: E402
from strategy_engine.feature_builder import FEATURE_ORDER       # noqa: E402
from strategy_engine.utils import load_config, pip_value_for     # noqa: E402


log = logging.getLogger("collector")


# ---------------------------------------------------------------------
# Broker symbol resolution
# ---------------------------------------------------------------------
def _find_broker_symbol(base: str, config: dict, mt5) -> str | None:
    aliases = config["symbols"][base].get("aliases", [base])
    for a in aliases:
        try:
            info = mt5.symbol_info(a)
            if info is not None:
                return a
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------
# Trade outcome simulator (for label generation)
# ---------------------------------------------------------------------
def _simulate_outcome(df_future: pd.DataFrame, entry: float, sl: float, tp: float, direction: int) -> int:
    """Return 1 (TP first), 0 (SL first), or -1 (neither)."""
    if df_future is None or len(df_future) == 0:
        return -1
    if direction == 1:
        for _, row in df_future.iterrows():
            if row["low"] <= sl:
                return 0
            if row["high"] >= tp:
                return 1
    else:
        for _, row in df_future.iterrows():
            if row["high"] >= sl:
                return 0
            if row["low"] <= tp:
                return 1
    return -1


# ---------------------------------------------------------------------
# Historical stream (per symbol)
# ---------------------------------------------------------------------
def collect_for_symbol(
    engine: StrategyEngine,
    symbol: str,
    broker_symbol: str,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    config: dict,
    future_bars_1m: int = 480,
) -> list[dict]:
    """Walk the 15m bars, evaluate a setup at each bar via the engine,
    keep only ``setup.valid`` samples, and label with future outcome."""
    samples: list[dict] = []
    n_15m = len(df_15m)
    min_ctx = 50
    if n_15m < min_ctx + 10:
        log.info("%s: not enough 15m bars (%d)", symbol, n_15m)
        return samples

    df_1h = df_1h.copy(); df_1h["time"] = df_1h["time"].astype(float)
    df_15m = df_15m.copy(); df_15m["time"] = df_15m["time"].astype(float)
    df_1m = df_1m.copy(); df_1m["time"] = df_1m["time"].astype(float)

    tp_hits = sl_hits = 0
    scanned = 0

    for i in range(min_ctx, n_15m - future_bars_1m // 15):
        cur_time = df_15m.iloc[i]["time"]
        slice_1h = df_1h[df_1h["time"] <= cur_time].tail(60)
        slice_15m = df_15m[df_15m["time"] <= cur_time].tail(60)
        slice_1m = df_1m[df_1m["time"] <= cur_time].tail(30)
        if len(slice_1h) < 30 or len(slice_15m) < 30 or len(slice_1m) < 10:
            continue
        scanned += 1

        md = MarketData(
            symbol=symbol,
            data_1h=slice_1h.to_dict("records"),
            data_15m=slice_15m.to_dict("records"),
            data_1m=slice_1m.to_dict("records"),
            timestamp=cur_time,
        )
        try:
            setup = engine.analyze(md)
        except Exception as e:
            log.debug("engine error @ %s: %s", cur_time, e)
            continue

        if not setup.valid or setup.stoploss.distance <= 0 or setup.takeprofit.distance <= 0:
            continue

        entry = setup.entry
        direction = setup.direction
        sl_price = setup.stoploss.price
        tp_price = setup.takeprofit.price

        # Get future 1m data for outcome
        future_1m = df_1m[df_1m["time"] > cur_time].head(future_bars_1m)
        if len(future_1m) < 60:
            continue

        outcome = _simulate_outcome(future_1m, entry, sl_price, tp_price, direction)
        if outcome == -1:
            continue

        feats = engine.build_features(setup)
        feats["outcome"] = outcome
        feats["symbol"] = symbol
        feats["timestamp"] = float(cur_time)
        feats["entry"] = float(entry)
        feats["sl_price"] = float(sl_price)
        feats["tp_price"] = float(tp_price)
        samples.append(feats)

        if outcome == 1:
            tp_hits += 1
        else:
            sl_hits += 1

        if scanned % 500 == 0:
            log.info("  %s scanned=%d setups=%d TP=%d SL=%d", symbol, scanned, len(samples), tp_hits, sl_hits)

    wr = tp_hits / max(1, tp_hits + sl_hits) * 100
    log.info("%s: %d samples (TP=%d SL=%d WR=%.1f%%)", symbol, len(samples), tp_hits, sl_hits, wr)
    return samples


# ---------------------------------------------------------------------
# Synthetic bar generator (used for the smoke-training fallback model
# and for CI-style testing without MT5).
# ---------------------------------------------------------------------
def synthetic_ohlc(bars: int, tf_seconds: int, base_price: float = 2000.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = int((datetime.now(tz=timezone.utc) - timedelta(seconds=bars * tf_seconds)).timestamp())
    times = np.array([start + i * tf_seconds for i in range(bars)], dtype=float)
    # Random walk with occasional impulses
    steps = rng.normal(0, base_price * 0.0007, size=bars)
    impulse_idx = rng.choice(bars, size=max(1, bars // 60), replace=False)
    steps[impulse_idx] *= 6.0
    closes = base_price + np.cumsum(steps)
    opens = np.concatenate([[base_price], closes[:-1]])
    high_noise = np.abs(rng.normal(0, base_price * 0.0004, size=bars))
    low_noise = np.abs(rng.normal(0, base_price * 0.0004, size=bars))
    highs = np.maximum(opens, closes) + high_noise
    lows = np.minimum(opens, closes) - low_noise
    df = pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows, "close": closes,
        "tick_volume": rng.integers(50, 500, size=bars),
    })
    return df


def _resample_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
    d = df_1m.copy()
    d["dt"] = pd.to_datetime(d["time"], unit="s")
    d = d.set_index("dt")
    o = d["open"].resample("15min").first()
    h = d["high"].resample("15min").max()
    l = d["low"].resample("15min").min()
    c = d["close"].resample("15min").last()
    v = d["tick_volume"].resample("15min").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "tick_volume": v}).dropna()
    # unit-agnostic (pandas 2.x may give datetime64[us] -> old code broke)
    out["time"] = (out.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    return out.reset_index(drop=True)[["time", "open", "high", "low", "close", "tick_volume"]]


def _resample_to_1h(df_1m: pd.DataFrame) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
    d = df_1m.copy()
    d["dt"] = pd.to_datetime(d["time"], unit="s")
    d = d.set_index("dt")
    o = d["open"].resample("1h").first()
    h = d["high"].resample("1h").max()
    l = d["low"].resample("1h").min()
    c = d["close"].resample("1h").last()
    v = d["tick_volume"].resample("1h").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "tick_volume": v}).dropna()
    # unit-agnostic (pandas 2.x may give datetime64[us] -> old code broke)
    out["time"] = (out.index - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
    return out.reset_index(drop=True)[["time", "open", "high", "low", "close", "tick_volume"]]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def _configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=None, help="Override symbol list")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip MT5, generate synthetic bars (dev/CI/smoke tests)")
    parser.add_argument("--from-csv", default=None, metavar="FOLDER",
                        help="Read history CSVs (from history_dump.mq5) instead of using MT5 Python bindings")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    config = load_config()
    engine = StrategyEngine(config)

    symbols = args.symbols or list(config["symbols"].keys())
    out_path = Path(args.out) if args.out else _ROOT / config["paths"]["training_dir"] / config["paths"]["training_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []

    if args.synthetic:
        log.info("=== SYNTHETIC MODE (dev/smoke) ===")
        for sym in symbols:
            base_price = {"XAUUSD": 2000, "NAS100": 17000, "US30": 38000,
                          "EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 150, "AUDUSD": 0.65}.get(sym, 1.0)
            df_1m = synthetic_ohlc(bars=int(60 * 24 * 30), tf_seconds=60,
                                   base_price=base_price, seed=hash(sym) & 0xFFFF)
            df_15m = _resample_to_15m(df_1m)
            df_1h = _resample_to_1h(df_1m)
            samples = collect_for_symbol(engine, sym, sym, df_1h, df_15m, df_1m, config,
                                         future_bars_1m=240)
            all_samples.extend(samples)
    elif args.from_csv:
        log.info("=== CSV MODE - reading from %s ===", args.from_csv)
        csv_dir = Path(args.from_csv)
        if not csv_dir.exists():
            log.error("CSV folder not found: %s", csv_dir)
            return
        for sym in symbols:
            f1h  = csv_dir / f"{sym}_H1.csv"
            f15m = csv_dir / f"{sym}_M15.csv"
            f1m  = csv_dir / f"{sym}_M1.csv"
            missing = [p.name for p in (f1h, f15m, f1m) if not p.exists()]
            if missing:
                log.warning("%s: missing %s in %s", sym, missing, csv_dir)
                continue
            log.info("=== %s ===", sym)
            df_1h  = pd.read_csv(f1h)
            df_15m = pd.read_csv(f15m)
            df_1m  = pd.read_csv(f1m)
            log.info("  loaded: 1h=%d  15m=%d  1m=%d", len(df_1h), len(df_15m), len(df_1m))
            samples = collect_for_symbol(engine, sym, sym, df_1h, df_15m, df_1m, config)
            all_samples.extend(samples)
    else:
        try:
            import MetaTrader5 as mt5   # type: ignore
        except ImportError:
            log.error("MetaTrader5 package not available. Use --synthetic for dev runs.")
            return

        if not mt5.initialize():
            log.error("Failed to initialise MT5 - is the terminal running?")
            return
        log.info("MT5 version: %s", mt5.version())

        end = datetime.now()
        start = end - timedelta(days=365 * args.years)

        for sym in symbols:
            broker = _find_broker_symbol(sym, config, mt5)
            if broker is None:
                log.warning("%s: broker symbol not found, skipping", sym)
                continue
            log.info("=== %s (%s) ===", sym, broker)
            rates_1h = mt5.copy_rates_range(broker, mt5.TIMEFRAME_H1, start, end)
            rates_15m = mt5.copy_rates_range(broker, mt5.TIMEFRAME_M15, start, end)
            rates_1m = mt5.copy_rates_range(broker, mt5.TIMEFRAME_M1, start, end)
            if not len(rates_1h) or not len(rates_15m) or not len(rates_1m):
                log.warning("%s: no bars from MT5", sym)
                continue
            df_1h = pd.DataFrame(rates_1h); df_15m = pd.DataFrame(rates_15m); df_1m = pd.DataFrame(rates_1m)
            samples = collect_for_symbol(engine, sym, broker, df_1h, df_15m, df_1m, config)
            all_samples.extend(samples)

        mt5.shutdown()

    if not all_samples:
        log.warning("No training samples produced")
        return

    df = pd.DataFrame(all_samples)
    # Ensure column order: features first (canonical), then metadata columns
    metadata_cols = [c for c in df.columns if c not in FEATURE_ORDER]
    df = df[list(FEATURE_ORDER) + [c for c in metadata_cols if c != "outcome"] + ["outcome"]]
    df.to_csv(out_path, index=False)
    log.info("Saved %d samples -> %s", len(df), out_path)
    if "outcome" in df.columns and len(df):
        log.info("Overall win rate: %.1f%%", df["outcome"].mean() * 100)


if __name__ == "__main__":
    main()
