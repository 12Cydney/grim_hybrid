"""
strategy_engine.utils
=====================
Small helpers used across modules.  Nothing here contains trading logic -
that lives in the specialised modules.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------
def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the central config.yaml.

    If ``path`` is None, we search for ``config.yaml`` starting from the
    caller module's directory upward.  Fails loudly if not found - config
    is mandatory.
    """
    if path is None:
        here = Path(__file__).resolve().parent
        for parent in [here, *here.parents]:
            candidate = parent / "config.yaml"
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError("config.yaml not found in project tree")

    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------
# DataFrame coercion
# ---------------------------------------------------------------------
_REQUIRED_COLS = ("time", "open", "high", "low", "close")


def to_ohlc_df(bars: Any) -> pd.DataFrame:
    """Coerce a list-of-dicts or DataFrame into a canonical OHLC frame.

    Guarantees columns: time, open, high, low, close, tick_volume.
    ``time`` is stored as float unix-seconds for deterministic hashing
    between collector and live paths.
    """
    if isinstance(bars, pd.DataFrame):
        df = bars.copy()
    else:
        df = pd.DataFrame(bars)

    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame missing columns: {missing}")

    # Normalise time to unix seconds float
    if np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = df["time"].astype("int64") // 10**9
    elif isinstance(df["time"].iloc[0], pd.Timestamp):
        df["time"] = df["time"].apply(lambda t: t.timestamp())

    df["time"] = df["time"].astype(float)

    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0

    for col in ("open", "high", "low", "close", "tick_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(_REQUIRED_COLS)).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------
def pip_value_for(symbol: str, config: Mapping) -> float:
    """Return pip value for a symbol, using aliases from config."""
    if not symbol:
        return 0.0001
    s = symbol.upper()

    # Direct match first
    for base, spec in config["symbols"].items():
        if base.upper() == s:
            return float(spec["pip_value"])

    # Alias / partial match
    for base, spec in config["symbols"].items():
        aliases = [a.upper() for a in spec.get("aliases", [])]
        if s in aliases or any(a in s or s in a for a in aliases):
            return float(spec["pip_value"])

    # Reasonable defaults from symbol name
    if "JPY" in s:
        return 0.01
    if "XAU" in s or "GOLD" in s:
        return 0.01
    if "US30" in s or "DJ" in s:
        return 1.0
    if "NAS" in s or "US100" in s or "USTEC" in s:
        return 0.1
    return 0.0001


def symbol_class(symbol: str, config: Mapping) -> str:
    s = symbol.upper()
    for base, spec in config["symbols"].items():
        aliases = [a.upper() for a in spec.get("aliases", [])]
        if base.upper() == s or s in aliases:
            return spec.get("class", "FX_MAJOR")
    if "JPY" in s:
        return "FX_JPY"
    return "FX_MAJOR"


# ---------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or b is None:
            return default
        return float(a) / float(b)
    except Exception:
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def unix_seconds(ts: Any) -> float:
    """Best-effort convert a timestamp to unix seconds (float)."""
    if ts is None:
        return datetime.now(tz=timezone.utc).timestamp()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, pd.Timestamp):
        return ts.timestamp()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    return float(ts)


def iter_last_n(seq: Iterable, n: int):
    lst = list(seq)
    return lst[-n:] if n <= len(lst) else lst
