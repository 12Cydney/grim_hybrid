"""
strategy_engine.sessions
========================
Trading session detection (London KZ, NY AM/PM Silver Bullet, broad
Asian/London/NY windows).  All times config-driven, DST-safe via pytz.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Mapping

import pytz

from strategy_engine.utils import unix_seconds


# ---------------------------------------------------------------------
# Broker clock -> UTC
# ---------------------------------------------------------------------
_NY = pytz.timezone("America/New_York")


def broker_offset_seconds(broker_ts: float, config: Mapping) -> float:
    """Broker-server-time minus UTC, in seconds, per ``config.broker_clock``.

    MT5 bar times and ``TimeCurrent()`` are broker SERVER time, not UTC.
    Modes:
      ny_plus_7 : server clock = New York wall clock + 7h (most FX brokers:
                  GMT+2 winter / GMT+3 summer, midnight = NY 17:00 close)
      fixed     : server clock = UTC + ``fixed_offset_hours``
      utc       : times are already UTC
    """
    bc = config.get("broker_clock", {}) or {}
    mode = str(bc.get("mode", "ny_plus_7")).lower()
    if mode == "utc":
        return 0.0
    if mode == "fixed":
        return float(bc.get("fixed_offset_hours", 0.0)) * 3600.0
    # ny_plus_7: broker wall time - 7h = NY wall time
    ny_wall = datetime.fromtimestamp(float(broker_ts) - 7 * 3600, tz=timezone.utc).replace(tzinfo=None)
    ny_local = _NY.localize(ny_wall, is_dst=False)
    utc_ts = ny_local.astimezone(timezone.utc).timestamp()
    return float(broker_ts) - utc_ts


def broker_to_utc(broker_ts, config: Mapping, gmt_offset: float | None = None) -> float:
    """Convert a broker-server timestamp to true UTC.

    ``gmt_offset`` (TimeCurrent() - TimeGMT(), sent live by the EA) is exact
    and wins; otherwise the ``broker_clock`` rule is used (history/training).
    """
    ts = unix_seconds(broker_ts)
    if gmt_offset is not None:
        return ts - float(gmt_offset)
    return ts - broker_offset_seconds(ts, config)


@dataclass
class SessionInfo:
    active: bool
    name: str                # 'LONDON_KZ', 'NY_AM_SB', 'NY_PM_SB', 'LONDON', 'NY', 'ASIAN', 'OFF', 'WEEKEND'
    silver_bullet: bool      # in one of the explicit Silver Bullet windows
    minutes_since_open: int  # into whichever session we matched (0 if OFF)
    weekday: int             # 0 = Monday, 6 = Sunday (NY tz)
    hour_ny: int
    minute_ny: int


def _parse_hhmm(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    return start <= t <= end


class SessionResolver:
    """Resolve which trading session a UTC/unix timestamp belongs to."""

    def __init__(self, config: Mapping):
        sc = config["sessions"]
        self.tz = pytz.timezone(sc["timezone"])
        self._london_kz = (_parse_hhmm(sc["london_kz"]["start"]), _parse_hhmm(sc["london_kz"]["end"]))
        self._ny_am_sb  = (_parse_hhmm(sc["ny_am_silver_bullet"]["start"]), _parse_hhmm(sc["ny_am_silver_bullet"]["end"]))
        self._ny_pm_sb  = (_parse_hhmm(sc["ny_pm_silver_bullet"]["start"]), _parse_hhmm(sc["ny_pm_silver_bullet"]["end"]))
        self._london    = (_parse_hhmm(sc["london_broad"]["start"]),        _parse_hhmm(sc["london_broad"]["end"]))
        self._ny        = (_parse_hhmm(sc["ny_broad"]["start"]),            _parse_hhmm(sc["ny_broad"]["end"]))
        self._asian     = (_parse_hhmm(sc["asian_broad"]["start"]),         _parse_hhmm(sc["asian_broad"]["end"]))

    def resolve(self, ts) -> SessionInfo:
        secs = unix_seconds(ts)
        dt_utc = datetime.fromtimestamp(secs, tz=timezone.utc)
        dt_ny = dt_utc.astimezone(self.tz)
        t = dt_ny.time()
        wd = dt_ny.weekday()

        if wd >= 5:
            return SessionInfo(False, "WEEKEND", False, 0, wd, dt_ny.hour, dt_ny.minute)

        # Silver bullet checks first (they're inside the broad windows)
        if _in_window(t, *self._london_kz):
            mins = (dt_ny.hour - self._london_kz[0].hour) * 60 + (dt_ny.minute - self._london_kz[0].minute)
            return SessionInfo(True, "LONDON_KZ", True, max(mins, 0), wd, dt_ny.hour, dt_ny.minute)
        if _in_window(t, *self._ny_am_sb):
            mins = (dt_ny.hour - self._ny_am_sb[0].hour) * 60 + (dt_ny.minute - self._ny_am_sb[0].minute)
            return SessionInfo(True, "NY_AM_SB", True, max(mins, 0), wd, dt_ny.hour, dt_ny.minute)
        if _in_window(t, *self._ny_pm_sb):
            mins = (dt_ny.hour - self._ny_pm_sb[0].hour) * 60 + (dt_ny.minute - self._ny_pm_sb[0].minute)
            return SessionInfo(True, "NY_PM_SB", True, max(mins, 0), wd, dt_ny.hour, dt_ny.minute)

        # Broad windows (used for scoring parity with legacy)
        if _in_window(t, *self._ny):
            mins = (dt_ny.hour - self._ny[0].hour) * 60 + (dt_ny.minute - self._ny[0].minute)
            return SessionInfo(True, "NY", False, max(mins, 0), wd, dt_ny.hour, dt_ny.minute)
        if _in_window(t, *self._london):
            mins = (dt_ny.hour - self._london[0].hour) * 60 + (dt_ny.minute - self._london[0].minute)
            return SessionInfo(True, "LONDON", False, max(mins, 0), wd, dt_ny.hour, dt_ny.minute)
        if _in_window(t, *self._asian):
            mins = (dt_ny.hour - self._asian[0].hour) * 60 + (dt_ny.minute - self._asian[0].minute)
            return SessionInfo(True, "ASIAN", False, max(mins, 0), wd, dt_ny.hour, dt_ny.minute)

        return SessionInfo(False, "OFF", False, 0, wd, dt_ny.hour, dt_ny.minute)
