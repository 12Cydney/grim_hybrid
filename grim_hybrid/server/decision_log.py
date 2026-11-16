"""
server.decision_log
===================
Audit trail: every PRICE_DATA_MTF decision is appended to
``logs/decisions_YYYYMMDD.csv`` (New York trading date) and every
model-load result is written to ``logs/model_status.json``.

Logging must never break trading - all writes swallow their errors.
"""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytz

COLUMNS = [
    "logged_at", "symbol", "action", "signal", "confidence", "setup_score", "setup_grade",
    "method", "model_type", "session", "liquidity_type", "direction", "htf_bias",
    "sl_kind", "sl_price", "tp_kind", "tp_price", "rr_planned", "volatility_regime",
    "trigger", "in_killzone", "poi_kind", "poi_tapped", "filter_failed",
    "risk_allowed", "lot_size", "risk_pct", "hold_reason",
]


class DecisionLog:
    def __init__(self, logs_dir: Path, tz: str = "America/New_York"):
        self.dir = Path(logs_dir)
        self.tz = pytz.timezone(tz)
        self._lock = threading.Lock()

    def _path(self, now: datetime) -> Path:
        """Today's file; if it was written with different columns (older
        version), continue in decisions_YYYYMMDD_2.csv, _3, ... instead."""
        stem = f"decisions_{now.astimezone(self.tz).strftime('%Y%m%d')}"
        header = ",".join(COLUMNS)
        path, k = self.dir / f"{stem}.csv", 1
        while path.exists():
            with path.open(encoding="utf-8") as fh:
                if fh.readline().strip() == header:
                    break
            k += 1
            path = self.dir / f"{stem}_{k}.csv"
        return path

    def append(self, response: dict[str, Any], now: datetime | None = None) -> Path | None:
        now = now or datetime.now(tz=timezone.utc)
        row = {k: response.get(k, "") for k in COLUMNS}
        row["logged_at"] = now.isoformat()
        try:
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                path = self._path(now)
                new = not path.exists()
                with path.open("a", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=COLUMNS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
            return path
        except Exception as e:   # pragma: no cover - never break trading on I/O
            print(f"[decision_log] write failed: {e}")
            return None

    def write_model_status(self, status: list[dict], all_versions_match: bool) -> Path | None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / "model_status.json"
            path.write_text(json.dumps({
                "checked_at": datetime.now(tz=timezone.utc).isoformat(),
                "models_reached": sum(1 for r in status if r["status"] == "REACHED"),
                "models_expected": len(status),
                "versions_ok": all_versions_match,
                "models": status,
            }, indent=2))
            return path
        except Exception as e:   # pragma: no cover
            print(f"[decision_log] model status write failed: {e}")
            return None
