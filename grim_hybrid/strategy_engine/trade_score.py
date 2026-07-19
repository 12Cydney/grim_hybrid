"""
strategy_engine.trade_score
===========================
Weighted per-component scoring.  Every sub-module returns a 0-1 score;
this file combines them into an overall setup grade using weights from
config.score_weights.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping


@dataclass
class TradeScore:
    bias: float = 0.0
    liquidity: float = 0.0
    displacement: float = 0.0
    fvg: float = 0.0
    order_block: float = 0.0
    session: float = 0.0
    volatility: float = 0.0
    premium_discount: float = 0.0
    ltf_confirmation: float = 0.0
    overall: float = 0.0
    grade: str = "F"
    description: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _grade_letter(overall: float) -> str:
    if overall >= 0.85:
        return "A+"
    if overall >= 0.75:
        return "A"
    if overall >= 0.65:
        return "B"
    if overall >= 0.55:
        return "C"
    if overall >= 0.45:
        return "D"
    return "F"


def combine(components: dict, config: Mapping) -> TradeScore:
    w = config["score_weights"]
    ts = TradeScore(
        bias=float(components.get("bias", 0.0)),
        liquidity=float(components.get("liquidity", 0.0)),
        displacement=float(components.get("displacement", 0.0)),
        fvg=float(components.get("fvg", 0.0)),
        order_block=float(components.get("order_block", 0.0)),
        session=float(components.get("session", 0.0)),
        volatility=float(components.get("volatility", 0.0)),
        premium_discount=float(components.get("premium_discount", 0.0)),
        ltf_confirmation=float(components.get("ltf_confirmation", 0.0)),
    )
    overall = (
        ts.bias * w["bias"]
        + ts.liquidity * w["liquidity"]
        + ts.displacement * w["displacement"]
        + ts.fvg * w["fvg"]
        + ts.order_block * w["order_block"]
        + ts.session * w["session"]
        + ts.volatility * w["volatility"]
        + ts.premium_discount * w["premium_discount"]
        + ts.ltf_confirmation * w["ltf_confirmation"]
    )
    ts.overall = float(round(overall, 4))
    ts.grade = _grade_letter(ts.overall)
    ts.description = describe(ts)
    return ts


def describe(ts: TradeScore) -> str:
    """Rich human-readable description of what the score means."""
    parts = []
    if ts.session >= 0.8:
        parts.append("in kill-zone")
    elif ts.session >= 0.4:
        parts.append("in trading session")
    else:
        parts.append("off session")

    if ts.bias >= 0.7:
        parts.append("HTF bias aligned")
    elif ts.bias >= 0.4:
        parts.append("mild HTF bias")

    if ts.liquidity >= 0.7:
        parts.append("liquidity swept strongly")
    elif ts.liquidity >= 0.4:
        parts.append("some liquidity taken")

    if ts.fvg >= 0.6:
        parts.append("fresh FVG confluence")
    if ts.order_block >= 0.6:
        parts.append("quality OB present")
    if ts.displacement >= 0.6:
        parts.append("displacement confirmed")
    if ts.premium_discount >= 0.8:
        parts.append("in OTE zone")
    elif ts.premium_discount >= 0.5:
        parts.append("acceptable price zone")

    if ts.ltf_confirmation >= 0.6:
        parts.append("LTF entry confirmed")

    if ts.volatility >= 0.7:
        parts.append("normal volatility")
    elif ts.volatility < 0.4:
        parts.append("poor volatility regime")

    grade_line = f"[{ts.grade} - {ts.overall:.2f}] "
    return grade_line + ", ".join(parts) if parts else grade_line + "no confluence"
