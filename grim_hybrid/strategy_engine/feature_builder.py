"""
strategy_engine.feature_builder
===============================

**Single source of ML features**.

Both the training collector and the live server produce feature vectors
by calling ``build_features(setup_result, config)``.  The schema returned
here MUST be identical in every code path.  If a feature isn't derivable
(e.g. LTF confirmation for a training sample) we still emit the key with
a 0-value placeholder, so the columns line up exactly.

The order of ``FEATURE_ORDER`` is the canonical training column order.
"""

from __future__ import annotations

from typing import Mapping

from strategy_engine.utils import safe_div


# Canonical feature order - DO NOT reorder, DO NOT remove.
# Anything new goes at the end; retrain to include it.
FEATURE_ORDER: tuple[str, ...] = (
    # Session / time
    "is_session", "silver_bullet", "session_ny", "session_london", "session_asian",
    "session_ny_am_sb", "session_ny_pm_sb", "session_london_kz",
    "minutes_since_open", "day_of_week", "hour_of_day",
    # Volatility
    "atr_15m", "atr_ratio_15m", "atr_1h", "atr_ratio_1h", "market_speed_15m",
    # HTF (1H)
    "htf_bias", "htf_bos_age", "htf_mss", "htf_choch",
    "htf_premium_discount", "htf_displacement_body_ratio",
    # 15M structure
    "m15_bos", "m15_bos_age", "m15_mss", "m15_broke_high", "m15_broke_low",
    "m15_break_size_atr",
    # 15M liquidity
    "m15_swept", "m15_sweep_depth_atr", "m15_sweep_direction",
    "m15_equal_highs", "m15_equal_lows", "m15_liquidity_score",
    # 15M FVG
    "m15_fvg", "m15_fvg_direction", "m15_fvg_size_atr", "m15_fvg_age", "m15_fvg_fill",
    # 15M OB
    "m15_ob", "m15_ob_direction", "m15_ob_body_atr", "m15_ob_age", "m15_ob_mitigated",
    # 15M displacement
    "m15_displacement", "m15_displacement_body_ratio", "m15_displacement_range_atr",
    # Premium/discount
    "pd_position_1h", "pd_zone_premium", "pd_zone_discount",
    "in_ote_long", "in_ote_short",
    # 1M confirmation
    "m1_confirmed", "m1_reversal_strength", "m1_displacement_body_ratio",
    "m1_bos", "m1_bos_age",
    # Component scores
    "score_bias", "score_liquidity", "score_displacement", "score_fvg",
    "score_ob", "score_session", "score_volatility", "score_pd",
    "score_ltf", "score_overall",
    # Direction / setup context (also inputs, informative for the model)
    "direction", "potential_direction",
    # SL / TP hints
    "sl_distance_atr", "tp_distance_atr", "rr_planned",
    # Volume
    "volume_ratio",
)


def _to_int(b) -> int:
    return 1 if bool(b) else 0


def build_features(setup, config: Mapping) -> dict[str, float]:
    """Deterministic dict of FEATURE_ORDER -> float from a SetupResult."""
    r = setup  # SetupResult
    atr_1h = r.atr_1h or 1e-9
    atr_15m = r.atr_15m or 1e-9

    session = r.session
    ms15 = r.structure_15m
    ms1h = r.structure_1h
    ms1m = r.structure_1m
    liq = r.liquidity
    fvg = r.fvg
    ob = r.order_block
    disp15 = r.displacement_15m
    disp1m = r.displacement_1m
    pd_ = r.premium_discount
    ts = r.score

    feats: dict[str, float] = {
        # Session / time
        "is_session": _to_int(session.active),
        "silver_bullet": _to_int(session.silver_bullet),
        "session_ny": _to_int(session.name in ("NY", "NY_AM_SB", "NY_PM_SB")),
        "session_london": _to_int(session.name in ("LONDON", "LONDON_KZ")),
        "session_asian": _to_int(session.name == "ASIAN"),
        "session_ny_am_sb": _to_int(session.name == "NY_AM_SB"),
        "session_ny_pm_sb": _to_int(session.name == "NY_PM_SB"),
        "session_london_kz": _to_int(session.name == "LONDON_KZ"),
        "minutes_since_open": float(session.minutes_since_open),
        "day_of_week": float(session.weekday),
        "hour_of_day": float(session.hour_ny),

        # Volatility
        "atr_15m": float(r.atr_15m),
        "atr_ratio_15m": float(r.vol_15m.atr_ratio),
        "atr_1h": float(r.atr_1h),
        "atr_ratio_1h": float(r.vol_1h.atr_ratio),
        "market_speed_15m": float(r.vol_15m.speed),

        # HTF (1H) structure
        "htf_bias": float(ms1h.bos_direction),
        "htf_bos_age": float(ms1h.bos_age),
        "htf_mss": float(ms1h.mss),
        "htf_choch": float(ms1h.choch),
        "htf_premium_discount": float(pd_.position),
        "htf_displacement_body_ratio": float(r.displacement_1h.body_ratio if r.displacement_1h else 0.0),

        # 15M structure
        "m15_bos": float(ms15.bos_direction),
        "m15_bos_age": float(ms15.bos_age),
        "m15_mss": float(ms15.mss),
        "m15_broke_high": _to_int(ms15.broke_high),
        "m15_broke_low": _to_int(ms15.broke_low),
        "m15_break_size_atr": safe_div(max(ms15.high_break_size, ms15.low_break_size), atr_15m, 0.0),

        # 15M liquidity
        "m15_swept": _to_int(liq.swept),
        "m15_sweep_depth_atr": safe_div(liq.sweep_depth, atr_15m, 0.0),
        "m15_sweep_direction": float(liq.sweep_direction),
        "m15_equal_highs": _to_int(liq.equal_highs),
        "m15_equal_lows": _to_int(liq.equal_lows),
        "m15_liquidity_score": float(ts.liquidity),

        # 15M FVG
        "m15_fvg": _to_int(fvg.present),
        "m15_fvg_direction": float(fvg.direction),
        "m15_fvg_size_atr": float(fvg.size_atr),
        "m15_fvg_age": float(fvg.age),
        "m15_fvg_fill": float(fvg.fill_pct),

        # 15M OB
        "m15_ob": _to_int(ob.present),
        "m15_ob_direction": float(ob.direction),
        "m15_ob_body_atr": float(ob.body_atr),
        "m15_ob_age": float(ob.age),
        "m15_ob_mitigated": _to_int(ob.mitigated),

        # 15M displacement
        "m15_displacement": _to_int(disp15.present),
        "m15_displacement_body_ratio": float(disp15.body_ratio),
        "m15_displacement_range_atr": float(disp15.range_atr),

        # Premium/discount
        "pd_position_1h": float(pd_.position),
        "pd_zone_premium": _to_int(pd_.zone == "PREMIUM"),
        "pd_zone_discount": _to_int(pd_.zone == "DISCOUNT"),
        "in_ote_long": _to_int(pd_.in_ote_long),
        "in_ote_short": _to_int(pd_.in_ote_short),

        # 1M confirmation
        "m1_confirmed": _to_int(r.ltf_confirmed),
        "m1_reversal_strength": float(r.ltf_reversal_strength),
        "m1_displacement_body_ratio": float(disp1m.body_ratio if disp1m else 0.0),
        "m1_bos": float(ms1m.bos_direction if ms1m else 0),
        "m1_bos_age": float(ms1m.bos_age if ms1m else 0),

        # Score components
        "score_bias": float(ts.bias),
        "score_liquidity": float(ts.liquidity),
        "score_displacement": float(ts.displacement),
        "score_fvg": float(ts.fvg),
        "score_ob": float(ts.order_block),
        "score_session": float(ts.session),
        "score_volatility": float(ts.volatility),
        "score_pd": float(ts.premium_discount),
        "score_ltf": float(ts.ltf_confirmation),
        "score_overall": float(ts.overall),

        # Direction
        "direction": float(r.direction),
        "potential_direction": float(r.direction),

        # SL/TP hints (in ATR units so features are volatility-relative)
        "sl_distance_atr": safe_div(r.stoploss.distance, atr_15m, 0.0),
        "tp_distance_atr": safe_div(r.takeprofit.distance, atr_15m, 0.0),
        "rr_planned": float(r.takeprofit.rr),

        # Volume
        "volume_ratio": float(r.volume_ratio),
    }

    # Ensure every key from FEATURE_ORDER is present (safety net)
    for k in FEATURE_ORDER:
        feats.setdefault(k, 0.0)

    # Return in canonical order
    return {k: float(feats[k]) for k in FEATURE_ORDER}


def feature_names() -> list[str]:
    return list(FEATURE_ORDER)
