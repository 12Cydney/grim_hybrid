# Grim Hybrid v1

Modular AI trading framework — one Strategy Engine, zero drift between
training and live.  Python 3.11+, MT5 EA client, TCP JSON transport.

## Layout
```
grim_hybrid/
├── config.yaml                       # single source of thresholds/risk/sessions
├── strategy_engine/                  # THE brain (imported by every component)
│   ├── engine.py                     # public entrypoint: analyze() -> SetupResult
│   ├── market_structure.py           # BOS / MSS / CHOCH / swings
│   ├── liquidity.py                  # BSL/SSL + volatility-driven sweeps
│   ├── displacement.py               # impulse / body-ratio / range-ATR
│   ├── fvg.py                        # FVG detect + size/age/fill %
│   ├── order_blocks.py               # OB detect + quality + mitigation
│   ├── premium_discount.py           # premium/discount/OTE
│   ├── sessions.py                   # London KZ, NY AM/PM SB, DST-safe
│   ├── volatility.py                 # ATR, avg candle, speed, regime
│   ├── stoploss.py                   # multi-location SL evaluator
│   ├── takeprofit.py                 # liquidity / PDH-PDL / ADR TP
│   ├── trade_score.py                # weighted per-component scoring
│   ├── feature_builder.py            # SetupResult -> ML feature vector
│   └── utils.py
├── server/silver_bullet_server.py    # thin TCP orchestrator (no logic)
├── collector/collect_training_data.py
├── trainer/train_model.py
├── risk/risk_engine.py
├── models/                           # xauusd_model.pkl, ..., fallback_model.pkl
├── mt5/GrimHybridEA_ZMQ.mq5          # execution-only EA
├── data/training/*.csv
└── tests/
```

## Install

```bash
pip install numpy pandas scikit-learn joblib pytz PyYAML xgboost
```

For MT5 data collection (Windows only):
```bash
pip install MetaTrader5
```

## Quick start (dev / no MT5)

```bash
# 1. Generate synthetic training data + a smoke-trained fallback model
python grim_hybrid/collector/collect_training_data.py --synthetic
python grim_hybrid/trainer/train_model.py

# 2. Run the server
python grim_hybrid/server/silver_bullet_server.py

# 3. Run tests
python -m pytest grim_hybrid/tests -q
```

## Production (with MT5)

```bash
# Open MT5 -> log into your broker -> then:
python grim_hybrid/collector/collect_training_data.py --years 5
python grim_hybrid/trainer/train_model.py
python grim_hybrid/server/silver_bullet_server.py
```

Attach `mt5/GrimHybridEA_ZMQ.mq5` to any chart and set `InpServerIP` to
the machine running the Python server.

## Core guarantees

- **One brain** — every downstream component calls
  `StrategyEngine.analyze(...)`. There is no other place where a trading
  decision is made.
- **No drift** — the collector and the live server both call
  `feature_builder.build_features(setup, config)`; feature columns are
  byte-identical.
- **No hardcoded pips** — sweep depth, FVG size, OB body etc. are all
  functions of ATR (see `config.yaml -> thresholds.*.atr_factor`).
- **Rich `SetupResult`** — every score, SL/TP candidate, session,
  displacement and volume metric is exposed for logging, dashboarding
  and future backtesting.
