# Grim Hybrid v1 — PRD

## Original problem statement
Refactor the monolithic Silver Bullet ML server into a modular framework
where ONE Strategy Engine is the single source of truth for the live server,
training data collector, trainer, and future backtester/optimizer/dashboard.
Zero logic drift between training and live.

## Architecture
```
MT5 EA (GrimHybridEA_ZMQ.mq5)  --TCP JSON-->  silver_bullet_server.py
                                                     |
                                             strategy_engine.analyze()  ← THE brain
                                                     |
                                             feature_builder.build_features()
                                                     |
                                                AI model (per-symbol + fallback)
                                                     |
                                                 risk_engine
                                                     |
                                     BUY / SELL / HOLD + SL price + TP price + size
```

## What's implemented (2026-07-19)
- `config.yaml` — single source of thresholds/risk/sessions/symbols (all dynamic, no hardcoded pips)
- **strategy_engine/** (13 modules, ~1500 LoC):
  - `engine.py` — public `StrategyEngine.analyze()` → rich `SetupResult`
  - `market_structure.py` — BOS / MSS / CHOCH / swings
  - `liquidity.py` — BSL / SSL + volatility-driven sweep detection
  - `displacement.py` — impulse / body-ratio / range-ATR
  - `fvg.py` — Fair Value Gap detect + size / age / fill %
  - `order_blocks.py` — bullish/bearish OB + quality + mitigation
  - `premium_discount.py` — premium / discount / OTE zone
  - `sessions.py` — London KZ, NY AM/PM Silver Bullet, DST-safe
  - `volatility.py` — ATR / avg candle / market speed / regime, all dynamic threshold helpers
  - `stoploss.py` — multi-location SL (liquidity / structure / OB / ATR) picks tightest valid
  - `takeprofit.py` — liquidity / PDH-PDL / RANGE / ADR TP with min R:R floor
  - `trade_score.py` — weighted 9-component 0-1 scoring + letter grade + rich description
  - `feature_builder.py` — single source of ML feature vector (73 features)
- **server/silver_bullet_server.py** — thin TCP orchestrator (no logic). Preserves v3.1 protocol; adds `setup_score`, `setup_grade`, `sl_kind`, `tp_kind`, `sl_price`, `tp_price`, `rr_planned`, `volatility_regime`.
- **collector/collect_training_data.py** — streams historical bars through the engine (MT5 or `--synthetic`). Guarantees byte-identical features vs live path.
- **trainer/train_model.py** — CSV → per-symbol XGBoost models + global fallback. Reads feature schema from `strategy_engine.feature_builder.FEATURE_ORDER`.
- **risk/risk_engine.py** — dynamic sizing (ATR / confidence / AI suggestion), exposure caps, breakeven + ATR trailing stop.
- **mt5/GrimHybridEA_ZMQ.mq5** — v1.0 dumb executor. Consumes new `sl_price`/`tp_price`/`sl_kind`/`tp_kind` from server. Zero strategy logic.
- **models/** — 7 per-symbol models + `fallback_model.pkl` (smoke-trained on synthetic; user will retrain with real MT5 history).
- **tests/** — 23 pytest cases covering every engine module, feature-builder determinism, risk engine guardrails, TCP end-to-end.

## Guarantees delivered
1. **One brain** — every downstream module imports `strategy_engine`.
2. **No drift** — a dedicated test asserts collector-produced features == live-produced features byte-for-byte.
3. **No hardcoded pips** — sweep depth, FVG size, OB body all volatility-relative.
4. **Rich SetupResult** — every score, SL/TP candidate, session, speed, and volume metric exposed on the wire.
5. **Auto-fallback** — if per-symbol confidence < threshold and fallback exists, use fallback (configurable).
6. **R:R floor** — takeprofit enforces `min_rr` (default 2.0, configurable).

## Test results (2026-07-19)
- 23/23 pytest pass (unit + end-to-end + risk).
- Live server smoke test: PING returns 7 models + fallback, feature_count=73. PRICE_DATA_MTF returns full decision JSON.
- Collector + trainer: 4,730 synthetic samples generated → 7 per-symbol + 1 fallback model produced.

## Backlog / next iterations
- **P1** Retrain with real MT5 CSVs once available (synthetic model F1 is intentionally low).
- **P2** Backtester module (reuses same engine, replays bars, produces P&L curve).
- **P2** Optimizer (walk-forward over `config.yaml.thresholds` search space).
- **P2** Read-only dashboard consuming the same engine (Flask/FastAPI + WebSocket).
- **P3** Persist per-EA account state (currently in-memory) for real risk continuity across restarts.
- **P3** MSS / CHOCH swing detection refinement (current implementation uses last-2-swings heuristic).

## How to run
```bash
cd /app/grim_hybrid
pip install -r requirements.txt

# Dev / smoke (no MT5)
python collector/collect_training_data.py --synthetic
python trainer/train_model.py
python server/silver_bullet_server.py

# Production (with MT5)
python collector/collect_training_data.py --years 5
python trainer/train_model.py
python server/silver_bullet_server.py
# Attach mt5/GrimHybridEA_ZMQ.mq5 to any chart, set InpServerIP.

# Tests
python -m pytest tests/ -q
```
