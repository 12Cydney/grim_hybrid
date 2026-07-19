# Grim Hybrid v1 — Complete User Guide

> **Read time:** ~15 minutes.  
> **Goal:** Take you from an empty machine to a live-trading Grim Hybrid setup with real AI models.

---

## Table of contents

1. [What you're running](#1-what-youre-running)
2. [The four programs and how they talk](#2-the-four-programs-and-how-they-talk)
3. [Machines you need](#3-machines-you-need)
4. [One-time install](#4-one-time-install)
5. [Step-by-step: from zero to live trading](#5-step-by-step-from-zero-to-live-trading)
6. [Understanding the config](#6-understanding-the-config)
7. [Reading the dashboard](#7-reading-the-dashboard)
8. [Retraining workflow](#8-retraining-workflow)
9. [Backup, rollback, and safety](#9-backup-rollback-and-safety)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. What you're running

Grim Hybrid is an ICT / Silver Bullet trading system split into four pieces
that all speak to **one strategy brain**. The brain lives in Python; the
executor lives in your MT5 terminal.

```
┌─────────────────┐   TCP JSON    ┌────────────────────────┐
│  MT5 EA (chart) │◄─────────────►│  Python server         │
│  - dashboard    │               │  - strategy engine     │
│  - executes     │               │  - AI models           │
│  - reports      │               │  - risk engine         │
└─────────────────┘               └────────────────────────┘
                                             ▲
                                             │ uses same brain
                                             ▼
                                  ┌────────────────────────┐
                                  │  Collector + Trainer   │
                                  │  (offline, historical) │
                                  └────────────────────────┘
```

**Why this shape?** Because the collector and the live server both call
the same `StrategyEngine.analyze()`. There is **zero chance** the AI
learns from one set of rules and trades under another.

---

## 2. The four programs and how they talk

| # | Program | Runs where | Purpose |
|---|---------|------------|---------|
| 1 | `strategy_engine/` | Python (import) | The brain. All ICT / SB rules live here. |
| 2 | `server/silver_bullet_server.py` | Python (24/7) | Thin TCP orchestrator. Loads models, calls brain, replies. |
| 3 | `collector/collect_training_data.py` | Python (occasional) | Streams historical bars through the brain, saves labeled setups to CSV. |
| 4 | `trainer/train_model.py` | Python (occasional) | Turns the CSV into 7 per-symbol XGBoost models + 1 fallback. |
| 5 | `mt5/GrimHybridEA_ZMQ.mq5` | MT5 (24/7) | Dashboard + dumb executor. Sends OHLC to server, executes returned decisions. |

**They talk over TCP** on port 5555 (configurable). Messages are one
newline-terminated JSON line:

- EA → server: `{"type":"PING"}` or `{"type":"PRICE_DATA_MTF", "symbol":..., "data_1h":[...], ...}`
- server → EA: `{"status":"success","action":"BUY","sl_price":...,"tp_price":...,"setup_grade":"A", ...}`

The EA never makes a trading decision on its own. It just executes what
the server tells it.

---

## 3. Machines you need

You have **two realistic deployment shapes**:

### Shape A — Everything on one Windows box (easiest)
- Windows PC with MT5 installed and logged into your broker.
- Python 3.11+ installed on the same PC.
- Server, collector, trainer, and MT5 all run there.
- Set `InpServerIP = "127.0.0.1"` in the EA.

### Shape B — Windows VPS for MT5, Linux/Mac for Python
- Windows VPS: only runs MT5 + the EA.
- Linux/Mac dev box or another VPS: runs the Python server (and collector/trainer when needed).
- Set `InpServerIP` in the EA to the Python-server machine's LAN or public IP.
- Open **port 5555 TCP** in the Python-server firewall.

> ⚠️ MT5 data collection **requires Windows** because the `MetaTrader5`
> Python package only works on Windows. But you can copy the resulting
> CSV/models to any OS. Shape A is what most people should start with.

---

## 4. One-time install

### 4.1 Get Python 3.11 or newer
- Windows: <https://www.python.org/downloads/windows/> — during install tick "Add Python to PATH".
- Check with `python --version`.

### 4.2 Get the code
Whatever way you got it (this repo, a zip, download from the platform), you
should end up with a folder like:

```
grim_hybrid/
├── config.yaml
├── strategy_engine/
├── server/
├── collector/
├── trainer/
├── risk/
├── models/
├── mt5/
├── data/
├── tests/
└── requirements.txt
```

### 4.3 Install Python dependencies
Open a terminal in the `grim_hybrid/` folder and run:

```bash
pip install -r requirements.txt
```

That installs numpy, pandas, scikit-learn, xgboost, joblib, pytz, PyYAML,
and pytest.

### 4.4 (Windows only, for real data) install MetaTrader5 bindings

```bash
pip install MetaTrader5
```

If it fails, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) then retry.

### 4.5 Install the EA in MT5
1. Open MT5 → menu **File → Open Data Folder** (opens Windows Explorer).
2. Navigate into `MQL5 → Experts`.
3. Copy `grim_hybrid/mt5/GrimHybridEA_ZMQ.mq5` into that folder.
4. Back in MT5, open the **Navigator** panel (Ctrl+N) → Expert Advisors →
   right-click → **Refresh**.
5. You should now see `GrimHybridEA_ZMQ` in the list.
6. In MT5 → **Tools → Options → Expert Advisors** tick:
   - ✅ Allow algorithmic trading
   - ✅ Allow DLL imports (needed for TCP)
   - ✅ Allow WebRequest for URL: add `http://127.0.0.1` (optional; harmless)

---

## 5. Step-by-step: from zero to live trading

Order matters. Do each step, then verify it worked, then move on.

### Step 1 — Sanity-check the code (2 min)

```bash
cd grim_hybrid
python -m pytest tests/ -q
```

Expected: **`23 passed`**. If anything fails, stop and fix before continuing.

### Step 2 — Collect training data (30–120 min depending on `--years`)

This is where you gather what the AI learns from.

**Option A: Real data (recommended, requires MT5 running):**

1. Open MT5 and make sure you're logged in.
2. In each symbol you want to train on (XAUUSD, NAS100, US30, EURUSD,
   GBPUSD, USDJPY, AUDUSD), open a chart and scroll left until you see
   several years of history. This forces MT5 to download the bars.
3. Run:
   ```bash
   python collector/collect_training_data.py --years 3
   ```
4. Watch the log — you'll see per-symbol progress like:
   ```
   === XAUUSD (XAUUSD) ===
     XAUUSD scanned=500 setups=42 TP=18 SL=24
     XAUUSD: 428 samples (TP=196 SL=232 WR=45.8%)
   ```
5. When it finishes, you'll have `data/training/training_data.csv`.

**Option B: Synthetic data (for testing only, no MT5 needed):**
```bash
python collector/collect_training_data.py --synthetic
```
This produces working models but they can't predict anything real. Only
useful to verify the pipeline compiles and runs end-to-end.

**Options you can pass:**
- `--symbols XAUUSD NAS100` — restrict to specific symbols
- `--years 5` — how much history to fetch (default 2)
- `--out path/to/other.csv` — override output path

### Step 3 — Train the models (1–5 min)

```bash
python trainer/train_model.py
```

Expected output:
```
Loaded 4730 rows from data/training/training_data.csv
--- Training XAUUSD (537 samples, 73 features) ---
XAUUSD | acc=0.612 prec=0.586 rec=0.541 f1=0.562
Saved xauusd_model.pkl (+ xauusd_model.json)
... (7 more)
Training complete. Per-symbol: ['XAUUSD', 'NAS100', ...]
```

After this you'll have 8 `.pkl` files in `models/`:
- `xauusd_model.pkl`
- `nas100_model.pkl`
- `us30_model.pkl`
- `eurusd_model.pkl`
- `gbpusd_model.pkl`
- `usdjpy_model.pkl`
- `audusd_model.pkl`
- `fallback_model.pkl` ← used when a per-symbol model isn't found or has low confidence

Each `.pkl` is accompanied by a `.json` metrics file so you can inspect
per-symbol accuracy without opening the pickle.

**What "good" numbers look like on real data:**
| Metric | Weak | OK | Good |
|--------|------|----|------|
| accuracy | < 0.55 | 0.55–0.62 | > 0.62 |
| F1 | < 0.40 | 0.40–0.55 | > 0.55 |
| precision | < 0.50 | 0.50–0.60 | > 0.60 |

If you see F1 = 0 with high accuracy → the model learned "always
predict SL". That means class imbalance in the CSV. Solutions in
[Troubleshooting](#10-troubleshooting).

### Step 4 — Start the Python server (30 sec)

```bash
python server/silver_bullet_server.py
```

Expected banner:
```
============================================================
GRIM HYBRID SILVER BULLET SERVER v1.0
============================================================
LOADING MODELS
  [OK]  XAUUSD: xauusd_model.pkl
  [OK]  NAS100: nas100_model.pkl
  ...
  [OK]  fallback: fallback_model.pkl
Models loaded: 8
TCP listening on: 0.0.0.0:5555
Feature count:    73
```

Keep this window open. In production, put it behind `nssm` (Windows) or
`systemd` (Linux) so it restarts automatically.

### Step 5 — Attach the EA to a chart (2 min)

1. In MT5 open a chart for the symbol you want to trade (e.g., XAUUSD).
2. Drag `GrimHybridEA_ZMQ` from Navigator onto the chart.
3. In the dialog:
   - **Common** tab → tick "Allow Algo Trading".
   - **Inputs** tab → set:
     - `InpServerIP` = `127.0.0.1` (or your Python-server IP)
     - `InpServerPort` = `5555`
     - `InpRiskPercent` = start with **0.5** or **1.0** for testing
     - `InpMinConfidence` = `0.65` (default)
     - `InpUseServerSLTP` = `true` (recommended — uses engine's intelligent SL/TP)
     - `InpUseTrailingStop` = `false` initially
   - Click **OK**.
4. Look for the smiley 🙂 in the top-right corner of the chart — that means
   the EA is active.
5. The dashboard panel should appear on the left of the chart, showing
   "Server: RUNNING", "Model: Loaded (7)", "Features: 73".

### Step 6 — Verify end-to-end (5 min)

On the Python server terminal you should see lines like:
```
[14:30:15] XAUUSD | HOLD | conf=0.0% | score=0.28 | ENGINE | N/A | No structure/liquidity trigger
[14:30:35] XAUUSD | HOLD | conf=42.3% | score=0.52 | ML | Symbol-specific (XAUUSD) | Confidence 42.3% below 65%
[14:31:15] XAUUSD | BUY  | conf=71.2% | score=0.78 | ML | Symbol-specific (XAUUSD) |
```

On the EA dashboard you should see:
- **Signal** field cycling between BUY / SELL / HOLD
- **Grade** letter with color
- **Score** bar filling up during confluence
- **SL Type** / **TP Type** showing which logic won (LIQUIDITY / STRUCTURE / OB / ADR / …)
- **Volatility** showing NORMAL / HIGH / LOW / EXTREME

When action = BUY or SELL AND confidence ≥ InpMinConfidence AND no
existing position → the EA fires an order.

### Step 7 — Go live (whenever you're ready)

- Start with `InpRiskPercent = 0.25` on a **demo account** for at least
  a week.
- Verify win rate roughly matches the trained metrics.
- Only then increase risk and switch to a live broker account.

---

## 6. Understanding the config

All tunables live in `config.yaml` — nothing is hardcoded in Python. Edit
this file, restart the server, and everything downstream picks up the
change.

### 6.1 Risk (`config.yaml → risk:`)
```yaml
risk:
  base_risk_pct: 1.0        # base % per trade
  max_risk_pct: 2.0         # upper clamp
  min_risk_pct: 0.25        # lower clamp
  max_concurrent_trades: 3
  max_daily_trades: 10
  max_daily_loss_pct: 4.0
  breakeven_r: 1.0          # move SL to entry at +1R
  trailing_start_r: 1.5     # start trailing at +1.5R
  min_rr: 2.0               # takeprofit R:R floor
  sl_min_atr: 0.5           # SL never tighter than 0.5 * ATR
  sl_max_atr: 4.0           # SL never wider than 4 * ATR
```

### 6.2 Dynamic thresholds (`config.yaml → thresholds:`)

The whole point of Grim Hybrid — **nothing is fixed in pips**. Everything
scales with ATR.
```yaml
thresholds:
  liquidity_sweep:
    atr_factor: 0.15        # sweep depth must be >= 0.15 * ATR(15m)
  fvg:
    min_size_atr_factor: 0.10   # min FVG size = 0.10 * ATR
  displacement:
    min_atr_multiple: 1.2       # impulse candle >= 1.2 * ATR
```

**Feel that XAUUSD sweeps are too small to catch?** Raise
`atr_factor` from `0.15` → `0.20`. Save. Restart server. Done. No code
changes.

### 6.3 Score weights (`config.yaml → score_weights:`)

Every setup gets a 0-1 sub-score for 9 categories. Overall score =
weighted sum. Raise the weight of what matters to your style:
```yaml
score_weights:
  bias: 0.15
  liquidity: 0.20        # ← liquidity-heavy trader? bump this
  displacement: 0.10
  fvg: 0.10
  order_block: 0.10
  session: 0.15
  volatility: 0.05
  premium_discount: 0.10
  ltf_confirmation: 0.05
```

### 6.4 Min setup score
```yaml
min_setup_score: 0.45      # below this the AI is never called
```
Higher = fewer trades, higher quality. Lower = more trades, noisier.

### 6.5 Sessions (times in New York time, DST-safe)
```yaml
sessions:
  london_kz:            { start: "02:00", end: "05:00" }
  ny_am_silver_bullet:  { start: "10:00", end: "11:00" }
  ny_pm_silver_bullet:  { start: "14:00", end: "15:00" }
```

---

## 7. Reading the dashboard

The MT5 dashboard has three sections:

### SYSTEM STATUS
| Field | Meaning |
|-------|---------|
| Server | RUNNING (green) / OFFLINE (red) |
| Heartbeat | Seconds since last successful PING |
| Model | Loaded (7) — 7 per-symbol models available |
| Features | 73 — the canonical feature schema |
| Version | Model version string (v1.0-grim-hybrid) |
| Last Signal | Seconds since last PRICE_DATA_MTF reply |

### CURRENT SIGNAL
| Field | Meaning |
|-------|---------|
| Symbol | The chart symbol |
| Session | LONDON_KZ / NY_AM_SB / NY_PM_SB / NY / LONDON / ASIAN / OFF / WEEKEND |
| Signal | BUY / SELL / HOLD (color-coded) |
| Grade | Letter grade for this setup: A+, A, B, C, D, F |
| Score | 0-1 overall weighted score |
| TP / SL bars | ML-predicted probabilities |
| Confidence | Winning-class probability from the model |
| R:R | Planned reward-to-risk (from server's TP evaluator) |
| SL Type | Where the SL is anchored: LIQUIDITY / STRUCTURE / ORDER_BLOCK / ATR |
| TP Type | Where the TP is anchored: LIQUIDITY / PDH / PDL / RANGE / ADR / RR_FLOOR |
| Volatility | LOW / NORMAL / HIGH / EXTREME |
| SL Dist | Absolute SL distance in pips (legacy field) |
| Method | ML / ML_FALLBACK / SCORE_ONLY / ENGINE |
| Reason | Why we're HOLDing (blank when we act) |

### TRADE INFO (when in a position)
| Field | Meaning |
|-------|---------|
| Status | IN TRADE / NO POSITION |
| Entry / SL / TP | The three levels |
| Current R | Live R-multiple |
| P/L | Floating P&L in account currency |
| Entry Conf | The confidence at the moment of entry |
| Duration | Minutes since entry |

---

## 8. Retraining workflow

You retrain when:
- New market regime (a currency stops behaving like it used to).
- You changed something in `config.yaml → thresholds` or `score_weights`.
- Real out-of-sample results diverge from trained metrics.
- Every 3–6 months as normal maintenance.

The cycle:
```
1. python collector/collect_training_data.py --years 3   # regenerate CSV
2. python trainer/train_model.py                         # retrain all models
3. Restart the Python server                             # reloads new .pkls
   (EA stays connected; you'll see "Model: Loaded (7)" refresh)
```

**Before overwriting**, back up your current models:
```bash
mkdir models/backup_2026-01-20 && cp models/*.pkl models/backup_2026-01-20/
```

---

## 9. Backup, rollback, and safety

### 9.1 What to back up regularly
- `config.yaml` (your tuning)
- `models/*.pkl` (your trained models)
- `data/training/training_data.csv` (in case you want to compare)

### 9.2 If a new model performs worse
- Stop server.
- Restore the previous model files from your backup folder.
- Start server.
- The EA doesn't need to know — it'll reconnect automatically.

### 9.3 Emergency kill switch
- Remove the EA from the chart in MT5 → all future signals stop.
- Or: stop the Python server. The EA will show "Server: OFFLINE" and
  will not open any new trades (existing positions keep their SL/TP).

### 9.4 Risk safety
- The server enforces `max_concurrent_trades`, `max_daily_trades`,
  `max_daily_loss_pct` before answering BUY/SELL.
- Also set MT5-side risk controls in **Tools → Options → Trade**.
- **Start on demo. Always.**

---

## 10. Troubleshooting

### "Server: OFFLINE" on the dashboard
1. Confirm the Python server is running (its terminal window is open).
2. In the same terminal, `curl` isn't installed on Windows by default; use:
   ```bash
   python -c "import socket, json; s = socket.create_connection(('127.0.0.1',5555)); s.sendall(b'{\"type\":\"PING\"}\n'); print(s.recv(4096))"
   ```
   You should get a JSON reply. If not, the server isn't reachable.
3. Firewall — allow TCP 5555 in Windows Defender.
4. In the EA inputs, confirm `InpServerIP` and `InpServerPort` match the
   server. If server is on same machine → `127.0.0.1`.

### "Model: Rules Only" on the dashboard
- Means no `.pkl` files were loaded. Either you skipped
  `train_model.py`, or the files are in the wrong folder.
- Check `ls models/` → you should see `fallback_model.pkl` at minimum.
- Server startup log will also list which files loaded.

### The EA never opens a trade
Order of things to check (most common first):
1. **Confidence gate** — `InpMinConfidence` might be too high. Try
   dropping to 0.55 to see if trades appear.
2. **Setup score gate** — server enforces `config.yaml → min_setup_score`.
   Try lowering it to 0.35.
3. **Session** — server only sees kill-zones as `silver_bullet=true`.
   Off-hours trades still fire, but with a lower session sub-score.
4. **Max spread** — check `InpMaxSpread`. Some brokers list spread in
   points × 10; try 5000.
5. **Watch server logs** — every request shows `action | conf | score |
   hold_reason`. The `hold_reason` tells you exactly what's blocking.

### Trainer says F1 = 0
- Class imbalance. Your synthetic run had ~94% SL labels.
- On real data this is much rarer. If it happens:
  - Collect more history (`--years 5`).
  - Lower `min_setup_score` in config before re-collecting → more diverse setups.
  - Investigate: `python -c "import pandas as pd; d = pd.read_csv('data/training/training_data.csv'); print(d['outcome'].value_counts())"` — if one class is < 20% of samples, you have imbalance.

### "MetaTrader5 not available" when collecting
- You're not on Windows, or MT5 isn't installed, or the `MetaTrader5`
  package isn't installed.
- For a smoke run without MT5: `python collector/collect_training_data.py --synthetic`

### Python server crashes on startup
```bash
tail -100 <wherever you redirected stderr>
```
Most common causes:
- `models/` folder missing → `mkdir -p models` and re-run trainer.
- `config.yaml` has bad YAML syntax → open in an editor, check indentation.
- Port 5555 already in use → change in `config.yaml → server.port`.

### Tests fail after editing engine code
- Run `python -m pytest tests/ -q --tb=short` to see which test.
- The `test_features_match_between_two_calls` test asserts determinism —
  if it fails you introduced randomness somewhere. Remove it.
- The `test_collector_and_live_produce_same_features` test is the
  zero-drift guarantee — never disable it.

---

## 11. FAQ

**Q: Do I need to retrain when I edit `config.yaml`?**  
Only when you change anything under `thresholds:` or `score_weights:` —
because those affect the features. Editing risk/session/models sections
does **not** require retraining.

**Q: Can I add a new symbol?**  
Yes. Add it under `symbols:` in `config.yaml` with its pip_value and
broker aliases, then run collector + trainer. A new `<symbol>_model.pkl`
appears in `models/`.

**Q: Can I run multiple EAs on multiple charts against one server?**  
Yes. The Python server is multithreaded (`ThreadingMixIn`). Each EA
opens its own TCP connection per request.

**Q: Can I add new features to the model?**  
Yes but with care. Append them to `FEATURE_ORDER` in
`strategy_engine/feature_builder.py` (append only, never reorder or
delete existing entries). Recollect + retrain. Existing models will
still work (they only use the columns they were trained on).

**Q: Where's the backtester?**  
Not in v1. It's on the P2 backlog. When it lands, it'll reuse the same
`StrategyEngine.analyze()` — same guarantee of zero drift.

**Q: Where do I put my broker's magic number / comment prefix?**  
The EA already sets `magic = InpMagicNumber` and comment
`"GrimHybrid_<GRADE>_<SESSION>_<CONF>"`. Change `InpMagicNumber` per
account so open positions are tracked independently.

**Q: How much does the AI actually contribute vs. the engine?**  
The engine filters setups (score gate). The AI ranks the survivors.
If the AI says "SL", the EA holds. If it says "TP" with confidence ≥
`InpMinConfidence`, the EA fires. On synthetic data the AI is useless;
on real data the AI is where your edge lives.

**Q: I want to force a trade for testing.**  
Two safe options:
1. Temporarily lower `InpMinConfidence` to `0.01`.
2. Or lower `min_setup_score` in `config.yaml` to `0.0` and lower
   `models.min_confidence` to `0.0`. **Reset after testing.**

---

## Quick reference card

```bash
# Sanity check
python -m pytest tests/ -q

# Regenerate training data
python collector/collect_training_data.py --years 3

# Retrain all models
python trainer/train_model.py

# Run the server
python server/silver_bullet_server.py

# Inspect a model's metrics
python -c "import joblib; m=joblib.load('models/xauusd_model.pkl')['metrics']; print(m)"

# Live-tail server decisions
python server/silver_bullet_server.py | tee -a logs/server.log
```

**Configuration files you'll touch most:**
- `config.yaml` — 90% of your tuning
- `mt5/GrimHybridEA_ZMQ.mq5` inputs (in the MT5 dialog) — 10%

**Files you should NEVER edit by hand:**
- `models/*.pkl` — they're binary artifacts
- `data/training/*.csv` — regenerate instead
- `strategy_engine/feature_builder.py FEATURE_ORDER` — only ever append, never reorder or delete

---

*If you hit anything not covered here, drop a note in the project chat with:
(1) the last 30 lines of the Python server log,
(2) the current EA dashboard values,
(3) the exact step that broke.*
