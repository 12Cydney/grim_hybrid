from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone

# Import grim_hybrid modules
import sys
_HERE = Path(__file__).resolve().parent
_GRIM_ROOT = _HERE.parent / "grim_hybrid"
if str(_GRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRIM_ROOT))

try:
    from grim_hybrid.strategy_engine.utils import load_config
    from grim_hybrid.strategy_engine import StrategyEngine, MarketData
    from grim_hybrid.strategy_engine.feature_builder import build_features
    from grim_hybrid.server.silver_bullet_server import ModelRegistry, runtime_versions, trained_versions, version_verdict, format_status_table
    from grim_hybrid.risk.risk_engine import RiskEngine, AccountState, PositionBook
    GRIM_AVAILABLE = True
except ImportError as e:
    GRIM_AVAILABLE = False
    print(f"Warning: Grim Hybrid modules not available: {e}")


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'grim_hybrid')]

# Create the main app without a prefix
app = FastAPI(title="Grim Hybrid API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize Grim Hybrid components
_config = None
_strategy_engine = None
_model_registry = None
_risk_engine = None
_position_book = None


def get_config():
    global _config
    if _config is None and GRIM_AVAILABLE:
        try:
            _config = load_config()
        except Exception as e:
            print(f"Failed to load config: {e}")
    return _config


def get_strategy_engine():
    global _strategy_engine
    if _strategy_engine is None and GRIM_AVAILABLE:
        cfg = get_config()
        if cfg:
            try:
                _strategy_engine = StrategyEngine(cfg)
            except Exception as e:
                print(f"Failed to create StrategyEngine: {e}")
    return _strategy_engine


def get_model_registry():
    global _model_registry
    if _model_registry is None and GRIM_AVAILABLE:
        cfg = get_config()
        if cfg:
            try:
                _model_registry = ModelRegistry(cfg)
            except Exception as e:
                print(f"Failed to create ModelRegistry: {e}")
    return _model_registry


def get_risk_engine():
    global _risk_engine, _position_book
    if _risk_engine is None and GRIM_AVAILABLE:
        cfg = get_config()
        if cfg:
            try:
                _risk_engine = RiskEngine(cfg)
                from server.decision_log import DecisionLog
                _position_book = PositionBook(_risk_engine, Path(cfg["paths"].get("logs_dir", "logs")) / "account_state.json")
            except Exception as e:
                print(f"Failed to create RiskEngine: {e}")
    return _risk_engine


def get_position_book():
    global _position_book
    if _position_book is None:
        get_risk_engine()
    return _position_book


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str


# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Grim Hybrid API Server", "version": "1.0.0", "status": "running"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)

    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()

    _ = await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)

    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])

    return status_checks


# =====================================================================
# Dashboard API Endpoints
# =====================================================================

@api_router.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    config = get_config()
    engine = get_strategy_engine()
    registry = get_model_registry()
    risk = get_risk_engine()
    book = get_position_book()

    status = {
        "status": "ok" if GRIM_AVAILABLE else "degraded",
        "grim_available": GRIM_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_loaded": config is not None,
        "engine_loaded": engine is not None,
        "models_loaded": registry is not None and len(registry.models) > 0,
        "model_count": len(registry.models) if registry else 0,
        "has_fallback": registry.fallback is not None if registry else False,
    }

    if book:
        state = book.state()
        status.update({
            "daily_pnl_pct": round(state.daily_pnl_pct, 4),
            "daily_trades": state.daily_trades,
            "open_trades": state.open_trades,
            "max_daily_trades": state.max_daily_trades,
            "max_daily_loss_pct": state.max_daily_loss_pct,
        })

    return status


@api_router.get("/models/status")
async def get_model_status():
    """Get model registry status"""
    registry = get_model_registry()
    if not registry:
        return []

    rows = []
    for r in registry.status:
        rows.append({
            "name": r["name"],
            "status": r["status"],
            "trained_sklearn": r["trained_sklearn"],
            "trained_xgboost": r["trained_xgboost"],
            "runtime_sklearn": r["runtime_sklearn"],
            "runtime_xgboost": r["runtime_xgboost"],
            "version_check": r["version_check"],
            "error": r.get("error", ""),
        })
    return rows


@api_router.get("/signals")
async def get_recent_signals(limit: int = 50):
    """Get recent trading signals from decision log"""
    try:
        config = get_config()
        if not config:
            return []

        logs_dir = Path(config["paths"].get("logs_dir", "logs"))
        decisions_file = logs_dir / "decisions.csv"

        if not decisions_file.exists():
            return []

        import pandas as pd
        df = pd.read_csv(decisions_file)

        # Get recent signals
        recent = df.tail(limit).to_dict('records')
        return recent
    except Exception as e:
        print(f"Error fetching signals: {e}")
        return []


@api_router.get("/risk/state")
async def get_risk_state():
    """Get current risk engine state"""
    book = get_position_book()
    risk = get_risk_engine()

    if not book or not risk:
        raise HTTPException(status_code=503, detail="Risk engine not available")

    state = book.state()

    return {
        "daily_pnl_pct": round(state.daily_pnl_pct, 4),
        "daily_trades": state.daily_trades,
        "open_trades": state.open_trades,
        "max_daily_trades": state.max_daily_trades,
        "max_daily_loss_pct": state.max_daily_loss_pct,
        "max_concurrent_trades": state.max_concurrent_trades,
        "base_risk_pct": state.base_risk_pct,
        "last_reset": state.last_reset,
    }


@api_router.get("/engine/analyze")
async def analyze_market(
    symbol: str = "XAUUSD",
    timeframe: str = "15m",
):
    """Run market analysis for a symbol"""
    engine = get_strategy_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Strategy engine not available")

    # This would require market data - for now return mock
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "status": "engine_ready",
        "message": "Market analysis endpoint ready. Provide market data via POST."
    }


@api_router.post("/engine/analyze")
async def analyze_market_post(data: dict):
    """Run market analysis with provided market data"""
    engine = get_strategy_engine()
    registry = get_model_registry()
    risk = get_risk_engine()
    book = get_position_book()

    if not engine:
        raise HTTPException(status_code=503, detail="Strategy engine not available")

    try:
        # Parse market data
        md = MarketData(
            symbol=data.get("symbol", "XAUUSD"),
            data_1h=data.get("data_1h", []),
            data_15m=data.get("data_15m", []),
            data_1m=data.get("data_1m", []),
            spread=float(data.get("spread", 0.0)),
            timestamp=data.get("timestamp"),
        )

        # Analyze
        setup = engine.analyze(md)

        if not setup.valid:
            return {
                "valid": False,
                "action": "HOLD",
                "hold_reason": setup.hold_reason or "No valid setup",
                "setup_score": float(setup.score.overall) if setup.score else 0.0,
            }

        # Build features
        features = engine.build_features(setup)

        # Get model prediction if available
        pack, model_type = None, "N/A"
        if registry:
            pack, model_type = registry.resolve(data.get("symbol", "XAUUSD"))

        prediction = 0
        confidence = 0.0
        method = "SCORE_ONLY"

        if pack is not None:
            try:
                from server.silver_bullet_server import _predict, decision_threshold
                pred, conf = _predict(pack, features)
                prediction = int(pred)
                confidence = float(conf)
                method = "ML"
            except Exception as e:
                print(f"Prediction error: {e}")

        # Determine action
        action = "HOLD"
        hold_reason = ""

        if pack is not None:
            min_conf = decision_threshold(get_config(), pack)
            if prediction == 1 and confidence >= min_conf:
                action = "BUY" if setup.direction == 1 else ("SELL" if setup.direction == -1 else "HOLD")
                hold_reason = "" if action != "HOLD" else "No direction"
            else:
                if confidence < min_conf:
                    hold_reason = f"Confidence {confidence:.1%} below {min_conf:.0%}"
                else:
                    hold_reason = "Model predicts SL"
        else:
            score = setup.score.overall if setup.score else 0.0
            if score >= 0.65:
                action = "BUY" if setup.direction == 1 else ("SELL" if setup.direction == -1 else "HOLD")
            else:
                hold_reason = f"Score {score:.2f} below threshold"

        # Risk guardrails
        risk_info = {"risk_allowed": True, "risk_pct": 0.0, "lot_size": 0.0}
        if action in ("BUY", "SELL") and risk and book:
            state = book.state()
            allow, why = risk.guardrails(state)
            if not allow:
                action, hold_reason = "HOLD", f"Risk: {why}"
                risk_info = {"risk_allowed": False}
            else:
                sl_p = setup.stoploss.price if setup.stoploss else 0.0
                tp_p = setup.takeprofit.price if setup.takeprofit else 0.0
                dec = risk.size(data.get("symbol", "XAUUSD"), setup.entry, sl_p, tp_p, confidence, state)
                risk_info = {"risk_allowed": dec.allow_trade, "risk_pct": float(dec.risk_pct),
                             "lot_size": float(dec.lot_size)}
                if not dec.allow_trade:
                    action, hold_reason = "HOLD", f"Risk: {dec.reason}"

        # Build response
        sl = setup.stoploss
        tp = setup.takeprofit
        pip = setup.pip_value or 0.0001
        sl_dist_pips = (sl.distance / pip) if (sl and sl.distance > 0) else 0.0

        return {
            "status": "success",
            "symbol": data.get("symbol", "XAUUSD"),
            "action": action,
            "confidence": confidence,
            "setup_score": float(setup.score.overall) if setup.score else 0.0,
            "setup_grade": setup.score.grade if setup.score else "F",
            "direction": int(setup.direction),
            "signal": 1 if action == "BUY" else (-1 if action == "SELL" else 0),
            "entry": float(setup.entry),
            "sl_price": float(sl.price) if sl else 0.0,
            "tp_price": float(tp.price) if tp else 0.0,
            "sl_distance": float(sl_dist_pips),
            "rr_planned": float(tp.rr) if tp else 0.0,
            "method": method,
            "model_type": model_type,
            "hold_reason": hold_reason,
            "risk": risk_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)