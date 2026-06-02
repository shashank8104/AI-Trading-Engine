"""
FastAPI application entry point.

Wires the complete real-time trading pipeline on startup:
    WebSocket ticks → asyncio.Queue → CandleEngine → FeatureEngine → MLEngine → SignalEngine

Lifecycle:
    startup  — init DB, load models, load instruments, start pipeline
    shutdown — flush candles, stop WebSocket, cleanup
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.api.routes import router, ws_manager
from app.config import get_settings
from app.core.candle_engine import CandleEngine
from app.core.data_ingest import DataIngestor
from app.core.feature_engine import FeatureEngine
from app.core.ml_engine import MLEngine
from app.core.signal_engine import SignalEngine
from app.db.crud import CandleCRUD, InstrumentCRUD
from app.db.database import _get_session_factory, init_db
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# ── Global pipeline components (accessed by routes for status) ────────────
ingestor: DataIngestor | None = None
candle_engine: CandleEngine | None = None
feature_engine: FeatureEngine | None = None
ml_engine: MLEngine | None = None
signal_engine: SignalEngine | None = None


# ── Pipeline callback ─────────────────────────────────────────────────────


async def on_candle_update(candle: dict) -> None:
    """Broadcast real-time open candle tick updates for NIFTY only."""
    settings = get_settings()
    if candle.get("instrument_token") == settings.NIFTY_INSTRUMENT_TOKEN:
        data = candle.copy()
        data["timestamp"] = data["timestamp"].isoformat()
        data["type"] = "CANDLE_UPDATE"
        await ws_manager.broadcast(data)


async def on_candle_close(candle: dict) -> None:
    """
    Pipeline callback: invoked when a candle interval closes.

    Full pipeline:
        1. Persist candle to PostgreSQL
        2. Update OI tracking for options instruments
        3. Compute features (NIFTY index candles only)
        4. Run ML ensemble prediction
        5. Evaluate signal rules and emit trade signal
    """
    settings = get_settings()

    # ── Step 1: Persist and optionally Broadcast candle ───────────
    if candle.get("instrument_token") == settings.NIFTY_INSTRUMENT_TOKEN:
        data = candle.copy()
        data["timestamp"] = data["timestamp"].isoformat()
        data["type"] = "CANDLE_CLOSE"
        await ws_manager.broadcast(data)

    session_factory = _get_session_factory()
    async with session_factory() as session:
        await CandleCRUD.insert_candle(session, candle.copy())

    # ── Step 2: Update OI for options instruments ─────────────────
    if feature_engine and candle.get("oi", 0) > 0:
        token = candle["instrument_token"]
        if token != settings.NIFTY_INSTRUMENT_TOKEN:
            if ingestor and ingestor._nfo_instruments.get(token):
                info = ingestor._nfo_instruments[token]
                feature_engine.update_options_oi(
                    token, candle["oi"], info["type"], info["strike"]
                )

    # ── Step 3: Compute features ──────────────────────────────────
    features = None
    if feature_engine:
        features = await feature_engine.compute_features(candle)

    # ── Step 4 & 5: ML prediction → Signal (NIFTY index only) ────
    if (
        features is not None
        and ml_engine is not None
        and ml_engine.is_loaded
        and signal_engine is not None
        and candle["instrument_token"] == settings.NIFTY_INSTRUMENT_TOKEN
    ):
        prediction = await ml_engine.predict(
            candle["instrument_token"], candle["interval"]
        )

        if prediction is not None:
            # ── Always persist prediction independent of signal logic ──
            session_factory = _get_session_factory()
            from app.db.crud import PredictionCRUD
            async with session_factory() as session:
                await PredictionCRUD.insert_prediction(
                    session,
                    {
                        "timestamp": candle["timestamp"],
                        "instrument_token": candle["instrument_token"],
                        "interval": candle["interval"],
                        "xgboost_bullish": prediction.xgboost_probs.get("BULLISH"),
                        "xgboost_bearish": prediction.xgboost_probs.get("BEARISH"),
                        "xgboost_neutral": prediction.xgboost_probs.get("NEUTRAL"),
                        "lstm_bullish": prediction.lstm_probs.get("BULLISH"),
                        "lstm_bearish": prediction.lstm_probs.get("BEARISH"),
                        "lstm_neutral": prediction.lstm_probs.get("NEUTRAL"),
                        "ensemble_bullish": prediction.ensemble_probs.get("BULLISH"),
                        "ensemble_bearish": prediction.ensemble_probs.get("BEARISH"),
                        "ensemble_neutral": prediction.ensemble_probs.get("NEUTRAL"),
                        "ensemble_confidence": prediction.ensemble_confidence,
                        "ensemble_direction": prediction.ensemble_direction,
                    },
                )

            pred_data = {
                "type": "PREDICTION",
                "ensemble_direction": prediction.ensemble_direction,
                "ensemble_confidence": prediction.ensemble_confidence,
                "timestamp": candle["timestamp"].isoformat()
            }
            await ws_manager.broadcast(pred_data)

            atr = features.get("atr_14")
            signal = await signal_engine.process_prediction(
                prediction, candle, atr=atr, features=features
            )

            if signal:
                sig_data = {
                    "type": "SIGNAL",
                    "direction": signal.direction,
                    "spread_type": signal.spread_type,
                    "confidence": signal.confidence,
                    "entry_price": signal.entry_price,
                    "target": signal.target,
                    "stop_loss": signal.stop_loss,
                    "timestamp": signal.timestamp.isoformat()
                }
                await ws_manager.broadcast(sig_data)


# ── Application Lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global ingestor, candle_engine, feature_engine, ml_engine, signal_engine

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger.info("starting_trading_system")

    # ── 1. Initialize database ────────────────────────────────────
    await init_db()
    logger.info("database_initialized")

    # ── 2. Initialize pipeline components ─────────────────────────
    loop = asyncio.get_running_loop()
    tick_queue = asyncio.Queue(maxsize=10_000)

    feature_engine = FeatureEngine()
    candle_engine = CandleEngine(
        tick_queue=tick_queue, 
        on_candle_close=on_candle_close,
        on_candle_update=on_candle_update
    )
    ingestor = DataIngestor(loop=loop, tick_queue=tick_queue)

    # ── 3. Load ML models ─────────────────────────────────────────
    ml_engine = MLEngine()
    ml_engine.load_models()

    signal_engine = SignalEngine()
    logger.info("ml_signal_engines_initialized")

    # ── 4. Load NFO instruments for dynamic ATM tracking ──────────
    session_factory = _get_session_factory()
    async with session_factory() as session:
        nearest_expiry = await InstrumentCRUD.get_nearest_expiry(session)
        if nearest_expiry:
            nfo_options = await InstrumentCRUD.get_nfo_options(
                session, strike_min=0, strike_max=100_000, expiry=nearest_expiry
            )
            nfo_map = {
                inst.instrument_token: {
                    "strike": inst.strike,
                    "type": inst.instrument_type,
                    "symbol": inst.tradingsymbol,
                }
                for inst in nfo_options
            }
            ingestor.set_nfo_instruments(nfo_map)

            # Initialize OI tracking in feature engine
            for inst in nfo_options:
                feature_engine.update_options_oi(
                    inst.instrument_token, 0, inst.instrument_type, inst.strike
                )

            logger.info(
                "nfo_instruments_loaded",
                count=len(nfo_map),
                expiry=str(nearest_expiry),
            )
        else:
            logger.warning(
                "no_nfo_instruments",
                hint="Run: python scripts/seed_instruments.py",
            )

    # ── 5. Start pipeline ─────────────────────────────────────────
    ingestor.start()
    await ingestor.backfill_today_history()
    candle_task = asyncio.create_task(candle_engine.run())
    logger.info("pipeline_started")

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("shutting_down")
    ingestor.stop()
    candle_engine.stop()
    await candle_engine.flush_all()
    candle_task.cancel()
    try:
        await candle_task
    except asyncio.CancelledError:
        pass
    logger.info("shutdown_complete")


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Trading Intelligence System",
    description="Real-time AI-powered decision support for NIFTY 50 options trading",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")

import os
html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

@app.get("/")
async def serve_frontend():
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Frontend not found."}
