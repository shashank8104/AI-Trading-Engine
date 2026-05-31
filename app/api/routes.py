"""
FastAPI REST API routes.

Endpoints serve data to the internal HTML dashboard and any other consumers.
All read operations — no mutations through the API.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.config import get_settings
from app.db.crud import CandleCRUD, PredictionCRUD, SignalCRUD
from app.db.database import get_db

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.active_connections.remove(connection)

ws_manager = ConnectionManager()


@router.get("/health")
async def health():
    """System health check."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@router.get("/status")
async def system_status():
    """
    Pipeline status overview.

    Returns WebSocket state, NIFTY spot, ATM strike,
    subscription count, and candle engine stats.
    """
    # Late import to avoid circular dependency with main module globals
    from app.main import candle_engine, ingestor, ml_engine, signal_engine

    return {
        "websocket_connected": ingestor.is_connected if ingestor else False,
        "nifty_spot": ingestor.nifty_spot if ingestor else 0,
        "current_atm": ingestor.current_atm if ingestor else 0,
        "subscribed_instruments": ingestor.subscribed_count if ingestor else 0,
        "candle_buffers_active": candle_engine.active_buffers if candle_engine else 0,
        "candles_emitted_total": candle_engine.candles_emitted if candle_engine else 0,
        "ml_models_loaded": ml_engine.is_loaded if ml_engine else False,
        "signals_generated": signal_engine.signals_generated if signal_engine else 0,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/candles/{interval}")
async def get_candles(
    interval: str,
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Get recent NIFTY candles for a given interval.

    Args:
        interval: "5m" or "15m"
        limit: number of candles (1–500, default 100)
    """
    settings = get_settings()
    candles = await CandleCRUD.get_recent_candles(
        db, settings.NIFTY_INSTRUMENT_TOKEN, interval, limit=limit
    )
    return [
        {
            "timestamp": c.timestamp.isoformat(),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "oi": c.oi,
        }
        for c in candles
    ]


@router.get("/signals/latest")
async def get_latest_signals(db: AsyncSession = Depends(get_db)):
    """Get all currently active trade signals."""
    signals = await SignalCRUD.get_active_signals(db)
    return [
        {
            "id": s.id,
            "timestamp": s.timestamp.isoformat(),
            "direction": s.direction,
            "spread_type": s.spread_type,
            "confidence": s.confidence,
            "entry_price": s.entry_price,
            "buy_strike": s.buy_strike,
            "sell_strike": s.sell_strike,
            "stop_loss": s.stop_loss,
            "target": s.target,
            "risk_reward": s.risk_reward,
            "status": s.status,
        }
        for s in signals
    ]


@router.get("/signals/history")
async def get_signal_history(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get signal history, newest first."""
    signals = await SignalCRUD.get_signal_history(db, limit=limit)
    return [
        {
            "id": s.id,
            "timestamp": s.timestamp.isoformat(),
            "direction": s.direction,
            "spread_type": s.spread_type,
            "confidence": s.confidence,
            "entry_price": s.entry_price,
            "target": s.target,
            "stop_loss": s.stop_loss,
            "status": s.status,
        }
        for s in signals
    ]


@router.get("/predictions/latest")
async def get_latest_prediction(db: AsyncSession = Depends(get_db)):
    """Get the most recent ML model prediction."""
    pred = await PredictionCRUD.get_latest_prediction(db)
    if not pred:
        return {"message": "No predictions yet"}
    return {
        "timestamp": pred.timestamp.isoformat(),
        "ensemble_direction": pred.ensemble_direction,
        "ensemble_confidence": pred.ensemble_confidence,
        "xgboost": {
            "bullish": pred.xgboost_bullish,
            "bearish": pred.xgboost_bearish,
            "neutral": pred.xgboost_neutral,
        },
        "lstm": {
            "bullish": pred.lstm_bullish,
            "bearish": pred.lstm_bearish,
            "neutral": pred.lstm_neutral,
        },
    }

@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint that clients can connect to."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for ping/pong if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
