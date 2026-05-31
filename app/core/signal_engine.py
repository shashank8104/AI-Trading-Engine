"""
Signal Engine: converts ML predictions into actionable trade signals.

Rules (strict per PRD):
    1. Confidence ≥ threshold (default 0.7)
    2. Both XGBoost and LSTM must agree on direction
    3. No signals in first 15 minutes (before 9:30) or last 30 minutes (after 15:00)
    4. Cooldown: minimum 2 candle intervals between signals
    5. Only BULLISH or BEARISH generate signals (NEUTRAL is ignored)

Signal output includes:
    - Direction, spread type, suggested strikes
    - Entry price, stop loss, target, risk-reward ratio
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

from app.config import get_settings
from app.core.ml_engine import PredictionResult
from app.db.crud import PredictionCRUD, SignalCRUD
from app.db.database import _get_session_factory
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Market time filters (IST)
_SIGNAL_START = time(9, 30)   # No signals before 9:30 (first 15 min)
_SIGNAL_END = time(15, 0)     # No signals after 15:00 (last 30 min)


@dataclass
class TradeSignal:
    """Complete trade signal recommendation."""

    timestamp: datetime
    direction: str           # BULLISH | BEARISH
    spread_type: str          # BULL_CALL_SPREAD | BEAR_PUT_SPREAD
    confidence: float
    entry_price: float        # Current NIFTY spot
    buy_strike: float         # Strike to buy
    sell_strike: float        # Strike to sell
    stop_loss: float
    target: float
    risk_reward: float


class SignalEngine:
    """
    Generates trade signals from ML ensemble predictions.

    Applies strict filtering rules before emitting a signal.
    """

    def __init__(self):
        self.settings = get_settings()
        self._last_signal_time: Optional[datetime] = None
        self._signals_generated = 0

    async def process_prediction(
        self,
        prediction: PredictionResult,
        candle: dict,
        atr: Optional[float] = None,
    ) -> Optional[TradeSignal]:
        """
        Evaluate a prediction and generate a signal if all rules pass.

        Args:
            prediction: ensemble output from MLEngine
            candle: the candle that triggered this prediction
            atr: current ATR value (for SL/target calculation)

        Returns:
            TradeSignal if all rules pass, None otherwise.
        """
        now = candle.get("timestamp", datetime.now())
        if isinstance(now, str):
            now = datetime.fromisoformat(now)

        # ── Rule 1: Direction must be actionable ──────────────────
        direction = prediction.ensemble_direction
        if direction == "NEUTRAL":
            logger.debug("signal_skipped", reason="neutral_prediction")
            return None

        # ── Rule 2: Both models must agree ────────────────────────
        if not prediction.models_agree:
            logger.info(
                "signal_skipped",
                reason="models_disagree",
                xgb=max(prediction.xgboost_probs, key=prediction.xgboost_probs.get),
                lstm=max(prediction.lstm_probs, key=prediction.lstm_probs.get),
            )
            return None

        # ── Rule 3: Confidence threshold ──────────────────────────
        confidence = prediction.ensemble_confidence
        threshold = self.settings.SIGNAL_CONFIDENCE_THRESHOLD
        if confidence < threshold:
            logger.info(
                "signal_skipped",
                reason="low_confidence",
                confidence=round(confidence, 4),
                threshold=threshold,
            )
            return None

        # ── Rule 4: Market timing filter ──────────────────────────
        tick_time = now.time()
        if tick_time < _SIGNAL_START:
            logger.debug("signal_skipped", reason="before_signal_window")
            return None
        if tick_time > _SIGNAL_END:
            logger.debug("signal_skipped", reason="after_signal_window")
            return None

        # ── Rule 5: Cooldown (2 candle intervals) ─────────────────
        if self._last_signal_time is not None:
            # Use 5-minute candles as the base interval for cooldown
            cooldown = timedelta(minutes=10)  # 2 × 5m candles
            if (now - self._last_signal_time) < cooldown:
                logger.debug(
                    "signal_skipped",
                    reason="cooldown",
                    last=self._last_signal_time.isoformat(),
                )
                return None

        # ── All rules passed — generate signal ────────────────────
        signal = self._build_signal(
            direction=direction,
            confidence=confidence,
            candle=candle,
            atr=atr,
            timestamp=now,
        )

        # Persist prediction and signal to database
        await self._persist(prediction, signal, candle)

        self._last_signal_time = now
        self._signals_generated += 1

        logger.info(
            "signal_generated",
            direction=signal.direction,
            spread=signal.spread_type,
            confidence=round(signal.confidence, 4),
            entry=signal.entry_price,
            sl=signal.stop_loss,
            target=signal.target,
            rr=signal.risk_reward,
        )

        return signal

    @property
    def signals_generated(self) -> int:
        return self._signals_generated

    # ── Signal Construction ───────────────────────────────────────

    def _build_signal(
        self,
        direction: str,
        confidence: float,
        candle: dict,
        atr: Optional[float],
        timestamp: datetime,
    ) -> TradeSignal:
        """
        Build a complete trade signal with spread recommendation.

        Uses ATR for stop-loss/target sizing if available,
        otherwise defaults to percentage-based calculation.
        """
        spot = candle.get("close", 0)
        step = self.settings.NIFTY_STRIKE_STEP
        atm = round(spot / step) * step

        # Default ATR if not provided (0.5% of spot)
        if atr is None or atr <= 0:
            atr = spot * 0.005

        if direction == "BULLISH":
            spread_type = "BULL_CALL_SPREAD"
            buy_strike = atm
            sell_strike = atm + step
            stop_loss = round(spot - (1.5 * atr), 2)
            target = round(spot + (2.0 * atr), 2)
        else:  # BEARISH
            spread_type = "BEAR_PUT_SPREAD"
            buy_strike = atm
            sell_strike = atm - step
            stop_loss = round(spot + (1.5 * atr), 2)
            target = round(spot - (2.0 * atr), 2)

        # Risk-reward ratio
        risk = abs(spot - stop_loss)
        reward = abs(target - spot)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

        return TradeSignal(
            timestamp=timestamp,
            direction=direction,
            spread_type=spread_type,
            confidence=round(confidence, 4),
            entry_price=round(spot, 2),
            buy_strike=buy_strike,
            sell_strike=sell_strike,
            stop_loss=stop_loss,
            target=target,
            risk_reward=risk_reward,
        )

    # ── Persistence ───────────────────────────────────────────────

    async def _persist(
        self,
        prediction: PredictionResult,
        signal: TradeSignal,
        candle: dict,
    ) -> None:
        """Save signal to database."""
        session_factory = _get_session_factory()
        async with session_factory() as session:
            # Save signal
            await SignalCRUD.insert_signal(
                session,
                {
                    "timestamp": signal.timestamp,
                    "direction": signal.direction,
                    "spread_type": signal.spread_type,
                    "confidence": signal.confidence,
                    "entry_price": signal.entry_price,
                    "buy_strike": signal.buy_strike,
                    "sell_strike": signal.sell_strike,
                    "stop_loss": signal.stop_loss,
                    "target": signal.target,
                    "risk_reward": signal.risk_reward,
                    "status": "ACTIVE",
                },
            )
