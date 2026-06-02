"""
Signal Engine: multi-factor confluence scoring for trade signal generation.

Strategy: Supertrend + Momentum Confluence
    Four independent signals are scored and weighted:
        1. Supertrend direction       (weight 1.5) — ATR-based trend confirmation
        2. RSI + MACD momentum        (weight 1.0) — momentum alignment
        3. Bollinger Band %B squeeze  (weight 1.0) — breakout positioning
        4. Volume spike confirmation  (weight 0.5) — volume validates the move

    Max possible absolute score = 4.0

    Signal fires when |confluence_score| >= effective_threshold:
        - Without ML agreement: threshold = 2.5
        - With ML agreement:    threshold = 2.0  (ML ensemble is a soft bonus)

    Timing rules:
        - No signals before 9:30 AM or after 3:00 PM IST
        - 10-minute cooldown between consecutive signals
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

from app.config import get_settings
from app.core.ml_engine import PredictionResult
from app.db.crud import SignalCRUD
from app.db.database import _get_session_factory
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Market time filters (IST)
_SIGNAL_START = time(9, 30)
_SIGNAL_END   = time(15, 0)

# Confluence parameters
_CONFLUENCE_THRESHOLD = 2.5   # Hard threshold without ML agreement
_ML_BOOST             = 0.5   # Reduces threshold to 2.0 when ML agrees
_COOLDOWN_MINUTES     = 10
_MIN_CONFIDENCE       = 0.60  # Minimum normalised confidence (filters weak signals)

# ATR multipliers for SL/Target sizing
# Smaller multipliers = reachable intraday targets on 5m NIFTY candles
_ATR_TARGET = 1.0   # Target  = 1.0 × ATR  (~25-35 pts on NIFTY)
_ATR_STOP   = 0.75  # SL      = 0.75 × ATR (~18-26 pts on NIFTY)
                    # Risk/Reward = 1.0 / 0.75 = 1.33:1


@dataclass
class TradeSignal:
    """Complete trade signal recommendation."""

    timestamp:    datetime
    direction:    str     # BULLISH | BEARISH
    spread_type:  str     # BULL_CALL_SPREAD | BEAR_PUT_SPREAD
    confidence:   float   # Normalized score mapped to 0.5–1.0
    entry_price:  float   # NIFTY spot at signal time
    buy_strike:   float
    sell_strike:  float
    stop_loss:    float
    target:       float
    risk_reward:  float


class SignalEngine:
    """
    Generates trade signals using multi-factor confluence scoring.

    The ML ensemble (XGBoost + LSTM) acts as a soft bonus — it lowers
    the confluence threshold from 2.5 → 2.0 when it agrees with the
    quantitative direction, but cannot alone trigger or block a signal.
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
        features: Optional[dict] = None,
    ) -> Optional[TradeSignal]:
        """
        Evaluate confluence score and generate a signal if all gates pass.

        Args:
            prediction:  ML ensemble output (used as soft bonus)
            candle:      the closed candle that triggered this evaluation
            atr:         ATR value for stop-loss/target sizing
            features:    computed feature dict from FeatureEngine
        """
        now = candle.get("timestamp", datetime.now())
        if isinstance(now, str):
            now = datetime.fromisoformat(now)

        # ── Gate 1: Confluence score ───────────────────────────────
        if not features:
            logger.warning("signal_skipped", reason="no_features_for_confluence")
            return None

        score, direction, ml_boost = self._compute_confluence_score(features, prediction)
        effective_threshold = _CONFLUENCE_THRESHOLD - ml_boost
        abs_score = abs(score)

        logger.info(
            "confluence_evaluated",
            score=round(score, 3),
            direction=direction,
            ml_boost=ml_boost,
            threshold=effective_threshold,
        )

        if direction == "NEUTRAL" or abs_score < effective_threshold:
            logger.info(
                "signal_skipped",
                reason="confluence_insufficient",
                score=round(abs_score, 3),
                threshold=effective_threshold,
            )
            return None

        # ── Gate 1b: Minimum confidence ───────────────────────────
        # Map abs_score → [0.5, 1.0] and reject if still too weak
        normalized_confidence = round(
            0.5 + 0.5 * min(
                (abs_score - effective_threshold) / (4.0 - effective_threshold), 1.0
            ),
            4,
        )
        if normalized_confidence < _MIN_CONFIDENCE:
            logger.info(
                "signal_skipped",
                reason="confidence_too_low",
                confidence=normalized_confidence,
                min_required=_MIN_CONFIDENCE,
            )
            return None

        # ── Gate 2: Market timing ──────────────────────────────────
        tick_time = now.time()
        if tick_time < _SIGNAL_START:
            logger.debug("signal_skipped", reason="before_signal_window")
            return None
        if tick_time > _SIGNAL_END:
            logger.debug("signal_skipped", reason="after_signal_window")
            return None

        # ── Gate 3: Cooldown ───────────────────────────────────────
        if self._last_signal_time is not None:
            if (now - self._last_signal_time) < timedelta(minutes=_COOLDOWN_MINUTES):
                logger.debug(
                    "signal_skipped",
                    reason="cooldown",
                    last=self._last_signal_time.isoformat(),
                )
                return None

        # ── All gates passed — build and persist signal ────────────
        # Confidence already computed above in Gate 1b
        signal = self._build_signal(
            direction=direction,
            confidence=normalized_confidence,
            candle=candle,
            atr=atr,
            timestamp=now,
        )

        await self._persist(signal)
        self._last_signal_time = now
        self._signals_generated += 1

        logger.info(
            "signal_generated",
            direction=signal.direction,
            spread=signal.spread_type,
            confluence_score=round(score, 3),
            confidence=signal.confidence,
            ml_boost=ml_boost,
            entry=signal.entry_price,
            sl=signal.stop_loss,
            target=signal.target,
            rr=signal.risk_reward,
        )

        return signal

    @property
    def signals_generated(self) -> int:
        return self._signals_generated

    # ── Confluence Scoring ─────────────────────────────────────────

    def _compute_confluence_score(
        self,
        features: dict,
        prediction: Optional[PredictionResult],
    ) -> tuple:
        """
        Compute multi-factor confluence score from feature values.

        Returns: (score: float, direction: str, ml_boost: float)
            score > 0 → BULLISH, score < 0 → BEARISH, score == 0 → NEUTRAL
        """
        score = 0.0

        # ── 1. Supertrend direction (weight 1.5) ───────────────────
        # +1.0 = price above Supertrend (bullish), -1.0 = below (bearish)
        st_dir = features.get("supertrend_direction") or 0.0
        score += 1.5 * st_dir

        # ── 2. RSI + MACD Momentum (weight 1.0) ───────────────────
        # Both RSI and MACD histogram must agree on direction
        rsi       = features.get("rsi_14") or 50.0
        macd_hist = features.get("macd_histogram") or 0.0
        if rsi > 55 and macd_hist > 0:
            momentum = 1.0     # Bullish momentum
        elif rsi < 45 and macd_hist < 0:
            momentum = -1.0    # Bearish momentum
        else:
            momentum = 0.0     # Mixed/neutral momentum
        score += 1.0 * momentum

        # ── 3. Bollinger Band %B (weight 1.0) ─────────────────────
        # > 0.8 → price near upper band (breakout), < 0.2 → near lower band
        bb_pct_b = features.get("bb_pct_b")
        if bb_pct_b is None:
            bb_signal = 0.0
        elif bb_pct_b > 0.8:
            bb_signal = 1.0    # Near upper band — bullish breakout
        elif bb_pct_b < 0.2:
            bb_signal = -1.0   # Near lower band — bearish breakdown
        else:
            bb_signal = 0.0
        score += 1.0 * bb_signal

        # ── 4. Volume spike confirmation (weight 0.5) ─────────────
        # Volume spike in the same direction as Supertrend = confirmation
        vol_ratio = features.get("volume_sma_ratio") or 1.0
        if vol_ratio and vol_ratio > 1.5 and st_dir != 0.0:
            score += 0.5 * st_dir   # Confirms trend direction

        # ── ML Soft Bonus ──────────────────────────────────────────
        # If ML ensemble agrees with the quantitative direction,
        # lower the effective threshold by 0.5 (from 2.5 to 2.0)
        ml_boost = 0.0
        if prediction and prediction.ensemble_direction != "NEUTRAL":
            if score > 0 and prediction.ensemble_direction == "BULLISH":
                ml_boost = _ML_BOOST
            elif score < 0 and prediction.ensemble_direction == "BEARISH":
                ml_boost = _ML_BOOST

        direction = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
        return score, direction, ml_boost

    # ── Signal Construction ────────────────────────────────────────

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

        Uses ATR for SL/target sizing. Default = 0.5% of spot if ATR unavailable.
        """
        spot = candle.get("close", 0)
        step = self.settings.NIFTY_STRIKE_STEP
        atm  = round(spot / step) * step

        if atr is None or atr <= 0:
            atr = spot * 0.005

        if direction == "BULLISH":
            spread_type = "BULL_CALL_SPREAD"
            buy_strike  = atm
            sell_strike = atm + step
            stop_loss   = round(spot - (_ATR_STOP   * atr), 2)
            target      = round(spot + (_ATR_TARGET  * atr), 2)
        else:  # BEARISH
            spread_type = "BEAR_PUT_SPREAD"
            buy_strike  = atm
            sell_strike = atm - step
            stop_loss   = round(spot + (_ATR_STOP   * atr), 2)
            target      = round(spot - (_ATR_TARGET  * atr), 2)

        risk      = abs(spot - stop_loss)
        reward    = abs(target - spot)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

        return TradeSignal(
            timestamp=timestamp,
            direction=direction,
            spread_type=spread_type,
            confidence=confidence,
            entry_price=round(spot, 2),
            buy_strike=buy_strike,
            sell_strike=sell_strike,
            stop_loss=stop_loss,
            target=target,
            risk_reward=risk_reward,
        )

    # ── Persistence ────────────────────────────────────────────────

    async def _persist(self, signal: TradeSignal) -> None:
        """Save signal to database."""
        session_factory = _get_session_factory()
        async with session_factory() as session:
            await SignalCRUD.insert_signal(
                session,
                {
                    "timestamp":   signal.timestamp,
                    "direction":   signal.direction,
                    "spread_type": signal.spread_type,
                    "confidence":  signal.confidence,
                    "entry_price": signal.entry_price,
                    "buy_strike":  signal.buy_strike,
                    "sell_strike": signal.sell_strike,
                    "stop_loss":   signal.stop_loss,
                    "target":      signal.target,
                    "risk_reward": signal.risk_reward,
                    "status":      "ACTIVE",
                },
            )
