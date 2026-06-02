"""
SQLAlchemy ORM models for the trading system.

Tables:
    candles      — 5m/15m OHLCV data
    features     — computed technical indicators per candle
    predictions  — ML model outputs (XGBoost + LSTM + ensemble)
    signals      — trade signal recommendations
    instruments  — cached NFO instrument metadata
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from app.db.database import Base


# ─── Candle ───────────────────────────────────────────────────────────────────


class Candle(Base):
    __tablename__ = "candles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_token = Column(Integer, nullable=False, index=True)
    interval = Column(String(4), nullable=False)        # "5m" | "15m"
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, default=0)
    oi = Column(BigInteger, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    features = relationship(
        "Feature", back_populates="candle", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_token", "interval", "timestamp", name="uq_candle"
        ),
        Index("ix_candle_lookup", "instrument_token", "interval", "timestamp"),
    )


# ─── Feature ─────────────────────────────────────────────────────────────────


class Feature(Base):
    __tablename__ = "features"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    candle_id = Column(BigInteger, ForeignKey("candles.id"), nullable=False)

    # Market features (from OHLCV + indicators)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    atr_14 = Column(Float)
    vwap = Column(Float)
    ema_9 = Column(Float)
    ema_21 = Column(Float)
    volume_sma_ratio = Column(Float)
    adx = Column(Float)

    # Options features (from OI data)
    pcr = Column(Float)
    max_pain = Column(Float)
    iv_percentile = Column(Float)
    atm_oi_change_ce = Column(Float)
    atm_oi_change_pe = Column(Float)

    # Derived price-action features
    body_wick_ratio = Column(Float)
    gap_pct = Column(Float)

    # Confluence scoring features (new strategy)
    supertrend_direction = Column(Float)   # +1.0 = bullish, -1.0 = bearish
    bb_pct_b = Column(Float)               # (close - lower) / (upper - lower), 0-1

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    candle = relationship("Candle", back_populates="features")

    __table_args__ = (Index("ix_feature_candle", "candle_id"),)


# ─── Prediction ──────────────────────────────────────────────────────────────


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    instrument_token = Column(Integer, nullable=False)
    interval = Column(String(4), nullable=False)

    # Individual model outputs
    xgboost_bullish = Column(Float)
    xgboost_bearish = Column(Float)
    xgboost_neutral = Column(Float)
    lstm_bullish = Column(Float)
    lstm_bearish = Column(Float)
    lstm_neutral = Column(Float)

    # Ensemble result
    ensemble_bullish = Column(Float)
    ensemble_bearish = Column(Float)
    ensemble_neutral = Column(Float)
    ensemble_confidence = Column(Float)
    ensemble_direction = Column(String(10))    # BULLISH | BEARISH | NEUTRAL

    created_at = Column(DateTime, default=datetime.utcnow)


# ─── Signal ───────────────────────────────────────────────────────────────────


class Signal(Base):
    __tablename__ = "signals"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    direction = Column(String(10), nullable=False)      # BULLISH | BEARISH
    spread_type = Column(String(30))                      # BULL_CALL_SPREAD | BEAR_PUT_SPREAD
    confidence = Column(Float, nullable=False)

    entry_price = Column(Float)
    buy_strike = Column(Float)
    sell_strike = Column(Float)
    stop_loss = Column(Float)
    target = Column(Float)
    risk_reward = Column(Float)

    status = Column(String(15), default="ACTIVE")         # ACTIVE | EXPIRED | EXECUTED | CANCELLED
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_signal_status", "status", "timestamp"),)


# ─── Instrument ──────────────────────────────────────────────────────────────


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_token = Column(Integer, nullable=False, unique=True, index=True)
    exchange_token = Column(Integer)
    tradingsymbol = Column(String(50), nullable=False)
    name = Column(String(50))
    exchange = Column(String(10), nullable=False)
    segment = Column(String(20))
    instrument_type = Column(String(10))                  # CE | PE | FUT | EQ
    strike = Column(Float, default=0)
    lot_size = Column(Integer, default=1)
    expiry = Column(DateTime)
    tick_size = Column(Float, default=0.05)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_instrument_nfo", "exchange", "instrument_type", "strike", "expiry"),
    )
