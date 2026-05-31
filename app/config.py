"""
Application configuration — single source of truth.

Loads from .env file. Fails fast if required keys are missing.
"""

from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """All configuration for the trading system."""

    # ── Angel One SmartAPI ────────────────────────────────────
    ANGEL_API_KEY: str
    ANGEL_CLIENT_ID: str
    ANGEL_MPIN: str          # 4-digit MPIN used in the Angel One app
    ANGEL_TOTP_SECRET: str   # Base32 secret from enable-totp page, NOT the 6-digit code

    # ── Database ──────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/trading_db"

    # ── NIFTY 50 ──────────────────────────────────────────────────
    NIFTY_INSTRUMENT_TOKEN: int = 26000  # Angel One NSE token for NIFTY 50
    NIFTY_STRIKE_STEP: int = 50

    # ── Signal Engine ─────────────────────────────────────────────
    SIGNAL_CONFIDENCE_THRESHOLD: float = 0.7
    CANDLE_INTERVALS: str = "5,15"

    # ── ML Models ─────────────────────────────────────────────────
    XGBOOST_MODEL_PATH: str = "models/xgboost_model.pkl"
    LSTM_MODEL_PATH: str = "models/lstm_model.pt"
    XGBOOST_WEIGHT: float = 0.6
    LSTM_WEIGHT: float = 0.4

    # ── Market Hours (IST, 24h) ───────────────────────────────────
    MARKET_OPEN: str = "09:15"
    MARKET_CLOSE: str = "15:30"

    # ── NFO Options ───────────────────────────────────────────────
    NFO_STRIKE_RANGE: int = 10
    NFO_ATM_UPDATE_MINUTES: int = 5

    # ── Logging ───────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    @property
    def candle_intervals_list(self) -> List[int]:
        """Parse comma-separated intervals into list of ints."""
        return [int(x.strip()) for x in self.CANDLE_INTERVALS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance. Call once at startup."""
    return Settings()
