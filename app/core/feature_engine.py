"""
Feature engineering: computes technical indicators from OHLCV candles.

Three feature groups:
    1. Market features  — RSI, MACD, Bollinger, ATR, VWAP, EMA, ADX
    2. Options features — PCR, Max Pain, ATM OI change
    3. Derived features — body/wick ratio, gap detection

Computed per closed candle for NIFTY 50 index only.
Requires ≥20 historical candles for meaningful indicator values.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from app.config import get_settings
from app.db.crud import CandleCRUD, FeatureCRUD
from app.db.database import _get_session_factory
from app.db.models import Candle
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum candles needed for indicator computation (RSI=14, BB=20, MACD=26)
FEATURE_WINDOW = 50


class FeatureEngine:
    """Computes feature vectors from completed candles."""

    def __init__(self):
        self.settings = get_settings()
        # Track OI for options features: token → {oi, prev_oi, type, strike}
        self._options_oi: Dict[int, dict] = {}

    # ── Public API ────────────────────────────────────────────────

    def update_options_oi(
        self, token: int, oi: int, instrument_type: str, strike: float
    ) -> None:
        """
        Update OI tracking for an options instrument.
        Called from the tick processor for each NFO tick.
        """
        prev = self._options_oi.get(token, {}).get("oi", oi)
        self._options_oi[token] = {
            "oi": oi,
            "prev_oi": prev,
            "type": instrument_type,
            "strike": strike,
        }

    async def compute_features(self, candle: dict) -> Optional[dict]:
        """
        Compute feature vector for a newly closed candle.

        Only processes NIFTY 50 index candles. Fetches recent history
        from DB, computes all indicator groups, and persists the result.

        Returns:
            Feature dict if computed, None if skipped.
        """
        token = candle["instrument_token"]
        interval = candle["interval"]

        # Only compute features for NIFTY 50 index candles
        if token != self.settings.NIFTY_INSTRUMENT_TOKEN:
            return None

        session_factory = _get_session_factory()
        async with session_factory() as session:
            # Fetch recent candles for indicator lookback
            recent = await CandleCRUD.get_recent_candles(
                session, token, interval, limit=FEATURE_WINDOW
            )

            if len(recent) < 20:
                logger.warning(
                    "insufficient_candles_for_features",
                    count=len(recent),
                    needed=20,
                )
                return None

            # Convert to DataFrame for indicator computation
            df = self._candles_to_df(recent)

            # Compute all feature groups
            features = self._compute_market_features(df)
            features.update(self._compute_options_features(candle))
            features.update(self._compute_derived_features(df))

            # Link to candle and persist
            latest = await CandleCRUD.get_latest_candle(session, token, interval)
            if latest:
                features["candle_id"] = latest.id
                feature_id = await FeatureCRUD.insert_feature(session, features)
                logger.info(
                    "features_computed",
                    candle_ts=candle["timestamp"].isoformat(),
                    feature_id=feature_id,
                    interval=interval,
                )

        return features

    # ── DataFrame Conversion ──────────────────────────────────────

    @staticmethod
    def _candles_to_df(candles: List[Candle]) -> pd.DataFrame:
        """Convert list of Candle ORM objects to a pandas DataFrame."""
        records = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "oi": c.oi,
            }
            for c in candles
        ]
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    # ── Market Features ───────────────────────────────────────────

    @staticmethod
    def _compute_market_features(df: pd.DataFrame) -> dict:
        """Compute market-based technical indicators from OHLCV data."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI (14-period)
        rsi = RSIIndicator(close=close, window=14)

        # MACD (12/26/9)
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)

        # Bollinger Bands (20-period, 2σ)
        bb = BollingerBands(close=close, window=20, window_dev=2)

        # ATR (14-period)
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)

        # EMA crossovers
        ema9 = EMAIndicator(close=close, window=9)
        ema21 = EMAIndicator(close=close, window=21)

        # ADX (14-period)
        adx = ADXIndicator(high=high, low=low, close=close, window=14)

        # VWAP (intraday approximation)
        typical_price = (high + low + close) / 3
        cum_vol = volume.cumsum()
        vwap = (typical_price * volume).cumsum() / cum_vol.replace(0, np.nan)

        # Volume ratio vs 20-period SMA
        vol_sma = volume.rolling(window=20).mean()
        vol_ratio = volume / vol_sma.replace(0, np.nan)

        return {
            "rsi_14": _safe_last(rsi.rsi()),
            "macd": _safe_last(macd.macd()),
            "macd_signal": _safe_last(macd.macd_signal()),
            "macd_histogram": _safe_last(macd.macd_diff()),
            "bb_upper": _safe_last(bb.bollinger_hband()),
            "bb_middle": _safe_last(bb.bollinger_mavg()),
            "bb_lower": _safe_last(bb.bollinger_lband()),
            "atr_14": _safe_last(atr.average_true_range()),
            "vwap": _safe_last(vwap),
            "ema_9": _safe_last(ema9.ema_indicator()),
            "ema_21": _safe_last(ema21.ema_indicator()),
            "volume_sma_ratio": _safe_last(vol_ratio),
            "adx": _safe_last(adx.adx()),
        }

    # ── Options Features ──────────────────────────────────────────

    def _compute_options_features(self, candle: dict) -> dict:
        """Compute options-derived features from tracked OI data."""
        ce_oi_total = 0
        pe_oi_total = 0
        atm_oi_change_ce = 0.0
        atm_oi_change_pe = 0.0

        spot = candle.get("close", 0)
        step = self.settings.NIFTY_STRIKE_STEP
        atm = round(spot / step) * step if spot > 0 else 0

        for info in self._options_oi.values():
            if info["type"] == "CE":
                ce_oi_total += info["oi"]
                if info["strike"] == atm:
                    atm_oi_change_ce = info["oi"] - info["prev_oi"]
            elif info["type"] == "PE":
                pe_oi_total += info["oi"]
                if info["strike"] == atm:
                    atm_oi_change_pe = info["oi"] - info["prev_oi"]

        pcr = pe_oi_total / ce_oi_total if ce_oi_total > 0 else 0.0
        max_pain = self._calculate_max_pain()

        return {
            "pcr": round(pcr, 4),
            "max_pain": max_pain,
            "iv_percentile": None,  # Phase 2: requires IV surface computation
            "atm_oi_change_ce": atm_oi_change_ce,
            "atm_oi_change_pe": atm_oi_change_pe,
        }

    def _calculate_max_pain(self) -> Optional[float]:
        """
        Calculate Max Pain strike from current OI data.

        Max Pain = strike where total intrinsic value loss to option
        buyers is minimized (i.e., maximized for option writers).
        """
        strikes: Dict[float, dict] = {}
        for info in self._options_oi.values():
            s = info["strike"]
            if s not in strikes:
                strikes[s] = {"ce_oi": 0, "pe_oi": 0}
            if info["type"] == "CE":
                strikes[s]["ce_oi"] = info["oi"]
            else:
                strikes[s]["pe_oi"] = info["oi"]

        if not strikes:
            return None

        strike_list = sorted(strikes.keys())
        min_pain = float("inf")
        max_pain_strike = None

        for target in strike_list:
            total_pain = 0
            for strike, oi_data in strikes.items():
                # CE is ITM when target > strike
                if target > strike:
                    total_pain += (target - strike) * oi_data["ce_oi"]
                # PE is ITM when target < strike
                if target < strike:
                    total_pain += (strike - target) * oi_data["pe_oi"]

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = target

        return max_pain_strike

    # ── Derived Features ──────────────────────────────────────────

    @staticmethod
    def _compute_derived_features(df: pd.DataFrame) -> dict:
        """Compute price-action derived features from the latest candle."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        # Body-to-wick ratio (measures candle conviction)
        body = abs(latest["close"] - latest["open"])
        upper_wick = latest["high"] - max(latest["close"], latest["open"])
        lower_wick = min(latest["close"], latest["open"]) - latest["low"]
        total_wick = upper_wick + lower_wick
        body_wick_ratio = body / total_wick if total_wick > 0 else 10.0

        # Gap percentage (vs previous close)
        gap_pct = (
            (latest["open"] - prev["close"]) / prev["close"] * 100
            if prev["close"] > 0
            else 0.0
        )

        return {
            "body_wick_ratio": round(body_wick_ratio, 4),
            "gap_pct": round(gap_pct, 4),
        }


def _safe_last(series: pd.Series) -> Optional[float]:
    """Safely extract the last value from a pandas Series."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return round(float(val), 6)
