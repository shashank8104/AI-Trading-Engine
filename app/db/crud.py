"""
Async CRUD operations for the trading database.

All methods are static — no instance state needed.
Uses PostgreSQL upsert (ON CONFLICT) for idempotent writes.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, desc, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Candle, Feature, Instrument, Prediction, Signal


# ─── Candle CRUD ──────────────────────────────────────────────────────────────


class CandleCRUD:
    """CRUD operations for OHLCV candle data."""

    @staticmethod
    async def insert_candle(session: AsyncSession, candle_data: dict) -> Optional[int]:
        """
        Insert a single candle. Skips on duplicate (token + interval + timestamp).
        Returns the candle ID or None if skipped.
        """
        stmt = (
            pg_insert(Candle)
            .values(**candle_data)
            .on_conflict_do_nothing(constraint="uq_candle")
            .returning(Candle.id)
        )
        result = await session.execute(stmt)
        await session.commit()
        row = result.fetchone()
        return row[0] if row else None

    @staticmethod
    async def insert_candles_batch(session: AsyncSession, candles: List[dict]) -> int:
        """
        Batch-insert candles. Skips duplicates. Returns count attempted.
        """
        if not candles:
            return 0
        stmt = pg_insert(Candle).values(candles).on_conflict_do_nothing(
            constraint="uq_candle"
        )
        await session.execute(stmt)
        await session.commit()
        return len(candles)

    @staticmethod
    async def get_recent_candles(
        session: AsyncSession,
        instrument_token: int,
        interval: str,
        limit: int = 100,
    ) -> List[Candle]:
        """Get most recent candles, returned in chronological order."""
        stmt = (
            select(Candle)
            .where(
                and_(
                    Candle.instrument_token == instrument_token,
                    Candle.interval == interval,
                )
            )
            .order_by(desc(Candle.timestamp))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(reversed(result.scalars().all()))

    @staticmethod
    async def get_latest_candle(
        session: AsyncSession,
        instrument_token: int,
        interval: str,
    ) -> Optional[Candle]:
        """Get the single most recent candle."""
        stmt = (
            select(Candle)
            .where(
                and_(
                    Candle.instrument_token == instrument_token,
                    Candle.interval == interval,
                )
            )
            .order_by(desc(Candle.timestamp))
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalars().first()


# ─── Feature CRUD ────────────────────────────────────────────────────────────


class FeatureCRUD:
    """CRUD operations for computed feature vectors."""

    @staticmethod
    async def insert_feature(session: AsyncSession, feature_data: dict) -> int:
        """Insert a feature row. Returns the new feature ID."""
        feature = Feature(**feature_data)
        session.add(feature)
        await session.commit()
        await session.refresh(feature)
        return feature.id

    @staticmethod
    async def get_latest_features(
        session: AsyncSession,
        instrument_token: int,
        interval: str,
        limit: int = 20,
    ) -> List[Feature]:
        """Get recent features joined with their candles, chronological order."""
        stmt = (
            select(Feature)
            .join(Candle, Feature.candle_id == Candle.id)
            .where(
                and_(
                    Candle.instrument_token == instrument_token,
                    Candle.interval == interval,
                )
            )
            .order_by(desc(Candle.timestamp))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(reversed(result.scalars().all()))


# ─── Signal CRUD ─────────────────────────────────────────────────────────────


class SignalCRUD:
    """CRUD operations for trade signals."""

    @staticmethod
    async def insert_signal(session: AsyncSession, signal_data: dict) -> int:
        """Insert a new trade signal. Returns the signal ID."""
        signal = Signal(**signal_data)
        session.add(signal)
        await session.commit()
        await session.refresh(signal)
        return signal.id

    @staticmethod
    async def get_active_signals(session: AsyncSession) -> List[Signal]:
        """Get all currently active signals, newest first."""
        stmt = (
            select(Signal)
            .where(Signal.status == "ACTIVE")
            .order_by(desc(Signal.timestamp))
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_signal_history(
        session: AsyncSession,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[Signal]:
        """Get signal history, optionally filtered by start time."""
        stmt = select(Signal).order_by(desc(Signal.timestamp)).limit(limit)
        if since:
            stmt = stmt.where(Signal.timestamp >= since)
        result = await session.execute(stmt)
        return list(result.scalars().all())


# ─── Prediction CRUD ────────────────────────────────────────────────────────


class PredictionCRUD:
    """CRUD operations for ML model predictions."""

    @staticmethod
    async def insert_prediction(session: AsyncSession, pred_data: dict) -> int:
        """Insert a prediction row. Returns the prediction ID."""
        pred = Prediction(**pred_data)
        session.add(pred)
        await session.commit()
        await session.refresh(pred)
        return pred.id

    @staticmethod
    async def get_latest_prediction(session: AsyncSession) -> Optional[Prediction]:
        """Get the single most recent prediction."""
        stmt = select(Prediction).order_by(desc(Prediction.timestamp)).limit(1)
        result = await session.execute(stmt)
        return result.scalars().first()


# ─── Instrument CRUD ─────────────────────────────────────────────────────────


class InstrumentCRUD:
    """CRUD operations for cached NFO instruments."""

    @staticmethod
    async def upsert_instruments(session: AsyncSession, instruments: List[dict]) -> int:
        """
        Upsert instruments by instrument_token.
        Returns count processed.
        """
        if not instruments:
            return 0
        for inst in instruments:
            stmt = (
                pg_insert(Instrument)
                .values(**inst)
                .on_conflict_do_update(
                    index_elements=["instrument_token"],
                    set_={k: v for k, v in inst.items() if k != "instrument_token"},
                )
            )
            await session.execute(stmt)
        await session.commit()
        return len(instruments)

    @staticmethod
    async def get_nfo_options(
        session: AsyncSession,
        strike_min: float,
        strike_max: float,
        expiry: Optional[datetime] = None,
    ) -> List[Instrument]:
        """Get NFO option instruments within a strike range."""
        filters = [
            Instrument.exchange == "NFO",
            Instrument.instrument_type.in_(["CE", "PE"]),
            Instrument.strike >= strike_min,
            Instrument.strike <= strike_max,
        ]
        if expiry:
            filters.append(Instrument.expiry == expiry)

        stmt = (
            select(Instrument).where(and_(*filters)).order_by(Instrument.strike)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_nearest_expiry(session: AsyncSession) -> Optional[datetime]:
        """Get the nearest future expiry date for NIFTY options."""
        stmt = (
            select(Instrument.expiry)
            .where(
                and_(
                    Instrument.exchange == "NFO",
                    Instrument.name == "NIFTY",
                    Instrument.instrument_type.in_(["CE", "PE"]),
                    Instrument.expiry >= datetime.now(),  # IST — matches naive expiry dates in DB
                )
            )
            .order_by(Instrument.expiry)
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalars().first()
