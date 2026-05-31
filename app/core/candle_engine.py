"""
Real-time candle aggregation engine.

Consumes raw ticks from an asyncio.Queue and aggregates them into
5m and 15m OHLCV candles aligned to market open (9:15 IST).

On candle close:
    1. Emits the completed candle dict via callback
    2. Caller persists to DB and triggers feature computation

Latency-critical path — minimal allocations, no DataFrame overhead.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Callable, Dict, Optional, Tuple

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CandleBuffer:
    """
    In-memory buffer for a single candle being formed.

    Tracks OHLCV from individual ticks. Volume is taken as the
    exchange-reported cumulative volume at each tick.
    """

    instrument_token: int
    interval: int  # minutes
    timestamp: datetime  # candle open time (floored to interval)
    open: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    close: float = 0.0
    volume: int = 0
    oi: int = 0
    tick_count: int = 0

    def update(self, price: float, volume: int, oi: int) -> None:
        """Update buffer with a new tick."""
        if self.tick_count == 0:
            self.open = price
            self.high = price
            self.low = price
        else:
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price
        self.close = price
        self.volume = volume  # cumulative from exchange
        self.oi = oi
        self.tick_count += 1

    def to_dict(self) -> dict:
        """Serialize to dict for DB insertion."""
        return {
            "instrument_token": self.instrument_token,
            "interval": f"{self.interval}m",
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "oi": self.oi,
        }


class CandleEngine:
    """
    Aggregates ticks into OHLCV candles for configured intervals.

    Args:
        tick_queue: asyncio.Queue receiving raw tick dicts from DataIngestor.
        on_candle_close: async callback invoked with completed candle dict.
    """

    def __init__(
        self,
        tick_queue: asyncio.Queue,
        on_candle_close: Optional[Callable] = None,
        on_candle_update: Optional[Callable] = None,
    ):
        self.settings = get_settings()
        self.tick_queue = tick_queue
        self.on_candle_close = on_candle_close
        self.on_candle_update = on_candle_update

        self.intervals = self.settings.candle_intervals_list  # e.g. [5, 15]

        # Buffers keyed by (instrument_token, interval_minutes)
        self._buffers: Dict[Tuple[int, int], CandleBuffer] = {}

        # Market hours (IST)
        h, m = self.settings.MARKET_OPEN.split(":")
        self._market_open = time(int(h), int(m))
        h, m = self.settings.MARKET_CLOSE.split(":")
        self._market_close = time(int(h), int(m))

        self._running = False
        self._candles_emitted = 0

    # ── Lifecycle ─────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop: consume ticks and aggregate into candles."""
        self._running = True
        logger.info("candle_engine_started", intervals=self.intervals)

        while self._running:
            try:
                tick = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)
                await self._process_tick(tick)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("candle_engine_error", error=str(e), exc_info=True)

    def stop(self) -> None:
        """Signal the engine to stop."""
        self._running = False
        logger.info("candle_engine_stopped", total_candles=self._candles_emitted)

    async def flush_all(self) -> None:
        """Flush all open buffers (call at market close)."""
        for buffer in self._buffers.values():
            if buffer.tick_count > 0:
                await self._emit_candle(buffer)
        self._buffers.clear()
        logger.info("candle_buffers_flushed")

    # ── Properties ────────────────────────────────────────────────

    @property
    def active_buffers(self) -> int:
        return len(self._buffers)

    @property
    def candles_emitted(self) -> int:
        return self._candles_emitted

    # ── Tick Processing ───────────────────────────────────────────

    async def _process_tick(self, tick: dict) -> None:
        """Process a single tick: validate, filter, update all interval buffers."""
        token = tick.get("instrument_token")
        price = tick.get("last_price")
        timestamp = tick.get("timestamp")
        volume = tick.get("volume", 0)
        oi = tick.get("oi", 0)

        if not all((token, price, timestamp)):
            return

        # Parse timestamp if needed
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Filter ticks outside market hours
        tick_time = timestamp.time()
        if tick_time < self._market_open or tick_time > self._market_close:
            return

        # Update buffers for each interval
        for interval in self.intervals:
            await self._update_buffer(token, interval, timestamp, price, volume, oi)

    async def _update_buffer(
        self,
        token: int,
        interval: int,
        tick_time: datetime,
        price: float,
        volume: int,
        oi: int,
    ) -> None:
        """Update or create a candle buffer. Emit on interval boundary crossing."""
        key = (token, interval)
        candle_start = self._floor_to_interval(tick_time, interval)

        buffer = self._buffers.get(key)

        if buffer is None:
            # First tick for this token/interval pair
            self._buffers[key] = CandleBuffer(
                instrument_token=token,
                interval=interval,
                timestamp=candle_start,
            )
            self._buffers[key].update(price, volume, oi)
            if self.on_candle_update:
                await self.on_candle_update(self._buffers[key].to_dict())
            return

        if candle_start > buffer.timestamp:
            # Interval boundary crossed — close previous candle
            if buffer.tick_count > 0:
                await self._emit_candle(buffer)

            # Start new buffer
            self._buffers[key] = CandleBuffer(
                instrument_token=token,
                interval=interval,
                timestamp=candle_start,
            )
            self._buffers[key].update(price, volume, oi)
            if self.on_candle_update:
                await self.on_candle_update(self._buffers[key].to_dict())
        else:
            # Same interval — aggregate
            buffer.update(price, volume, oi)
            if self.on_candle_update:
                await self.on_candle_update(buffer.to_dict())

    async def _emit_candle(self, buffer: CandleBuffer) -> None:
        """Emit a completed candle via callback."""
        candle = buffer.to_dict()
        self._candles_emitted += 1

        logger.info(
            "candle_closed",
            token=candle["instrument_token"],
            interval=candle["interval"],
            ts=candle["timestamp"].isoformat(),
            o=candle["open"],
            h=candle["high"],
            l=candle["low"],
            c=candle["close"],
            v=candle["volume"],
        )

        if self.on_candle_close:
            await self.on_candle_close(candle)

    # ── Interval Alignment ────────────────────────────────────────

    @staticmethod
    def _floor_to_interval(dt: datetime, interval_minutes: int) -> datetime:
        """
        Floor a datetime to the nearest interval boundary,
        aligned to market open at 9:15 IST.

        Examples (5m interval):
            09:17 → 09:15
            09:20 → 09:20
            09:23 → 09:20

        Examples (15m interval):
            09:17 → 09:15
            09:30 → 09:30
            09:42 → 09:30
        """
        # Minutes since midnight
        total_minutes = dt.hour * 60 + dt.minute
        # Market open = 9:15 = 555 minutes
        market_open_minutes = 9 * 60 + 15
        # Minutes elapsed since market open
        elapsed = total_minutes - market_open_minutes
        if elapsed < 0:
            elapsed = 0
        # Floor to interval
        floored_elapsed = (elapsed // interval_minutes) * interval_minutes
        floored_minutes = market_open_minutes + floored_elapsed

        return dt.replace(
            hour=floored_minutes // 60,
            minute=floored_minutes % 60,
            second=0,
            microsecond=0,
        )
