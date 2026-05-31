"""
Angel One SmartAPI data ingestion — replaces Zerodha KiteTicker.

Design:
    SmartWebSocketV2 runs on its own thread (internally uses websocket-client).
    Ticks are bridged to asyncio via loop.call_soon_threadsafe -> asyncio.Queue.

    On connect, subscribes to NIFTY 50 index in SNAP_QUOTE mode.
    Every NFO_ATM_UPDATE_MINUTES, recalculates ATM strike and
    subscribes/unsubscribes +-NFO_STRIKE_RANGE option instruments.

    The public interface is identical to the previous KiteTicker-based
    DataIngestor so app/main.py requires zero changes.

Angel One exchange types:
    1 = NSE (index)
    2 = NFO (options/futures)

Subscription modes:
    1 = LTP
    2 = QUOTE (OHLC + volume)
    3 = SNAP_QUOTE (full, including OI and market depth)
"""

import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Set

import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Angel One exchange segment constants
EXCHANGE_NSE = 1
EXCHANGE_NFO = 2
MODE_SNAP_QUOTE = 3


class DataIngestor:
    """
    Angel One WebSocket data ingestor.

    Identical public interface to the previous Kite-based ingestor.
    Drop-in replacement — app/main.py is unchanged.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, tick_queue: asyncio.Queue):
        self.settings = get_settings()
        self.loop = loop
        self.tick_queue = tick_queue

        # Angel One session — generated fresh via TOTP on start()
        self._smart: Optional[SmartConnect] = None
        self._auth_token: str = ""
        self._feed_token: str = ""
        self._sws: Optional[SmartWebSocketV2] = None
        self._ws_thread: Optional[threading.Thread] = None

        # State
        self._subscribed_tokens: Set[str] = set()   # Angel One tokens are strings
        self._nifty_spot: float = 0.0
        self._current_atm: float = 0.0
        self._last_atm_update: Optional[datetime] = None
        self._nfo_instruments: Dict[int, dict] = {}  # keyed by int token (DB key)
        self._tick_count: int = 0
        self._connected: bool = False

        # NIFTY token as string for Angel One WebSocket
        self._nifty_token_str: str = str(self.settings.NIFTY_INSTRUMENT_TOKEN)

    # ── Public API ────────────────────────────────────────────────

    def set_nfo_instruments(self, instruments: Dict[int, dict]) -> None:
        """
        Set available NFO instruments for dynamic subscription.

        Args:
            instruments: dict keyed by instrument_token (int), values must have
                         'strike', 'type' (CE/PE), 'symbol' keys.
        """
        self._nfo_instruments = instruments
        logger.info("nfo_instruments_loaded", count=len(instruments))

    async def backfill_today_history(self) -> None:
        """
        Fetch today's 5m history from Angel One API and insert into DB.
        Called once at startup to pre-populate candles since 9:15 AM.
        """
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

        if now < market_open or now.weekday() >= 5:
            return

        if not self._smart:
            logger.warning("backfill_skipped_no_session")
            return

        logger.info("starting_historical_backfill", from_date=market_open.isoformat())

        try:
            import functools

            params = {
                "exchange": "NSE",
                "symboltoken": "99926000",  # Historical token for NIFTY 50
                "interval": "FIVE_MINUTE",
                "fromdate": market_open.strftime("%Y-%m-%d %H:%M"),
                "todate": now.strftime("%Y-%m-%d %H:%M"),
            }

            records = await self.loop.run_in_executor(
                None,
                functools.partial(self._smart.getCandleData, params),
            )

            if not records or not records.get("data"):
                logger.info("no_historical_data_for_today")
                return

            from app.db.crud import CandleCRUD
            from app.db.database import _get_session_factory

            token_int = self.settings.NIFTY_INSTRUMENT_TOKEN
            candles = []
            for r in records["data"]:
                # Angel One candle format: [timestamp, open, high, low, close, volume]
                ts = datetime.strptime(r[0][:19], "%Y-%m-%dT%H:%M:%S")
                candles.append(
                    {
                        "instrument_token": token_int,
                        "interval": "5m",
                        "timestamp": ts,
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": int(r[5]),
                        "oi": 0,
                    }
                )

            session_factory = _get_session_factory()
            async with session_factory() as session:
                count = await CandleCRUD.insert_candles_batch(session, candles)
            logger.info("backfilled_today_history", count=count)

        except Exception as e:
            logger.error("backfill_history_error", error=str(e))

    def start(self) -> None:
        """Authenticate via TOTP and start the WebSocket in a background thread."""
        self._authenticate()
        self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self._ws_thread.start()
        logger.info("websocket_started")

    def stop(self) -> None:
        """Gracefully stop the WebSocket connection."""
        self._connected = False
        if self._sws:
            try:
                self._sws.close_connection()
            except Exception:
                pass
        logger.info("websocket_stopped", total_ticks=self._tick_count)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def nifty_spot(self) -> float:
        return self._nifty_spot

    @property
    def current_atm(self) -> float:
        return self._current_atm

    @property
    def subscribed_count(self) -> int:
        return len(self._subscribed_tokens)

    # ── Authentication ─────────────────────────────────────────────

    def _authenticate(self) -> None:
        """Generate fresh Angel One session using TOTP. No user interaction needed."""
        s = self.settings
        totp_code = pyotp.TOTP(s.ANGEL_TOTP_SECRET).now()

        smart = SmartConnect(api_key=s.ANGEL_API_KEY)

        try:
            data = smart.generateSession(s.ANGEL_CLIENT_ID, s.ANGEL_MPIN, totp_code)
        except Exception as e:
            logger.error("angel_auth_failed", error=str(e))
            raise

        if not data or not data.get("status"):
            msg = data.get("message", "Unknown") if data else "No response"
            raise RuntimeError(f"Angel One login failed: {msg}")

        self._smart = smart
        self._auth_token = data["data"]["jwtToken"]
        self._feed_token = smart.getfeedToken()

        logger.info(
            "angel_authenticated",
            client=s.ANGEL_CLIENT_ID,
            feed_token_prefix=self._feed_token[:8],
        )

    # ── WebSocket ─────────────────────────────────────────────────

    def _run_websocket(self) -> None:
        """Initialize and connect SmartWebSocketV2 (blocking — runs in its own thread)."""
        s = self.settings

        self._sws = SmartWebSocketV2(
            auth_token=self._auth_token,
            api_key=s.ANGEL_API_KEY,
            client_code=s.ANGEL_CLIENT_ID,
            feed_token=self._feed_token,
            max_retry_attempt=5,
        )

        self._sws.on_data = self._on_data
        self._sws.on_open = self._on_open
        self._sws.on_error = self._on_error
        self._sws.on_close = self._on_close

        try:
            self._sws.connect()
        except Exception as e:
            logger.error("websocket_connect_error", error=str(e))

    def _on_open(self, wsapp) -> None:
        """Called when WebSocket connects. Subscribe to NIFTY 50 index."""
        self._connected = True
        logger.info("websocket_connected")

        token_list = [
            {"exchangeType": EXCHANGE_NSE, "tokens": [self._nifty_token_str]}
        ]
        self._sws.subscribe(
            correlation_id="nifty_index",
            mode=MODE_SNAP_QUOTE,
            token_list=token_list,
        )
        self._subscribed_tokens.add(self._nifty_token_str)
        logger.info("subscribed_nifty", token=self._nifty_token_str)

    def _on_data(self, wsapp, message) -> None:
        """
        Called on every tick. Normalize to standard dict and push to asyncio queue.

        Angel One SNAP_QUOTE tick fields (relevant subset):
            token           -> instrument token (string)
            last_traded_price -> last price * 100 (paise) for equity, actual for index
            volume_trade_for_the_day -> cumulative volume
            open_interest   -> OI (for NFO instruments)
            exchange_timestamp -> epoch ms
        """
        self._tick_count += 1

        try:
            token_str = str(message.get("token", ""))
            token_int = int(token_str) if token_str else 0

            # Angel One sends prices in paise (1/100 rupee) for equity;
            # for NSE index (NIFTY), it sends the actual float value.
            # last_traded_price is already a float in the SDK's parsed message.
            last_price = float(message.get("last_traded_price", 0))

            # Sanity: if price looks like paise (e.g. 2450000 for 24500), convert
            if last_price > 100_000 and token_str == self._nifty_token_str:
                last_price = last_price / 100.0

            tick = {
                "instrument_token": token_int,
                "last_price": last_price,
                "volume": int(message.get("volume_trade_for_the_day", 0)),
                "oi": int(message.get("open_interest", 0)),
                "timestamp": datetime.now(),
            }

            # Update NIFTY spot for ATM tracking
            if token_str == self._nifty_token_str:
                self._nifty_spot = last_price
                self._maybe_update_atm()

            # Thread-safe push to asyncio queue
            try:
                self.loop.call_soon_threadsafe(self.tick_queue.put_nowait, tick)
            except asyncio.QueueFull:
                pass  # Drop if overloaded

        except Exception as e:
            logger.error("tick_parse_error", error=str(e))

    def _on_error(self, wsapp, error) -> None:
        logger.error("websocket_error", error=str(error))

    def _on_close(self, wsapp) -> None:
        self._connected = False
        logger.warning("websocket_closed")

    # ── ATM Tracking & Dynamic Subscription ───────────────────────

    def _maybe_update_atm(self) -> None:
        """
        Check if ATM strike has shifted. Update NFO subscriptions if so.
        Debounced to NFO_ATM_UPDATE_MINUTES interval.
        """
        now = datetime.now()
        interval_sec = self.settings.NFO_ATM_UPDATE_MINUTES * 60

        if (
            self._last_atm_update is not None
            and (now - self._last_atm_update).total_seconds() < interval_sec
        ):
            return

        if self._nifty_spot <= 0 or not self._nfo_instruments:
            return

        step = self.settings.NIFTY_STRIKE_STEP
        new_atm = round(self._nifty_spot / step) * step

        if new_atm == self._current_atm and self._last_atm_update is not None:
            self._last_atm_update = now
            return

        old_atm = self._current_atm
        self._current_atm = new_atm
        self._last_atm_update = now

        logger.info("atm_updated", old=old_atm, new=new_atm, spot=self._nifty_spot)

        # Compute desired option tokens: +-strike_range around ATM
        strike_range = self.settings.NFO_STRIKE_RANGE
        desired_strikes = set(
            new_atm + (i * step) for i in range(-strike_range, strike_range + 1)
        )

        desired_token_strs: Set[str] = {self._nifty_token_str}
        desired_token_ints: Set[int] = set()

        for token_int, info in self._nfo_instruments.items():
            if info.get("strike") in desired_strikes:
                desired_token_strs.add(str(token_int))
                desired_token_ints.add(token_int)

        if not self._sws:
            return

        # Unsubscribe tokens no longer needed (NFO only, keep NIFTY)
        nfo_currently_subbed = self._subscribed_tokens - {self._nifty_token_str}
        to_unsub = nfo_currently_subbed - desired_token_strs
        if to_unsub:
            self._sws.unsubscribe(
                correlation_id="nfo_unsub",
                mode=MODE_SNAP_QUOTE,
                token_list=[{"exchangeType": EXCHANGE_NFO, "tokens": list(to_unsub)}],
            )
            logger.info("unsubscribed_nfo", count=len(to_unsub))

        # Subscribe new tokens
        new_nfo_tokens = desired_token_strs - {self._nifty_token_str} - self._subscribed_tokens
        if new_nfo_tokens:
            self._sws.subscribe(
                correlation_id="nfo_sub",
                mode=MODE_SNAP_QUOTE,
                token_list=[
                    {"exchangeType": EXCHANGE_NFO, "tokens": list(new_nfo_tokens)}
                ],
            )
            logger.info(
                "subscribed_nfo",
                count=len(new_nfo_tokens),
                atm=new_atm,
                total=len(desired_token_strs),
            )

        self._subscribed_tokens = desired_token_strs
