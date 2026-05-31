"""
Fetch historical NIFTY candle data from Angel One SmartAPI or CSV into PostgreSQL.

Supports two modes:
    1. Angel One API: python scripts/fetch_historical.py --days 60
    2. CSV import:    python scripts/fetch_historical.py --csv data/nifty_5m.csv

CSV format: timestamp,open,high,low,close,volume[,oi]
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import pyotp
import pandas as pd
from SmartApi import SmartConnect

from app.config import get_settings
from app.db.crud import CandleCRUD
from app.db.database import _get_session_factory, init_db


def _get_smart_client(settings) -> SmartConnect:
    """Authenticate with Angel One via TOTP and return a ready SmartConnect client."""
    totp_code = pyotp.TOTP(settings.ANGEL_TOTP_SECRET).now()
    smart = SmartConnect(api_key=settings.ANGEL_API_KEY)
    data = smart.generateSession(
        settings.ANGEL_CLIENT_ID, settings.ANGEL_MPIN, totp_code
    )
    if not data or not data.get("status"):
        msg = data.get("message", "Unknown") if data else "No response"
        raise RuntimeError(f"Angel One login failed: {msg}")
    print(f"  Authenticated as {settings.ANGEL_CLIENT_ID}")
    return smart


async def fetch_from_api(days: int) -> None:
    """
    Fetch historical NIFTY candle data from Angel One SmartAPI.

    Angel One interval strings: FIVE_MINUTE, FIFTEEN_MINUTE
    Date format: YYYY-MM-DD HH:MM
    Max range per request: 30 days for minute-level data.
    """
    settings = get_settings()
    smart = _get_smart_client(settings)

    # Angel One uses a specific token for historical index data
    token_str = "99926000"  # Historical token for NIFTY 50
    token_int = settings.NIFTY_INSTRUMENT_TOKEN  # Keep 26000 for DB storage
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    await init_db()
    session_factory = _get_session_factory()

    # Angel One allows max ~30 days per minute-level request — chunk at 28 days
    chunk_days = 28

    for interval_label, interval_api in [("5m", "FIVE_MINUTE"), ("15m", "FIFTEEN_MINUTE")]:
        print(f"\n{'-' * 50}")
        print(f"Fetching {interval_label} candles for last {days} days...")

        current_from = from_date
        total_inserted = 0

        while current_from < to_date:
            current_to = min(current_from + timedelta(days=chunk_days), to_date)

            params = {
                "exchange": "NSE",
                "symboltoken": token_str,
                "interval": interval_api,
                "fromdate": current_from.strftime("%Y-%m-%d %H:%M"),
                "todate": current_to.strftime("%Y-%m-%d %H:%M"),
            }

            try:
                resp = smart.getCandleData(params)
            except Exception as e:
                print(f"  WARNING: API error ({current_from.date()} -> {current_to.date()}): {e}")
                current_from = current_to
                continue

            if not resp or not resp.get("data"):
                current_from = current_to
                continue

            candles = []
            for r in resp["data"]:
                # Format: ["2024-05-01T09:15:00+05:30", open, high, low, close, volume]
                try:
                    ts_str = r[0][:19]  # strip timezone
                    ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                    candles.append(
                        {
                            "instrument_token": token_int,
                            "interval": interval_label,
                            "timestamp": ts,
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                            "close": float(r[4]),
                            "volume": int(r[5]),
                            "oi": 0,
                        }
                    )
                except Exception:
                    continue

            if candles:
                async with session_factory() as session:
                    await CandleCRUD.insert_candles_batch(session, candles)
                total_inserted += len(candles)

            print(
                f"  {current_from.date()} -> {current_to.date()}: {len(candles)} candles"
            )
            current_from = current_to

        print(f"Done: Total {interval_label}: {total_inserted} candles stored")


async def import_from_csv(csv_path: str) -> None:
    """
    Import candle data from a CSV file.

    Expected columns: timestamp, open, high, low, close, volume [, oi]
    Interval is auto-detected from the first two timestamps.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found — {csv_path}")
        sys.exit(1)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    settings = get_settings()
    token = settings.NIFTY_INSTRUMENT_TOKEN

    # Auto-detect interval
    if len(df) >= 2:
        delta_minutes = (
            df["timestamp"].iloc[1] - df["timestamp"].iloc[0]
        ).total_seconds() / 60
        interval = f"{int(delta_minutes)}m"
    else:
        interval = "5m"

    print(f"  Detected interval: {interval}")
    print(f"  Rows: {len(df)}")
    print(
        f"  Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
    )

    await init_db()
    session_factory = _get_session_factory()

    candles = []
    for _, row in df.iterrows():
        candles.append(
            {
                "instrument_token": token,
                "interval": interval,
                "timestamp": row["timestamp"].to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0)),
                "oi": int(row.get("oi", 0)),
            }
        )

    BATCH_SIZE = 1000
    async with session_factory() as session:
        for i in range(0, len(candles), BATCH_SIZE):
            batch = candles[i : i + BATCH_SIZE]
            await CandleCRUD.insert_candles_batch(session, batch)
            print(f"  Inserted {min(i + BATCH_SIZE, len(candles))}/{len(candles)}")

    print(f"Done: {len(candles)} candles imported")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical NIFTY data for training"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--days", type=int, help="Fetch last N days from Angel One API"
    )
    group.add_argument(
        "--csv", type=str, help="Import from CSV file"
    )

    args = parser.parse_args()

    if args.days:
        asyncio.run(fetch_from_api(args.days))
    elif args.csv:
        asyncio.run(import_from_csv(args.csv))


if __name__ == "__main__":
    main()
