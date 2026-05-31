"""
Seed NFO instruments from Angel One OpenAPIScripMaster.json.

Downloads the Angel One instrument master (updated daily) and inserts
all NIFTY option contracts (OPTIDX) into the local instruments table.

Usage:
    python scripts/seed_instruments.py

No credentials required — the ScripMaster file is public.
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import httpx
import pandas as pd

from app.db.crud import InstrumentCRUD
from app.db.database import _get_session_factory, init_db

SCRIP_MASTER_URL = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
)


def download_scrip_master() -> pd.DataFrame:
    """Download and return the full Angel One instrument master as a DataFrame."""
    print("Downloading Angel One ScripMaster...")
    with httpx.Client(timeout=30) as client:
        resp = client.get(SCRIP_MASTER_URL)
        resp.raise_for_status()

    data = resp.json()
    df = pd.DataFrame(data)
    print(f"  Total instruments downloaded: {len(df)}")
    return df


def filter_nifty_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to NIFTY index options only.

    Angel One fields:
        name            -> underlying name (e.g. "NIFTY")
        instrumenttype  -> "OPTIDX" for index options
        symbol          -> trading symbol (e.g. "NIFTY26MAY2522500CE")
        token           -> instrument token (string, convert to int)
        strike          -> strike price (stored as float string, e.g. "22500.000000")
        optiontype      -> "CE" or "PE"
        expiry          -> expiry date string (e.g. "29MAY2025")
        exch_seg        -> exchange segment ("NFO")
        lotsize         -> lot size
    """
    nifty = df[
        (df["name"] == "NIFTY") & (df["instrumenttype"] == "OPTIDX")
    ].copy()

    print(f"  NIFTY option contracts found: {len(nifty)}")
    return nifty


def parse_expiry(expiry_str: str):
    """Parse Angel One expiry format (e.g. '29MAY2025') to date."""
    try:
        return datetime.strptime(expiry_str.strip(), "%d%b%Y").date()
    except Exception:
        return None


async def seed(df: pd.DataFrame) -> None:
    """Insert filtered instruments into the database."""
    await init_db()
    session_factory = _get_session_factory()

    instruments = []
    skipped = 0

    for _, row in df.iterrows():
        expiry = parse_expiry(str(row.get("expiry", "")))
        if expiry is None:
            skipped += 1
            continue

        try:
            token = int(str(row["token"]).strip())
            strike = float(str(row["strike"]).strip())
        except (ValueError, KeyError):
            skipped += 1
            continue

        instruments.append(
            {
                "instrument_token": token,
                "exchange_token": token,          # Angel One uses same token
                "tradingsymbol": str(row.get("symbol", "")).strip(),
                "name": "NIFTY",
                "expiry": expiry,
                "strike": strike,
                "instrument_type": str(row.get("optiontype", "")).strip(),  # CE or PE
                "segment": "NFO-OPT",
                "exchange": "NFO",
                "lot_size": int(float(str(row.get("lotsize", 25)))),
            }
        )

    if not instruments:
        print("ERROR: No valid instruments to insert.")
        return

    print(f"  Inserting {len(instruments)} instruments ({skipped} skipped)...")

    BATCH = 500
    total_inserted = 0

    async with session_factory() as session:
        for i in range(0, len(instruments), BATCH):
            batch = instruments[i : i + BATCH]
            await InstrumentCRUD.upsert_instruments(session, batch)
            total_inserted += len(batch)
            print(f"  Progress: {min(total_inserted, len(instruments))}/{len(instruments)}")

    print(f"\nDone: {len(instruments)} NIFTY option contracts seeded into DB.")


def main():
    df_all = download_scrip_master()
    df_nifty = filter_nifty_options(df_all)

    if df_nifty.empty:
        print("ERROR: No NIFTY options found in ScripMaster.")
        sys.exit(1)

    asyncio.run(seed(df_nifty))


if __name__ == "__main__":
    main()
