"""Migrate DB instrument tokens from Kite (256265) to Angel One (26000)."""
import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import text
from app.db.database import _get_engine

async def migrate():
    engine = _get_engine()
    async with engine.begin() as conn:
        r1 = await conn.execute(
            text("UPDATE candles SET instrument_token=26000 WHERE instrument_token=256265")
        )
        r2 = await conn.execute(
            text("UPDATE predictions SET instrument_token=26000 WHERE instrument_token=256265")
        )
        print(f"Candles migrated: {r1.rowcount}")
        print(f"Predictions migrated: {r2.rowcount}")
        print("Token migration complete")

asyncio.run(migrate())
