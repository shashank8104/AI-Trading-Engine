import asyncio
from sqlalchemy import text
from app.db.database import _get_engine

async def check():
    engine = _get_engine()
    async with engine.begin() as conn:
        c = await conn.execute(text("SELECT timestamp FROM candles ORDER BY timestamp DESC LIMIT 1"))
        print(f"Latest candle: {c.scalar()}")
        p = await conn.execute(text("SELECT timestamp FROM predictions ORDER BY timestamp DESC LIMIT 1"))
        print(f"Latest prediction: {p.scalar()}")
        s = await conn.execute(text("SELECT timestamp FROM signals ORDER BY timestamp DESC LIMIT 1"))
        print(f"Latest signal: {s.scalar()}")

asyncio.run(check())
