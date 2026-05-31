import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    engine = create_async_engine('postgresql+asyncpg://postgres:shanky8104@localhost:5432/trading_db')
    async with engine.begin() as conn:
        await conn.execute(text('TRUNCATE signals, predictions;'))
    print('Tables truncated successfully.')

asyncio.run(main())
