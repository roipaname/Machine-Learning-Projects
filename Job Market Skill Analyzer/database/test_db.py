# test_db.py
import asyncio
from sqlalchemy import text
from src.database.session import init_db, get_db_context, close_db

async def main():
    await init_db()  # verifies connection

    async with get_db_context() as db:
        result = await db.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
        tables = result.scalars().all()
        print("Tables found:", tables)

    await close_db()

asyncio.run(main())