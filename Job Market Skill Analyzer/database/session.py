from config.settings import (
    DB_NAME,DB_HOST,
    DB_PASSWORD,DB_PORT,DB_USER,
    DB_POOL_SIZE,DB_MAX_OVERFLOW,
    DB_ECHO,APP_ENV,DB_POOL_TIMEOUT,
    SQLALCHEMY_URL
)
from sqlalchemy.pool import NullPool
from __future__ import annotations
from loguru import logger
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from slqalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)



# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
 

def build_engine()->AsyncEngine:

    kwargs=dict(
      echo=DB_ECHO,
      echo_pool=DB_POOL_SIZE,
      future=True
    )

    # Use NullPool for scripts / testing to avoid "event loop closed" errors
    if APP_ENV=="testing":
        kwargs['poolclass']=NullPool
    else:
        kwargs.update(
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_pre_ping=True,   # detect stale connections
        )

        return create_async_engine(SQLALCHEMY_URL,**kwargs)
    

engine:AsyncEngine=build_engine()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal:async_sessionmaker[AsyncSession]= async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # keep attrs accessible after commit
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a transactional AsyncSession.
    Rolls back automatically on exception; commits on clean exit.
 
    Example
    -------
    @router.post("/jobs")
    async def create_job(db: AsyncSession = Depends(get_db)):
        ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for use outside FastAPI (scripts, CLIs, schedulers).
 
    Example
    -------
    async with get_db_context() as db:
        result = await db.execute(select(Skill))
        skills = result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
 
