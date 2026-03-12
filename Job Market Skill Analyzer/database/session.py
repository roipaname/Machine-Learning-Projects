from config.settings import (
    DB_NAME,DB_HOST,
    DB_PASSWORD,DB_PORT,DB_USER,
    DB_POOL_SIZE,DB_MAX_OVERFLOW,
    DB_ECHO,APP_ENV,DB_POOL_TIMEOUT,
    
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

        return create_async_engine()
