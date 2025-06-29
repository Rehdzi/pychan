from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
import os
import logging
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables")

# Use QueuePool with reasonable pool size and timeout
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,  # Set to False in production
    pool_size=5,  # Reasonable default for moderate traffic
    max_overflow=10,  # Allow 10 more connections when pool is full
    pool_timeout=30,  # Timeout in seconds when waiting for a connection
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True  # Check connection validity before using
)

# Extend AsyncSession with a method to check if transaction is in progress
class ExtendedAsyncSession(AsyncSession):
    def in_transaction(self) -> bool:
        """Check if this session is in an active transaction.
        
        SQLAlchemy's AsyncSession tracks its transaction state differently
        than we previously assumed. This implements a proper check.
        """
        return self._transaction is not None

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=ExtendedAsyncSession,  # Use our extended class
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[ExtendedAsyncSession, None]:
    """Dependency for getting a database session with proper error handling."""
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        await session.rollback()
        raise
    finally:
        await session.close()