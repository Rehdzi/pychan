import redis.asyncio as redis
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

# Create Redis client with connection pooling
redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,  # Better for JSON handling
    max_connections=10,     # Maximum number of connections in the pool
    socket_timeout=5,       # Socket timeout in seconds
    socket_connect_timeout=2,  # Socket connect timeout
    retry_on_timeout=True,  # Retry on timeout
    health_check_interval=30  # Seconds between health checks
)

# Test connection at startup
async def test_redis_connection():
    try:
        await redis_client.ping()
        logger.info("Redis connection successful")
    except redis.RedisError as e:
        logger.error(f"Redis connection error: {e}")
        # Don't raise, let the app start anyway