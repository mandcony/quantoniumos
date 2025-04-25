"""
Quantonium OS - Redis Configuration

This module provides a centralized mechanism for establishing and managing Redis connections.
It includes graceful fallbacks and error handling to ensure the application works even without Redis.
"""

import os
import logging
import time
from functools import wraps

# Configure logger
logger = logging.getLogger("quantonium_redis")

# Check if Redis is available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not found. Redis-based features will be disabled.")

# Redis connection parameters
REDIS_URL = os.environ.get('REDIS_URL')
REDIS_POOL = None  # Global connection pool

# Rate limiting defaults
DEFAULT_RATE_LIMIT = 60  # 60 requests per minute
DEFAULT_RATE_WINDOW = 60  # 1 minute window


def get_redis_connection():
    """
    Get a Redis connection from the pool or create one if needed.
    
    Returns:
        redis.Redis: Redis client instance or None if Redis is not available
    """
    global REDIS_POOL
    
    if not REDIS_AVAILABLE or not REDIS_URL:
        return None
    
    try:
        # Create connection pool if it doesn't exist
        if not REDIS_POOL:
            REDIS_POOL = redis.ConnectionPool.from_url(
                REDIS_URL,
                max_connections=10,
                socket_timeout=3,
                socket_connect_timeout=3,
                health_check_interval=30
            )
            logger.info("Redis connection pool initialized")
        
        # Get connection from pool
        client = redis.Redis(connection_pool=REDIS_POOL)
        
        # Test connection
        client.ping()
        return client
    
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        REDIS_POOL = None  # Reset pool on error
        return None


def check_redis_health():
    """
    Check if Redis is available and working.
    
    Returns:
        bool: True if Redis is healthy, False otherwise
    """
    client = get_redis_connection()
    if not client:
        return False
    
    try:
        # Check basic operations
        key = "health_check"
        client.set(key, "1", ex=10)
        value = client.get(key)
        return value == b"1"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


def with_redis(fallback_value=None):
    """
    Decorator that provides a Redis client to a function or falls back gracefully.
    
    Args:
        fallback_value: Value to return if Redis is not available
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = get_redis_connection()
            if not client:
                return fallback_value
            
            try:
                return func(client, *args, **kwargs)
            except redis.RedisError as e:
                logger.error(f"Redis operation error: {e}")
                return fallback_value
        
        return wrapper
    
    return decorator


@with_redis(fallback_value=None)
def get_rate_limit(client, key, limit=DEFAULT_RATE_LIMIT, window=DEFAULT_RATE_WINDOW):
    """
    Check and update rate limit for a key.
    
    Args:
        client: Redis client
        key: Unique identifier (e.g. IP address)
        limit: Maximum number of requests allowed in the window
        window: Time window in seconds
        
    Returns:
        tuple: (exceeded, remaining, reset_time) or None if Redis is not available
    """
    pipe = client.pipeline()
    
    # Keys for rate limiting
    count_key = f"ratelimit:{key}:count"
    reset_key = f"ratelimit:{key}:reset"
    
    # Get current time
    now = int(time.time())
    
    # Get current values
    pipe.get(count_key)
    pipe.get(reset_key)
    results = pipe.execute()
    
    # Extract values
    current_count = int(results[0]) if results[0] else 0
    reset_time = int(results[1]) if results[1] else (now + window)
    
    # Check if window has expired
    if now > reset_time:
        # Start new window
        pipe.set(count_key, 1)
        pipe.set(reset_key, now + window)
        pipe.expire(count_key, window)
        pipe.expire(reset_key, window)
        pipe.execute()
        
        return False, limit - 1, now + window
    
    # Increment counter
    pipe.incr(count_key)
    pipe.execute()
    
    # Check if limit has been exceeded
    if current_count >= limit:
        return True, 0, reset_time
    
    return False, limit - current_count - 1, reset_time