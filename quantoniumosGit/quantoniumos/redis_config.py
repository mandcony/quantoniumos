"""
QuantoniumOS - Redis Clustering Configuration
Implements distributed Redis backend for enterprise-grade rate limiting and caching
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis
from redis.exceptions import ConnectionError, TimeoutError
from redis.sentinel import Sentinel

logger = logging.getLogger("quantonium_redis")


class QuantoniumRedisCluster:
    """
    Enterprise Redis clustering implementation for QuantoniumOS
    Provides high availability, automatic failover, and distributed rate limiting
    """

    def __init__(self):
        self.redis_client = None
        self.sentinel = None
        self.cluster_nodes = self._get_cluster_config()
        self.master_name = os.environ.get("REDIS_MASTER_NAME", "quantonium-master")
        self.connect()

    def _get_cluster_config(self) -> List[tuple]:
        """Get Redis cluster configuration from environment"""
        redis_sentinels = os.environ.get("REDIS_SENTINELS", "")
        if redis_sentinels:
            # Format: "host1:port1,host2:port2,host3:port3"
            nodes = []
            for node in redis_sentinels.split(","):
                host, port = node.strip().split(":")
                nodes.append((host, int(port)))
            return nodes

        # Fallback to single Redis instance
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", 6379))
        return [(redis_host, redis_port)]

    def connect(self) -> bool:
        """Establish connection to Redis cluster with automatic failover"""
        try:
            if len(self.cluster_nodes) > 1:
                # Use Redis Sentinel for HA
                self.sentinel = Sentinel(
                    self.cluster_nodes,
                    sentinel_kwargs={
                        "password": os.environ.get("REDIS_SENTINEL_PASSWORD"),
                        "socket_timeout": 5.0,
                        "socket_connect_timeout": 5.0,
                    },
                )
                self.redis_client = self.sentinel.master_for(
                    self.master_name,
                    password=os.environ.get("REDIS_PASSWORD"),
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            else:
                # Single Redis instance
                self.redis_client = redis.Redis(
                    host=self.cluster_nodes[0][0],
                    port=self.cluster_nodes[0][1],
                    password=os.environ.get("REDIS_PASSWORD"),
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

            # Test connection
            self.redis_client.ping()
            logger.info("Redis cluster connection established successfully")
            self._check_and_enforce_security()
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis cluster: {e}")
            return False

    def _check_and_enforce_security(self):
        """Check Redis configuration for security best practices."""
        if not self.redis_client:
            return

        try:
            # Check if FLUSHDB and FLUSHALL are disabled
            config = self.redis_client.config_get("rename-command")
            renamed_commands = config.get("rename-command", "")

            if isinstance(renamed_commands, dict):  # redis-py >= 4.2
                if (
                    "FLUSHDB" not in renamed_commands
                    or renamed_commands.get("FLUSHDB") != ""
                ):
                    logger.critical(
                        "SECURITY WARNING: Redis command 'FLUSHDB' is not disabled. It is highly recommended to disable it in production by setting 'rename-command FLUSHDB \"\"' in your redis.conf."
                    )
                if (
                    "FLUSHALL" not in renamed_commands
                    or renamed_commands.get("FLUSHALL") != ""
                ):
                    logger.critical(
                        "SECURITY WARNING: Redis command 'FLUSHALL' is not disabled. It is highly recommended to disable it in production by setting 'rename-command FLUSHALL \"\"' in your redis.conf."
                    )

            # Fallback for older redis-py or different response formats
            elif isinstance(renamed_commands, str) and renamed_commands:
                commands = dict(
                    zip(
                        renamed_commands.split(",")[::2],
                        renamed_commands.split(",")[1::2],
                    )
                )
                if commands.get("FLUSHDB") != "":
                    logger.critical(
                        "SECURITY WARNING: Redis command 'FLUSHDB' is not disabled."
                    )
                if commands.get("FLUSHALL") != "":
                    logger.critical(
                        "SECURITY WARNING: Redis command 'FLUSHALL' is not disabled."
                    )

        except Exception as e:
            logger.error(f"Could not check Redis security configuration: {e}")

    def get_rate_limit_key(self, identifier: str, endpoint: str) -> str:
        """Generate rate limit key for distributed tracking"""
        return f"quantonium:ratelimit:{endpoint}:{identifier}"

    def check_rate_limit(
        self, identifier: str, endpoint: str, limit: int, window: int
    ) -> tuple:
        """
        Check rate limit using sliding window algorithm
        Returns (allowed: bool, remaining: int, reset_time: datetime)
        """
        if not self.redis_client:
            return True, limit, datetime.now()  # Fail open if Redis unavailable

        key = self.get_rate_limit_key(identifier, endpoint)
        now = datetime.now()
        window_start = now - timedelta(seconds=window)

        try:
            pipe = self.redis_client.pipeline()

            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start.timestamp())

            # Count current requests in window
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(now.timestamp()): now.timestamp()})

            # Set expiration
            pipe.expire(key, window)

            results = pipe.execute()
            current_count = results[1] + 1  # +1 for the request we just added

            if current_count <= limit:
                remaining = limit - current_count
                reset_time = now + timedelta(seconds=window)
                return True, remaining, reset_time
            else:
                # Remove the request we added since it's over limit
                self.redis_client.zrem(key, str(now.timestamp()))
                remaining = 0
                reset_time = now + timedelta(seconds=window)
                return False, remaining, reset_time

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, limit, now  # Fail open

    def log_security_event(self, event_data: Dict[str, Any]) -> bool:
        """Store security events for behavioral analysis"""
        if not self.redis_client:
            return False

        try:
            event_key = (
                f"quantonium:security:events:{datetime.now().strftime('%Y%m%d')}"
            )
            event_data["timestamp"] = datetime.now().isoformat()

            self.redis_client.lpush(event_key, json.dumps(event_data))
            self.redis_client.expire(event_key, 86400 * 30)  # Keep for 30 days
            return True

        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            return False

    def get_user_behavior_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user behavior profile for anomaly detection"""
        if not self.redis_client:
            return {}

        try:
            profile_key = f"quantonium:behavior:{user_id}"
            profile_data = self.redis_client.hgetall(profile_key)

            if profile_data:
                return {
                    "request_count": int(profile_data.get("request_count", 0)),
                    "last_seen": profile_data.get("last_seen"),
                    "endpoints_accessed": json.loads(
                        profile_data.get("endpoints_accessed", "[]")
                    ),
                    "average_session_duration": float(
                        profile_data.get("avg_session_duration", 0)
                    ),
                    "risk_score": float(profile_data.get("risk_score", 0)),
                }
            return {}

        except Exception as e:
            logger.error(f"Failed to get behavior profile: {e}")
            return {}

    def update_user_behavior(
        self, user_id: str, endpoint: str, session_duration: float = 0
    ) -> bool:
        """Update user behavior profile for ML-based anomaly detection"""
        if not self.redis_client:
            return False

        try:
            profile_key = f"quantonium:behavior:{user_id}"
            pipe = self.redis_client.pipeline()

            # Increment request count
            pipe.hincrby(profile_key, "request_count", 1)

            # Update last seen
            pipe.hset(profile_key, "last_seen", datetime.now().isoformat())

            # Update endpoints accessed
            current_endpoints = self.redis_client.hget(
                profile_key, "endpoints_accessed"
            )
            if current_endpoints:
                endpoints = json.loads(current_endpoints)
            else:
                endpoints = []

            if endpoint not in endpoints:
                endpoints.append(endpoint)
                pipe.hset(profile_key, "endpoints_accessed", json.dumps(endpoints))

            # Update session duration if provided
            if session_duration > 0:
                current_avg = float(
                    self.redis_client.hget(profile_key, "avg_session_duration") or 0
                )
                request_count = int(
                    self.redis_client.hget(profile_key, "request_count") or 1
                )
                new_avg = (
                    (current_avg * (request_count - 1)) + session_duration
                ) / request_count
                pipe.hset(profile_key, "avg_session_duration", new_avg)

            # Set expiration (90 days)
            pipe.expire(profile_key, 86400 * 90)

            pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Failed to update user behavior: {e}")
            return False

    def cache_quantum_computation(
        self, computation_id: str, result: Dict[str, Any], ttl: int = 3600
    ) -> bool:
        """Cache quantum computation results for performance optimization"""
        if not self.redis_client:
            return False

        try:
            cache_key = f"quantonium:cache:quantum:{computation_id}"
            self.redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            return True

        except Exception as e:
            logger.error(f"Failed to cache quantum computation: {e}")
            return False

    def get_cached_computation(self, computation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached quantum computation result"""
        if not self.redis_client:
            return None

        try:
            cache_key = f"quantonium:cache:quantum:{computation_id}"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                return json.loads(cached_data)
            return None

        except Exception as e:
            logger.error(f"Failed to get cached computation: {e}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of Redis cluster"""
        health_status = {
            "redis_connected": False,
            "sentinel_connected": False,
            "cluster_info": {},
            "memory_usage": {},
            "response_time_ms": 0,
        }

        if not self.redis_client:
            return health_status

        try:
            start_time = datetime.now()

            # Basic connectivity test
            self.redis_client.ping()
            health_status["redis_connected"] = True

            # Response time
            end_time = datetime.now()
            health_status["response_time_ms"] = (
                end_time - start_time
            ).total_seconds() * 1000

            # Memory usage
            info = self.redis_client.info("memory")
            health_status["memory_usage"] = {
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "used_memory_peak": info.get("used_memory_peak"),
                "used_memory_peak_human": info.get("used_memory_peak_human"),
            }

            # Sentinel status if using HA setup
            if self.sentinel:
                try:
                    sentinel_info = self.sentinel.discover_master(self.master_name)
                    health_status["sentinel_connected"] = True
                    health_status["cluster_info"] = {
                        "master_host": sentinel_info[0],
                        "master_port": sentinel_info[1],
                    }
                except Exception:
                    health_status["sentinel_connected"] = False

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        return health_status


# Global Redis cluster instance
redis_cluster = QuantoniumRedisCluster()


# Backward compatibility functions for existing middleware
def get_redis_connection():
    """Get Redis connection for backward compatibility"""
    if redis_cluster.redis_client:
        return redis_cluster.redis_client
    return None


def get_rate_limit(identifier: str, limit: int, window: int):
    """Rate limiting function for backward compatibility"""
    try:
        allowed, remaining, reset_time = redis_cluster.check_rate_limit(
            identifier, "general", limit, window
        )
        exceeded = not allowed
        return exceeded, remaining, reset_time
    except Exception:
        return None


# Configuration flags
REDIS_AVAILABLE = redis_cluster.redis_client is not None
REDIS_URL = os.environ.get("REDIS_URL", "")
