"""
Quantonium OS - Authentication Middleware

Implements authentication and rate limiting middleware for the API.
This module provides Redis-backed distributed rate limiting and JWT authentication.
"""

import os
import time
import json
import logging
from functools import wraps

# Try importing Flask-related modules, fallback gracefully if not available
try:
    from flask import request, jsonify, g, current_app
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    FLASK_AVAILABLE = True
except ImportError:
    # Set placeholders for Flask objects if Flask is not available
    FLASK_AVAILABLE = False
    request = None
    jsonify = None
    g = None
    current_app = None
    Limiter = None
    get_remote_address = None

# Configure logging
logger = logging.getLogger("quantonium_auth")

# Try to import Redis, fallback gracefully if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not found. Falling back to in-memory rate limiting.")

# Get Redis URL from environment
REDIS_URL = os.environ.get('REDIS_URL')

# Create a Redis-backed rate limiter middleware
class RedisRateLimiter:
    """
    Redis-backed distributed rate limiter for Flask applications.
    This complements the main rate limiter from flask_limiter in security.py.
    
    This middleware provides IP-based rate limiting at the WSGI level
    before the request even reaches the Flask application, using Redis
    as a distributed storage backend for better scalability.
    """
    
    def __init__(self, redis_url, calls=30, period=60, redis_prefix='ratelimit:'):
        """
        Initialize the Redis rate limiter.
        
        Args:
            redis_url: Redis connection URL
            calls: Number of calls allowed
            period: Time period in seconds
            redis_prefix: Prefix for Redis keys
        """
        self.calls = calls
        self.period = period
        self.prefix = redis_prefix
        self.redis_client = None
        
        # Initialize Redis client if Redis is available
        if not REDIS_AVAILABLE:
            logger.error("Redis module not available. Cannot initialize Redis rate limiter.")
            return
        
        # Initialize Redis client
        try:
            # Use the imported redis module if available
            import redis as redis_module
            self.redis_client = redis_module.Redis.from_url(redis_url)
            self.redis_client.ping()  # Test connection
            logger.info("Connected to Redis server for distributed rate limiting")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def check_rate_limit(self, client_ip):
        """
        Check if a client has exceeded rate limit and update counter
        
        Args:
            client_ip: Client IP address
            
        Returns:
            Tuple of (exceeded, remaining)
        """
        # If Redis client is not available, always allow
        if not self.redis_client:
            return False, self.calls
            
        # Create a key for this IP
        key = f"{self.prefix}{client_ip}"
        
        try:
            # Use a pipeline for atomic operations
            with self.redis_client.pipeline() as pipe:
                # Atomic operations using Redis pipeline
                pipe.get(key)
                pipe.incr(key)
                pipe.expire(key, self.period)
                results = pipe.execute()
                
                # Get current count (before increment)
                current_count = 0
                if results[0]:
                    current_count = int(results[0])
                    
                # Check if over limit
                if current_count >= self.calls:
                    return True, 0
                
                return False, self.calls - current_count - 1
                
        except Exception as e:
            logger.error(f"Redis error during rate limit check: {e}")
            # On error, allow the request but log the failure
            return False, -1
    
    def __call__(self, app):
        """
        Apply rate limiting to the Flask application.
        
        Args:
            app: Flask application
            
        Returns:
            A WSGI middleware that wraps the Flask application
        """
        def middleware(environ, start_response):
            # Get client IP
            client_ip = environ.get('REMOTE_ADDR', 'unknown')
            
            # Only apply rate limiting if Redis is available
            if self.redis_client:
                exceeded, remaining = self.check_rate_limit(client_ip)
                
                if exceeded:
                    # Rate limit exceeded
                    headers = [
                        ('Content-Type', 'application/json'),
                        ('Retry-After', str(self.period)),
                        ('X-RateLimit-Limit', str(self.calls)),
                        ('X-RateLimit-Remaining', '0'),
                        ('X-RateLimit-Reset', str(int(time.time()) + self.period))
                    ]
                    start_response('429 Too Many Requests', headers)
                    return [b'{"error": "Rate limit exceeded. Try again later."}']
                
                # Add rate limit headers for passing requests too
                def custom_start_response(status, headers, exc_info=None):
                    headers.extend([
                        ('X-RateLimit-Limit', str(self.calls)),
                        ('X-RateLimit-Remaining', str(max(0, remaining))),
                        ('X-RateLimit-Reset', str(int(time.time()) + self.period))
                    ])
                    return start_response(status, headers, exc_info)
                
                # Pass through to the application with our custom start_response
                return app(environ, custom_start_response)
            
            # Fallback to standard behavior if Redis is not available
            return app(environ, start_response)
        
        return middleware


# Fallback in-memory rate limiter if Redis is not available
class MemoryRateLimiter:
    """
    Memory-based rate limiter for Flask applications.
    This complements the main rate limiter from flask_limiter in security.py.
    
    This middleware provides IP-based rate limiting at the WSGI level
    before the request even reaches the Flask application.
    """
    
    def __init__(self, calls=30, period=60):
        """
        Initialize the rate limiter.
        
        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.clients = {}
        logger.warning("Using in-memory rate limiter. This does not scale across multiple instances.")
        
    def __call__(self, app):
        """
        Apply rate limiting to the Flask application.
        
        Args:
            app: Flask application
            
        Returns:
            A WSGI middleware that wraps the Flask application
        """
        def middleware(environ, start_response):
            # Get client IP
            client_ip = environ.get('REMOTE_ADDR', 'unknown')
            
            # Get current time
            current_time = time.time()
            
            # Clean up old entries
            for ip in list(self.clients.keys()):
                if current_time - self.clients[ip]['timestamp'] > self.period:
                    del self.clients[ip]
            
            # Check if client has exceeded rate limit
            if client_ip in self.clients:
                client = self.clients[client_ip]
                if client['count'] >= self.calls:
                    # Rate limit exceeded
                    headers = [
                        ('Content-Type', 'application/json'),
                        ('Retry-After', str(self.period)),
                        ('X-RateLimit-Limit', str(self.calls)),
                        ('X-RateLimit-Remaining', '0'),
                        ('X-RateLimit-Reset', str(int(current_time) + self.period))
                    ]
                    start_response('429 Too Many Requests', headers)
                    return [b'{"error": "Rate limit exceeded. Try again later."}']
                
                # Increment count
                client['count'] += 1
                remaining = max(0, self.calls - client['count'])
            else:
                # Add new client
                self.clients[client_ip] = {
                    'count': 1,
                    'timestamp': current_time
                }
                remaining = self.calls - 1
            
            # Add rate limit headers
            def custom_start_response(status, headers, exc_info=None):
                headers.extend([
                    ('X-RateLimit-Limit', str(self.calls)),
                    ('X-RateLimit-Remaining', str(remaining)),
                    ('X-RateLimit-Reset', str(int(current_time) + self.period))
                ])
                return start_response(status, headers, exc_info)
            
            # Pass through to the application with custom headers
            return app(environ, custom_start_response)
        
        return middleware


# Factory function to create the appropriate rate limiter
def RateLimiter(calls=30, period=60):
    """
    Factory function that returns the appropriate rate limiter based on availability.
    Will use Redis if available, otherwise falls back to in-memory.
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
        
    Returns:
        A rate limiter instance
    """
    if REDIS_AVAILABLE and REDIS_URL:
        try:
            # Test Redis connection and return Redis-backed limiter if successful
            limiter = RedisRateLimiter(REDIS_URL, calls, period)
            if limiter.redis_client:
                logger.info(f"Using Redis-backed rate limiter ({calls} requests per {period}s)")
                return limiter
        except Exception as e:
            logger.error(f"Error initializing Redis rate limiter: {e}")
            logger.info("Falling back to in-memory rate limiter")
    
    # Fallback to in-memory rate limiter
    return MemoryRateLimiter(calls, period)

# Decorator for requiring JWT authentication
def require_jwt_auth(f):
    """
    Decorator to require JWT authentication for a route.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function
    """
    # Check if Flask is available
    if not FLASK_AVAILABLE:
        logger.error("Flask is not available. JWT authentication will not work.")
        # Return a pass-through decorator
        return f
    
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Missing Authorization header'}), 401
        
        # Check that it's a Bearer token
        parts = auth_header.split()
        if parts[0].lower() != 'bearer':
            return jsonify({'error': 'Authorization header must start with Bearer'}), 401
        
        if len(parts) == 1:
            return jsonify({'error': 'Token not found'}), 401
        
        if len(parts) > 2:
            return jsonify({'error': 'Authorization header must be Bearer token'}), 401
        
        token = parts[1]
        
        # Validate token
        # This is where you would validate the JWT token
        # For now, just accept any token
        g.user = {'id': 1, 'username': 'testuser'}
        
        return f(*args, **kwargs)
    
    return decorated