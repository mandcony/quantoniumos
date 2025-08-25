"""
QuantoniumOS Auth Middleware

Authentication and authorization middleware for QuantoniumOS.
"""

from flask import request, jsonify, g
import time
from functools import wraps
from collections import defaultdict
from typing import Dict, Any, Callable

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record this request
        self.requests[client_id].append(current_time)
        return True
    
    def get_reset_time(self, client_id: str) -> float:
        """Get time when rate limit resets"""
        if not self.requests[client_id]:
            return time.time()
        
        oldest_request = min(self.requests[client_id])
        return oldest_request + self.window_seconds

def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """Rate limiting decorator"""
    limiter = RateLimiter(max_requests, window_seconds)
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = request.remote_addr
            
            if not limiter.is_allowed(client_id):
                reset_time = limiter.get_reset_time(client_id)
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': int(reset_time - time.time())
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_auth(f: Callable) -> Callable:
    """Require authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for session or API key
        if 'user_id' not in request.session:
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return jsonify({'error': 'Authentication required'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def require_admin(f: Callable) -> Callable:
    """Require admin privileges decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check authentication first
        if 'user_id' not in request.session:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Check admin status (simplified for demo)
        user_id = request.session.get('user_id')
        if user_id != 'admin':
            return jsonify({'error': 'Admin privileges required'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

def log_request():
    """Log incoming requests"""
    g.start_time = time.time()
    
def log_response(response):
    """Log outgoing responses"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        print(f"Request: {request.method} {request.path} - {response.status_code} - {duration:.3f}s")
    return response
