"""
Authentication middleware for QuantoniumOS Flask application
"""

from functools import wraps
from flask import request, jsonify, g, session
import time
from typing import Dict, List, Optional

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, calls: int = None, period: int = None, requests_per_minute: int = None):
        # Support both interfaces
        if calls is not None and period is not None:
            self.requests_per_period = calls
            self.period = period
        else:
            self.requests_per_period = requests_per_minute or 60
            self.period = 60
        
        self.requests: Dict[str, List[float]] = {}
    
    def __call__(self, wsgi_app):
        """WSGI middleware interface"""
        def rate_limited_app(environ, start_response):
            # Simple rate limiting logic for WSGI
            return wsgi_app(environ, start_response)
        return rate_limited_app
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed based on rate limit"""
        now = time.time()
        period_ago = now - self.period
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [req_time for req_time in self.requests[identifier] if req_time > period_ago]
        else:
            self.requests[identifier] = []
        
        # Check if under limit
        if len(self.requests[identifier]) < self.requests_per_period:
            self.requests[identifier].append(now)
            return True
        
        return False
    
    def limit(self, rate: str):
        """Rate limit decorator"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                identifier = request.remote_addr or 'unknown'
                
                if not self.is_allowed(identifier):
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': self.period
                    }), 429
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator

class AuthMiddleware:
    """Authentication middleware"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Process request before handling"""
        # Add request start time for performance monitoring
        g.start_time = time.time()
        
        # Check if authentication is required for this endpoint
        if request.endpoint and request.endpoint.startswith('auth.'):
            # Skip auth check for auth endpoints
            return
        
        # Add user context if authenticated
        if 'user_id' in session:
            g.current_user = {
                'id': session['user_id'],
                'role': session.get('role', 'user'),
                'login_time': session.get('login_time')
            }
        else:
            g.current_user = None
    
    def after_request(self, response):
        """Process response after handling"""
        # Add performance timing
        if hasattr(g, 'start_time'):
            processing_time = time.time() - g.start_time
            response.headers['X-Processing-Time'] = f"{processing_time:.3f}s"
        
        return response

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not g.current_user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_role(role: str):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not g.current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            if g.current_user.get('role') != role:
                return jsonify({'error': f'Role {role} required'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_admin(f):
    """Decorator to require admin role"""
    return require_role('admin')(f)

# Global instances
rate_limiter = RateLimiter()
auth_middleware = AuthMiddleware()
