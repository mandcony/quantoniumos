"""
Quantonium OS - Authentication Middleware

Implements authentication and rate limiting middleware for the API.
"""

from functools import wraps
from flask import request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Create a rate limiter middleware
class RateLimiter:
    """
    Secondary rate limiter for Flask applications.
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
            import time
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
                        ('Retry-After', str(self.period))
                    ]
                    start_response('429 Too Many Requests', headers)
                    return [b'{"error": "Rate limit exceeded. Try again later."}']
                
                # Increment count
                client['count'] += 1
            else:
                # Add new client
                self.clients[client_ip] = {
                    'count': 1,
                    'timestamp': current_time
                }
            
            # Pass through to the application
            return app(environ, start_response)
        
        return middleware

# Decorator for requiring JWT authentication
def require_jwt_auth(f):
    """
    Decorator to require JWT authentication for a route.
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function
    """
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