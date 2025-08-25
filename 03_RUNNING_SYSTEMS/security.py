"""
QuantoniumOS Security Configuration

Security middleware and configuration for QuantoniumOS.
"""

from flask import Flask, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import hashlib
import time
from functools import wraps
from typing import Dict, Any, Optional, Callable

def configure_security(app: Flask) -> None:
    """Configure security middleware for Flask app"""
    
    # Set secure session configuration
    app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
    
    # Set security headers
    @app.after_request
    def set_security_headers(response):
        """Set security headers on all responses"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        return response
    
    # Rate limiting storage (in production, use Redis or database)
    rate_limit_storage = {}
    
    @app.before_request
    def rate_limit():
        """Simple rate limiting middleware"""
        client_ip = request.remote_addr
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 3600  # 1 hour window
        rate_limit_storage[client_ip] = [
            timestamp for timestamp in rate_limit_storage.get(client_ip, [])
            if timestamp > cutoff_time
        ]
        
        # Check rate limit (100 requests per hour)
        if len(rate_limit_storage.get(client_ip, [])) >= 100:
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        # Record this request
        if client_ip not in rate_limit_storage:
            rate_limit_storage[client_ip] = []
        rate_limit_storage[client_ip].append(current_time)
    
    print("✅ Security middleware configured")

def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_api_key(f: Callable) -> Callable:
    """Decorator to require API key for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        # In production, validate against database
        # For demo, accept any non-empty key
        if not api_key.strip():
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password: str) -> str:
    """Hash password securely"""
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return check_password_hash(password_hash, password)

def generate_session_token() -> str:
    """Generate secure session token"""
    return secrets.token_urlsafe(32)

def generate_api_key() -> str:
    """Generate secure API key"""
    return secrets.token_urlsafe(32)

def validate_input(data: Dict[str, Any], required_fields: list) -> Optional[str]:
    """Validate input data has required fields"""
    for field in required_fields:
        if field not in data or not data[field]:
            return f"Missing required field: {field}"
    return None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove or replace dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    return filename

def check_file_type(filename: str, allowed_extensions: set) -> bool:
    """Check if file type is allowed"""
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in allowed_extensions

class SecurityAudit:
    """Security audit logging"""
    
    def __init__(self):
        self.events = []
    
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details,
            'ip': request.remote_addr if request else 'system'
        }
        self.events.append(event)
        
        # In production, write to secure log file or database
        print(f"🔐 Security Event: {event_type} - {details}")
    
    def get_recent_events(self, limit: int = 100) -> list:
        """Get recent security events"""
        return self.events[-limit:]

# Global security audit instance
security_audit = SecurityAudit()
