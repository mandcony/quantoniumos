"""
Security configuration for QuantoniumOS Flask application
"""

from flask import Flask
import os

def configure_security(app: Flask):
    """Configure security settings for the Flask app"""
    
    # Basic security headers
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        return response
    
    # Session configuration
    app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    
    # Configure secret key
    if not app.config.get('SECRET_KEY'):
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    print("✅ Security configuration applied")
    
    # Create mock security objects for compatibility
    class MockTalisman:
        def __init__(self):
            self.enabled = True
    
    class MockLimiter:
        def __init__(self):
            self.enabled = True
        
        def limit(self, limit_string):
            """Mock rate limit decorator"""
            def decorator(f):
                return f
            return decorator
    
    talisman = MockTalisman()
    limiter = MockLimiter()
    
    return talisman, limiter

def validate_api_key(api_key: str) -> bool:
    """Validate API key (placeholder implementation)"""
    # This is a placeholder - implement proper API key validation
    valid_keys = [
        'quantonium-dev-key',
        'quantonium-admin-key'
    ]
    return api_key in valid_keys

def generate_csrf_token() -> str:
    """Generate CSRF token"""
    import secrets
    return secrets.token_urlsafe(32)

def verify_csrf_token(token: str, session_token: str) -> bool:
    """Verify CSRF token"""
    return token == session_token

class SecurityConfig:
    """Security configuration constants"""
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # 1 hour
    
    # API Keys
    API_KEY_HEADER = 'X-API-Key'
    
    # Authentication
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 900  # 15 minutes
    
    # Encryption
    ENCRYPTION_ALGORITHM = 'RFT-AES-256'
    KEY_DERIVATION_ITERATIONS = 100000
