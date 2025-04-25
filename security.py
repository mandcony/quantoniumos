"""
Quantonium OS - Security Configuration

Implements security middleware, headers, and protection mechanisms.
"""

import os
from flask import request, abort, Flask
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

# Configure logger
logger = logging.getLogger("quantonium_security")
logger.setLevel(logging.INFO)

# CSP policy that only blocks iframe embedding but allows everything else to work
CSP_POLICY = {
    'frame-ancestors': ["'none'"]  # Only block iframe embedding
}

# Configure CORS
def get_cors_origins():
    """
    Get allowed CORS origins from environment variable
    Returns list of allowed origins
    """
    origins_env = os.environ.get('CORS_ORIGINS', '')
    if not origins_env:
        return []
    
    # Split by comma and strip whitespace
    return [origin.strip() for origin in origins_env.split(',')]

def cors_check():
    """
    Check if the request's Origin header is in the allowed list
    If not, abort with 403 Forbidden
    """
    origin = request.headers.get('Origin')
    if not origin:
        return
    
    allowed_origins = get_cors_origins()
    if allowed_origins and origin not in allowed_origins:
        logger.warning(f"Rejected cross-origin request from {origin}")
        abort(403, f"Origin {origin} not allowed")

def configure_security(app: Flask):
    """
    Configure all security middleware for the Flask app
    """
    # Determine if we're in development mode
    is_development = os.environ.get('FLASK_ENV') == 'development' or app.debug
    
    # Store is_development in app config for access in route handlers
    app.config['IS_DEVELOPMENT'] = is_development
    
    # Set up Talisman for HTTPS, HSTS, and CSP
    # Note: We enforce HTTPS everywhere now except localhost development
    local_development = is_development and (
        os.environ.get('REPLIT_ENVIRONMENT') is None or 
        os.environ.get('LOCAL_DEVELOPMENT') == 'true'
    )
    
    talisman = Talisman(
        app,
        content_security_policy=CSP_POLICY,
        content_security_policy_nonce_in=['script-src'],
        force_https=not local_development,  # Force HTTPS except for local development
        force_https_permanent=True,
        force_file_save=True,
        frame_options='SAMEORIGIN',  # Allow only same-origin embedding
        frame_options_allow_from=None,
        strict_transport_security=True,  # Always enable HSTS
        strict_transport_security_preload=True,
        strict_transport_security_max_age=31536000,  # 1 year
        referrer_policy='no-referrer',
        session_cookie_secure=not local_development,  # Use secure cookies except in local dev
        session_cookie_http_only=True
    )
    
    # Set up rate limiting - use Redis if available, otherwise memory
    redis_url = os.environ.get('REDIS_URL')
    storage_uri = redis_url if redis_url else "memory://"
    
    # More strict rate limits in production, more lenient in development
    if is_development:
        default_limits = ["300 per minute", "10000 per day"]
    else:
        default_limits = ["120 per minute", "3000 per day"]
        
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=default_limits,
        storage_uri=storage_uri,
        strategy="fixed-window"  # Standard algorithm
    )
    
    # Add CORS checking to before_request
    @app.before_request
    def before_request():
        cors_check()
    
    # Add permissions policy header to all responses
    @app.after_request
    def set_additional_headers(response):
        response.headers['Permissions-Policy'] = 'interest-cohort=()'
        
        # Add CORS headers with strict origin checking
        origin = request.headers.get('Origin')
        allowed_origins = get_cors_origins()
        
        # If we have allowed origins and the request has an origin header
        if allowed_origins and origin:
            # Only set CORS headers if the origin is allowed
            if origin in allowed_origins:
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key, Authorization'
                response.headers['Access-Control-Allow-Credentials'] = 'true'
                response.headers['Vary'] = 'Origin'  # Vary header is important for caching
            # If no allowed origins are configured, default to same-origin policy (don't set CORS headers)
        elif app.config['IS_DEVELOPMENT']:
            # In development mode only, allow all origins for easier testing
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key, Authorization'
        
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Only allow same-origin iframe embedding
        response.headers['X-Content-Type-Options'] = 'nosniff'  # Prevent MIME type sniffing
        return response
    
    return talisman, limiter