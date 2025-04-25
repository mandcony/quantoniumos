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
    
    # If no origins are configured, use safe defaults (self and localhost)
    if not origins_env:
        # Include the Replit environment domain and localhost for development
        default_origins = []
        replit_domains = os.environ.get('REPLIT_DOMAINS', '')
        if replit_domains:
            # Replit domains are separated by commas
            for domain in replit_domains.split(','):
                # Only use HTTPS versions
                default_origins.append(f"https://{domain.strip()}")
        
        # Local development exception - only for direct testing
        if os.environ.get('LOCAL_DEVELOPMENT') == 'true':
            default_origins.extend([
                'https://localhost:5000',
                'https://localhost:3000'
            ])
        return default_origins
    
    # Split by comma and strip whitespace
    return [origin.strip() for origin in origins_env.split(',')]

def cors_check():
    """
    Check if the request's Origin header is in the allowed list
    If not, abort with 403 Forbidden
    
    Exception: API endpoints and static resources are allowed from any origin.
    """
    origin = request.headers.get('Origin')
    if not origin:
        return  # Same-origin requests always allowed
    
    # Allow API endpoints and static resources from any origin
    api_endpoints = request.path.startswith('/api/')
    static_resources = request.path.startswith('/static/') or request.path.startswith('/wave_ui/')
    if api_endpoints or static_resources:
        return
    
    # For all other endpoints, check if the origin is in the allowed list
    allowed_origins = get_cors_origins()
    if allowed_origins and origin not in allowed_origins:
        logger.warning(f"Rejected cross-origin request from {origin} to {request.path}")
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
    # We enforce HTTPS everywhere now with a limited exception for local development
    is_local_development = os.environ.get('LOCAL_DEVELOPMENT') == 'true'
    if is_local_development:
        logger.debug("Local development mode detected")
    
    talisman = Talisman(
        app,
        content_security_policy=CSP_POLICY,
        content_security_policy_nonce_in=['script-src'],
        force_https=True,  # Force HTTPS by default
        force_https_permanent=True,
        force_file_save=True,
        frame_options='SAMEORIGIN',  # Allow only same-origin embedding
        frame_options_allow_from=None,
        strict_transport_security=True,  # Always enable HSTS
        strict_transport_security_preload=True,
        strict_transport_security_max_age=31536000,  # 1 year
        referrer_policy='no-referrer',
        session_cookie_secure=True,  # Always use secure cookies
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
        
        # Special handling for api endpoints that need to be accessible
        api_endpoints = request.path.startswith('/api/')
        static_resources = request.path.startswith('/static/') or request.path.startswith('/wave_ui/')
        
        # If this is a same-origin request (no Origin header), always allow it
        if not origin:
            # No CORS headers needed for same-origin requests
            pass
        # If this is an allowed origin, add appropriate headers
        elif allowed_origins and origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key, Authorization'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Vary'] = 'Origin'  # Vary header is important for caching
        # Specific allowances for API endpoints and static resources
        elif api_endpoints or static_resources:
            # For API and static resources, allow cross-origin access but not credentials
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key, Authorization'
            response.headers['Vary'] = 'Origin'
        # In development mode, be more permissive
        elif app.config['IS_DEVELOPMENT']:
            # In development mode only, allow all origins for easier testing
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key, Authorization'
        
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Only allow same-origin iframe embedding
        response.headers['X-Content-Type-Options'] = 'nosniff'  # Prevent MIME type sniffing
        return response
    
    return talisman, limiter