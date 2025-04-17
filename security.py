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

# Default CSP policy - restrict to same origin by default
CSP_POLICY = {
    'default-src': "'self'",
    'img-src': ["'self'", 'https:'],
    'script-src': ["'self'"],
    'style-src': ["'self'"],
    'font-src': ["'self'"],
    'connect-src': ["'self'"],
    'worker-src': ["'self'"],
    'frame-ancestors': "'none'",
    'form-action': "'self'",
    'base-uri': "'self'",
    'object-src': "'none'"
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
    
    # Set up Talisman for HTTPS, HSTS, and CSP
    talisman = Talisman(
        app,
        content_security_policy=CSP_POLICY,
        content_security_policy_nonce_in=['script-src'],
        force_https=not is_development,  # Only force HTTPS in production
        force_https_permanent=not is_development,
        force_file_save=True,
        frame_options='DENY',
        frame_options_allow_from=None,
        strict_transport_security=not is_development,  # Only enable HSTS in production
        strict_transport_security_preload=not is_development,
        strict_transport_security_max_age=31536000,
        referrer_policy='no-referrer',
        session_cookie_secure=not is_development,  # Only secure cookies in production
        session_cookie_http_only=True
    )
    
    # Set up rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["60 per minute"],
        storage_uri="memory://",
        strategy="fixed-window"
    )
    
    # Add CORS checking to before_request
    @app.before_request
    def before_request():
        cors_check()
    
    # Add permissions policy header to all responses
    @app.after_request
    def set_additional_headers(response):
        response.headers['Permissions-Policy'] = 'interest-cohort=()'
        return response
    
    return talisman, limiter