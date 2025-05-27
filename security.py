"""
Quantonium OS - Security Configuration

Implements security middleware, headers, and protection mechanisms.
"""

import os
import platform
from flask import request, abort, Flask, g, jsonify
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

# Import enhanced security logging
from utils.security_logger import (
    log_security_event, SecurityEventType, SecurityOutcome,
    log_rate_limit_exceeded, log_suspicious_activity
)

# Configure logger
logger = logging.getLogger("quantonium_security")
logger.setLevel(logging.INFO)

# CSP policy that allows iframe embedding from the same site
CSP_POLICY = {
    'frame-ancestors': ["'self'"]  # Allow same-origin embedding
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
        # Log security event with enhanced context
        log_security_event(
            event_type=SecurityEventType.ACCESS_DENIED,
            message=f"Rejected cross-origin request from {origin} to {request.path}",
            outcome=SecurityOutcome.BLOCKED,
            level=logging.WARNING,
            target_resource=request.path,
            metadata={
                "origin": origin,
                "ip_address": request.remote_addr,
                "allowed_origins": allowed_origins,
                "user_agent": request.headers.get('User-Agent', 'Unknown')
            },
            security_labels=["cors-violation"]
        )
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
        frame_options=None,  # Disable X-Frame-Options to allow embedding
        frame_options_allow_from='*',
        strict_transport_security=True,  # Always enable HSTS
        strict_transport_security_preload=True,
        strict_transport_security_max_age=31536000,  # 1 year
        referrer_policy='no-referrer',
        session_cookie_secure=True,  # Always use secure cookies
        session_cookie_http_only=True
    )
    
    # Set up rate limiting with Redis for distributed systems
    redis_url = os.environ.get('REDIS_URL')
    
    # Configure more detailed Redis options if available
    if redis_url:
        logger.info("Configuring Flask-Limiter with Redis backend")
        storage_uri = redis_url
        storage_options = {
            "connection_pool": True,      # Use connection pooling
            "socket_connect_timeout": 3,  # Timeout for initial connection
            "socket_timeout": 3,          # Timeout for operations
            "health_check_interval": 30   # Check health of connection periodically
        }
    else:
        logger.warning("Redis URL not found. Using memory storage for rate limits - will not work across multiple instances.")
        storage_uri = "memory://"
        storage_options = {}
    
    # More strict rate limits in production, more lenient in development
    if is_development:
        default_limits = ["300 per minute", "10000 per day"]
    else:
        default_limits = ["120 per minute", "3000 per day"]
    
    # Create a simple limiter with basic configuration
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=default_limits
    )

    # Register a custom error handler for 429 errors
    @app.errorhandler(429)
    def ratelimit_error_handler(e):
        """Custom rate limit exceeded handler with enhanced security logging"""
        # Log the rate limit event with detailed context
        log_rate_limit_exceeded(
            client_ip=request.remote_addr,
            resource=request.path,
            message=f"Rate limit exceeded: {str(e)}",
            metadata={
                "user_agent": request.headers.get('User-Agent', 'Unknown'),
                "method": request.method,
                "referrer": request.referrer,
                "endpoint": request.endpoint
            }
        )
        
        # Create a custom JSON response
        response = jsonify({
            "error": "Rate limit exceeded",
            "message": str(e),
            "status_code": 429,
            "retry_after": 60
        })
        response.status_code = 429
        
        # Add Retry-After header to help legitimate clients
        response.headers['Retry-After'] = '60'
            
        return response
    
    # Apply API limits with decorators rather than direct route application
    # These will be used when routes are registered
    encryption_limit = limiter.shared_limit("60 per minute", scope="encryption_endpoints")
    quantum_limit = limiter.shared_limit("30 per minute", scope="quantum_endpoints")
    
    # Note: The following are just definitions that will be used by route decorators
    # Exempt status and health endpoints that should never be rate limited
    app.config['RATELIMIT_EXEMPT_ROUTES'] = [
        '/api/health',
        '/status',
        '/api/stream/wave'  # Streaming endpoints should not be rate limited
    ]
    
    # Add CORS checking and suspicious path filtering to before_request
    @app.before_request
    def before_request():
        # Block suspicious scanning attempts
        suspicious_paths = [
            '/wp-admin/', '/wordpress/', '/wp-content/', '/wp-includes/',
            '/admin/', '/administrator/', '/phpmyadmin/', '/phpMyAdmin/',
            '/config.php', '/setup.php', '/install.php', '/wp-config.php',
            '/.env', '/.git/', '/backup/', '/tmp/', '/temp/',
            '/cgi-bin/', '/scripts/', '/HNAP1/', '/boaform/',
            '/vendor/', '/node_modules/', '/uploads/', '/files/'
        ]
        
        request_path = request.path.lower()
        for suspicious_path in suspicious_paths:
            if suspicious_path in request_path:
                # Log the suspicious activity
                log_suspicious_activity(
                    activity_type="path_scanning",
                    client_ip=request.remote_addr or "unknown",
                    resource=request.path,
                    message=f"Blocked suspicious path scan: {request.path}",
                    metadata={
                        "user_agent": request.headers.get('User-Agent', 'Unknown'),
                        "method": request.method,
                        "referrer": request.referrer,
                        "matched_pattern": suspicious_path
                    }
                )
                abort(404)  # Return 404 instead of 403 to not reveal blocking
        
        # Block requests with suspicious query parameters
        suspicious_params = ['eval', 'exec', 'system', 'passthru', 'shell_exec', 'cmd', 'wget', 'curl']
        query_string = request.query_string.decode('utf-8', errors='ignore').lower()
        for param in suspicious_params:
            if param in query_string:
                log_suspicious_activity(
                    activity_type="malicious_parameters",
                    client_ip=request.remote_addr or "unknown",
                    resource=request.path,
                    message=f"Blocked request with suspicious parameter: {param}",
                    metadata={
                        "user_agent": request.headers.get('User-Agent', 'Unknown'),
                        "query_string": request.query_string.decode('utf-8', errors='ignore'),
                        "matched_parameter": param
                    }
                )
                abort(404)
        
        # Check for suspicious referrer patterns
        referrer = request.referrer or ''
        suspicious_referrers = [
            '.google.fr', '.google.com.fake', '.bing.fake', '.yahoo.fake',
            'malicious-site.com', 'phishing-site.org'
        ]
        
        # Also check for fake search engine user agents
        user_agent = request.headers.get('User-Agent', '').lower()
        fake_bot_patterns = [
            'googlebot' if 'google' not in referrer.lower() else '',
            'bingbot' if 'bing' not in referrer.lower() else '',
            'yandexbot' if 'yandex' not in referrer.lower() else ''
        ]
        
        for suspicious_ref in suspicious_referrers:
            if suspicious_ref in referrer.lower():
                log_suspicious_activity(
                    activity_type="suspicious_referrer",
                    client_ip=request.remote_addr or "unknown",
                    resource=request.path,
                    message=f"Blocked request with suspicious referrer: {referrer}",
                    metadata={
                        "user_agent": user_agent,
                        "referrer": referrer,
                        "matched_pattern": suspicious_ref
                    }
                )
                abort(404)
        
        # Check for bot impersonation
        for bot_pattern in fake_bot_patterns:
            if bot_pattern and bot_pattern in user_agent:
                log_suspicious_activity(
                    activity_type="bot_impersonation",
                    client_ip=request.remote_addr or "unknown",
                    resource=request.path,
                    message=f"Blocked fake bot impersonation: {bot_pattern}",
                    metadata={
                        "user_agent": user_agent,
                        "referrer": referrer,
                        "impersonated_bot": bot_pattern
                    }
                )
                abort(404)
        
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
        
        # Remove X-Frame-Options header to allow embedding in iframes
        # response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-Content-Type-Options'] = 'nosniff'  # Prevent MIME type sniffing
        return response
    
    return talisman, limiter