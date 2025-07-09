"""
Quantonium OS - JWT Authentication Module

Implements JWT-based auth with API keys for the Quantonium API.
"""

import jwt
from functools import wraps
from flask import request, g, jsonify
from werkzeug.security import check_password_hash
from auth.models import APIKey, APIKeyAuditLog

# Authentication headers
API_KEY_HEADER = "X-API-Key"
JWT_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "

def authenticate_key(api_key):
    """
    Authenticate an API key and return the corresponding APIKey object
    
    Args:
        api_key: The raw API key string
        
    Returns:
        The APIKey object if valid, otherwise None
    """
    # Fast fail if no key
    if not api_key:
        return None
    
    # Extract prefix to find the right key record
    try:
        prefix = api_key.split('.')[0]
    except (AttributeError, IndexError):
        return None
    
    # Query for keys with matching prefix
    possible_keys = APIKey.query.filter_by(key_prefix=prefix, is_active=True, revoked=False).all()
    
    # Check each possible key
    for key in possible_keys:
        if key.verify_key(api_key):
            # Update usage stats
            key.update_last_used()
            return key
    
    return None

def get_token_from_request():
    """Extract JWT token from Authorization header"""
    auth_header = request.headers.get(JWT_HEADER)
    
    if auth_header and auth_header.startswith(BEARER_PREFIX):
        return auth_header[len(BEARER_PREFIX):]
    
    return None

def get_api_key_from_request():
    """Extract API key from X-API-Key header"""
    return request.headers.get(API_KEY_HEADER)

def verify_token(token):
    """
    Verify a JWT token and return the corresponding APIKey
    
    Args:
        token: JWT token string
        
    Returns:
        Tuple of (APIKey object, decoded token payload) if valid,
        otherwise (None, None)
    """
    if not token:
        return None, None
    
    try:
        # Decode without verification to get the key ID
        unverified = jwt.decode(token, options={"verify_signature": False})
        key_id = unverified.get('sub')
        
        if not key_id:
            return None, None
        
        # Find the key
        key = APIKey.query.filter_by(key_id=key_id, is_active=True).first()
        
        if not key:
            return None, None
        
        # Verify with the key's secret, ensuring kid matches
        payload = key.verify_token(token, required_kid=key_id)
        
        if not payload:
            return None, None
        
        # Update usage stats
        key.update_last_used()
        
        return key, payload
        
    except jwt.PyJWTError:
        return None, None

def get_current_api_key():
    """
    Get the authenticated API key from the current request
    
    First tries JWT token, then falls back to API key header
    Stores the key in g.api_key for access in views
    
    Returns:
        The authenticated APIKey object or None
    """
    # Check if we've already authenticated
    if hasattr(g, 'api_key'):
        return g.api_key
    
    # Try JWT token first
    token = get_token_from_request()
    
    if token:
        key, payload = verify_token(token)
        
        if key:
            # Store in g for later access
            g.api_key = key
            g.jwt_payload = payload
            
            # Create audit log
            APIKeyAuditLog.log(
                key,
                'api_request',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string if request.user_agent else None,
                request_path=request.path,
                details=f"Method: {request.method}"
            )
            
            return key
    
    # Fall back to API key header
    api_key = get_api_key_from_request()
    
    if api_key:
        key = authenticate_key(api_key)
        
        if key:
            # Store in g for later access
            g.api_key = key
            
            # Create audit log
            APIKeyAuditLog.log(
                key,
                'api_request',
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string if request.user_agent else None,
                request_path=request.path,
                details=f"Method: {request.method}"
            )
            
            return key
    
    return None

def require_jwt_auth(f):
    """
    Decorator to require JWT authentication
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        key = get_current_api_key()
        
        if not key:
            return jsonify({
                "error": "Unauthorized",
                "message": "Valid API authentication required",
                "code": 401
            }), 401
        
        return f(*args, **kwargs)
    
    return decorated

def require_permission(permission):
    """
    Decorator to require a specific permission
    Must be used after @require_jwt_auth
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            key = get_current_api_key()
            
            if not key or not key.has_permission(permission):
                return jsonify({
                    "error": "Forbidden",
                    "message": f"Missing required permission: {permission}",
                    "code": 403
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated
    
    return decorator

def require_admin(f):
    """
    Decorator to require admin privileges
    Must be used after @require_jwt_auth
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        key = get_current_api_key()
        
        if not key or not key.is_admin:
            return jsonify({
                "error": "Forbidden",
                "message": "Admin privileges required",
                "code": 403
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated