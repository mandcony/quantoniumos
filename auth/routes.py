"""
Quantonium OS - Auth API Routes

API endpoints for key management and authentication.
"""

from flask import Blueprint, request, jsonify, g
from auth.models import APIKey, APIKeyAuditLog
from auth.jwt_auth import (
    require_jwt_auth, 
    require_admin, 
    get_current_api_key,
    authenticate_key
)
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

# Create blueprint
auth_api = Blueprint("auth_api", __name__, url_prefix="/api/auth")

# Request models
class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    expires_in_days: Optional[int] = None
    permissions: str = Field("api:read api:write", min_length=1)
    is_admin: bool = False

class RevokeKeyRequest(BaseModel):
    key_id: str = Field(..., min_length=1)
    reason: Optional[str] = None

# Response models
class APIKeyResponse(BaseModel):
    key_id: str
    name: str
    description: Optional[str]
    is_admin: bool
    is_active: bool
    revoked: bool
    permissions: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    use_count: int

class CreateKeyResponse(BaseModel):
    key: APIKeyResponse
    api_key: str
    token: str
    message: str

class TokenResponse(BaseModel):
    token: str
    expires_in: int
    token_type: str = "Bearer"
    key_id: str
    permissions: str
    is_admin: bool
    
# API Routes

@auth_api.route("/token", methods=["POST"])
def create_token():
    """
    Create a JWT token from an API key
    
    This allows clients to get a short-lived token for subsequent
    requests without sending the API key each time.
    """
    # Get the API key from header
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        return jsonify({
            "error": "Unauthorized",
            "message": "Missing X-API-Key header",
            "code": 401
        }), 401
    
    # Authenticate the key
    key = authenticate_key(api_key)
    
    if not key:
        return jsonify({
            "error": "Unauthorized",
            "message": "Invalid API key",
            "code": 401
        }), 401
    
    # Create a token
    token = key.create_token(expiry_minutes=60)
    
    # Log token creation
    APIKeyAuditLog.log(
        key,
        "token_created",
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string if request.user_agent else None
    )
    
    # Return the token
    return jsonify({
        "token": token,
        "expires_in": 3600,  # 60 minutes in seconds
        "token_type": "Bearer",
        "key_id": key.key_id,
        "permissions": key.permissions,
        "is_admin": key.is_admin
    })

@auth_api.route("/keys", methods=["GET"])
@require_jwt_auth
@require_admin
def list_keys():
    """
    List all API keys (admin only)
    """
    # Get all keys
    keys = APIKey.query.all()
    
    # Convert to response format
    result = []
    for key in keys:
        result.append({
            "key_id": key.key_id,
            "name": key.name,
            "description": key.description,
            "is_admin": key.is_admin,
            "is_active": key.is_active,
            "revoked": key.revoked,
            "permissions": key.permissions,
            "created_at": key.created_at.isoformat() if key.created_at else None,
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
            "use_count": key.use_count
        })
    
    return jsonify({"keys": result})

@auth_api.route("/keys", methods=["POST"])
@require_jwt_auth
@require_admin
def create_key():
    """
    Create a new API key (admin only)
    """
    # Parse and validate request
    data = CreateKeyRequest(**request.get_json())
    
    # Create key
    key, raw_key = APIKey.create(
        name=data.name,
        description=data.description,
        expires_in_days=data.expires_in_days,
        permissions=data.permissions,
        is_admin=data.is_admin
    )
    
    # Log key creation
    admin_key = get_current_api_key()
    APIKeyAuditLog.log(
        key,
        "created",
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string if request.user_agent else None,
        details=f"Created by admin key {admin_key.key_id}"
    )
    
    # Generate a token for immediate use
    token = key.create_token(expiry_minutes=60)
    
    # Build response
    response = {
        "key": {
            "key_id": key.key_id,
            "name": key.name,
            "description": key.description,
            "is_admin": key.is_admin,
            "is_active": key.is_active,
            "revoked": key.revoked,
            "permissions": key.permissions,
            "created_at": key.created_at.isoformat() if key.created_at else None,
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
            "use_count": key.use_count
        },
        "api_key": raw_key,
        "token": token,
        "message": "API key created successfully. Save the complete key now, it will not be displayed again."
    }
    
    return jsonify(response), 201

@auth_api.route("/keys/<key_id>/revoke", methods=["POST"])
@require_jwt_auth
@require_admin
def revoke_key(key_id):
    """
    Revoke an API key (admin only)
    """
    # Find the key
    key = APIKey.query.filter_by(key_id=key_id).first()
    
    if not key:
        return jsonify({
            "error": "Not Found",
            "message": f"No API key found with ID {key_id}",
            "code": 404
        }), 404
    
    # Can't revoke own key
    admin_key = get_current_api_key()
    if key.key_id == admin_key.key_id:
        return jsonify({
            "error": "Forbidden",
            "message": "Cannot revoke your own API key",
            "code": 403
        }), 403
    
    # Parse request if any
    reason = None
    if request.is_json:
        data = RevokeKeyRequest(**request.get_json())
        reason = data.reason
    
    # Revoke the key
    key.revoke()
    
    # Log key revocation
    APIKeyAuditLog.log(
        key,
        "revoked",
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string if request.user_agent else None,
        details=f"Revoked by admin key {admin_key.key_id}" + (f": {reason}" if reason else "")
    )
    
    return jsonify({
        "message": f"API key {key_id} revoked successfully",
        "key_id": key.key_id,
        "revoked_at": key.revoked_at.isoformat()
    })

@auth_api.route("/keys/<key_id>/rotate", methods=["POST"])
@require_jwt_auth
@require_admin
def rotate_key(key_id):
    """
    Rotate an API key (admin only)
    
    Creates a new key with the same permissions, deactivates the old one.
    """
    # Find the key
    key = APIKey.query.filter_by(key_id=key_id).first()
    
    if not key:
        return jsonify({
            "error": "Not Found",
            "message": f"No API key found with ID {key_id}",
            "code": 404
        }), 404
    
    # Rotate the key
    new_key, raw_key = key.rotate()
    
    # Log key rotation
    admin_key = get_current_api_key()
    APIKeyAuditLog.log(
        key,
        "rotated",
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string if request.user_agent else None,
        details=f"Rotated to {new_key.key_id} by admin key {admin_key.key_id}"
    )
    
    # Generate a token for immediate use
    token = new_key.create_token(expiry_minutes=60)
    
    # Build response
    response = {
        "message": f"API key {key_id} rotated successfully",
        "old_key_id": key.key_id,
        "new_key": {
            "key_id": new_key.key_id,
            "name": new_key.name,
            "description": new_key.description,
            "is_admin": new_key.is_admin,
            "is_active": new_key.is_active,
            "revoked": new_key.revoked,
            "permissions": new_key.permissions,
            "created_at": new_key.created_at.isoformat() if new_key.created_at else None,
            "expires_at": new_key.expires_at.isoformat() if new_key.expires_at else None
        },
        "api_key": raw_key,
        "token": token
    }
    
    return jsonify(response)

@auth_api.route("/profile", methods=["GET"])
@require_jwt_auth
def get_profile():
    """
    Get the profile of the current API key
    """
    key = get_current_api_key()
    
    return jsonify({
        "key_id": key.key_id,
        "name": key.name,
        "permissions": key.permissions,
        "is_admin": key.is_admin,
        "created_at": key.created_at.isoformat() if key.created_at else None,
        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
        "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
        "use_count": key.use_count
    })