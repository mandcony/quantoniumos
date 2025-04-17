"""
Quantonium OS - Authentication Module

Contains the authentication framework for the Quantonium API.
"""

from auth.jwt_auth import require_jwt_auth, get_current_api_key, authenticate_key
from auth.models import APIKey, APIKeyAuditLog, db