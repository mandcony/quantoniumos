"""
Quantonium OS - Authentication Module

Contains the authentication framework for the Quantonium API.
"""

__version__ = "0.3.0-rc1"

import os
import logging
import secrets

# Configure logging
logger = logging.getLogger(__name__)

def initialize_auth(app=None):
    """
    Initialize the authentication system.
    This includes setting up the master encryption key if not present.
    
    Args:
        app: Optional Flask app to configure
    """
    from auth.secret_manager import get_master_key
    
    # Check if master key exists, which will generate one if missing
    master_key = get_master_key()
    
    # If app is provided, configure it
    if app:
        # Import here to avoid circular imports
        from auth.models import db
        from auth.jwt_auth import require_jwt_auth, get_current_api_key, authenticate_key
        
        # Initialize database with app
        db.init_app(app)

# Export commonly used functions
from auth.jwt_auth import require_jwt_auth, get_current_api_key, authenticate_key
from auth.models import APIKey, APIKeyAuditLog, db