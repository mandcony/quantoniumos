"""
QuantoniumOS Auth Package

Authentication and authorization components for QuantoniumOS.
Contains user management, token handling, and security features.
"""

__version__ = "1.0.0"

# Import core components
from .database import db, initialize_auth
from .routes import auth_api

__all__ = ['db', 'initialize_auth', 'auth_api']
