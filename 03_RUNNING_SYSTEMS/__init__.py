"""
QuantoniumOS Running Systems Package
===================================

This package contains the operational web server and application components
for QuantoniumOS, including:

- Flask web application (main.py)
- Authentication system (auth/)
- API routes and quantum endpoints (routes/, routes_quantum.py)
- Security middleware and audit systems (security.py, middleware/)
- Configuration management (env_loader.py)
- Logging utilities (utils/)

The system provides:
- RESTful API endpoints for quantum operations
- True RFT quantum computing interface
- User authentication and session management
- Rate limiting and security monitoring
- JSON-structured logging

Usage:
    python main.py  # Start web server on port 8080
"""

__version__ = "1.0.0"

# Core exports
from .auth import db, initialize_auth
from .routes import api, encrypt, decrypt
from .routes_quantum import quantum_api
from .security import configure_security
from .env_loader import config

__all__ = [
    'db', 'initialize_auth', 'api', 'encrypt', 'decrypt', 
    'quantum_api', 'configure_security', 'config'
]
