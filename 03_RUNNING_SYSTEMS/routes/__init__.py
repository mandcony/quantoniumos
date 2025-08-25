"""
QuantoniumOS Routes Package

Web routing and API endpoint definitions for QuantoniumOS.
Contains route handlers and API implementations.
"""

__version__ = "1.0.0"

# Import core route components
from .core import api, encrypt, decrypt, sample_entropy, entropy_stream

__all__ = ['api', 'encrypt', 'decrypt', 'sample_entropy', 'entropy_stream']
