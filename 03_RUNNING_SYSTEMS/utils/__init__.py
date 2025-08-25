"""
QuantoniumOS Utils Package

Utility functions and helper modules for QuantoniumOS
running systems and applications.
"""

__version__ = "1.0.0"

from .json_logger import setup_json_logger
from .security_logger import setup_security_logger

__all__ = ['setup_json_logger', 'setup_security_logger']
