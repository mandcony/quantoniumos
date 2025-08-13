"""
QuantoniumOS - Quantum-resistant cryptographic operating system.

This package provides quantum-resistant encryption, formal verification,
and cryptographic security utilities.
"""

__version__ = "0.1.0"
__all__ = ["core", "security", "utils", "auth", "api"]

# Ensure proper namespace package structure
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
