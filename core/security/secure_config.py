"""
Secure configuration management for QuantoniumOS.

This module provides production-safe configuration that eliminates dangerous
entropy bypasses and debug hooks in production builds.
"""

import os

class SecureConfig:
    """Configuration manager that prevents dangerous entropy bypasses in production."""

    @staticmethod
    def is_debug_build() -> bool:
        """Check if this is a debug build."""
        return __debug__

    @staticmethod
    def allow_test_entropy() -> bool:
        """Only allow test entropy in debug builds with explicit environment variable."""
        if not __debug__:
            return False  # Never allow in production builds
        return os.environ.get('QUANTONIUM_ALLOW_TEST_ENTROPY', '').lower() == 'true'

    @staticmethod
    def get_entropy_source() -> str:
        """Get the entropy source, always defaulting to secure system entropy in production."""
        if SecureConfig.allow_test_entropy():
            test_source = os.environ.get('QUANTONIUM_ENTROPY_SOURCE', 'system')
            if test_source in ['system', 'test']:
                return test_source
        return 'system'  # Always system entropy in production or invalid test source

    @staticmethod
    def get_crypto_debug_level() -> int:
        """Get cryptographic debugging level (0 in production)."""
        if not __debug__:
            return 0  # No debug output in production builds
        level = os.environ.get('QUANTONIUM_CRYPTO_DEBUG', '0')
        try:
            return max(0, min(3, int(level)))  # Clamp to valid range
        except ValueError:
            return 0

    @staticmethod
    def validate_production_safety() -> bool:
        """Validate that configuration is safe for production use."""
        if __debug__:
            return False  # Debug builds are not production-safe

        # Ensure no dangerous environment variables are set
        dangerous_vars = [
            'QUANTONIUM_ALLOW_TEST_ENTROPY',
            'QUANTONIUM_ENTROPY_SOURCE',
            'QUANTONIUM_CRYPTO_DEBUG'
        ]

        for var in dangerous_vars:
            if os.environ.get(var):
                return False

        return True

# Production safety check on import
if not __debug__ and not SecureConfig.validate_production_safety():
    raise RuntimeError("Unsafe configuration detected in production build")
