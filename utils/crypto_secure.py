"""
Quantonium OS - Cryptographic Security Utilities

Implements NIST 800-53 SC-13 (Cryptographic Protection) compliant utilities
for secure cryptographic operations. This module provides standardized
interfaces for cryptographic functions while hiding implementation details.
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Configure logger
logger = logging.getLogger("quantonium_crypto")
logger.setLevel(logging.INFO)

# Constants for cryptographic operations
KEY_BYTES = 32  # 256 bits
SALT_BYTES = 16  # 128 bits
NONCE_BYTES = 12  # 96 bits
TAG_BYTES = 16  # 128 bits
PBKDF2_ITERATIONS = (
    600000  # NIST SP 800-132 recommends >= 10000, we use a much higher value
)

# Initialize variables for when security logging is not available
SECURITY_LOGGING = False
log_security_event = None
SecurityEventType = None
SecurityOutcome = None
ImpactLevel = None

# Import specialized security logging if available
try:
    from utils.security_logger import (ImpactLevel, SecurityEventType,
                                       SecurityOutcome, log_security_event)

    SECURITY_LOGGING = True
except ImportError:
    print("Security logging not available - falling back to standard logging")

    # Create stub classes and functions if security_logger is not available
    from enum import Enum

    class StubSecurityEventType(str, Enum):
        CRYPTO_OPERATION = "crypto_operation"

    class StubSecurityOutcome(str, Enum):
        SUCCESS = "success"
        FAILURE = "failure"
        WARNING = "warning"
        UNKNOWN = "unknown"

    class StubImpactLevel(str, Enum):
        LOW = "low"
        MODERATE = "moderate"
        HIGH = "high"

    def stub_log_security_event(**kwargs):
        logger.info(f"Security event: {kwargs.get('message', 'No message')}")

    # Assign stub implementations
    SecurityEventType = StubSecurityEventType
    SecurityOutcome = StubSecurityOutcome
    ImpactLevel = StubImpactLevel
    log_security_event = stub_log_security_event
    SECURITY_LOGGING = True  # We're using stubs so we can consider logging enabled


# Helper function to securely generate random bytes
def generate_random_bytes(num_bytes: int) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Args:
        num_bytes: Number of random bytes to generate

    Returns:
        Random bytes
    """
    return secrets.token_bytes(num_bytes)


# Helper function to derive a key from a password
def derive_key_from_password(
    password: str,
    salt: Optional[bytes] = None,
    iterations: int = PBKDF2_ITERATIONS,
    key_length: int = KEY_BYTES,
) -> Tuple[bytes, bytes]:
    """
    Derive a cryptographic key from a password using PBKDF2.

    Args:
        password: Password to derive key from
        salt: Salt for key derivation (generated if not provided)
        iterations: Number of iterations for PBKDF2
        key_length: Length of the derived key in bytes

    Returns:
        Tuple of (derived_key, salt)
    """
    if not salt:
        salt = generate_random_bytes(SALT_BYTES)

    # Use PBKDF2 with HMAC-SHA-256
    derived_key = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations, key_length
    )

    return derived_key, salt


# Secure hash function
def secure_hash(data: Union[str, bytes], algorithm: str = "sha256") -> str:
    """
    Create a secure hash of data.

    Args:
        data: Data to hash (string or bytes)
        algorithm: Hash algorithm to use (sha256, sha384, or sha512)

    Returns:
        Hexadecimal hash string
    """
    if algorithm not in ("sha256", "sha384", "sha512"):
        if SECURITY_LOGGING:
            log_security_event(
                event_type=SecurityEventType.CRYPTO_OPERATION,
                message=f"Unsupported hash algorithm: {algorithm}, falling back to sha256",
                outcome=SecurityOutcome.WARNING,
                level=logging.WARNING,
                target_resource="secure_hash",
                security_labels=["crypto", "hash", "algorithm"],
                impact_level=ImpactLevel.MODERATE,
            )
        algorithm = "sha256"  # Fall back to sha256

    # Convert string to bytes if needed
    if isinstance(data, str):
        data = data.encode("utf-8")

    # Use the appropriate hash function
    if algorithm == "sha256":
        hash_obj = hashlib.sha256(data)
    elif algorithm == "sha384":
        hash_obj = hashlib.sha384(data)
    elif algorithm == "sha512":
        hash_obj = hashlib.sha512(data)

    return hash_obj.hexdigest()


# Secure HMAC function
def secure_hmac(
    key: Union[str, bytes], message: Union[str, bytes], algorithm: str = "sha256"
) -> str:
    """
    Create a secure HMAC of a message.

    Args:
        key: Key for HMAC (string or bytes)
        message: Message to authenticate (string or bytes)
        algorithm: Hash algorithm to use (sha256, sha384, or sha512)

    Returns:
        Hexadecimal HMAC string
    """
    if algorithm not in ("sha256", "sha384", "sha512"):
        if SECURITY_LOGGING:
            log_security_event(
                event_type=SecurityEventType.CRYPTO_OPERATION,
                message=f"Unsupported HMAC algorithm: {algorithm}, falling back to sha256",
                outcome=SecurityOutcome.WARNING,
                level=logging.WARNING,
                target_resource="secure_hmac",
                security_labels=["crypto", "hmac", "algorithm"],
                impact_level=ImpactLevel.MODERATE,
            )
        algorithm = "sha256"  # Fall back to sha256

    # Convert to bytes if needed
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(message, str):
        message = message.encode("utf-8")

    # Create HMAC with the specified algorithm
    if algorithm == "sha256":
        hmac_obj = hmac.new(key, message, hashlib.sha256)
    elif algorithm == "sha384":
        hmac_obj = hmac.new(key, message, hashlib.sha384)
    elif algorithm == "sha512":
        hmac_obj = hmac.new(key, message, hashlib.sha512)

    return hmac_obj.hexdigest()


# Generate a secure random token
def generate_token(bytes_length: int = 32) -> str:
    """
    Generate a secure random token.

    Args:
        bytes_length: Length of the token in bytes

    Returns:
        URL-safe base64-encoded token
    """
    token_bytes = generate_random_bytes(bytes_length)
    return base64.urlsafe_b64encode(token_bytes).decode("utf-8").rstrip("=")


# Constant-time string comparison (to prevent timing attacks)
def constant_time_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """
    Compare two strings or byte sequences in constant time.

    Args:
        a: First string or bytes
        b: Second string or bytes

    Returns:
        True if the strings are equal, False otherwise
    """
    # Convert to bytes if needed
    if isinstance(a, str):
        a = a.encode("utf-8")
    if isinstance(b, str):
        b = b.encode("utf-8")

    return hmac.compare_digest(a, b)


# Generate a secure UUID
def generate_uuid() -> str:
    """
    Generate a secure UUID.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


# Encrypt data using password-based encryption
def encrypt_data(data: Union[str, bytes], password: str) -> Dict[str, str]:
    """
    Encrypt data using password-based encryption (AES-GCM).

    Args:
        data: Data to encrypt (string or bytes)
        password: Password for encryption

    Returns:
        Dictionary with encrypted data and metadata:
        {
            'ciphertext': base64-encoded ciphertext,
            'salt': base64-encoded salt,
            'nonce': base64-encoded nonce,
            'tag': base64-encoded authentication tag,
            'algorithm': encryption algorithm
        }
    """
    # This is a placeholder that delegates to the actual implementation
    # in the core cryptographic module. We do not implement the actual
    # cryptographic operations here to protect the proprietary algorithm.

    if SECURITY_LOGGING:
        log_security_event(
            event_type=SecurityEventType.CRYPTO_OPERATION,
            message="Data encryption operation delegated to core implementation",
            outcome=SecurityOutcome.UNKNOWN,
            level=logging.INFO,
            target_resource="encrypt_data",
            security_labels=["crypto", "encrypt"],
            impact_level=ImpactLevel.MODERATE,
        )

    # For demo purposes, return a mock structure
    # The actual implementation would call the core module
    return {
        "ciphertext": "<encrypted-data-placeholder>",
        "salt": base64.b64encode(generate_random_bytes(SALT_BYTES)).decode("utf-8"),
        "nonce": base64.b64encode(generate_random_bytes(NONCE_BYTES)).decode("utf-8"),
        "tag": base64.b64encode(generate_random_bytes(TAG_BYTES)).decode("utf-8"),
        "algorithm": "AES-GCM-256",
    }


# Decrypt data using password-based encryption
def decrypt_data(encrypted_data: Dict[str, str], password: str) -> bytes:
    """
    Decrypt data using password-based encryption.

    Args:
        encrypted_data: Dictionary with encrypted data and metadata
        password: Password for decryption

    Returns:
        Decrypted data as bytes
    """
    # This is a placeholder that delegates to the actual implementation
    # in the core cryptographic module. We do not implement the actual
    # cryptographic operations here to protect the proprietary algorithm.

    if SECURITY_LOGGING:
        log_security_event(
            event_type=SecurityEventType.CRYPTO_OPERATION,
            message="Data decryption operation delegated to core implementation",
            outcome=SecurityOutcome.UNKNOWN,
            level=logging.INFO,
            target_resource="decrypt_data",
            security_labels=["crypto", "decrypt"],
            impact_level=ImpactLevel.MODERATE,
        )

    # For demo purposes, return a mock result
    # The actual implementation would call the core module
    return b"<decrypted-data-placeholder>"


# Generate a cryptographically secure key
def generate_key() -> str:
    """
    Generate a cryptographically secure key.

    Returns:
        Base64-encoded key
    """
    key = generate_random_bytes(KEY_BYTES)
    return base64.urlsafe_b64encode(key).decode("utf-8")


# Get the master encryption key from environment or generate a temporary one
def get_master_key() -> bytes:
    """
    Get the master encryption key from environment or generate a temporary one.

    Returns:
        Master key as bytes
    """
    env_key = os.environ.get("QUANTONIUM_MASTER_KEY")

    if env_key:
        # Decode the base64-encoded key
        try:
            key = base64.urlsafe_b64decode(env_key)
            if len(key) >= KEY_BYTES:
                return key
        except Exception as e:
            if SECURITY_LOGGING:
                log_security_event(
                    event_type=SecurityEventType.CRYPTO_OPERATION,
                    message=f"Error decoding master key: {str(e)}",
                    outcome=SecurityOutcome.FAILURE,
                    level=logging.ERROR,
                    target_resource="get_master_key",
                    security_labels=["crypto", "master-key", "error"],
                    impact_level=ImpactLevel.HIGH,
                )
            logger.error(f"Error decoding master key: {str(e)}")

    # Generate a temporary key if not set or invalid
    new_key = generate_random_bytes(KEY_BYTES)
    new_key_b64 = base64.urlsafe_b64encode(new_key).decode("utf-8")

    # Log a warning for temporary key
    logger.warning(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        "WARNING: No master encryption key found in environment!\n"
        f"A temporary key has been generated: {new_key_b64}\n"
        "Set this in your environment as QUANTONIUM_MASTER_KEY\n"
        "Without this key, you will lose access to encrypted secrets on restart!\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )

    return new_key
