"""
Quantonium OS - API Utility Functions

Includes:
- API key validation
- SHA-256 response signing
- Timestamp injection
"""

import hashlib
import os
import time

from flask import jsonify, request

API_KEY_HEADER = "X-API-Key"
API_SECRET = os.environ.get("QUANTONIUM_API_KEY", "default_dev_key")  # Replit Secrets


def validate_api_key(req: request) -> bool:
    """
    Check if the API key in the request headers matches the expected secret.

    Returns:
        True if valid, False otherwise.
    """
    key = req.headers.get(API_KEY_HEADER)
    return key == API_SECRET


def sign_response(payload: dict) -> dict:
    """
    Add timestamp and SHA-256 hash to the response payload.

    Args:
        payload: JSON-compatible dictionary

    Returns:
        Signed dictionary with timestamp and hash
    """
    timestamp = int(time.time())
    payload["timestamp"] = timestamp

    raw = str(sorted(payload.items())).encode()
    hash_digest = hashlib.sha256(raw).hexdigest()

    payload["sha256"] = hash_digest
    return payload


def reject_unauthorized():
    return jsonify({"error": "Unauthorized", "code": 403}), 403
