"""
Quantonium OS - Key Rotation Service

This module provides functions for automated key rotation based on schedules.
"""

import logging
import os
from datetime import datetime, timedelta

from auth.models import APIKey, APIKeyAuditLog, db
from auth.secret_manager import get_keys_due_for_rotation

# Setup logging
logger = logging.getLogger(__name__)

# Default number of days before a key is rotated
DEFAULT_ROTATION_DAYS = 90


def rotate_keys_by_age(max_age_days=DEFAULT_ROTATION_DAYS):
    """
    Rotate API keys that are older than the specified age.

    Args:
        max_age_days: Maximum age of keys in days before rotation

    Returns:
        List of rotated key IDs
    """
    cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
    old_keys = APIKey.query.filter(
        APIKey.created_at < cutoff_date,
        APIKey.is_active == True,
        APIKey.revoked == False,
    ).all()

    rotated_keys = []

    for key in old_keys:
        try:
            # Rotate the key
            new_key, raw_key = key.rotate(keep_name=True)

            # Log the rotation
            APIKeyAuditLog.log(
                key,
                "rotated_by_age",
                details=f"Key rotated after {max_age_days} days. New key: {new_key.key_id}",
            )

            rotated_keys.append(key.key_id)
            logger.info(f"Rotated key {key.key_id} due to age ({max_age_days} days)")

        except Exception as e:
            logger.error(f"Failed to rotate key {key.key_id}: {str(e)}")

    return rotated_keys


def rotate_scheduled_keys():
    """
    Rotate keys that are scheduled for rotation.

    Returns:
        List of rotated key IDs
    """
    due_keys = get_keys_due_for_rotation()
    rotated_keys = []

    for key_id in due_keys:
        key = APIKey.query.filter_by(key_id=key_id).first()

        if not key:
            logger.warning(f"Scheduled key {key_id} not found, skipping rotation")
            continue

        try:
            # Rotate the key
            new_key, raw_key = key.rotate(keep_name=True)

            # Log the rotation
            APIKeyAuditLog.log(
                key,
                "rotated_by_schedule",
                details=f"Key rotated according to schedule. New key: {new_key.key_id}",
            )

            rotated_keys.append(key.key_id)
            logger.info(f"Rotated key {key.key_id} according to schedule")

        except Exception as e:
            logger.error(f"Failed to rotate scheduled key {key.key_id}: {str(e)}")

    return rotated_keys


def rotate_expired_keys():
    """
    Rotate keys that are about to expire.
    This ensures continuity of service by creating new keys before old ones expire.

    Returns:
        List of rotated key IDs
    """
    # Get keys that expire within the next 7 days
    soon = datetime.utcnow() + timedelta(days=7)
    expiring_keys = APIKey.query.filter(
        APIKey.expires_at.isnot(None),
        APIKey.expires_at < soon,
        APIKey.expires_at > datetime.utcnow(),
        APIKey.is_active == True,
        APIKey.revoked == False,
    ).all()

    rotated_keys = []

    for key in expiring_keys:
        try:
            # Calculate new expiry (same duration from now)
            original_duration = (key.expires_at - key.created_at).days

            # Rotate the key
            new_key, raw_key = key.rotate(keep_name=True)

            # Update expiry on new key
            new_key.expires_at = datetime.utcnow() + timedelta(days=original_duration)
            db.session.commit()

            # Log the rotation
            APIKeyAuditLog.log(
                key,
                "rotated_expiring",
                details=f"Key rotated due to upcoming expiration. New key: {new_key.key_id}",
            )

            rotated_keys.append(key.key_id)
            logger.info(f"Rotated key {key.key_id} due to upcoming expiration")

        except Exception as e:
            logger.error(f"Failed to rotate expiring key {key.key_id}: {str(e)}")

    return rotated_keys


def run_key_rotation():
    """
    Run all key rotation functions.

    Returns:
        Dictionary with counts of keys rotated by each method
    """
    results = {
        "by_age": len(rotate_keys_by_age()),
        "by_schedule": len(rotate_scheduled_keys()),
        "expiring": len(rotate_expired_keys()),
    }

    return results
