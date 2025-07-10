"""
Quantonium OS - Authentication Models

This module defines the SQLAlchemy models for API keys and related authentication entities.
"""

import os
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import jwt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (JSON, Boolean, Column, DateTime, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from werkzeug.security import check_password_hash, generate_password_hash

from auth.secret_manager import (decrypt_secret, encrypt_secret,
                                 generate_jwt_secret)

db = SQLAlchemy()

# --- Key Management Constants ---
ACTIVE_KEY_STATUS = "active"
SUPERSEDED_KEY_STATUS = "superseded"
REVOKED_KEY_STATUS = "revoked"


class APIKey(db.Model):
    """API Key model for authenticated API access"""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    key_id = Column(
        String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    key_prefix = Column(String(8), nullable=False)
    key_hash = Column(String(256), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    # Key permissions
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    revoked = Column(Boolean, default=False)

    # Limits and scope
    rate_limit = Column(Integer, default=3600)  # Requests per hour
    permissions = Column(
        String(255), default="api:read"
    )  # Space-separated permission list

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    last_rotated_at = Column(DateTime, nullable=True)

    # Usage statistics
    use_count = Column(Integer, default=0)

    # JWT secret management for key rotation
    # Stores a list of dictionaries: [{"kid": "...", "secret": "...", "status": "...", "expires": "..."}]
    jwt_secrets = Column(JSON, nullable=False)

    # Audit log relationship - one key can have many audit logs
    audit_logs = relationship("APIKeyAuditLog", back_populates="api_key")

    @staticmethod
    def generate_key():
        """Generate a new random API key with prefix"""
        raw_key = secrets.token_urlsafe(42)
        prefix = raw_key[:8]
        return f"{prefix}.{raw_key}"

    @classmethod
    def create(
        cls,
        name,
        description=None,
        expires_in_days=None,
        permissions="api:read",
        is_admin=False,
    ):
        """
        Create a new API key with generated values

        Args:
            name: Name of the key (e.g., "Production Server")
            description: Optional description
            expires_in_days: Optional expiration in days
            permissions: Space-separated permission strings
            is_admin: Whether this key has admin privileges

        Returns:
            Tuple of (APIKey object, raw_key)
        """
        # Generate key material
        raw_key = cls.generate_key()
        prefix = raw_key.split(".")[0]

        # Hash the key for storage
        key_hash = generate_password_hash(raw_key)

        # Generate JWT secret for this key and encrypt it
        jwt_secret = generate_jwt_secret()
        encrypted_jwt_secret = encrypt_secret(jwt_secret)

        # Create the initial secret entry with a unique key ID (kid)
        kid = str(uuid.uuid4())
        secrets_list = [
            {
                "kid": kid,
                "secret": encrypted_jwt_secret,
                "status": ACTIVE_KEY_STATUS,
                "created": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            }
        ]

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create key record
        key = cls(
            key_prefix=prefix,
            key_hash=key_hash,
            name=name,
            description=description,
            is_admin=is_admin,
            permissions=permissions,
            expires_at=expires_at,
            jwt_secrets=secrets_list,
        )

        # Add and return
        db.session.add(key)
        db.session.commit()

        # Log the creation of the API key
        APIKeyAuditLog.log(api_key=key, action="created", details="API key created")

        return key, raw_key

    def verify_key(self, api_key):
        """Verify if the provided key matches this record"""
        if not self.is_active or self.revoked:
            return False

        if self.expires_at and self.expires_at < datetime.utcnow():
            return False

        return check_password_hash(self.key_hash, api_key)

    def update_last_used(self):
        """Update the last used timestamp and use count"""
        self.last_used_at = datetime.utcnow()
        self.use_count += 1
        db.session.commit()

    def revoke(self):
        """Revoke this API key"""
        self.revoked = True
        self.revoked_at = datetime.utcnow()
        self.is_active = False
        db.session.commit()

    def create_token(self, expiry_minutes=15):
        """Create a JWT token for this API key"""
        now = datetime.utcnow()
        expiry = now + timedelta(minutes=expiry_minutes)

        active_secret_info = self.get_active_secret()
        if not active_secret_info:
            raise ValueError("No active JWT secret found for this API key.")

        kid = active_secret_info["kid"]

        payload = {
            "sub": self.key_id,
            "kid": kid,  # Use the specific kid for this secret
            "prefix": self.key_prefix,
            "permissions": self.permissions,
            "is_admin": self.is_admin,
            "iat": now,
            "exp": expiry,
        }

        # Decrypt the JWT secret for use
        secret = decrypt_secret(active_secret_info["secret"])
        if not secret:
            raise ValueError("Failed to decrypt JWT secret.")

        token = jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"kid": kid},  # Explicitly set kid in header
        )

        return token

    def get_secret_by_kid(self, kid: str) -> Dict[str, Any]:
        """Find a secret dictionary by its kid."""
        for secret_info in self.jwt_secrets:
            if secret_info.get("kid") == kid:
                return secret_info
        return None

    def get_active_secret(self) -> Dict[str, Any]:
        """Get the currently active secret."""
        for secret_info in self.jwt_secrets:
            if secret_info.get("status") == ACTIVE_KEY_STATUS:
                return secret_info
        return None

    def verify_token(self, token: str, required_kid: str) -> Dict[str, Any]:
        """Verify a JWT token created by this key, ensuring the KID matches."""
        try:
            secret_info = self.get_secret_by_kid(required_kid)
            if not secret_info:
                return None  # KID from token not found for this API key

            # Decrypt the JWT secret for verification
            secret = decrypt_secret(secret_info["secret"])
            if not secret:
                return None  # Decryption failed

            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                options={"require": ["exp", "iat", "sub", "kid"]},
            )

            # Basic validation
            if payload["sub"] != self.key_id or payload["kid"] != required_kid:
                return None

            # Check if key was revoked after token was issued
            if self.revoked and self.revoked_at:
                token_iat = datetime.utcfromtimestamp(payload["iat"])
                if token_iat < self.revoked_at:
                    return None

            return payload
        except jwt.PyJWTError:
            return None

    def rotate_jwt_secret(self) -> str:
        """
        Rotates the JWT secret for this API key.
        The old key is marked as 'superseded' and a new 'active' key is created.
        Returns the KID of the new active secret.
        """
        now = datetime.utcnow()
        new_kid = str(uuid.uuid4())
        new_secret = encrypt_secret(generate_jwt_secret())

        new_secrets_list = []
        for s in self.jwt_secrets:
            if s["status"] == ACTIVE_KEY_STATUS:
                s["status"] = SUPERSEDED_KEY_STATUS
                s["expires"] = (
                    now + timedelta(days=7)
                ).isoformat()  # Grace period for old tokens
            new_secrets_list.append(s)

        new_secrets_list.append(
            {
                "kid": new_kid,
                "secret": new_secret,
                "status": ACTIVE_KEY_STATUS,
                "created": now.isoformat(),
                "expires": (now + timedelta(days=90)).isoformat(),
            }
        )

        # Prune old superseded keys
        self.jwt_secrets = [
            s for s in new_secrets_list if datetime.fromisoformat(s["expires"]) > now
        ]
        db.session.commit()
        return new_kid

    def rotate(self):
        """
        Rotate the API key by creating a new key and deactivating the old one.

        Returns:
            Tuple of (new APIKey object, new raw key)
        """
        # Generate new key material
        new_raw_key = self.generate_key()
        new_key_prefix = new_raw_key.split(".")[0]
        new_key_hash = generate_password_hash(new_raw_key)

        # Create a new JWT secret for the new key and encrypt it
        jwt_secret = generate_jwt_secret()
        encrypted_jwt_secret = encrypt_secret(jwt_secret)
        kid = str(uuid.uuid4())
        secrets_list = [
            {
                "kid": kid,
                "secret": encrypted_jwt_secret,
                "status": ACTIVE_KEY_STATUS,
                "created": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            }
        ]

        # Create new key record with same permissions and metadata
        new_key = APIKey(
            key_prefix=new_key_prefix,
            key_hash=new_key_hash,
            name=f"{self.name} (rotated)",
            description=f"Rotated from key {self.key_id} at {datetime.utcnow().isoformat()}",
            is_admin=self.is_admin,
            permissions=self.permissions,
            rate_limit=self.rate_limit,
            jwt_secrets=secrets_list,
        )

        # Deactivate the old key
        self.is_active = False
        self.revoked = True
        self.revoked_at = datetime.utcnow()

        # Add the new key and update old key
        db.session.add(new_key)
        db.session.commit()

        # Log the rotation event
        APIKeyAuditLog.log(
            api_key=self,
            action="rotated",
            details=f"Key rotated to new key {new_key.key_id}",
        )
        APIKeyAuditLog.log(
            api_key=new_key,
            action="created",
            details=f"Created by rotation from key {self.key_id}",
        )

        return new_key, new_raw_key

    def has_permission(self, permission):
        """Check if this key has a specific permission"""
        if not self.is_active or self.revoked:
            return False

        if self.is_admin:
            return True

        return permission in self.permissions.split()

    def __repr__(self):
        return f"<APIKey {self.key_id} ({self.name})>"


class APIKeyAuditLog(db.Model):
    """Audit log for API key actions"""

    __tablename__ = "api_key_audit_logs"

    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"))
    key_id = Column(String(36), nullable=False)
    action = Column(String(50), nullable=False)  # created, revoked, rotated, used, etc.
    timestamp = Column(DateTime, default=func.now())
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    request_path = Column(String(255), nullable=True)
    status_code = Column(Integer, nullable=True)
    details = Column(Text, nullable=True)

    # Relationship back to API key
    api_key = relationship("APIKey", back_populates="audit_logs")

    @classmethod
    def log(
        cls,
        api_key,
        action,
        ip_address=None,
        user_agent=None,
        request_path=None,
        status_code=None,
        details=None,
    ):
        """
        Create an audit log entry for an API key action

        Args:
            api_key: The APIKey object
            action: The action being performed
            ip_address: Client IP address
            user_agent: Client user agent
            request_path: Request path
            status_code: HTTP status code
            details: Additional details
        """
        log_entry = cls(
            api_key_id=api_key.id,
            key_id=api_key.key_id,
            action=action,
            ip_address=ip_address,
            user_agent=user_agent,
            request_path=request_path,
            status_code=status_code,
            details=details,
        )

        db.session.add(log_entry)
        db.session.commit()

        return log_entry

    def __repr__(self):
        return f"<APIKeyAuditLog {self.key_id} {self.action} at {self.timestamp}>"
