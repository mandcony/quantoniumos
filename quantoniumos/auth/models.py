"""
Quantonium OS - Authentication Models

This module defines the SQLAlchemy models for API keys and related authentication entities.
"""

import os
import uuid
import secrets
from datetime import datetime, timedelta
import jwt
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from auth.secret_manager import encrypt_secret, decrypt_secret, generate_jwt_secret, schedule_key_rotation

db = SQLAlchemy()

class APIKey(db.Model):
    """API Key model for authenticated API access"""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    key_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
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
    permissions = Column(String(255), default="api:read")  # Space-separated permission list
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    
    # Usage statistics
    use_count = Column(Integer, default=0)
    
    # JWT secret used for signing tokens
    jwt_secret = Column(Text, nullable=False)
    
    # Audit log relationship - one key can have many audit logs
    audit_logs = relationship("APIKeyAuditLog", back_populates="api_key")
    
    @staticmethod
    def generate_key():
        """Generate a new random API key with prefix"""
        raw_key = secrets.token_urlsafe(42)
        prefix = raw_key[:8]
        return f"{prefix}.{raw_key}"
    
    @classmethod
    def create(cls, name, description=None, expires_in_days=None, permissions="api:read", is_admin=False):
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
        prefix = raw_key.split('.')[0]
        
        # Hash the key for storage
        key_hash = generate_password_hash(raw_key)
        
        # Generate JWT secret for this key and encrypt it
        jwt_secret = generate_jwt_secret()
        encrypted_jwt_secret = encrypt_secret(jwt_secret)
        
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
            jwt_secret=encrypted_jwt_secret
        )
        
        # Schedule key rotation if expiration is set
        if expires_in_days:
            # Schedule at 75% of expiry time or 90 days, whichever is shorter
            rotation_days = min(int(expires_in_days * 0.75), 90)
            schedule_key_rotation(str(uuid.uuid4()), rotation_days)
        
        # Add and return
        db.session.add(key)
        db.session.commit()
        
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
        
        payload = {
            'sub': self.key_id,
            'kid': self.key_id,
            'prefix': self.key_prefix, 
            'permissions': self.permissions,
            'is_admin': self.is_admin,
            'iat': now,
            'exp': expiry
        }
        
        # Decrypt the JWT secret for use
        secret = decrypt_secret(self.jwt_secret)
        if not secret:
            # Fallback to using the encrypted value directly if decryption fails
            # This maintains backward compatibility with existing keys
            secret = self.jwt_secret
            
        token = jwt.encode(
            payload,
            secret,
            algorithm='HS256'
        )
        
        return token
    
    def verify_token(self, token):
        """Verify a JWT token created by this key"""
        try:
            # Decrypt the JWT secret for verification
            secret = decrypt_secret(self.jwt_secret)
            if not secret:
                # Fallback to using the encrypted value directly if decryption fails
                # This maintains backward compatibility with existing keys
                secret = self.jwt_secret
                
            payload = jwt.decode(
                token,
                secret,
                algorithms=['HS256']
            )
            
            # Basic validation
            if payload['sub'] != self.key_id:
                return False
                
            # Check if key was revoked after token was issued
            if self.revoked and self.revoked_at:
                token_iat = datetime.utcfromtimestamp(payload['iat'])
                if token_iat < self.revoked_at:
                    return False
            
            return payload
        except jwt.PyJWTError:
            return False
            
    def rotate(self, keep_name=True):
        """
        Rotate this API key, generating a new one to replace it
        Returns the new API key
        """
        # Create a new key with the same attributes
        new_key, raw_key = self.__class__.create(
            name=self.name if keep_name else f"{self.name} (Rotated)",
            description=f"Rotated from key {self.key_id} on {datetime.utcnow().isoformat()}",
            expires_in_days=(self.expires_at - datetime.utcnow()).days if self.expires_at else None,
            permissions=self.permissions,
            is_admin=self.is_admin
        )
        
        # Mark the old key as rotated
        self.description = f"{self.description or ''}\nRotated to {new_key.key_id} on {datetime.utcnow().isoformat()}"
        self.is_active = False
        db.session.commit()
        
        return new_key, raw_key
    
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
    __tablename__ = 'api_key_audit_logs'
    
    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey('api_keys.id'))
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
    def log(cls, api_key, action, ip_address=None, user_agent=None, request_path=None, status_code=None, details=None):
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
            details=details
        )
        
        db.session.add(log_entry)
        db.session.commit()
        
        return log_entry
    
    def __repr__(self):
        return f"<APIKeyAuditLog {self.key_id} {self.action} at {self.timestamp}>"