# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

"""
Quantonium OS - Authentication Testing

Test suite for the authentication framework.
"""

import time
import unittest
from datetime import datetime, timedelta
import jwt
from auth.jwt_auth import authenticate_key, verify_token
# Import authentication modules
from auth.models import APIKey, APIKeyAuditLog, db
from flask import Flask
from werkzeug.security import check_password_hash

class TestAuthentication(unittest.TestCase):
    """Test suite for the authentication framework"""

    def setUp(self):
        """Set up test app and database"""
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        self.app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

        # Initialize db
        db.init_app(self.app)

        # Create all tables
        with self.app.app_context():
            db.create_all()

    def test_api_key_creation(self):
        """Test API key creation and validation"""
        with self.app.app_context():
            # Create a test key
            key, raw_key = APIKey.create(
                name="Test Key",
                description="Created for testing",
                permissions="api:read api:write",
                is_admin=False,
            )

            # Check key was created correctly
            self.assertIsNotNone(key)
            self.assertIsNotNone(raw_key)
            self.assertEqual(key.name, "Test Key")
            self.assertEqual(key.permissions, "api:read api:write")
            self.assertFalse(key.is_admin)
            self.assertTrue(key.is_active)
            self.assertFalse(key.revoked)

            # Verify key format
            prefix, rest = raw_key.split(".")
            self.assertEqual(prefix, key.key_prefix)
            self.assertEqual(len(prefix), 8)

            # Verify hashing
            self.assertTrue(check_password_hash(key.key_hash, raw_key))

            # Test API key authentication
            authenticated_key = authenticate_key(raw_key)
            self.assertIsNotNone(authenticated_key)
            self.assertEqual(authenticated_key.id, key.id)

            # Test invalid key
            invalid_key = authenticate_key("invalid-key")
            self.assertIsNone(invalid_key)

    def test_api_key_rotation(self):
        """Test API key rotation"""
        with self.app.app_context():
            # Create a test key
            key, raw_key = APIKey.create(
                name="Rotation Test", permissions="api:read", is_admin=False
            )

            # Rotate the key
            new_key, new_raw_key = key.rotate()

            # Verify old key is now inactive
            self.assertFalse(key.is_active)

            # Verify new key is active
            self.assertTrue(new_key.is_active)
            self.assertEqual(new_key.permissions, key.permissions)
            self.assertEqual(new_key.is_admin, key.is_admin)

            # Verify old key can no longer authenticate
            old_auth = authenticate_key(raw_key)
            self.assertIsNone(old_auth)

            # Verify new key authenticates properly
            new_auth = authenticate_key(new_raw_key)
            self.assertIsNotNone(new_auth)
            self.assertEqual(new_auth.id, new_key.id)

    def test_api_key_revocation(self):
        """Test API key revocation"""
        with self.app.app_context():
            # Create a test key
            key, raw_key = APIKey.create(
                name="Revocation Test", permissions="api:read", is_admin=False
            )

            # Authenticate before revocation
            auth_before = authenticate_key(raw_key)
            self.assertIsNotNone(auth_before)

            # Revoke the key
            key.revoke()

            # Verify key is now revoked and inactive
            self.assertTrue(key.revoked)
            self.assertFalse(key.is_active)
            self.assertIsNotNone(key.revoked_at)

            # Verify revoked key can no longer authenticate
            auth_after = authenticate_key(raw_key)
            self.assertIsNone(auth_after)

    def test_jwt_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        with self.app.app_context():
            # Create a test key
            key, raw_key = APIKey.create(
                name="JWT Test", permissions="api:read api:write", is_admin=False
            )

            # Create a token
            token = key.create_token(expiry_minutes=15)
            self.assertIsNotNone(token)

            # Verify the token
            api_key, payload = verify_token(token)
            self.assertIsNotNone(api_key)
            self.assertIsNotNone(payload)
            self.assertEqual(api_key.id, key.id)
            self.assertEqual(payload["sub"], key.key_id)
            self.assertEqual(payload["permissions"], key.permissions)

            # Test token with revoked key
            key.revoke()
            revoked_api_key, revoked_payload = verify_token(token)
            self.assertIsNone(revoked_api_key)
            self.assertIsNone(revoked_payload)

            # Test expired token
            expired_key, _ = APIKey.create(
                name="Expired Token Test", permissions="api:read", is_admin=False
            )

            # Create token with custom payload including very short expiry
            now = datetime.utcnow()
            exp = now + timedelta(seconds=1)
            payload = {"sub": expired_key.key_id, "exp": exp, "iat": now}
            short_token = jwt.encode(payload, expired_key.jwt_secret, algorithm="HS256")

            # Wait for token to expire
            time.sleep(2)

            # Verify expired token
            expired_api_key, expired_payload = verify_token(short_token)
            self.assertIsNone(expired_api_key)
            self.assertIsNone(expired_payload)

    def test_api_key_audit_logging(self):
        """Test API key audit logging"""
        with self.app.app_context():
            # Create a test key
            key, raw_key = APIKey.create(
                name="Audit Test", permissions="api:read", is_admin=False
            )

            # Create some audit logs
            APIKeyAuditLog.log(
                key, "tested", ip_address="127.0.0.1", details="Test log entry"
            )
            APIKeyAuditLog.log(key, "verified", details="Another test")

            # Check logs were created
            logs = APIKeyAuditLog.query.filter_by(api_key_id=key.id).all()
            self.assertEqual(len(logs), 3)  # Including creation log

            # Check relationship works
            key_logs = key.audit_logs
            self.assertEqual(len(key_logs), 3)

            # Check log data
            test_log = APIKeyAuditLog.query.filter_by(action="tested").first()
            self.assertEqual(test_log.key_id, key.key_id)
            self.assertEqual(test_log.ip_address, "127.0.0.1")
            self.assertEqual(test_log.details, "Test log entry")

    def test_permissions_check(self):
        """Test API key permission checking"""
        with self.app.app_context():
            # Create regular key with limited permissions
            regular_key, _ = APIKey.create(
                name="Regular Key", permissions="api:read", is_admin=False
            )

            # Create admin key
            admin_key, _ = APIKey.create(
                name="Admin Key", permissions="api:read", is_admin=True
            )

            # Check permissions
            self.assertTrue(regular_key.has_permission("api:read"))
            self.assertFalse(regular_key.has_permission("api:write"))
            self.assertFalse(regular_key.has_permission("api:admin"))

            # Admin should have all permissions regardless
            self.assertTrue(admin_key.has_permission("api:read"))
            self.assertTrue(admin_key.has_permission("api:write"))
            self.assertTrue(admin_key.has_permission("api:admin"))

            # Revoked key should have no permissions
            regular_key.revoke()
            self.assertFalse(regular_key.has_permission("api:read"))

if __name__ == "__main__":
    unittest.main()
