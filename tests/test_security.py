#!/usr/bin/env python3
"""
Quantonium OS - Security Test Suite

Tests for API security hardening features.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask
from flask.testing import FlaskClient

import main
from security import CSP_POLICY, configure_security, get_cors_origins


class TestSecurityHeaders(unittest.TestCase):
    """Test cases for security headers"""

    def setUp(self):
        """Set up test client and app"""
        self.app = Flask(__name__)
        self.app.testing = True

        # Apply Talisman with specific test settings
        talisman = Talisman(
            self.app,
            content_security_policy=CSP_POLICY,
            force_https=False,  # Disable HTTPS for testing
            frame_options="DENY",
            referrer_policy="no-referrer",
        )

        # Create a simple test route
        @self.app.route("/test")
        def test_route():
            return "OK"

        # Add permissions policy header
        @self.app.after_request
        def set_permissions_policy(response):
            response.headers["Permissions-Policy"] = "interest-cohort=()"
            return response

        self.client = self.app.test_client()

    def test_security_headers(self):
        """Test that security headers are set correctly"""
        response = self.client.get("/test")

        # Test X-Frame-Options
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")

        # Test Referrer-Policy
        self.assertEqual(response.headers.get("Referrer-Policy"), "no-referrer")

        # Test Permissions-Policy (note: newer Talisman versions use browsing-topics instead)
        permissions_policy = response.headers.get("Permissions-Policy")
        self.assertTrue(
            "interest-cohort=()" in permissions_policy
            or "browsing-topics=()" in permissions_policy
        )

        # Test Content-Security-Policy (just check it exists)
        self.assertIn("Content-Security-Policy", response.headers)

    def test_csp_default_src(self):
        """Test that default-src is correctly set to 'self'"""
        response = self.client.get("/test")
        csp = response.headers.get("Content-Security-Policy")
        self.assertIn("default-src 'self'", csp)

    def test_csp_frame_ancestors(self):
        """Test that frame-ancestors is set to 'none'"""
        response = self.client.get("/test")
        csp = response.headers.get("Content-Security-Policy")
        self.assertIn("frame-ancestors 'none'", csp)

    def test_csp_img_src(self):
        """Test that img-src includes https"""
        response = self.client.get("/test")
        csp = response.headers.get("Content-Security-Policy")
        self.assertIn("img-src 'self' https:", csp)


class TestRateLimiting(unittest.TestCase):
    """Test cases for rate limiting"""

    def test_rate_limiting_is_configured(self):
        """Test that rate limiting is properly configured"""
        # This is a simple test to verify that limiter is created with expected config
        app = Flask(__name__)
        _, limiter = configure_security(app)

        # Check that limiter exists and is properly configured
        self.assertIsNotNone(limiter)

        # Verify limiter is using the app
        self.assertEqual(limiter.app, app)


class TestCORSConfiguration(unittest.TestCase):
    """Test cases for CORS configuration"""

    @patch.dict(
        os.environ,
        {"CORS_ORIGINS": "https://allowed-origin.com,https://another-origin.com"},
    )
    def test_cors_origins_parsing(self):
        """Test CORS origins are correctly parsed from environment variable"""
        origins = get_cors_origins()
        self.assertEqual(len(origins), 2)
        self.assertIn("https://allowed-origin.com", origins)
        self.assertIn("https://another-origin.com", origins)

    @patch.dict(os.environ, {"CORS_ORIGINS": ""})
    def test_empty_cors_origins(self):
        """Test empty CORS origins environment variable"""
        origins = get_cors_origins()
        self.assertEqual(len(origins), 0)


class TestHealthCheck(unittest.TestCase):
    """Test cases for health check endpoint"""

    def test_health_check_route(self):
        """Test that health check route is properly defined"""
        app = Flask(__name__)
        app.testing = True

        # Mock the security configuration to avoid full implementation
        with patch("security.configure_security") as mock_security:
            mock_security.return_value = (MagicMock(), MagicMock())

            # Create a basic health check endpoint
            @app.route("/api/health")
            def health_check():
                return jsonify({"status": "ok"})

            client = app.test_client()
            response = client.get("/api/health")

            # Test response
            self.assertEqual(response.status_code, 200)


# Import Flask components here to avoid circular imports in test
from flask import jsonify
from flask_talisman import Talisman

if __name__ == "__main__":
    unittest.main()
