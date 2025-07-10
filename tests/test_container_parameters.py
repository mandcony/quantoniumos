#!/usr/bin/env python3
"""
Quantonium OS - Container Parameters API Test Suite

Tests for the container parameters extraction API endpoint.
"""

import json
import os
import sys
import unittest

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.encryption.geometric_waveform_hash import generate_waveform_hash
from main import create_app


class TestContainerParametersAPI(unittest.TestCase):
    """Test cases for container parameters API endpoint"""

    def setUp(self):
        """Set up test client and test data"""
        # Create a test Flask app
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Generate test hash for testing the extraction
        self.test_amplitude = 0.6427
        self.test_phase = 0.8315
        self.test_hash = generate_waveform_hash(self.test_amplitude, self.test_phase)

    def test_extract_parameters_success(self):
        """Test parameter extraction with a valid hash"""
        response = self.client.post(
            "/api/container/parameters",
            data=json.dumps({"hash": self.test_hash}),
            content_type="application/json",
        )

        # Check response
        self.assertEqual(response.status_code, 200)

        # Parse JSON
        data = json.loads(response.data)

        # Check structure
        self.assertTrue(data["success"])
        self.assertAlmostEqual(data["amplitude"], self.test_amplitude, places=4)
        self.assertAlmostEqual(data["phase"], self.test_phase, places=4)

    def test_extract_parameters_invalid_hash(self):
        """Test parameter extraction with an invalid hash"""
        response = self.client.post(
            "/api/container/parameters",
            data=json.dumps({"hash": "invalid_hash_format"}),
            content_type="application/json",
        )

        # Check response
        self.assertEqual(response.status_code, 200)

        # Parse JSON
        data = json.loads(response.data)

        # Should return failure
        self.assertFalse(data["success"])
        self.assertIn("message", data)
        self.assertIn("Invalid hash format", data["message"])

    def test_extract_parameters_missing_hash(self):
        """Test parameter extraction with missing hash parameter"""
        response = self.client.post(
            "/api/container/parameters",
            data=json.dumps({}),
            content_type="application/json",
        )

        # Check response
        self.assertEqual(response.status_code, 200)

        # Parse JSON
        data = json.loads(response.data)

        # Should return failure
        self.assertFalse(data["success"])
        self.assertIn("message", data)
        self.assertIn("Missing hash parameter", data["message"])


if __name__ == "__main__":
    unittest.main()
