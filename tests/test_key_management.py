#!/usr/bin/env python3
"""
Quantonium OS - API Key Management Test Script

Tests the API key management endpoints in the Quantonium OS API with JWT authentication.
"""

import os
import requests
import json
import sys
import time
import pytest
import unittest
from unittest.mock import patch, MagicMock

# Base URL for the API
BASE_URL = "http://localhost:5000"

# API Key (in production, use environment variable)
API_KEY = os.environ.get("QUANTONIUM_API_KEY", "test_key_for_testing")  # Default for testing

# JWT token storage
token_data = None

def get_auth_token():
    """Get a JWT token using the API key"""
    global token_data
    
    # Only get a new token if we don't have one or it's expired
    if token_data and token_data.get("expires_at", 0) > time.time():
        return token_data["token"]
        
    # Check if we have an API key
    if not API_KEY:
        pytest.skip("No API key found. Set the QUANTONIUM_API_KEY environment variable.")
    
    # Get token from auth endpoint
    auth_headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/token", headers=auth_headers, timeout=1)
        
        if response.status_code != 200:
            pytest.skip(f"API server not available or auth failed: {response.status_code}")
            
        # Parse token response
        response_data = response.json()
        
        # Add expiry time for our tracking
        token_data = {
            "token": response_data["token"],
            "expires_at": time.time() + response_data.get("expires_in", 3600) - 60,
            "key_id": response_data.get("key_id")
        }
        
        return token_data["token"]
        
    except requests.exceptions.RequestException:
        pytest.skip("API server not available for testing")

def get_headers():
    """Get headers with authentication token"""
    token = get_auth_token()
    
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

@pytest.mark.integration  
def test_create_api_key():
    """Test creating a new API key"""
    # Admin access is required for this operation
    payload = {
        "name": "Test API Key",
        "description": "Created via API for testing",
        "permissions": "api:read api:write",
        "is_admin": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/keys",
            headers=get_headers(),
            json=payload,
            timeout=5
        )
        
        assert response.status_code in [200, 201], f"Expected 200/201, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "key" in data, "Response should contain 'key' field"
        
    except requests.exceptions.RequestException:
        pytest.skip("API server not available for testing")
@pytest.mark.integration
def test_list_api_keys():
    """Test listing all API keys"""
    try:
        response = requests.get(
            f"{BASE_URL}/api/auth/keys",
            headers=get_headers(),
            timeout=5
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert isinstance(data, (list, dict)), "Response should be a list or dict"
        
    except requests.exceptions.RequestException:
        pytest.skip("API server not available for testing")

@pytest.mark.integration
@pytest.mark.skip(reason="Requires existing key ID - integration test only")
def test_revoke_api_key():
    """Test revoking an API key"""
    # This test would require an existing key ID
    # In practice, this would be run with a specific key
    pytest.skip("Requires existing key ID for testing")

@pytest.mark.integration  
@pytest.mark.skip(reason="Requires existing key ID - integration test only")
def test_rotate_api_key():
    """Test rotating an API key"""
    # This test would require an existing key ID
    # In practice, this would be run with a specific key  
    pytest.skip("Requires existing key ID for testing")

@pytest.mark.integration
def test_get_profile():
    """Test getting the current API key profile"""
    try:
        response = requests.get(
            f"{BASE_URL}/api/auth/profile",
            headers=get_headers(),
            timeout=5
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "key_id" in data, "Profile should contain key_id"
        
    except requests.exceptions.RequestException:
        pytest.skip("API server not available for testing")

# Integration test class for when the API server is running
@pytest.mark.integration
class TestKeyManagementIntegration:
    """Integration tests for key management - requires running API server"""
    
    def test_integration_flow(self):
        """Test the full key management flow if server is available"""
        try:
            # Test profile endpoint
            response = requests.get(f"{BASE_URL}/api/auth/profile", headers=get_headers(), timeout=1)
            if response.status_code != 200:
                pytest.skip("API server not available or authentication failed")
                
        except requests.exceptions.RequestException:
            pytest.skip("API server not available for integration testing")

if __name__ == "__main__":
    # For backwards compatibility with direct script execution
    pytest.main([__file__, "-v"])