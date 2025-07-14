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

# Base URL for the API
BASE_URL = "http://localhost:5000"

# API Key (in production, use environment variable)
API_KEY = os.environ.get("QUANTONIUM_API_KEY", "")  # Get from environment variable

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
        print("Error: No API key found. Set the QUANTONIUM_API_KEY environment variable.")
        print("You can create a key using the CLI tool: python -m auth.cli create --name 'Test Key'")
        sys.exit(1)
    
    # Get token from auth endpoint
    auth_headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/token", headers=auth_headers)
        
        if response.status_code != 200:
            print(f"Error getting auth token: {response.status_code}")
            print(response.text)
            sys.exit(1)
            
        # Parse token response
        token_data = response.json()
        
        # Add expiry time for our tracking
        token_data["expires_at"] = time.time() + token_data["expires_in"] - 60  # Expire 60s early to be safe
        
        print(f"✅ Successfully acquired auth token for key {token_data['key_id']}")
        return token_data["token"]
        
    except Exception as e:
        print(f"Error getting auth token: {str(e)}")
        sys.exit(1)

def get_headers():
    """Get headers with authentication token"""
    token = get_auth_token()
    
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

def test_create_api_key():
    """Test creating a new API key"""
    # Admin access is required for this operation
    payload = {
        "name": "Test API Key",
        "description": "Created via API for testing",
        "permissions": "api:read api:write",
        "is_admin": False
    }
    
    response = requests.post(
        f"{BASE_URL}/api/auth/keys",
        headers=get_headers(),
        json=payload
    )
    
    print("\n=== Create API Key Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
        
        # Save the key ID and raw key for later tests
        if response.status_code == 201:
            key_id = data.get("key", {}).get("key_id")
            api_key = data.get("api_key")
            print(f"⚠️  IMPORTANT: Save this API key: {api_key}")
            return key_id, api_key
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")
    
    return None, None

def test_list_api_keys():
    """Test listing all API keys"""
    response = requests.get(
        f"{BASE_URL}/api/auth/keys",
        headers=get_headers()
    )
    
    print("\n=== List API Keys Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")

def test_revoke_api_key(key_id):
    """Test revoking an API key"""
    if not key_id:
        print("No key ID provided, skipping revoke test")
        return
    
    response = requests.post(
        f"{BASE_URL}/api/auth/keys/{key_id}/revoke",
        headers=get_headers(),
        json={"reason": "Testing key revocation"}
    )
    
    print("\n=== Revoke API Key Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")

def test_rotate_api_key(key_id):
    """Test rotating an API key"""
    if not key_id:
        print("No key ID provided, skipping rotate test")
        return
    
    response = requests.post(
        f"{BASE_URL}/api/auth/keys/{key_id}/rotate",
        headers=get_headers()
    )
    
    print("\n=== Rotate API Key Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
        
        # Save the new key information
        if response.status_code == 200:
            new_key_id = data.get("new_key", {}).get("key_id")
            new_api_key = data.get("api_key")
            print(f"⚠️  IMPORTANT: Save this new API key: {new_api_key}")
            return new_key_id, new_api_key
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")
    
    return None, None

def test_get_profile():
    """Test getting the current API key profile"""
    response = requests.get(
        f"{BASE_URL}/api/auth/profile",
        headers=get_headers()
    )
    
    print("\n=== API Key Profile Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")

if __name__ == "__main__":
    print("🔑 Quantonium OS API Key Management Test Suite")
    print("=============================================")
    
    # The test sequence matters here - we create a key, list keys, get profile,
    # then revoke the key we created, and finally rotate it
    
    test_get_profile()
    test_list_api_keys()
    
    # Create a key and save its ID
    key_id, api_key = test_create_api_key()
    
    if key_id:
        # Now that we have a key, we can test rotating and revoking it
        new_key_id, new_api_key = test_rotate_api_key(key_id)
        
        # If rotation worked, revoke the new key
        if new_key_id:
            test_revoke_api_key(new_key_id)
        else:
            # Fall back to revoking the original key
            test_revoke_api_key(key_id)
    
    print("\n✅ Key management tests completed")