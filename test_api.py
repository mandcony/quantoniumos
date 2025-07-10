#!/usr/bin/env python3
"""
Quantonium OS - API Test Script

This script demonstrates how to properly access the Quantonium OS API
with JWT authentication using the new security framework.
"""

import json
import os
import sys
import time

import requests

# Base URL for the API
BASE_URL = "http://localhost:5000"

# API Key (in production, use environment variable)
API_KEY = os.environ.get("QUANTONIUM_API_KEY", "")  # Get from environment variable

# JWT token storage
token_data = None

# Default headers
headers = {"Content-Type": "application/json"}


def get_auth_token():
    """Get a JWT token using the API key"""
    global token_data

    # Only get a new token if we don't have one or it's expired
    if token_data and token_data.get("expires_at", 0) > time.time():
        return token_data["token"]

    # Check if we have an API key
    if not API_KEY:
        print(
            "Error: No API key found. Set the QUANTONIUM_API_KEY environment variable."
        )
        print(
            "You can create a key using the CLI tool: python -m auth.cli create --name 'Test Key'"
        )
        sys.exit(1)

    # Get token from auth endpoint
    auth_headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}

    try:
        response = requests.post(f"{BASE_URL}/api/auth/token", headers=auth_headers)

        if response.status_code != 200:
            print(f"Error getting auth token: {response.status_code}")
            print(response.text)
            sys.exit(1)

        # Parse token response
        token_data = response.json()

        # Add expiry time for our tracking
        token_data["expires_at"] = (
            time.time() + token_data["expires_in"] - 60
        )  # Expire 60s early to be safe

        print(f"✅ Successfully acquired auth token for key {token_data['key_id']}")
        return token_data["token"]

    except Exception as e:
        print(f"Error getting auth token: {str(e)}")
        sys.exit(1)


def get_headers():
    """Get headers with authentication token"""
    token = get_auth_token()

    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def test_root_endpoint():
    """Test the root endpoint for API status."""
    # Using the /status endpoint which is accessible without API key
    response = requests.get(f"{BASE_URL}/status")
    print("\n=== API Status ===")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_encrypt():
    """Test the encryption endpoint."""
    payload = {"plaintext": "hello quantum world", "key": "symbolic-key"}
    response = requests.post(
        f"{BASE_URL}/api/encrypt", headers=get_headers(), json=payload
    )
    print("\n=== Encryption Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
        # Return the ciphertext for use in decrypt test
        if response.status_code == 200:
            return response.json().get("ciphertext")
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")
    return None


def test_decrypt(ciphertext=None):
    """Test the decryption endpoint."""
    if ciphertext is None:
        # Use a placeholder if no ciphertext is provided
        ciphertext = "dO7kLB0s/zNeZ3ZtVUPLHw=="

    payload = {"ciphertext": ciphertext, "key": "symbolic-key"}
    response = requests.post(
        f"{BASE_URL}/api/decrypt", headers=get_headers(), json=payload
    )
    print("\n=== Decryption Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")


def test_rft():
    """Test the Resonance Fourier Transform endpoint."""
    payload = {"waveform": [0.1, 0.5, 0.9, 0.5, 0.1, 0.5, 0.9, 0.5]}
    response = requests.post(
        f"{BASE_URL}/api/simulate/rft", headers=get_headers(), json=payload
    )
    print("\n=== RFT Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")


def test_entropy():
    """Test the entropy generation endpoint."""
    payload = {"amount": 32}
    response = requests.post(
        f"{BASE_URL}/api/entropy/sample", headers=get_headers(), json=payload
    )
    print("\n=== Entropy Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")


def test_container():
    """Test the container unlock endpoint."""
    payload = {"waveform": [0.2, 0.7, 0.3], "hash": "d6a88f4f..."}
    response = requests.post(
        f"{BASE_URL}/api/container/unlock", headers=get_headers(), json=payload
    )
    print("\n=== Container Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")


def test_without_auth():
    """Test what happens with no authentication."""
    # Headers without API key
    no_auth_headers = {"Content-Type": "application/json"}
    # Try to access a protected API endpoint without auth
    response = requests.get(f"{BASE_URL}/api/encrypt", headers=no_auth_headers)
    print("\n=== No Authentication Test ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print(f"Response text: {response.text}")


if __name__ == "__main__":
    print("🧠 Quantonium OS API Test Suite")
    print("==============================")

    # Run all tests
    test_root_endpoint()

    # Run encrypt and decrypt in sequence to test roundtrip functionality
    ciphertext = test_encrypt()
    test_decrypt(ciphertext)

    test_rft()
    test_entropy()
    test_container()
    test_without_auth()

    print("\n✅ Tests completed")
