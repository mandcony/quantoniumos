#!/usr/bin/env python3
"""
Quantonium OS - API Test Script

This script demonstrates how to properly access the Quantonium OS API
with the correct authentication headers.
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:5000"

# API Key (in production, use environment variable)
API_KEY = "default_dev_key"

# Headers for API requests
headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY  # This is required for authentication
}

def test_root_endpoint():
    """Test the root endpoint for API status."""
    # Using the /status endpoint which is accessible without API key
    response = requests.get(f"{BASE_URL}/status")
    print("\n=== API Status ===")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_encrypt():
    """Test the encryption endpoint."""
    payload = {
        "plaintext": "hello quantum world", 
        "key": "symbolic-key"
    }
    response = requests.post(
        f"{BASE_URL}/encrypt", 
        headers=headers, 
        json=payload
    )
    print("\n=== Encryption Test ===")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_rft():
    """Test the Resonance Fourier Transform endpoint."""
    payload = {
        "waveform": [0.1, 0.5, 0.9, 0.5, 0.1, 0.5, 0.9, 0.5]
    }
    response = requests.post(
        f"{BASE_URL}/simulate/rft", 
        headers=headers, 
        json=payload
    )
    print("\n=== RFT Test ===")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_entropy():
    """Test the entropy generation endpoint."""
    payload = {
        "amount": 32
    }
    response = requests.post(
        f"{BASE_URL}/entropy/sample", 
        headers=headers, 
        json=payload
    )
    print("\n=== Entropy Test ===")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_container():
    """Test the container unlock endpoint."""
    payload = {
        "waveform": [0.2, 0.7, 0.3],
        "hash": "d6a88f4f..."
    }
    response = requests.post(
        f"{BASE_URL}/container/unlock", 
        headers=headers, 
        json=payload
    )
    print("\n=== Container Test ===")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_without_auth():
    """Test what happens with no authentication."""
    # Headers without API key
    no_auth_headers = {"Content-Type": "application/json"}
    response = requests.get(f"{BASE_URL}/", headers=no_auth_headers)
    print("\n=== No Authentication Test ===")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("🧠 Quantonium OS API Test Suite")
    print("==============================")
    
    # Run all tests
    test_root_endpoint()
    test_encrypt()
    test_rft()
    test_entropy()
    test_container()
    test_without_auth()
    
    print("\n✅ Tests completed")