"""
QuantoniumOS API Test
Tests the basic API endpoints in the simplified version
"""

import requests
import json
import time
from pprint import pprint

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, method="GET", data=None, headers=None):
    """Test an API endpoint and return the result"""
    url = f"{BASE_URL}{endpoint}"
    print(f"Testing {method} {url}...")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=5)
        else:
            print(f"Method {method} not supported")
            return None
        
        if response.status_code >= 200 and response.status_code < 300:
            print(f"✅ {method} {endpoint} - {response.status_code}")
            return response.json()
        else:
            print(f"❌ {method} {endpoint} - {response.status_code}")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {e}")
        return None

def main():
    """Run all API tests"""
    print("🔍 Testing QuantoniumOS API...")
    
    # Test basic endpoints
    home = test_endpoint("/")
    if home:
        print(f"App Name: {home.get('name')}")
        print(f"Status: {home.get('status')}")
    
    health = test_endpoint("/api/health")
    if health:
        print(f"Health Status: {health.get('status')}")
    
    status = test_endpoint("/api/status")
    if status:
        print(f"Services:")
        for service, state in status.get('services', {}).items():
            print(f"  - {service}: {state}")
    
    version = test_endpoint("/api/version")
    if version:
        print(f"Version: {version.get('version')}")
        print(f"Build: {version.get('build')}")
    
    print("\n✨ API testing complete")

if __name__ == "__main__":
    main()
