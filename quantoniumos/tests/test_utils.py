"""
Shared testing utilities for QuantoniumOS
Reduces code duplication between test files
"""

import os
import time
import requests
import logging

logger = logging.getLogger("test_utils")

def get_api_key_from_env():
    """Get API key from environment variables"""
    return os.environ.get("QUANTONIUM_API_KEY")

def get_auth_token(base_url, api_key, timeout=10):
    """Get JWT token using API key - shared between tests"""
    if not api_key:
        return None
    
    try:
        headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        response = requests.post(f"{base_url}/api/auth/token", 
                               headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            logger.error(f"Failed to get auth token: {response.status_code}")
            return None
            
        token_data = response.json()
        logger.info(f"Successfully acquired auth token for key {token_data.get('key_id')}")
        return token_data.get("token")
        
    except Exception as e:
        logger.error(f"Error getting auth token: {str(e)}")
        return None

def get_test_headers(base_url=None, api_key=None, token=None):
    """Get headers for API testing"""
    headers = {"Content-Type": "application/json"}
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif api_key:
        headers["X-API-Key"] = api_key
    elif base_url and api_key:
        # Get token automatically
        token = get_auth_token(base_url, api_key)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    
    return headers

def wait_for_health_check(base_url, timeout=60, interval=2):
    """Wait for server health check to pass"""
    logger.info(f"Waiting for server at {base_url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is ready!")
                return True
        except requests.RequestException:
            pass
        
        time.sleep(interval)
    
    logger.error(f"❌ Server at {base_url} did not become available within {timeout}s")
    return False

def test_basic_imports():
    """Test that basic imports work - minimal test for CI pipeline"""
    import flask
    import json
    import requests
    assert True  # Basic test to ensure pytest can run

def test_environment_setup():
    """Test that environment is set up correctly"""
    import sys
    import os
    
    # Check Python version
    assert sys.version_info >= (3, 9), "Python 3.9+ required"
    
    # Check that we're in the right directory
    assert os.path.exists("main.py"), "main.py should exist"
    assert os.path.exists("requirements.txt"), "requirements.txt should exist"

def test_test_utils_functions():
    """Test that test utility functions work"""
    api_key = get_api_key_from_env()
    headers = get_test_headers()
    
    # These should not crash
    assert isinstance(headers, dict)
    assert "Content-Type" in headers
