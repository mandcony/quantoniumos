#!/usr/bin/env python
"""
QuantoniumOS Route Validation Script

This script tests all routes and services in the QuantoniumOS application
to ensure they are working correctly.
"""

import os
import sys
import json
import time
import logging
import requests
import base64
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantonium_routes_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("route_validator")

# Constants
DEFAULT_HOST = "http://localhost:5000"
AUTH_TOKEN = None  # Will be populated by login
TIMEOUT = 10  # seconds
MAX_PARALLEL_TESTS = 5

# Test status tracking
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "details": []
}

def log_test_result(route: str, method: str, status: str, response: Any = None, error: str = None):
    """Log test result and add to the results tracking"""
    test_results["total"] += 1
    
    if status == "PASS":
        test_results["passed"] += 1
        logger.info(f"✅ {method} {route} - PASS")
    elif status == "FAIL":
        test_results["failed"] += 1
        logger.error(f"❌ {method} {route} - FAIL: {error}")
    elif status == "SKIP":
        test_results["skipped"] += 1
        logger.warning(f"⚠️ {method} {route} - SKIPPED: {error}")
    
    result_detail = {
        "route": route,
        "method": method,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    
    if response:
        if isinstance(response, requests.Response):
            try:
                result_detail["response"] = response.json() if response.headers.get("content-type") == "application/json" else {"status_code": response.status_code, "text": response.text[:100] + "..."}
            except:
                result_detail["response"] = {"status_code": response.status_code, "text": response.text[:100] + "..."}
        else:
            result_detail["response"] = response
    
    if error:
        result_detail["error"] = error
    
    test_results["details"].append(result_detail)

def make_request(route: str, method: str = "GET", data: Dict = None, 
                 headers: Dict = None, auth_required: bool = True,
                 expected_status: int = 200, expected_keys: List[str] = None) -> Tuple[bool, Any, str]:
    """Make a request to the specified route and validate the response"""
    if headers is None:
        headers = {}
    
    # Add authentication if required
    if auth_required and AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    url = f"{args.host}{route}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=TIMEOUT)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=TIMEOUT)
        else:
            return False, None, f"Unsupported method: {method}"
        
        # Check status code
        if response.status_code != expected_status:
            return False, response, f"Expected status {expected_status}, got {response.status_code}"
        
        # Try to parse JSON response
        try:
            resp_data = response.json()
        except:
            # Not all endpoints return JSON
            resp_data = response.text
            
            # If we expect JSON keys but couldn't parse JSON, that's an error
            if expected_keys:
                return False, response, "Expected JSON response but got non-JSON"
            
            # Otherwise, it's fine
            return True, response, None
        
        # Check for expected keys
        if expected_keys:
            for key in expected_keys:
                if key not in resp_data:
                    return False, response, f"Expected key '{key}' not found in response"
        
        return True, response, None
    except requests.exceptions.RequestException as e:
        return False, None, f"Request error: {str(e)}"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def test_route(route: str, method: str = "GET", data: Dict = None, 
               headers: Dict = None, auth_required: bool = True,
               expected_status: int = 200, expected_keys: List[str] = None):
    """Test a specific route and log the result"""
    
    success, response, error = make_request(
        route, method, data, headers, auth_required, expected_status, expected_keys
    )
    
    if success:
        log_test_result(route, method, "PASS", response)
        return True
    else:
        log_test_result(route, method, "FAIL", response, error)
        return False

def authenticate():
    """Authenticate and get a JWT token"""
    global AUTH_TOKEN
    
    logger.info("Authenticating to get JWT token...")
    
    # Default test credentials (should be configured for your app)
    auth_data = {
        "username": "test_user",
        "password": "test_password"
    }
    
    # Try to read credentials from environment
    if "QUANTONIUM_TEST_USER" in os.environ and "QUANTONIUM_TEST_PASSWORD" in os.environ:
        auth_data["username"] = os.environ["QUANTONIUM_TEST_USER"]
        auth_data["password"] = os.environ["QUANTONIUM_TEST_PASSWORD"]
    
    success, response, error = make_request(
        "/auth/login", 
        method="POST", 
        data=auth_data, 
        auth_required=False,
        expected_keys=["token"]
    )
    
    if success:
        try:
            AUTH_TOKEN = response.json().get("token")
            logger.info("Authentication successful")
            return True
        except:
            logger.error("Failed to extract token from authentication response")
            return False
    else:
        logger.error(f"Authentication failed: {error}")
        return False

def generate_test_data():
    """Generate test data for various endpoints"""
    return {
        "encrypt": {
            "plaintext": "Test plaintext message for encryption",
            "key": base64.b64encode(os.urandom(16)).decode('utf-8')
        },
        "decrypt": {
            # Will be populated after encrypt test
        },
        "rft": {
            "signal": [1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5]
        },
        "entropy": {
            "size": 8
        },
        "container": {
            "label": "test_container",
            "payload": "Test payload for container",
            "amplitude": 1.5,
            "phase": 0.5
        }
    }

def run_core_api_tests():
    """Test core API endpoints"""
    logger.info("Testing core API endpoints...")
    
    # Root status (public)
    test_route("/api/", auth_required=False)
    
    # Version and status
    test_route("/api/status")
    test_route("/api/version")
    
    # Ping with token validation
    test_route("/api/ping", expected_keys=["message", "timestamp"])

def run_encryption_tests(test_data):
    """Test encryption endpoints"""
    logger.info("Testing encryption endpoints...")
    
    # Test encryption
    encrypt_success = test_route(
        "/api/encrypt", 
        method="POST",
        data=test_data["encrypt"],
        expected_keys=["ciphertext", "signature"]
    )
    
    if encrypt_success:
        # Prepare decryption test with encrypted data
        try:
            # We need to get the actual response to use it for decryption
            _, response, _ = make_request(
                "/api/encrypt", 
                method="POST",
                data=test_data["encrypt"]
            )
            
            test_data["decrypt"] = {
                "ciphertext": response.json()["ciphertext"],
                "key": test_data["encrypt"]["key"]
            }
            
            # Test decryption
            test_route(
                "/api/decrypt",
                method="POST",
                data=test_data["decrypt"],
                expected_keys=["plaintext", "signature"]
            )
        except Exception as e:
            log_test_result(
                "/api/decrypt", 
                "POST", 
                "SKIP", 
                error=f"Skipped because encryption test data extraction failed: {str(e)}"
            )
    else:
        log_test_result(
            "/api/decrypt", 
            "POST", 
            "SKIP", 
            error="Skipped because encryption test failed"
        )

def run_quantum_tests(test_data):
    """Test quantum endpoints"""
    logger.info("Testing quantum endpoints...")
    
    # RFT (Resonance Fourier Transform)
    test_route(
        "/api/rft",
        method="POST",
        data=test_data["rft"],
        expected_keys=["frequency_domain"]
    )
    
    # IRFT (Inverse Resonance Fourier Transform)
    # Only if RFT test succeeded
    try:
        _, response, _ = make_request(
            "/api/rft", 
            method="POST",
            data=test_data["rft"]
        )
        
        irft_data = {
            "frequency_domain": response.json()["frequency_domain"]
        }
        
        test_route(
            "/api/irft",
            method="POST",
            data=irft_data,
            expected_keys=["time_domain"]
        )
    except Exception as e:
        log_test_result(
            "/api/irft", 
            "POST", 
            "SKIP", 
            error=f"Skipped because RFT test data extraction failed: {str(e)}"
        )
    
    # Entropy
    test_route(
        "/api/entropy",
        method="POST",
        data=test_data["entropy"],
        expected_keys=["entropy_values"]
    )

def run_container_tests(test_data):
    """Test container endpoints"""
    logger.info("Testing container endpoints...")
    
    # Create container
    create_success = test_route(
        "/api/container/create",
        method="POST",
        data=test_data["container"],
        expected_keys=["container_id", "resonance_key"]
    )
    
    if create_success:
        # We need to get the container details for other tests
        try:
            _, response, _ = make_request(
                "/api/container/create", 
                method="POST",
                data=test_data["container"]
            )
            
            container_id = response.json()["container_id"]
            
            # Get container
            test_route(
                f"/api/container/{container_id}",
                expected_keys=["container_id", "label", "entropy_score"]
            )
            
            # Unlock container
            unlock_data = {
                "container_id": container_id,
                "amplitude": test_data["container"]["amplitude"],
                "phase": test_data["container"]["phase"]
            }
            
            test_route(
                "/api/container/unlock",
                method="POST",
                data=unlock_data,
                expected_keys=["payload", "signature"]
            )
            
            # List containers
            test_route(
                "/api/containers",
                expected_keys=["containers"]
            )
        except Exception as e:
            logger.error(f"Failed to extract container details: {str(e)}")
            log_test_result(
                "/api/container/unlock", 
                "POST", 
                "SKIP", 
                error="Skipped because container creation data extraction failed"
            )
    else:
        log_test_result(
            "/api/container/unlock", 
            "POST", 
            "SKIP", 
            error="Skipped because container creation failed"
        )

def run_auth_tests():
    """Test authentication endpoints"""
    logger.info("Testing authentication endpoints...")
    
    # Login endpoint was already tested in authenticate()
    
    # Test register (if enabled)
    register_data = {
        "username": f"test_user_{int(time.time())}",
        "password": "test_password_123",
        "email": f"test_{int(time.time())}@example.com"
    }
    
    # This might fail if registration is disabled or requires admin approval
    register_result = test_route(
        "/auth/register",
        method="POST",
        data=register_data,
        auth_required=False,
        expected_status=201  # Created
    )
    
    # Test user profile (requires authentication)
    test_route(
        "/auth/profile",
        expected_keys=["username", "email"]
    )
    
    # Test refresh token
    test_route(
        "/auth/refresh",
        method="POST",
        expected_keys=["token"]
    )

def run_api_key_tests():
    """Test API key management endpoints"""
    logger.info("Testing API key management endpoints...")
    
    # Create API key
    create_key_data = {
        "name": f"test_key_{int(time.time())}",
        "scopes": ["read", "write"]
    }
    
    create_key_success = test_route(
        "/auth/apikeys",
        method="POST",
        data=create_key_data,
        expected_keys=["key_id", "api_key"]
    )
    
    if create_key_success:
        try:
            # Get the key_id to use in other tests
            _, response, _ = make_request(
                "/auth/apikeys",
                method="POST",
                data=create_key_data
            )
            
            key_id = response.json()["key_id"]
            api_key = response.json()["api_key"]
            
            # List API keys
            test_route(
                "/auth/apikeys",
                expected_keys=["keys"]
            )
            
            # Get specific API key
            test_route(
                f"/auth/apikeys/{key_id}",
                expected_keys=["key_id", "name", "scopes"]
            )
            
            # Test endpoint with API key
            test_headers = {
                "X-API-Key": api_key
            }
            
            test_route(
                "/api/status",
                headers=test_headers,
                auth_required=False  # We're using API key instead
            )
            
            # Revoke API key
            test_route(
                f"/auth/apikeys/{key_id}",
                method="DELETE",
                expected_status=204  # No Content
            )
        except Exception as e:
            logger.error(f"Failed to extract API key details: {str(e)}")
    else:
        log_test_result(
            "/auth/apikeys (GET, DELETE)", 
            "VARIOUS", 
            "SKIP", 
            error="Skipped because API key creation failed"
        )

def run_benchmark_tests():
    """Test benchmark endpoints"""
    logger.info("Testing benchmark endpoints...")
    
    # Run quick benchmark
    test_route(
        "/api/benchmark/quick",
        expected_keys=["results", "timestamp"]
    )
    
    # Benchmark history
    test_route(
        "/api/benchmark/history",
        expected_keys=["benchmarks"]
    )

def run_all_tests():
    """Run all tests in sequence"""
    start_time = time.time()
    logger.info(f"Starting QuantoniumOS route validation against {args.host}")
    
    # Authenticate first
    if not authenticate():
        logger.error("Authentication failed, cannot proceed with tests requiring auth")
    
    # Generate test data
    test_data = generate_test_data()
    
    # Run all test groups
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Run tests in parallel would be:
        # test_futures = []
        # test_futures.append(executor.submit(run_core_api_tests))
        # test_futures.append(executor.submit(run_encryption_tests, test_data))
        # ...
        
        # But for better debugging and to avoid race conditions, we run them sequentially
        run_core_api_tests()
        run_encryption_tests(test_data)
        run_quantum_tests(test_data)
        run_container_tests(test_data)
        run_auth_tests()
        run_api_key_tests()
        run_benchmark_tests()
    
    # Report results
    duration = time.time() - start_time
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Total Tests: {test_results['total']}")
    logger.info(f"Passed: {test_results['passed']} ({test_results['passed']/test_results['total']*100:.1f}%)")
    logger.info(f"Failed: {test_results['failed']} ({test_results['failed']/test_results['total']*100:.1f}%)")
    logger.info(f"Skipped: {test_results['skipped']} ({test_results['skipped']/test_results['total']*100:.1f}%)")
    logger.info(f"Time: {duration:.2f} seconds")
    
    # Write detailed results to file
    with open('route_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Detailed results written to route_test_results.json")
    
    # Return overall success status
    return test_results["failed"] == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QuantoniumOS Route Validation Tool')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'Host URL (default: {DEFAULT_HOST})')
    args = parser.parse_args()
    
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
