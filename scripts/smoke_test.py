#!/usr/bin/env python3
"""
Quantonium OS - End-to-End Smoke Tests

This script performs end-to-end tests of core API functionality to verify
that the system is working as expected after deployment. It uses a real API
key to authenticate and tests the entire request/response cycle.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smoke_test")

# Default base URL for local testing
DEFAULT_BASE_URL = "http://localhost:5000"


def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run E2E smoke tests for Quantonium OS API"
    )
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL of the API")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--skip-auth", action="store_true", help="Skip tests requiring authentication"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def get_api_key(args):
    """Get API key from args or environment"""
    if args.api_key:
        return args.api_key

    api_key = os.environ.get("QUANTONIUM_API_KEY")
    if api_key:
        return api_key

    if args.skip_auth:
        logger.warning("No API key provided, skipping authenticated tests")
        return None

    logger.error(
        "No API key provided. Set QUANTONIUM_API_KEY environment variable or use --api-key"
    )
    sys.exit(1)


def get_auth_token(base_url, api_key):
    """Get JWT token using API key"""
    if not api_key:
        return None

    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    try:
        response = requests.post(f"{base_url}/api/auth/token", headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to get auth token: {response.status_code}")
            logger.error(response.text)
            return None

        token_data = response.json()
        logger.info(
            f"Successfully acquired auth token for key {token_data.get('key_id')}"
        )
        return token_data.get("token")
    except Exception as e:
        logger.error(f"Error getting auth token: {str(e)}")
        return None


def get_headers(token=None, api_key=None):
    """Get request headers with authentication"""
    headers = {"Content-Type": "application/json"}

    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif api_key:
        headers["X-API-Key"] = api_key

    return headers


def test_health_check(base_url):
    """Test the health check endpoint"""
    logger.info("Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code != 200:
            logger.error(
                f"Health check failed with status code: {response.status_code}"
            )
            return False

        data = response.json()
        logger.info(f"Health check successful! API version: {data.get('version')}")
        return True
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return False


def test_metrics(base_url, headers):
    """Test the metrics endpoint"""
    logger.info("Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/api/metrics", headers=headers)
        if response.status_code != 200:
            logger.error(
                f"Metrics test failed with status code: {response.status_code}"
            )
            return False

        data = response.json()
        logger.info(
            f"Metrics test successful! Process memory: {data.get('process', {}).get('memory_rss_bytes', 'N/A')} bytes"
        )
        return True
    except Exception as e:
        logger.error(f"Metrics test error: {str(e)}")
        return False


def test_encryption(base_url, headers):
    """Test the encryption endpoint"""
    logger.info("Testing encryption endpoint...")
    try:
        payload = {
            "plaintext": f"Test message at {datetime.now().isoformat()}",
            "key": "smoke-test-key",
        }

        response = requests.post(
            f"{base_url}/api/encrypt", headers=headers, json=payload
        )
        if response.status_code != 200:
            logger.error(
                f"Encryption test failed with status code: {response.status_code}"
            )
            logger.error(response.text)
            return None

        data = response.json()
        ciphertext = data.get("ciphertext")
        hash_value = data.get("hash")
        logger.info(f"Encryption test successful! Generated hash: {hash_value}")
        return ciphertext, hash_value
    except Exception as e:
        logger.error(f"Encryption test error: {str(e)}")
        return None


def test_decryption(base_url, headers, ciphertext):
    """Test the decryption endpoint"""
    if not ciphertext:
        logger.warning("Skipping decryption test (no ciphertext)")
        return False

    logger.info("Testing decryption endpoint...")
    try:
        payload = {"ciphertext": ciphertext, "key": "smoke-test-key"}

        response = requests.post(
            f"{base_url}/api/decrypt", headers=headers, json=payload
        )
        if response.status_code != 200:
            logger.error(
                f"Decryption test failed with status code: {response.status_code}"
            )
            logger.error(response.text)
            return False

        data = response.json()
        plaintext = data.get("plaintext")
        logger.info(f"Decryption test successful! Result: {plaintext}")
        return True
    except Exception as e:
        logger.error(f"Decryption test error: {str(e)}")
        return False


def test_openapi_spec(base_url):
    """Test the OpenAPI spec endpoint"""
    logger.info("Testing OpenAPI spec endpoint...")
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code != 200:
            logger.error(
                f"OpenAPI spec test failed with status code: {response.status_code}"
            )
            return False

        data = response.json()
        if not data.get("openapi"):
            logger.error("Invalid OpenAPI spec (missing 'openapi' field)")
            return False

        version = data.get("info", {}).get("version")
        title = data.get("info", {}).get("title")
        endpoints = len(data.get("paths", {}))
        logger.info(
            f"OpenAPI spec test successful! API: {title}, Version: {version}, Endpoints: {endpoints}"
        )
        return True
    except Exception as e:
        logger.error(f"OpenAPI spec test error: {str(e)}")
        return False


def test_docs_ui(base_url):
    """Test the API docs UI endpoint"""
    logger.info("Testing API docs UI endpoint...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code != 200:
            logger.error(
                f"API docs UI test failed with status code: {response.status_code}"
            )
            return False

        if "swagger-ui" not in response.text:
            logger.error("Invalid API docs UI (missing Swagger UI)")
            return False

        logger.info("API docs UI test successful!")
        return True
    except Exception as e:
        logger.error(f"API docs UI test error: {str(e)}")
        return False


def run_tests(args):
    """Run all smoke tests"""
    base_url = args.url.rstrip("/")
    api_key = get_api_key(args) if not args.skip_auth else None
    token = get_auth_token(base_url, api_key) if api_key else None
    headers = get_headers(token, api_key)

    results = {}

    # Public endpoint tests (no auth required)
    results["health_check"] = test_health_check(base_url)
    results["openapi_spec"] = test_openapi_spec(base_url)
    results["docs_ui"] = test_docs_ui(base_url)

    # Authenticated endpoint tests
    if api_key or token:
        # Test encryption/decryption flow
        encryption_result = test_encryption(base_url, headers)
        if encryption_result:
            ciphertext, hash_value = encryption_result
            results["encryption"] = True
            results["decryption"] = test_decryption(base_url, headers, ciphertext)
        else:
            results["encryption"] = False
            results["decryption"] = False

        # Test metrics endpoint
        results["metrics"] = test_metrics(base_url, headers)

    # Print summary
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Summary: {success_count}/{total_count} tests passed")

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {name}")

    logger.info("=" * 50)

    # Return appropriate exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    args = setup_args()

    # Set verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    sys.exit(run_tests(args))
