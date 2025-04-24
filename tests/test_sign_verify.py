"""
Quantonium OS - Authentication & Non-Repudiation Test Suite

This module tests the wave_hmac-based authentication system for digital signatures,
verifying both successful signature generation/verification and tamper detection.
"""

import sys
import os
import pytest
import json
import base64
from typing import Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the authentication functions
from encryption.resonance_encrypt import wave_hmac, encrypt, decrypt
from api.routes import sign_payload, verify_signature

# Test constants
TEST_PRIVATE_PHASE = "TEST_PRIVATE_PHASE_KEY_DO_NOT_USE_IN_PRODUCTION"
TEST_MESSAGES = [
    "Hello, Quantonium!",
    "This is a test of the wave_hmac signature system",
    "{'data': 'JSON compatible payload'}",
    "Binary data can be encoded as base64",
    "1234567890" * 10,  # Longer message
    ""  # Empty message
]


@pytest.fixture
def monkeypatch_env(monkeypatch):
    """Setup environment variables for testing"""
    monkeypatch.setenv("QUANTONIUM_PRIVATE_PHASE", TEST_PRIVATE_PHASE)
    yield


def test_wave_hmac_basics():
    """Test basic properties of the wave_hmac function"""
    # Same message and key should produce the same HMAC
    hmac1 = wave_hmac("test message", "key1")
    hmac2 = wave_hmac("test message", "key1")
    assert hmac1 == hmac2, "wave_hmac is not deterministic"
    
    # Different messages should produce different HMACs
    hmac3 = wave_hmac("different message", "key1")
    assert hmac1 != hmac3, "wave_hmac is not sensitive to message changes"
    
    # Different keys should produce different HMACs
    hmac4 = wave_hmac("test message", "key2")
    assert hmac1 != hmac4, "wave_hmac is not sensitive to key changes"
    
    # HMAC output should be non-empty and a string
    assert isinstance(hmac1, str)
    assert len(hmac1) > 0


@pytest.mark.parametrize("message", TEST_MESSAGES)
def test_sign_verify_happy_path(message, monkeypatch_env):
    """Test that signing and verification work correctly for valid signatures"""
    # Sign the message
    signature_result = sign_payload(message)
    
    # Verify that the signature result has expected structure
    assert isinstance(signature_result, dict)
    assert "header" in signature_result
    assert "payload" in signature_result
    assert "signature" in signature_result
    assert "phi" in signature_result
    
    # Verify the signature
    verification_result = verify_signature(signature_result)
    
    # Check verification succeeded
    assert verification_result["verified"] is True
    assert verification_result["message"] == message


def test_verify_tampered_payload(monkeypatch_env):
    """Test that verification fails when the payload is tampered with"""
    original_message = "Original message that will be tampered with"
    
    # Sign the original message
    signature_result = sign_payload(original_message)
    
    # Tamper with the payload
    tampered_result = signature_result.copy()
    tampered_result["payload"] = base64.b64encode("Tampered message".encode()).decode()
    
    # Verify should fail
    verification_result = verify_signature(tampered_result)
    assert verification_result["verified"] is False
    
    
def test_verify_tampered_signature(monkeypatch_env):
    """Test that verification fails when the signature is tampered with"""
    message = "Message with a signature that will be tampered with"
    
    # Sign the message
    signature_result = sign_payload(message)
    
    # Tamper with the signature
    tampered_result = signature_result.copy()
    original_sig = tampered_result["signature"]
    tampered_result["signature"] = original_sig[:-5] + "XXXXX"  # Change last 5 characters
    
    # Verify should fail
    verification_result = verify_signature(tampered_result)
    assert verification_result["verified"] is False


def test_verify_tampered_header(monkeypatch_env):
    """Test that verification fails when the header is tampered with"""
    message = "Message with a header that will be tampered with"
    
    # Sign the message
    signature_result = sign_payload(message)
    
    # Tamper with the header
    tampered_result = signature_result.copy()
    header_dict = json.loads(base64.b64decode(tampered_result["header"]).decode())
    header_dict["alg"] = "FAKE"  # Change algorithm
    tampered_result["header"] = base64.b64encode(json.dumps(header_dict).encode()).decode()
    
    # Verify should fail
    verification_result = verify_signature(tampered_result)
    assert verification_result["verified"] is False


def test_verify_tampered_phi(monkeypatch_env):
    """Test that verification fails when the phase information is tampered with"""
    message = "Message with phase that will be tampered with"
    
    # Sign the message
    signature_result = sign_payload(message)
    
    # Tamper with the phi value
    tampered_result = signature_result.copy()
    tampered_result["phi"] = "0.12345"  # Change phase value
    
    # Verify should fail
    verification_result = verify_signature(tampered_result)
    assert verification_result["verified"] is False


def test_sign_with_custom_phase():
    """Test signing with a custom phase value"""
    message = "Message signed with custom phase"
    custom_phase = "CUSTOM_PHASE_FOR_TESTING"
    
    # Override environment for this test
    os.environ["QUANTONIUM_PRIVATE_PHASE"] = custom_phase
    
    # Sign with custom phase
    signature_result = sign_payload(message)
    
    # Verify works with correct phase
    verification_result = verify_signature(signature_result)
    assert verification_result["verified"] is True
    
    # Change phase and verify should fail
    os.environ["QUANTONIUM_PRIVATE_PHASE"] = "WRONG_PHASE"
    verification_result = verify_signature(signature_result)
    assert verification_result["verified"] is False
    
    # Reset environment
    if "QUANTONIUM_PRIVATE_PHASE" in os.environ:
        del os.environ["QUANTONIUM_PRIVATE_PHASE"]


if __name__ == "__main__":
    # Run some basic tests
    print("Testing wave_hmac and sign/verify...")
    
    # Setup test environment
    os.environ["QUANTONIUM_PRIVATE_PHASE"] = TEST_PRIVATE_PHASE
    
    # Test basic HMAC functionality
    print("\nBasic HMAC tests:")
    message = "Test message"
    key = "Test key"
    hmac = wave_hmac(message, key)
    print(f"wave_hmac('{message}', '{key}') = {hmac}")
    
    # Test sign/verify
    print("\nSign/verify tests:")
    for i, message in enumerate(TEST_MESSAGES):
        print(f"\nTest {i+1}: '{message[:30]}{'...' if len(message) > 30 else ''}'")
        
        # Sign
        signature_result = sign_payload(message)
        print(f"  Signature: {signature_result['signature'][:20]}...")
        
        # Verify
        verification_result = verify_signature(signature_result)
        print(f"  Verified: {verification_result['verified']}")
        
        # Tamper and verify
        tampered = signature_result.copy()
        if message:
            tampered_message = message[:-1] + ("X" if message[-1] != "X" else "Y")
            tampered["payload"] = base64.b64encode(tampered_message.encode()).decode()
        else:
            tampered["payload"] = base64.b64encode(b"tampered").decode()
            
        tampered_verify = verify_signature(tampered)
        print(f"  Tampered verify: {tampered_verify['verified']} (should be False)")
        assert tampered_verify["verified"] is False, "Tamper detection failed!"
    
    print("\nAll tests passed!")