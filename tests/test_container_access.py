"""
Quantonium OS - Container Access Control Test Suite

This module tests the secure container access mechanisms that enforce the
lock-and-key principle where only the specific hash that created a container
can unlock it when paired with the correct key.
"""

import sys
import os
import json
from typing import Dict, List, Any
import hashlib
import random
import time

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary modules
from core.encryption.resonance_encrypt import encrypt_symbolic, decrypt_symbolic
from core.encryption.geometric_waveform_hash import wave_hash
from orchestration.resonance_manager import (
    register_container,
    get_container_by_hash,
    check_container_access,
    verify_container_key
)

# Test constants
TEST_DATA = "0123456789abcdef0123456789abcdef"  # 32 chars (128 bits) hex
TEST_KEY = "fedcba9876543210fedcba9876543210"  # 32 chars (128 bits) hex


def generate_test_data_set(count=5):
    """Generate a set of test plaintext and key values"""
    data_set = []
    for i in range(count):
        # Generate random hex strings of the right length
        plaintext = ''.join(random.choice('0123456789abcdef') for _ in range(32))
        key = ''.join(random.choice('0123456789abcdef') for _ in range(32))
        data_set.append((plaintext, key))
    return data_set


def test_container_creation_and_access():
    """Test creating containers and accessing them with the correct hash and key"""
    print("\nTest: Container creation and valid access")
    
    # Generate test data
    data_set = generate_test_data_set()
    
    # Create containers for each data set
    containers = []
    for plaintext, key in data_set:
        # Encrypt to get the hash/ciphertext
        result = encrypt_symbolic(plaintext, key)
        
        # Register a container with this hash
        register_container(
            hash_value=result["hash"],
            plaintext=plaintext,
            ciphertext=result["ciphertext"],
            key=key
        )
        
        containers.append({
            "plaintext": plaintext,
            "key": key,
            "hash": result["hash"],
            "ciphertext": result["ciphertext"]
        })
        
        print(f"  Created container with hash: {result['hash'][:8]}...")
    
    # Try to access each container with the correct hash and key
    for container in containers:
        # Verify the key is valid for this container
        key_valid = verify_container_key(container["hash"], container["key"])
        assert key_valid, f"Key validation failed for hash {container['hash'][:8]}..."
        
        # Check container access
        access_granted = check_container_access(container["hash"])
        assert access_granted, f"Access validation failed for hash {container['hash'][:8]}..."
        
        # Get the container and check its contents
        stored_container = get_container_by_hash(container["hash"])
        assert stored_container is not None, f"Container not found for hash {container['hash'][:8]}..."
        assert stored_container["plaintext"] == container["plaintext"], "Plaintext mismatch"
        
        print(f"  Successfully accessed container with hash: {container['hash'][:8]}...")
    
    print("  All containers accessed successfully with correct hash and key")
    return containers


def test_container_tamper_prevention():
    """Test that containers cannot be accessed with incorrect hash or key"""
    print("\nTest: Container tamper prevention")
    
    # Generate test data and create a container
    plaintext = TEST_DATA
    key = TEST_KEY
    
    # Encrypt to get the hash/ciphertext
    result = encrypt_symbolic(plaintext, key)
    
    # Register a container with this hash
    register_container(
        hash_value=result["hash"],
        plaintext=plaintext,
        ciphertext=result["ciphertext"],
        key=key
    )
    
    print(f"  Created container with hash: {result['hash'][:8]}...")
    
    # 1. Try with the wrong key
    wrong_key = ''.join(random.choice('0123456789abcdef') for _ in range(32))
    key_valid = verify_container_key(result["hash"], wrong_key)
    assert not key_valid, "Key validation should fail with incorrect key"
    print("  ✓ Access denied with incorrect key")
    
    # 2. Try with a non-existent hash
    wrong_hash = ''.join(random.choice('0123456789abcdef') for _ in range(32))
    access_granted = check_container_access(wrong_hash)
    assert not access_granted, "Access validation should fail with incorrect hash"
    print("  ✓ Access denied with incorrect hash")
    
    # 3. Try with correct hash but incorrect key
    key_valid = verify_container_key(result["hash"], wrong_key)
    assert not key_valid, "Key validation should fail with incorrect key"
    print("  ✓ Access denied with correct hash but incorrect key")
    
    # 4. Verify that access is granted with correct hash and key
    key_valid = verify_container_key(result["hash"], key)
    assert key_valid, "Key validation should succeed with correct key"
    access_granted = check_container_access(result["hash"])
    assert access_granted, "Access validation should succeed with correct hash"
    print("  ✓ Access granted with correct hash and key")


def test_container_collision_resistance():
    """Test that container hashes are collision resistant"""
    print("\nTest: Container collision resistance")
    
    # Generate a set of plaintext values with minor differences
    plaintexts = []
    base_text = "0123456789abcdef0123456789abcdef"
    
    # Generate variations by flipping single bits
    for i in range(8):
        # Flip a bit in position i
        char_pos = i * 4
        char = base_text[char_pos]
        # Simple way to flip a hex digit
        flipped = hex(15 - int(char, 16))[2:]
        plaintexts.append(base_text[:char_pos] + flipped + base_text[char_pos+1:])
    
    # Add the base text itself
    plaintexts.append(base_text)
    
    # Generate a hash for each plaintext using the same key
    key = TEST_KEY
    hashes = []
    
    for plaintext in plaintexts:
        result = encrypt_symbolic(plaintext, key)
        hashes.append(result["hash"])
        
        # Register a container for each hash
        register_container(
            hash_value=result["hash"],
            plaintext=plaintext,
            ciphertext=result["ciphertext"],
            key=key
        )
        
        print(f"  Created container with plaintext: {plaintext[:8]}... hash: {result['hash'][:8]}...")
    
    # Verify that all hashes are unique (collision resistance)
    unique_hashes = set(hashes)
    assert len(unique_hashes) == len(hashes), "Hash collision detected"
    print(f"  ✓ All {len(hashes)} hashes are unique")
    
    # Verify that each container can only be unlocked with its specific hash and key
    for i, plaintext in enumerate(plaintexts):
        for j, hash_value in enumerate(hashes):
            # This container should only be accessible with its own hash
            container = get_container_by_hash(hash_value)
            if i == j:
                # Correct hash for this plaintext
                assert container["plaintext"] == plaintext, f"Plaintext mismatch for hash {hash_value[:8]}..."
            else:
                # Wrong hash for this plaintext
                assert container["plaintext"] != plaintext, f"Incorrect plaintext match for hash {hash_value[:8]}..."
    
    print("  ✓ Each container can only be accessed with its specific hash")


def run_all_tests():
    """Run all container access tests"""
    print("\n=== Quantonium OS Container Access Tests ===\n")
    
    try:
        containers = test_container_creation_and_access()
        test_container_tamper_prevention()
        test_container_collision_resistance()
        
        print("\nAll container access tests passed! The lock-and-key mechanism is working correctly.")
        return True, containers
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        return False, None


if __name__ == "__main__":
    success, containers = run_all_tests()
    if not success:
        sys.exit(1)