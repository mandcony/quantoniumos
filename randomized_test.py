#!/usr/bin/env python3
"""
Quantonium OS - Randomized Test Script

This script demonstrates the Quantonium OS architecture using randomized test data
rather than user input. This helps protect the proprietary algorithms while still
showing the system's capabilities.
"""

import os
import random
import string
import base64
import hashlib
import time
import json
import requests
from typing import List, Dict, Any

# Base URL for the API
BASE_URL = "http://localhost:5000"

# API Key (in production, use environment variable)
API_KEY = "default_dev_key"

# Headers for API requests
headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY  # This is required for authentication
}

def generate_random_string(length: int = 32) -> str:
    """Generate a random string of fixed length."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def generate_random_waveform(length: int = 8) -> List[float]:
    """Generate a random waveform (list of float values between 0 and 1)."""
    return [random.random() for _ in range(length)]

def test_encryption_decryption_cycle(iterations: int = 5):
    """Test multiple encryption/decryption cycles with random data."""
    print("\n=== Randomized Encryption/Decryption Test ===")
    
    success_count = 0
    for i in range(iterations):
        # Generate random plaintext and key
        plaintext = generate_random_string(random.randint(10, 100))
        key = generate_random_string(random.randint(8, 32))
        
        # Encrypt
        encrypt_payload = {
            "plaintext": plaintext,
            "key": key
        }
        encrypt_response = requests.post(
            f"{BASE_URL}/api/encrypt", 
            headers=headers, 
            json=encrypt_payload
        )
        
        if encrypt_response.status_code != 200:
            print(f"Iteration {i+1}: Encryption failed with status {encrypt_response.status_code}")
            continue
        
        # Get ciphertext
        ciphertext = encrypt_response.json().get("ciphertext")
        
        # Decrypt
        decrypt_payload = {
            "ciphertext": ciphertext,
            "key": key
        }
        decrypt_response = requests.post(
            f"{BASE_URL}/api/decrypt", 
            headers=headers, 
            json=decrypt_payload
        )
        
        if decrypt_response.status_code != 200:
            print(f"Iteration {i+1}: Decryption failed with status {decrypt_response.status_code}")
            continue
            
        # Check if decrypted text matches original
        decrypted_text = decrypt_response.json().get("plaintext")
        if decrypted_text == plaintext:
            success_count += 1
            print(f"Iteration {i+1}: ✅ Success - Round trip encryption/decryption works")
        else:
            print(f"Iteration {i+1}: ❌ Failure - Decrypted text does not match original")
            print(f"  Original: {plaintext[:30]}..." if len(plaintext) > 30 else f"  Original: {plaintext}")
            print(f"  Decrypted: {decrypted_text[:30]}..." if len(decrypted_text) > 30 else f"  Decrypted: {decrypted_text}")
    
    print(f"\nSuccess rate: {success_count}/{iterations} ({success_count/iterations*100:.2f}%)")

def test_randomized_rft(iterations: int = 5):
    """Test Resonance Fourier Transform with random waveforms."""
    print("\n=== Randomized RFT Test ===")
    
    for i in range(iterations):
        # Generate random waveform of random length (between 8 and 20 points)
        waveform = generate_random_waveform(random.randint(8, 20))
        
        # Call RFT API
        payload = {"waveform": waveform}
        response = requests.post(
            f"{BASE_URL}/api/simulate/rft", 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 200:
            print(f"Iteration {i+1}: ✅ Success - RFT completed")
            # Print a sample of the frequencies found (first 3)
            frequencies = response.json().get("frequencies", {})
            sample_keys = list(frequencies.keys())[:3]
            print(f"  Sample frequencies: {', '.join(sample_keys)}...")
        else:
            print(f"Iteration {i+1}: ❌ Failure - RFT failed with status {response.status_code}")

def test_randomized_entropy(iterations: int = 5):
    """Test entropy generation with random sizes."""
    print("\n=== Randomized Entropy Generation Test ===")
    
    for i in range(iterations):
        # Generate random amount between 16 and 128 bytes
        amount = random.randint(16, 128)
        
        # Call entropy API
        payload = {"amount": amount}
        response = requests.post(
            f"{BASE_URL}/api/entropy/sample", 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 200:
            entropy = response.json().get("entropy", "")
            print(f"Iteration {i+1}: ✅ Success - Generated {len(entropy)} chars of entropy for {amount} bytes request")
            # Try to decode the base64 entropy and calculate its actual size
            try:
                decoded = base64.b64decode(entropy)
                print(f"  Actual entropy size: {len(decoded)} bytes")
            except:
                print("  (Could not decode entropy)")
        else:
            print(f"Iteration {i+1}: ❌ Failure - Entropy generation failed with status {response.status_code}")

def test_randomized_container(iterations: int = 5):
    """Test container unlocking with random waveforms (most should fail)."""
    print("\n=== Randomized Container Test ===")
    
    for i in range(iterations):
        # Generate random waveform and hash
        waveform = generate_random_waveform(random.randint(3, 10))
        random_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        
        # Call container API
        payload = {
            "waveform": waveform,
            "hash": random_hash
        }
        response = requests.post(
            f"{BASE_URL}/api/container/unlock", 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json().get("unlocked", False)
            if result:
                print(f"Iteration {i+1}: ✅ Unexpected Success - Container unlocked (rare with random data)")
            else:
                print(f"Iteration {i+1}: ✓ Expected Failure - Container remained locked")
        else:
            print(f"Iteration {i+1}: ❌ API Failure - Container API failed with status {response.status_code}")

def full_randomized_test():
    """Run a full suite of randomized tests."""
    print("🧠 Quantonium OS Randomized Test Suite")
    print("=====================================")
    
    # Status check
    response = requests.get(f"{BASE_URL}/api")
    if response.status_code == 200:
        print("\n✅ API Status Check: Operational")
    else:
        print("\n❌ API Status Check: Not responding properly")
        print("Stopping tests.")
        return
    
    # Run all randomized tests
    test_encryption_decryption_cycle()
    test_randomized_rft()
    test_randomized_entropy()
    test_randomized_container()
    
    print("\n✅ Randomized tests completed")
    print("This demonstrates architecture functionality using random data rather than exposing any real user inputs to the system.")

if __name__ == "__main__":
    full_randomized_test()