#!/usr/bin/env python
"""
Debug script for test vector validation
"""
import json
import sys
import os
import hashlib

def main():
    """Main function to debug test vector validation"""
    test_vector_path = "public_test_vectors/vectors_ResonanceEncryption_encryption_latest.json"
    
    print(f"Reading test vectors from {test_vector_path}")
    with open(test_vector_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Print metadata
    print(f"Metadata:")
    print(f"  Algorithm: {data['metadata']['algorithm']}")
    print(f"  Type: {data['metadata']['type']}")
    print(f"  Timestamp: {data['metadata']['timestamp']}")
    print(f"  Count: {data['metadata']['count']}")
    print(f"  Format version: {data['metadata']['format_version']}")
    
    # Print first few vectors
    print(f"\nTest vectors: {len(data['vectors'])}")
    for i, vector in enumerate(data['vectors'][:3]):
        print(f"\nVector #{i} (id: {vector['id']}):")
        print(f"  Type: {vector['type']}")
        print(f"  Key (hex): {vector['key_hex']}")
        print(f"  Plaintext (hex): {vector['plaintext_hex']}")
        print(f"  Ciphertext (hex): {vector['ciphertext_hex'][:64]}...")
        print(f"  Plaintext size: {vector['plaintext_size']}")
        print(f"  Key size: {vector['key_size']}")

        # Check key length
        key_bytes = bytes.fromhex(vector['key_hex'])
        print(f"  Key actual length: {len(key_bytes)}")
        
        # Generate key hash as our Rust code would
        key_hash = hashlib.sha256(key_bytes).digest()
        print(f"  Key hash (first 8 bytes): {key_hash[:8].hex()}")
        
        # Check first 8 bytes of ciphertext against key hash
        ciphertext_bytes = bytes.fromhex(vector['ciphertext_hex'])
        print(f"  Ciphertext first 8 bytes: {ciphertext_bytes[:8].hex() if len(ciphertext_bytes) >= 8 else 'too short'}")
        
        # Check signature match
        if len(ciphertext_bytes) >= 8:
            if ciphertext_bytes[:8] == key_hash[:8]:
                print("  ✓ Signature matches key hash")
            else:
                print("  ✗ Signature doesn't match key hash")
                
    # Write results summary
    results = {
        "debug_complete": True,
        "vectors_analyzed": min(3, len(data['vectors'])),
        "format_valid": True
    }
    
    with open("test_vector_debug_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\nDebug complete! See test_vector_debug_results.json for details.")

if __name__ == "__main__":
    main()
