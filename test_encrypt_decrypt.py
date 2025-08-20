#!/usr/bin/env python3
"""
Quick test to verify encrypt/decrypt functions work correctly
"""

import sys
import os

# Add crypto path
sys.path.insert(0, '06_CRYPTOGRAPHY')

def test_encrypt_decrypt():
    """Test that encrypt/decrypt functions work properly"""
    print("🔬 Testing Encrypt/Decrypt Functionality")
    print("=" * 50)
    
    try:
        from resonance_encryption import encrypt_symbolic, decrypt_symbolic
        
        # Test message and key
        test_message = "Secret quantum message using resonance encryption"
        test_key = "resonance_quantum_key_2024"
        
        print(f"Original Message: {test_message}")
        print(f"Key: {test_key}")
        print()
        
        # Test encryption
        encrypt_result = encrypt_symbolic(test_message, test_key)
        print(f"Encryption Success: {encrypt_result.get('success')}")
        
        if encrypt_result.get("success"):
            ciphertext = encrypt_result.get("ciphertext")
            print(f"Ciphertext: {ciphertext[:100]}...")
            print(f"Ciphertext Length: {len(ciphertext)} characters")
            print()
            
            # Test decryption
            decrypt_result = decrypt_symbolic(ciphertext, test_key)
            print(f"Decryption Success: {decrypt_result.get('success')}")
            
            if decrypt_result.get("success"):
                decrypted_message = decrypt_result.get("plaintext")
                print(f"Decrypted Message: {decrypted_message}")
                print()
                
                # Verify they match
                if test_message == decrypted_message:
                    print("✅ SUCCESS: Original and decrypted messages match!")
                    print("✅ Encrypt/Decrypt functions are working correctly")
                else:
                    print("❌ FAILED: Messages do not match")
                    print(f"Expected: {test_message}")
                    print(f"Got:      {decrypted_message}")
            else:
                print("❌ Decryption failed")
                print(f"Error: {decrypt_result.get('error')}")
        else:
            print("❌ Encryption failed")
            print(f"Error: {encrypt_result.get('error')}")
            
        # Test with wrong key
        print("\n🔐 Testing with wrong key...")
        wrong_decrypt = decrypt_symbolic(ciphertext, "wrong_key")
        if wrong_decrypt.get("success"):
            print("⚠️  WARNING: Decryption with wrong key succeeded (security issue)")
        else:
            print("✅ Correctly rejected wrong key")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_encrypt_decrypt()
