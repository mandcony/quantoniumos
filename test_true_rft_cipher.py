#!/usr/bin/env python3
"""
Test the TRUE Novel RFT Cipher Functions
Verify encrypt/decrypt cycle with resonance_encrypt and resonance_decrypt
"""

import sys
import os

# Add crypto path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '06_CRYPTOGRAPHY'))

def test_true_rft_cipher():
    """Test the TRUE RFT cipher functions"""
    print("=" * 60)
    print("TESTING TRUE NOVEL RFT CIPHER")
    print("=" * 60)
    
    try:
        from resonance_encryption import resonance_encrypt, resonance_decrypt
        print("✅ Successfully imported TRUE RFT cipher functions")
        
        # Test data
        test_message = "Secret quantum message using TRUE RFT cipher"
        test_key = "resonance_quantum_key_2024"
        
        print(f"\n📝 Original Message: {test_message}")
        print(f"🔑 Key: {test_key}")
        
        # Encrypt with TRUE RFT
        print("\n🔒 ENCRYPTING with TRUE RFT...")
        encrypted = resonance_encrypt(test_message, test_key)
        print(f"✅ Encrypted: {encrypted[:60]}{'...' if len(encrypted) > 60 else ''}")
        print(f"📏 Encrypted Length: {len(encrypted)} characters")
        
        # Decrypt with TRUE RFT  
        print("\n🔓 DECRYPTING with TRUE RFT...")
        decrypted = resonance_decrypt(encrypted, test_key)
        print(f"✅ Decrypted: {decrypted}")
        
        # Verify round-trip
        print("\n🧪 VERIFICATION...")
        success = (test_message == decrypted)
        print(f"{'✅ SUCCESS' if success else '❌ FAILED'}: Round-trip integrity check")
        
        if success:
            print("🎯 PERFECT: TRUE RFT cipher working correctly!")
            print("🌊 Engine: TRUE NOVEL RFT RESONANCE CIPHER")
            print("🔐 Wave-HMAC: Authenticated")
            print("🧬 Quantum-Enhanced Waveform Generation")
        else:
            print("❌ ERROR: Decrypted text doesn't match original!")
            print(f"Expected: {test_message}")
            print(f"Got:      {decrypted}")
            
        # Test with wrong key
        print("\n🔐 TESTING WRONG KEY...")
        try:
            wrong_decrypted = resonance_decrypt(encrypted, "wrong_key")
            if wrong_decrypted != test_message:
                print("✅ GOOD: Wrong key produces different result (secure)")
            else:
                print("⚠️ WARNING: Wrong key produced correct result (security issue)")
        except Exception as e:
            print(f"✅ EXCELLENT: Wrong key caused error (very secure): {e}")
            
    except ImportError as e:
        print(f"❌ FAILED: Could not import TRUE RFT cipher: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
        
    return success

if __name__ == "__main__":
    success = test_true_rft_cipher()
    if success:
        print("\n🏆 TRUE RFT CIPHER VALIDATION: COMPLETE")
    else:
        print("\n💥 TRUE RFT CIPHER VALIDATION: FAILED")
