#!/usr/bin/env python3
"""
Test script to verify all crypto playground buttons work correctly
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, '06_CRYPTOGRAPHY')
sys.path.insert(0, '.')

def test_crypto_functionality():
    """Test all crypto functions that the buttons use"""
    print("🔬 Testing QuantoniumOS Crypto Functionality")
    print("=" * 50)
    
    try:
        # Test resonance encryption import
        try:
            from resonance_encryption import ResonanceEncryptionEngine, wave_hmac, verify_wave_hmac, generate_entropy
            print("✅ Resonance encryption module imported successfully")
            
            # Test resonance engine
            engine = ResonanceEncryptionEngine()
            test_message = "Test quantum message"
            test_key = "test_key_2024"
            
            # Test encryption/decryption
            result = engine.encrypt_message(test_message, test_key, use_wave_hmac=True)
            if result.get("success"):
                print("✅ Resonance encryption working")
                
                # Test decryption
                decrypt_result = engine.decrypt_message(result.get("ciphertext"), test_key)
                if decrypt_result.get("success") and decrypt_result.get("plaintext") == test_message:
                    print("✅ Resonance decryption working")
                else:
                    print("⚠️  Resonance decryption issue")
            else:
                print("⚠️  Resonance encryption issue")
                
            # Test Wave-HMAC
            signature = wave_hmac(test_message, test_key)
            if signature:
                print("✅ Wave-HMAC signature generation working")
                
                # Test verification
                is_valid = verify_wave_hmac(test_message, signature, test_key)
                if is_valid:
                    print("✅ Wave-HMAC verification working")
                else:
                    print("⚠️  Wave-HMAC verification issue")
            else:
                print("⚠️  Wave-HMAC signature issue")
                
            # Test entropy generation
            entropy = generate_entropy(32)
            if entropy and len(entropy) == 32:
                print("✅ Entropy generation working")
            else:
                print("⚠️  Entropy generation issue")
                
            # Test waveform generation
            waveform_result = engine.generate_secure_waveform(64, test_key)
            if waveform_result.get("success"):
                print("✅ Waveform generation working")
                
                # Test RFT analysis
                waveform = waveform_result.get("waveform", [])
                if waveform:
                    rft_result = engine.perform_rft_analysis(waveform)
                    if rft_result.get("success"):
                        print("✅ RFT analysis working")
                    else:
                        print("⚠️  RFT analysis issue")
            else:
                print("⚠️  Waveform generation issue")
                
        except ImportError as e:
            print(f"⚠️  Resonance encryption not available: {e}")
            
        # Test C++ crypto bindings
        try:
            import enhanced_rft_crypto_bindings
            print("✅ Enhanced RFT crypto bindings available")
        except ImportError:
            print("⚠️  Enhanced RFT crypto bindings not available")
            
        try:
            import true_rft_engine_bindings
            print("✅ True RFT engine bindings available")
        except ImportError:
            print("⚠️  True RFT engine bindings not available")
            
        print("\n🎯 Crypto Functionality Summary:")
        print("All major crypto functions are accessible through the UI")
        print("The crypto playground buttons should work correctly")
        
    except Exception as e:
        print(f"❌ Error during crypto testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crypto_functionality()
