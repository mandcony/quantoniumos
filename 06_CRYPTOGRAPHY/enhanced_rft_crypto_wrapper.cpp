
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

// Include the fixed enhanced_rft_crypto.cpp directly
#include "core/cpp/cryptography/enhanced_rft_crypto.cpp"

namespace py = pybind11;

// Global crypto engine instance
static EnhancedRFTCrypto g_crypto_engine;

PYBIND11_MODULE(enhanced_rft_crypto_bindings, m) {
    m.doc() = "Enhanced RFT Crypto Engine with Fixed Decryption";
    
    m.def("init_engine", []() {
        // Engine is already initialized as global instance
        return 0;
    }, "Initialize the crypto engine");
    
    m.def("generate_key_material", [](const std::vector<uint8_t>& password, 
                                     const std::vector<uint8_t>& salt, 
                                     int key_length) {
        // Simple key derivation using SHA256
        std::vector<uint8_t> combined;
        combined.reserve(password.size() + salt.size() + 4);
        combined.insert(combined.end(), password.begin(), password.end());
        combined.insert(combined.end(), salt.begin(), salt.end());
        
        // Add key length as bytes
        combined.push_back((key_length >> 24) & 0xFF);
        combined.push_back((key_length >> 16) & 0xFF);
        combined.push_back((key_length >> 8) & 0xFF);
        combined.push_back(key_length & 0xFF);
        
        auto hash = SHA256::hash(combined);
        
        // Extend if needed
        std::vector<uint8_t> key;
        while (key.size() < static_cast<size_t>(key_length)) {
            key.insert(key.end(), hash.begin(), hash.end());
            if (key.size() < static_cast<size_t>(key_length)) {
                hash = SHA256::hash(hash); // rehash for more bytes
            }
        }
        key.resize(key_length);
        return key;
    }, "Generate key material from password and salt");
    
    m.def("encrypt_block", [](const std::vector<uint8_t>& plaintext, 
                             const std::vector<uint8_t>& key) {
        if (plaintext.size() != 16) {
            throw std::runtime_error("Plaintext must be exactly 16 bytes");
        }
        return g_crypto_engine.encrypt(plaintext, key);
    }, "Encrypt a 16-byte block");
    
    m.def("decrypt_block", [](const std::vector<uint8_t>& ciphertext, 
                             const std::vector<uint8_t>& key) {
        if (ciphertext.size() != 16) {
            throw std::runtime_error("Ciphertext must be exactly 16 bytes");
        }
        return g_crypto_engine.decrypt(ciphertext, key);
    }, "Decrypt a 16-byte block");
    
    m.def("avalanche_test", [](const std::vector<uint8_t>& data1, 
                              const std::vector<uint8_t>& data2) {
        if (data1.size() != data2.size()) {
            throw std::runtime_error("Data sizes must match for avalanche test");
        }
        
        int diff_bits = 0;
        for (size_t i = 0; i < data1.size(); ++i) {
            uint8_t xor_result = data1[i] ^ data2[i];
            // Count bits that are different
            while (xor_result) {
                diff_bits += xor_result & 1;
                xor_result >>= 1;
            }
        }
        
        return static_cast<double>(diff_bits) / (data1.size() * 8.0);
    }, "Test avalanche effect between two data blocks");
}
