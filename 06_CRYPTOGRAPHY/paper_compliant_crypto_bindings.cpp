#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <random>

// Include our enhanced RFT crypto implementation
#include "enhanced_rft_crypto.cpp"

namespace py = pybind11;

// Global engine instance for state management
static EnhancedRFTCrypto* global_engine = nullptr;
static std::vector<uint8_t> current_key;

// Paper-compliant interface functions
void init_engine() {
    if (!global_engine) {
        global_engine = new EnhancedRFTCrypto();
    }
}

py::bytes generate_key_material(py::bytes input_bytes, py::bytes salt_bytes, int output_length) {
    init_engine();
    
    std::string input_str = input_bytes;
    std::string salt_str = salt_bytes;
    
    // Combine input and salt for key derivation
    std::vector<uint8_t> combined;
    combined.insert(combined.end(), input_str.begin(), input_str.end());
    combined.insert(combined.end(), salt_str.begin(), salt_str.end());
    
    // Use enhanced mixing to generate key material
    std::vector<uint8_t> key_material;
    key_material.reserve(output_length);
    
    // Generate blocks until we have enough material
    while (key_material.size() < output_length) {
        // Create a temporary key from current material
        std::vector<uint8_t> temp_key = combined;
        temp_key.resize(32, 0x42); // Pad to 32 bytes
        
        // Add round counter for uniqueness
        uint32_t round = key_material.size() / 16;
        temp_key[0] ^= round & 0xFF;
        temp_key[1] ^= (round >> 8) & 0xFF;
        temp_key[2] ^= (round >> 16) & 0xFF;
        temp_key[3] ^= (round >> 24) & 0xFF;
        
        // Encrypt a known plaintext to generate pseudo-random material
        std::vector<uint8_t> plaintext(16, 0);
        for (int i = 0; i < 16; i++) {
            plaintext[i] = (i + round) & 0xFF;
        }
        
        auto encrypted = global_engine->encrypt(plaintext, temp_key);
        
        // Take first 16 bytes of encrypted output
        int bytes_to_take = std::min(16, output_length - (int)key_material.size());
        key_material.insert(key_material.end(), encrypted.begin(), encrypted.begin() + bytes_to_take);
    }
    
    // Store generated key for encrypt/decrypt operations
    current_key = std::vector<uint8_t>(key_material.begin(), key_material.begin() + std::min(32, output_length));
    
    return py::bytes(std::string(key_material.begin(), key_material.begin() + output_length));
}

py::bytes encrypt_block(py::bytes plaintext_bytes, py::bytes key_bytes) {
    init_engine();
    
    std::string plaintext_str = plaintext_bytes;
    std::string key_str = key_bytes;
    
    std::vector<uint8_t> plaintext(plaintext_str.begin(), plaintext_str.end());
    std::vector<uint8_t> key(key_str.begin(), key_str.end());
    
    // Paper specifies 16-byte blocks - pad if needed
    if (plaintext.size() != 16) {
        plaintext.resize(16, 0x80); // ISO/IEC 7816-4 padding
    }
    
    // Ensure key is proper length
    if (key.size() < 32) {
        key.resize(32, 0x42); // Pad key if too short
    }
    
    auto result = global_engine->encrypt(plaintext, key);
    
    // Return exactly 16 bytes for block cipher interface
    return py::bytes(std::string(result.begin(), result.begin() + 16));
}

py::bytes decrypt_block(py::bytes ciphertext_bytes, py::bytes key_bytes) {
    init_engine();
    
    std::string ciphertext_str = ciphertext_bytes;
    std::string key_str = key_bytes;
    
    std::vector<uint8_t> ciphertext(ciphertext_str.begin(), ciphertext_str.end());
    std::vector<uint8_t> key(key_str.begin(), key_str.end());
    
    // Paper specifies 16-byte blocks
    if (ciphertext.size() != 16) {
        throw std::invalid_argument("Ciphertext must be exactly 16 bytes for block operation");
    }
    
    // Ensure key is proper length
    if (key.size() < 32) {
        key.resize(32, 0x42); // Pad key if too short
    }
    
    auto result = global_engine->decrypt(ciphertext, key);
    
    // Return exactly 16 bytes for block cipher interface
    return py::bytes(std::string(result.begin(), result.begin() + 16));
}

double avalanche_test(py::bytes key1_bytes, py::bytes key2_bytes) {
    init_engine();
    
    std::string key1_str = key1_bytes;
    std::string key2_str = key2_bytes;
    
    std::vector<uint8_t> key1(key1_str.begin(), key1_str.end());
    std::vector<uint8_t> key2(key2_str.begin(), key2_str.end());
    
    // Ensure keys are same length for fair comparison
    size_t max_len = std::max(key1.size(), key2.size());
    key1.resize(max_len, 0x42);
    key2.resize(max_len, 0x42);
    
    // Test with standard 16-byte plaintext
    std::vector<uint8_t> plaintext(16, 0);
    for (int i = 0; i < 16; i++) {
        plaintext[i] = i;
    }
    
    auto result1 = global_engine->encrypt(plaintext, key1);
    auto result2 = global_engine->encrypt(plaintext, key2);
    
    // Count bit differences in first 16 bytes
    int different_bits = 0;
    int total_bits = 0;
    
    for (size_t i = 0; i < std::min(result1.size(), result2.size()) && i < 16; i++) {
        uint8_t xor_val = result1[i] ^ result2[i];
        total_bits += 8;
        
        // Count set bits in XOR result
        for (int bit = 0; bit < 8; bit++) {
            if (xor_val & (1 << bit)) {
                different_bits++;
            }
        }
    }
    
    // Return avalanche ratio (should be ~0.5 for good cipher)
    return total_bits > 0 ? (double)different_bits / total_bits : 0.0;
}

// Python module definition with paper-compliant interface
PYBIND11_MODULE(enhanced_rft_crypto_bindings, m) {
    m.doc() = "Enhanced RFT Crypto C++ Engine Python Bindings";
    
    m.def("init_engine", &init_engine, "Initialize the RFT crypto engine");
    m.def("generate_key_material", &generate_key_material, "Generate key material from input and salt");
    m.def("encrypt_block", &encrypt_block, "Encrypt a 16-byte block");
    m.def("decrypt_block", &decrypt_block, "Decrypt a 16-byte block");
    m.def("avalanche_test", &avalanche_test, "Test avalanche effect between two keys");
}
