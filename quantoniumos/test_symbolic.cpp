#include <iostream>
#include "symbolic_eigenvector.h"

int main() {
    std::cout << "Testing QuantoniumOS symbolic eigenvector implementation...\n";

    // Test encode/decode resonance
    const char* test_data = "QuantoniumOS";
    char encoded[1024];
    int encoded_len = 0;
    
    try {
        encode_resonance(test_data, encoded, &encoded_len);
        std::cout << "Encoded: " << encoded << std::endl;
        
        char decoded[1024];
        int decoded_len = 0;
        decode_resonance(encoded, decoded, &decoded_len);
        std::cout << "Decoded: " << decoded << std::endl;
        
        if (strcmp(test_data, decoded) == 0) {
            std::cout << "✓ Encode/decode test passed\n";
        } else {
            std::cout << "✗ Encode/decode test failed\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in encode/decode test: " << e.what() << std::endl;
    }
    
    // Test eigenstate entropy
    const int size = 8;
    double entropy[size];
    try {
        generate_eigenstate_entropy(size, entropy);
        std::cout << "Eigenstate entropy: [";
        for (int i = 0; i < size; i++) {
            std::cout << entropy[i];
            if (i < size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    } catch (const std::exception& e) {
        std::cerr << "Error in entropy test: " << e.what() << std::endl;
    }
    
    // Test quantum superposition
    double state1[size] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // |0⟩
    double state2[size] = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // |1⟩
    double output[size];
    
    try {
        quantum_superposition(state1, state2, size, 1.0, 1.0, output);
        std::cout << "Quantum superposition: [";
        for (int i = 0; i < size; i++) {
            std::cout << output[i];
            if (i < size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    } catch (const std::exception& e) {
        std::cerr << "Error in superposition test: " << e.what() << std::endl;
    }
    
    std::cout << "Tests completed.\n";
    return 0;
}
