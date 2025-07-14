#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>
#include "symbolic_eigenvector.h"

// Helper function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// Helper to validate vector contents
bool validate_vector(double* vec, int size, const std::vector<double>& expected, double epsilon = 1e-5) {
    for (int i = 0; i < size && i < expected.size(); i++) {
        if (!approx_equal(vec[i], expected[i], epsilon)) {
            std::cout << "  At index " << i << ": Expected " << expected[i] 
                      << ", got " << vec[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "=== QuantoniumOS Rigorous Validation Tests ===\n";
    bool all_tests_passed = true;

    // Test 1: Encode/Decode Resonance - Should not be trivial
    {
        std::cout << "\nTest 1: Encode/Decode Resonance\n";
        const char* test_data = "QuantoniumOS";
        char encoded[1024];
        int encoded_len = 0;
        
        try {
            // Encode the data
            encode_resonance(test_data, encoded, &encoded_len);
            std::cout << "  Encoded: " << encoded << std::endl;
            
            // Verify encoding is not trivial (should not contain the original string directly)
            std::string encoded_str(encoded);
            bool is_trivial = encoded_str.find(test_data) != std::string::npos;
            if (is_trivial) {
                std::cout << "  ✗ FAIL: Encoding appears to be trivial\n";
                all_tests_passed = false;
            } else {
                std::cout << "  ✓ Encoding is non-trivial\n";
            }
            
            // Verify decoding works correctly
            char decoded[1024];
            int decoded_len = 0;
            decode_resonance(encoded, decoded, &decoded_len);
            std::cout << "  Decoded: " << decoded << std::endl;
            
            if (strcmp(test_data, decoded) == 0) {
                std::cout << "  ✓ Decode test passed\n";
            } else {
                std::cout << "  ✗ FAIL: Decode test failed\n";
                all_tests_passed = false;
            }
            
            // Verify encoding changes with input
            const char* different_data = "DifferentInput";
            char different_encoded[1024];
            int different_encoded_len = 0;
            encode_resonance(different_data, different_encoded, &different_encoded_len);
            
            if (strcmp(encoded, different_encoded) != 0) {
                std::cout << "  ✓ Different inputs produce different encodings\n";
            } else {
                std::cout << "  ✗ FAIL: Different inputs produce same encoding\n";
                all_tests_passed = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in encode/decode test: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    // Test 2: Eigenstate Entropy - Verify expected properties
    {
        std::cout << "\nTest 2: Eigenstate Entropy\n";
        const int size = 8;
        double entropy[size];
        
        try {
            generate_eigenstate_entropy(size, entropy);
            
            // Print values
            std::cout << "  Entropy values: [";
            for (int i = 0; i < size; i++) {
                std::cout << entropy[i];
                if (i < size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // Entropy values should be non-negative
            bool all_non_negative = true;
            for (int i = 0; i < size; i++) {
                if (entropy[i] < 0) {
                    all_non_negative = false;
                    break;
                }
            }
            
            if (all_non_negative) {
                std::cout << "  ✓ All entropy values are non-negative\n";
            } else {
                std::cout << "  ✗ FAIL: Some entropy values are negative\n";
                all_tests_passed = false;
            }
            
            // Total entropy should be meaningful (not all zeros)
            double total_entropy = 0;
            for (int i = 0; i < size; i++) {
                total_entropy += entropy[i];
            }
            
            if (total_entropy > 0.01) {
                std::cout << "  ✓ Total entropy is meaningful: " << total_entropy << std::endl;
            } else {
                std::cout << "  ✗ FAIL: Total entropy is too small: " << total_entropy << std::endl;
                all_tests_passed = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in entropy test: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    // Test 3: Quantum Superposition - Verify mathematical properties
    {
        std::cout << "\nTest 3: Quantum Superposition\n";
        const int size = 8;
        double state1[size] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // |0⟩
        double state2[size] = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // |1⟩
        double output[size];
        
        try {
            // Equal superposition with normalization
            quantum_superposition(state1, state2, size, 1.0, 1.0, output);
            
            std::cout << "  Superposition: [";
            for (int i = 0; i < size; i++) {
                std::cout << output[i];
                if (i < size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // Expected values for equal superposition should be 1/sqrt(2) ≈ 0.7071... for first two elements
            double expected_val = 1.0 / std::sqrt(2.0);
            
            if (approx_equal(output[0], expected_val, 0.01) && approx_equal(output[1], expected_val, 0.01)) {
                std::cout << "  ✓ Equal superposition has correct values\n";
            } else {
                std::cout << "  ✗ FAIL: Equal superposition values incorrect. "
                          << "Expected ~" << expected_val << " for first two elements.\n";
                all_tests_passed = false;
            }
            
            // Test normalization - sum of squares should be 1
            double sum_squares = 0.0;
            for (int i = 0; i < size; i++) {
                sum_squares += output[i] * output[i];
            }
            
            if (approx_equal(sum_squares, 1.0, 0.01)) {
                std::cout << "  ✓ Superposition is properly normalized\n";
            } else {
                std::cout << "  ✗ FAIL: Superposition not normalized. Sum of squares = " 
                          << sum_squares << " (should be 1.0)\n";
                all_tests_passed = false;
            }
            
            // Test unequal weights - should prioritize the state with larger coefficient
            quantum_superposition(state1, state2, size, 2.0, 1.0, output);
            if (std::abs(output[0]) > std::abs(output[1])) {
                std::cout << "  ✓ Unequal coefficients correctly affect amplitudes\n";
            } else {
                std::cout << "  ✗ FAIL: Unequal coefficients don't correctly affect amplitudes\n";
                all_tests_passed = false;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in superposition test: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    // Test 4: Hadamard Transform - Verify specific mathematical properties
    {
        std::cout << "\nTest 4: Hadamard Transform\n";
        const int size = 4;
        double input[size] = {1.0, 0.0, 0.0, 0.0}; // |0⟩ state
        double output[size];
        
        try {
            hadamard_transform(input, size, output);
            
            std::cout << "  Hadamard applied to |0⟩: [";
            for (int i = 0; i < size; i++) {
                std::cout << output[i];
                if (i < size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // For H|0⟩, all amplitudes should be equal = 1/sqrt(N)
            double expected = 1.0 / std::sqrt(size);
            bool all_equal = true;
            
            for (int i = 0; i < size; i++) {
                if (!approx_equal(std::abs(output[i]), expected, 0.01)) {
                    all_equal = false;
                    break;
                }
            }
            
            if (all_equal) {
                std::cout << "  ✓ Hadamard on |0⟩ creates equal superposition\n";
            } else {
                std::cout << "  ✗ FAIL: Hadamard on |0⟩ does not create equal superposition\n";
                all_tests_passed = false;
            }
            
            // Apply Hadamard again - should return to original state
            double twice_transformed[size];
            hadamard_transform(output, size, twice_transformed);
            
            // H²|0⟩ = |0⟩ (up to normalization)
            bool returns_to_original = true;
            double norm = std::abs(twice_transformed[0]);
            std::vector<double> expected_result = {norm, 0, 0, 0};
            
            if (validate_vector(twice_transformed, size, expected_result, 0.01)) {
                std::cout << "  ✓ H² returns to original state (up to normalization)\n";
            } else {
                std::cout << "  ✗ FAIL: H² does not return to original state\n";
                all_tests_passed = false;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in Hadamard test: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    // Test 5: Resonance Signature - Verify it produces consistent results
    {
        std::cout << "\nTest 5: Resonance Signature\n";
        const int in_size = 8;
        const int sig_size = 4;
        double data[in_size] = {1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5};
        double signature1[sig_size];
        double signature2[sig_size];
        
        try {
            // Generate signature
            generate_resonance_signature(data, in_size, signature1, sig_size);
            
            std::cout << "  Signature: [";
            for (int i = 0; i < sig_size; i++) {
                std::cout << signature1[i];
                if (i < sig_size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // Signature should have meaningful values (not all zeros)
            double sum_abs = 0.0;
            for (int i = 0; i < sig_size; i++) {
                sum_abs += std::abs(signature1[i]);
            }
            
            if (sum_abs > 0.01) {
                std::cout << "  ✓ Signature has meaningful values\n";
            } else {
                std::cout << "  ✗ FAIL: Signature values are too small\n";
                all_tests_passed = false;
            }
            
            // Same input should produce same signature (consistency)
            generate_resonance_signature(data, in_size, signature2, sig_size);
            bool consistent = true;
            
            for (int i = 0; i < sig_size; i++) {
                if (!approx_equal(signature1[i], signature2[i])) {
                    consistent = false;
                    break;
                }
            }
            
            if (consistent) {
                std::cout << "  ✓ Signature generation is consistent\n";
            } else {
                std::cout << "  ✗ FAIL: Signature generation is not consistent\n";
                all_tests_passed = false;
            }
            
            // Different inputs should produce different signatures
            double different_data[in_size] = {0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4};
            double different_signature[sig_size];
            
            generate_resonance_signature(different_data, in_size, different_signature, sig_size);
            bool all_same = true;
            
            for (int i = 0; i < sig_size; i++) {
                if (!approx_equal(signature1[i], different_signature[i])) {
                    all_same = false;
                    break;
                }
            }
            
            if (!all_same) {
                std::cout << "  ✓ Different inputs produce different signatures\n";
            } else {
                std::cout << "  ✗ FAIL: Different inputs produce identical signatures\n";
                all_tests_passed = false;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in resonance signature test: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    // Final results
    std::cout << "\n=== Test Summary ===\n";
    if (all_tests_passed) {
        std::cout << "✓ ALL TESTS PASSED: QuantoniumOS implementation is valid\n";
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED: See details above\n";
        return 1;
    }
}
