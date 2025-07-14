#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>
#include <cstring>
#include "quantoniumos/secure_core/include/symbolic_eigenvector.h"

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

// Forward declarations for the validation functions we'll need
inline void validate_inputs(const double* ptr, int size, const char* name) {
    if (!ptr) throw std::invalid_argument(std::string(name) + " pointer is null");
    if (size <= 0) throw std::invalid_argument(std::string(name) + " size must be positive");
}

int main() {
    std::cout << "=== QuantoniumOS Simple Validation Tests ===\n";
    bool all_tests_passed = true;

    // Test 1: Encode/Decode Resonance
    {
        std::cout << "\nTest 1: Encode/Decode Resonance\n";
        const char* test_data = "QuantoniumOS";
        char encoded[1024];
        int encoded_len = 0;
        
        try {
            // Encode the data
            encode_resonance(test_data, encoded, &encoded_len);
            std::cout << "  Encoded: " << encoded << std::endl;
            
            // Verify decoding works correctly
            char decoded[1024];
            int decoded_len = 0;
            decode_resonance(encoded, decoded, &decoded_len);
            std::cout << "  Decoded: " << decoded << std::endl;
            
            if (strcmp(test_data, decoded) == 0) {
                std::cout << "  ✓ Encode/decode test passed\n";
            } else {
                std::cout << "  ✗ FAIL: Encode/decode test failed\n";
                all_tests_passed = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in encode/decode test: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    // Test 2: U (State Update Function)
    {
        std::cout << "\nTest 2: U (State Update Function)\n";
        const int size = 4;
        double state[size] = {1.0, 2.0, 3.0, 4.0};
        double derivative[size] = {0.1, 0.2, 0.3, 0.4};
        double dt = 0.5;
        double output[size];
        
        try {
            U(state, derivative, size, output, dt);
            
            std::cout << "  Result: [";
            for (int i = 0; i < size; i++) {
                std::cout << output[i];
                if (i < size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // Expected: state + dt * derivative
            std::vector<double> expected = {
                state[0] + dt * derivative[0],
                state[1] + dt * derivative[1],
                state[2] + dt * derivative[2],
                state[3] + dt * derivative[3]
            };
            
            if (validate_vector(output, size, expected)) {
                std::cout << "  ✓ U function test passed\n";
            } else {
                std::cout << "  ✗ FAIL: U function test failed\n";
                all_tests_passed = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in U function test: " << e.what() << std::endl;
            all_tests_passed = false;
        }
    }
    
    // Test 3: T (Transform Function)
    {
        std::cout << "\nTest 3: T (Transform Function)\n";
        const int size = 4;
        double state[size] = {1.0, 2.0, 3.0, 4.0};
        double transform[size] = {0.5, 0.5, 0.5, 0.5};
        double output[size];
        
        try {
            T(state, transform, size, output);
            
            std::cout << "  Result: [";
            for (int i = 0; i < size; i++) {
                std::cout << output[i];
                if (i < size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            
            // Expected: state * transform (element-wise)
            std::vector<double> expected = {
                state[0] * transform[0],
                state[1] * transform[1],
                state[2] * transform[2],
                state[3] * transform[3]
            };
            
            if (validate_vector(output, size, expected)) {
                std::cout << "  ✓ T function test passed\n";
            } else {
                std::cout << "  ✗ FAIL: T function test failed\n";
                all_tests_passed = false;
            }
        } catch (const std::exception& e) {
            std::cerr << "  ✗ FAIL: Error in T function test: " << e.what() << std::endl;
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
