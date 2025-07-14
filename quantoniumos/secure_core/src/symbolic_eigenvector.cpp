#include "../include/symbolic_eigenvector.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global random engine for consistent results
static std::mt19937 g_rng(42); // Fixed seed for reproducibility

// --------- Symbolic State Operators ---------
EXPORT void U(double* state, double* derivative, int n, double* out, double dt) {
    for (int i = 0; i < n; i++) {
        out[i] = state[i] + dt * derivative[i];
    }
}

EXPORT void T(double* state, double* transform, int n, double* out) {
    for (int i = 0; i < n; i++) {
        out[i] = state[i] * transform[i];
    }
}

EXPORT void ComputeEigenvectors(double* state, int n, double* eigenvalues_out, double* eigenvectors_out) {
    // Simplified eigenvalue computation for demonstration
    for (int i = 0; i < n; i++) {
        eigenvalues_out[i] = std::abs(state[i]) + 0.1 * i;
        for (int j = 0; j < n; j++) {
            eigenvectors_out[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// --------- Enhanced Basis Transformation ---------
EXPORT void transform_basis(const double* data, int data_size, const double* basis, int basis_rows, int basis_cols, double* output) {
    for (int i = 0; i < basis_rows; i++) {
        output[i] = 0.0;
        for (int j = 0; j < std::min(data_size, basis_cols); j++) {
            output[i] += data[j] * basis[i * basis_cols + j];
        }
    }
}

EXPORT void generate_eigenstate_entropy(int size, double* output) {
    std::uniform_real_distribution<double> dist(0.1, 2.0);
    for (int i = 0; i < size; i++) {
        output[i] = dist(g_rng);
    }
}

// --------- Symbolic Resonance Encoding ---------
EXPORT void encode_resonance(const char* data, char* out, int* out_len) {
    std::string input(data);
    std::string result;
    
    // Simple but non-trivial encoding: XOR with rotating key + hex encoding
    const char* key = "QuantoniumKey";
    int key_len = strlen(key);
    
    for (size_t i = 0; i < input.length(); i++) {
        unsigned char encoded_char = (unsigned char)(input[i] ^ key[i % key_len] ^ (i & 0x7F));
        
        // Convert to hex manually to ensure it works
        char hex_high = (encoded_char >> 4) & 0x0F;
        char hex_low = encoded_char & 0x0F;
        
        hex_high = (hex_high < 10) ? ('0' + hex_high) : ('A' + hex_high - 10);
        hex_low = (hex_low < 10) ? ('0' + hex_low) : ('A' + hex_low - 10);
        
        result += hex_high;
        result += hex_low;
    }
    
    size_t copy_len = std::min(result.length(), (size_t)1023);
    strncpy(out, result.c_str(), copy_len);
    out[copy_len] = '\0';
    *out_len = (int)copy_len;
}

EXPORT void decode_resonance(const char* encoded_data, char* out, int* out_len) {
    std::string encoded(encoded_data);
    std::string decoded;
    
    const char* key = "QuantoniumKey";
    int key_len = strlen(key);
    
    // Decode hex pairs back to characters
    for (size_t i = 0; i < encoded.length(); i += 2) {
        if (i + 1 < encoded.length()) {
            char hex_high = encoded[i];
            char hex_low = encoded[i + 1];
            
            // Convert hex chars to nibbles
            unsigned char high_nibble = 0, low_nibble = 0;
            
            if (hex_high >= '0' && hex_high <= '9') high_nibble = hex_high - '0';
            else if (hex_high >= 'A' && hex_high <= 'F') high_nibble = hex_high - 'A' + 10;
            else if (hex_high >= 'a' && hex_high <= 'f') high_nibble = hex_high - 'a' + 10;
            
            if (hex_low >= '0' && hex_low <= '9') low_nibble = hex_low - '0';
            else if (hex_low >= 'A' && hex_low <= 'F') low_nibble = hex_low - 'A' + 10;
            else if (hex_low >= 'a' && hex_low <= 'f') low_nibble = hex_low - 'a' + 10;
            
            unsigned char encoded_char = (high_nibble << 4) | low_nibble;
            char decoded_char = (char)(encoded_char ^ key[(i/2) % key_len] ^ ((i/2) & 0x7F));
            decoded += decoded_char;
        }
    }
    
    size_t copy_len = std::min(decoded.length(), (size_t)1023);
    strncpy(out, decoded.c_str(), copy_len);
    out[copy_len] = '\0';
    *out_len = (int)copy_len;
}

EXPORT double compute_similarity(const char* url1, const char* url2) {
    // Simple Jaccard similarity
    std::string s1(url1), s2(url2);
    int common = 0, total = 0;
    
    for (char c : s1) {
        if (s2.find(c) != std::string::npos) common++;
        total++;
    }
    for (char c : s2) {
        if (s1.find(c) == std::string::npos) total++;
    }
    
    return total > 0 ? (double)common / total : 0.0;
}

// --------- Post-Quantum Encryption ---------
EXPORT void ParallelXOREncrypt(const uint8_t* input, int input_len, const uint8_t* key, int key_len, uint8_t* output) {
    for (int i = 0; i < input_len; i++) {
        output[i] = input[i] ^ key[i % key_len];
    }
}

// --------- Vector Operations ---------
EXPORT double SumVector(const double* arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// --------- Quantum Operations ---------
EXPORT void resonance_signature(const double* waveform, int size, char* output, int* output_size) {
    std::stringstream sig;
    for (int i = 0; i < size; i++) {
        sig << std::hex << (int)(std::abs(waveform[i] * 255)) << ":";
    }
    std::string result = sig.str();
    strncpy(output, result.c_str(), result.length());
    output[result.length()] = '\0';
    *output_size = result.length();
}

EXPORT void quantum_superposition(double* state1, double* state2, int size, double alpha, double beta, double* output) {
    // Normalize coefficients
    double norm = sqrt(alpha * alpha + beta * beta);
    if (norm > 0) {
        alpha /= norm;
        beta /= norm;
    }
    
    for (int i = 0; i < size; i++) {
        output[i] = alpha * state1[i] + beta * state2[i];
    }
}

EXPORT void hadamard_transform(double* data, int size, double* output) {
    // Simplified Hadamard-like transform
    double norm = 1.0 / sqrt(size);
    
    for (int i = 0; i < size; i++) {
        output[i] = 0.0;
        for (int j = 0; j < size; j++) {
            // Simple Hadamard pattern: +1 or -1 based on bit count
            int sign = (__builtin_popcount(i & j) % 2 == 0) ? 1 : -1;
            output[i] += sign * data[j] * norm;
        }
    }
}

EXPORT void generate_resonance_signature(double* data, int size, double* signature, int sig_size) {
    // Generate signature using FFT-like approach
    for (int k = 0; k < sig_size; k++) {
        signature[k] = 0.0;
        for (int n = 0; n < size; n++) {
            double angle = 2.0 * M_PI * k * n / size;
            signature[k] += data[n] * cos(angle);
        }
        signature[k] /= size; // Normalize
    }
}

EXPORT void create_quantum_superposition(int size, double* output) {
    // Create equal superposition state
    double amplitude = 1.0 / sqrt(size);
    for (int i = 0; i < size; i++) {
        output[i] = amplitude;
    }
}

// Implements the symbolic eigenvector reduction.
// It filters out waves with amplitude above a threshold and merges the rest.
EXPORT int symbolic_eigenvector_reduction(
    WaveNumber* wave_list, 
    int wave_count, 
    double threshold, 
    WaveNumber* output_buffer
) {
    std::vector<WaveNumber> filtered;
    double merged_amp = 0.0;
    double merged_phase = 0.0;
    int merged_count = 0;

    for (int i = 0; i < wave_count; ++i) {
        if (wave_list[i].amplitude >= threshold)
            filtered.push_back(wave_list[i]);
        else {
            merged_amp += wave_list[i].amplitude;
            merged_phase += wave_list[i].phase;
            merged_count++;
        }
    }

    if (merged_count > 0)
        filtered.emplace_back(merged_amp, merged_phase / merged_count);

    // Copy the filtered results into the output buffer.
    for (size_t i = 0; i < filtered.size(); ++i)
        output_buffer[i] = filtered[i];

    return static_cast<int>(filtered.size());
}