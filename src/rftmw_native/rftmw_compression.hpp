/*
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
 * under LICENSE-CLAIMS-NC.md (research/education only). Commercial
 * rights require a separate patent license from the author.
 *
 * rftmw_compression.hpp - RFTMW Compression Pipeline
 * ===================================================
 *
 * High-performance compression using RFTMW + quantization + entropy coding.
 */

#pragma once

#include "rftmw_core.hpp"
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace rftmw {

// ============================================================================
// Quantization
// ============================================================================

struct QuantizationParams {
    int precision_bits = 16;      // Bits per coefficient
    double dead_zone = 0.0;       // Values below this become zero
    bool use_log_scale = false;   // Logarithmic quantization for wide dynamic range
};

/**
 * Quantize complex coefficients to fixed-point integers.
 */
inline std::vector<int32_t> quantize_coefficients(
    const ComplexVec& coeffs,
    const QuantizationParams& params,
    double& scale_factor,
    double& max_value
) {
    size_t n = coeffs.size();
    std::vector<int32_t> quantized(n * 2);  // real + imag interleaved
    
    // Find max magnitude for scaling
    max_value = 0.0;
    for (const auto& c : coeffs) {
        max_value = std::max(max_value, std::abs(c.real()));
        max_value = std::max(max_value, std::abs(c.imag()));
    }
    
    if (max_value < 1e-30) max_value = 1.0;  // Avoid division by zero
    
    // Scale factor: map [-max, max] to [-2^(b-1), 2^(b-1)-1]
    int32_t max_int = (1 << (params.precision_bits - 1)) - 1;
    scale_factor = static_cast<double>(max_int) / max_value;
    
    for (size_t i = 0; i < n; ++i) {
        double re = coeffs[i].real();
        double im = coeffs[i].imag();
        
        // Apply dead zone
        if (std::abs(re) < params.dead_zone) re = 0.0;
        if (std::abs(im) < params.dead_zone) im = 0.0;
        
        // Quantize
        quantized[2*i]     = static_cast<int32_t>(std::round(re * scale_factor));
        quantized[2*i + 1] = static_cast<int32_t>(std::round(im * scale_factor));
        
        // Clamp to valid range
        quantized[2*i]     = std::clamp(quantized[2*i], -max_int, max_int);
        quantized[2*i + 1] = std::clamp(quantized[2*i + 1], -max_int, max_int);
    }
    
    return quantized;
}

/**
 * Dequantize fixed-point integers back to complex coefficients.
 */
inline ComplexVec dequantize_coefficients(
    const std::vector<int32_t>& quantized,
    double scale_factor
) {
    size_t n = quantized.size() / 2;
    ComplexVec coeffs(n);
    
    double inv_scale = 1.0 / scale_factor;
    
    for (size_t i = 0; i < n; ++i) {
        double re = static_cast<double>(quantized[2*i]) * inv_scale;
        double im = static_cast<double>(quantized[2*i + 1]) * inv_scale;
        coeffs[i] = Complex(re, im);
    }
    
    return coeffs;
}

// ============================================================================
// Simple ANS (Asymmetric Numeral Systems) Implementation
// ============================================================================

/**
 * Table-based ANS encoder for fast entropy coding.
 */
class ANSEncoder {
private:
    static constexpr size_t TABLE_LOG = 12;
    static constexpr size_t TABLE_SIZE = 1 << TABLE_LOG;
    
    std::vector<uint32_t> symbol_table_;
    std::vector<uint32_t> freq_table_;
    std::vector<uint32_t> cumfreq_table_;
    size_t alphabet_size_;
    
public:
    explicit ANSEncoder(const std::vector<uint32_t>& frequencies)
        : alphabet_size_(frequencies.size())
    {
        // Normalize frequencies to sum to TABLE_SIZE
        uint64_t total = std::accumulate(frequencies.begin(), frequencies.end(), 0ULL);
        
        freq_table_.resize(alphabet_size_);
        cumfreq_table_.resize(alphabet_size_ + 1);
        cumfreq_table_[0] = 0;
        
        uint32_t assigned = 0;
        for (size_t i = 0; i < alphabet_size_; ++i) {
            // Ensure at least 1 for non-zero frequencies
            freq_table_[i] = std::max(1U, 
                static_cast<uint32_t>((frequencies[i] * TABLE_SIZE + total/2) / total));
            cumfreq_table_[i + 1] = cumfreq_table_[i] + freq_table_[i];
        }
        
        // Adjust to exactly TABLE_SIZE
        int32_t diff = static_cast<int32_t>(TABLE_SIZE) - static_cast<int32_t>(cumfreq_table_.back());
        if (diff != 0 && alphabet_size_ > 0) {
            freq_table_[0] += diff;
            for (size_t i = 1; i <= alphabet_size_; ++i) {
                cumfreq_table_[i] += diff;
            }
        }
        
        // Build symbol table for decoding
        symbol_table_.resize(TABLE_SIZE);
        for (size_t s = 0; s < alphabet_size_; ++s) {
            for (uint32_t i = cumfreq_table_[s]; i < cumfreq_table_[s + 1]; ++i) {
                symbol_table_[i] = s;
            }
        }
    }
    
    /**
     * Encode a sequence of symbols.
     * Returns compressed data + final state.
     */
    std::pair<std::vector<uint8_t>, uint64_t> encode(const std::vector<uint32_t>& symbols) {
        std::vector<uint8_t> output;
        output.reserve(symbols.size() * 2);  // Approximate
        
        uint64_t state = TABLE_SIZE;  // Initial state
        
        // Encode in reverse order (ANS is LIFO)
        for (auto it = symbols.rbegin(); it != symbols.rend(); ++it) {
            uint32_t s = *it;
            if (s >= alphabet_size_) {
                throw std::runtime_error("Symbol out of range");
            }
            
            uint32_t freq = freq_table_[s];
            uint32_t cumfreq = cumfreq_table_[s];
            
            // Renormalize: emit bytes while state is too large
            while (state >= (TABLE_SIZE * freq)) {
                output.push_back(state & 0xFF);
                state >>= 8;
            }
            
            // Update state: state' = (state / freq) * TABLE_SIZE + (state % freq) + cumfreq
            state = ((state / freq) << TABLE_LOG) + (state % freq) + cumfreq;
        }
        
        // Emit final state
        for (int i = 0; i < 8; ++i) {
            output.push_back((state >> (i * 8)) & 0xFF);
        }
        
        // Reverse output (since we wrote in reverse)
        std::reverse(output.begin(), output.end());
        
        return {output, state};
    }
    
    /**
     * Decode a sequence of symbols.
     */
    std::vector<uint32_t> decode(const std::vector<uint8_t>& data, size_t num_symbols) {
        std::vector<uint32_t> symbols(num_symbols);
        
        // Read initial state from end of data
        if (data.size() < 8) {
            throw std::runtime_error("Data too short");
        }
        
        uint64_t state = 0;
        for (int i = 0; i < 8; ++i) {
            state |= static_cast<uint64_t>(data[i]) << (i * 8);
        }
        
        size_t byte_pos = 8;
        
        // Decode symbols
        for (size_t i = 0; i < num_symbols; ++i) {
            uint32_t slot = state & (TABLE_SIZE - 1);
            uint32_t s = symbol_table_[slot];
            symbols[i] = s;
            
            uint32_t freq = freq_table_[s];
            uint32_t cumfreq = cumfreq_table_[s];
            
            // Update state
            state = freq * (state >> TABLE_LOG) + slot - cumfreq;
            
            // Renormalize: read bytes while state is too small
            while (state < TABLE_SIZE && byte_pos < data.size()) {
                state = (state << 8) | data[byte_pos++];
            }
        }
        
        return symbols;
    }
    
    // Accessors for serialization
    const std::vector<uint32_t>& frequencies() const { return freq_table_; }
    size_t alphabet_size() const { return alphabet_size_; }
};

// ============================================================================
// RFTMW Compression Pipeline
// ============================================================================

struct CompressionResult {
    std::vector<uint8_t> data;      // Compressed bytes
    size_t original_size;           // Original number of samples
    double scale_factor;            // Quantization scale
    double max_value;               // Max coefficient magnitude
    QuantizationParams quant_params;
    std::vector<uint32_t> frequencies; // Symbol frequencies for ANS
};

class RFTMWCompressor {
private:
    RFTMWEngine engine_;
    QuantizationParams quant_params_;
    
public:
    explicit RFTMWCompressor(
        size_t max_size = 65536,
        QuantizationParams params = {}
    )
        : engine_(max_size)
        , quant_params_(params)
    {}
    
    /**
     * Compress real-valued data.
     */
    CompressionResult compress(const RealVec& input) {
        CompressionResult result;
        result.original_size = input.size();
        result.quant_params = quant_params_;
        
        // 1. Forward RFTMW transform
        ComplexVec coeffs = engine_.forward(input);
        
        // 2. Quantize
        std::vector<int32_t> quantized = quantize_coefficients(
            coeffs, quant_params_, result.scale_factor, result.max_value
        );
        
        // 3. Convert to unsigned for ANS (shift to positive range)
        int32_t offset = 1 << (quant_params_.precision_bits - 1);
        std::vector<uint32_t> symbols(quantized.size());
        for (size_t i = 0; i < quantized.size(); ++i) {
            symbols[i] = static_cast<uint32_t>(quantized[i] + offset);
        }
        
        // 4. Compute symbol frequencies
        size_t alphabet = 1 << quant_params_.precision_bits;
        result.frequencies.resize(alphabet, 0);
        for (uint32_t s : symbols) {
            if (s < alphabet) {
                result.frequencies[s]++;
            }
        }
        
        // Ensure no zero frequencies for ANS
        for (auto& f : result.frequencies) {
            if (f == 0) f = 1;
        }
        
        // 5. ANS encode
        ANSEncoder encoder(result.frequencies);
        auto [compressed, final_state] = encoder.encode(symbols);
        result.data = std::move(compressed);
        
        return result;
    }
    
    /**
     * Decompress to real-valued data.
     */
    RealVec decompress(const CompressionResult& compressed) {
        // 1. ANS decode
        ANSEncoder encoder(compressed.frequencies);
        size_t num_symbols = compressed.original_size * 2;  // real + imag
        std::vector<uint32_t> symbols = encoder.decode(compressed.data, num_symbols);
        
        // 2. Convert back to signed
        int32_t offset = 1 << (compressed.quant_params.precision_bits - 1);
        std::vector<int32_t> quantized(symbols.size());
        for (size_t i = 0; i < symbols.size(); ++i) {
            quantized[i] = static_cast<int32_t>(symbols[i]) - offset;
        }
        
        // 3. Dequantize
        ComplexVec coeffs = dequantize_coefficients(quantized, compressed.scale_factor);
        
        // 4. Inverse RFTMW transform
        RealVec output = engine_.inverse(coeffs);
        
        return output;
    }
    
    // Accessors
    const QuantizationParams& quantization_params() const { return quant_params_; }
    void set_quantization_params(const QuantizationParams& params) { quant_params_ = params; }
};

} // namespace rftmw
