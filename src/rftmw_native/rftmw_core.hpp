/*
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
 * under LICENSE-CLAIMS-NC.md (research/education only). Commercial
 * rights require a separate patent license from the author.
 *
 * rftmw_core.hpp - Φ-RFT Transform C++ Core
 * ==========================================
 *
 * High-performance RFTMW implementation with SIMD acceleration.
 * This is the heavy-lifting engine; Python calls this via pybind11.
 *
 * Architecture:
 *   ASM kernels (rft_kernel_asm.asm, feistel_round48.asm, etc.)
 *       ↓
 *   C++ SIMD wrappers (rftmw_asm_kernels.hpp)
 *       ↓
 *   C++ engine (this file)
 *       ↓
 *   Python bindings (rftmw_python.cpp via pybind11)
 *
 * Assembly Kernels Available:
 *   - rft_transform_asm: Unitary RFT transform (matrix-vector)
 *   - rft_basis_multiply_asm: Matrix-vector with transpose
 *   - rft_quantum_gate_asm: Quantum gate application
 *   - qsc_symbolic_compression_asm: Million-qubit symbolic compression
 *   - feistel_encrypt_batch_asm: 48-round Feistel cipher (9.2 MB/s target)
 */

#pragma once

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#define RFTMW_HAS_AVX2 1
#else
#define RFTMW_HAS_AVX2 0
#endif

#ifdef __AVX512F__
#define RFTMW_HAS_AVX512 1
#else
#define RFTMW_HAS_AVX512 0
#endif

#ifdef __FMA__
#define RFTMW_HAS_FMA 1
#else
#define RFTMW_HAS_FMA 0
#endif

// Check if ASM kernels are available (set by build system)
#ifndef RFTMW_ENABLE_ASM
#define RFTMW_ENABLE_ASM 0
#endif

#if RFTMW_ENABLE_ASM
#include "rftmw_asm_kernels.hpp"
#endif

namespace rftmw {

// Golden ratio constant (compile-time)
constexpr double PHI = 1.6180339887498948482045868343656;
constexpr double PHI_INV = 0.6180339887498948482045868343656;
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;
using RealVec = std::vector<double>;

// ============================================================================
// SIMD-Accelerated Phase Modulation Kernels
// ============================================================================

#if RFTMW_HAS_AVX2

/**
 * AVX2-accelerated golden phase computation.
 * Computes: phase[k] = 2π * φ^(k/n) for k = 0..n-1
 */
inline void compute_golden_phases_avx2(double* phases, size_t n) {
    const double log_phi = std::log(PHI);
    const double inv_n = 1.0 / static_cast<double>(n);
    
    // Process 4 doubles at a time with AVX2
    __m256d v_two_pi = _mm256_set1_pd(TWO_PI);
    __m256d v_log_phi = _mm256_set1_pd(log_phi);
    __m256d v_inv_n = _mm256_set1_pd(inv_n);
    
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        __m256d v_k = _mm256_set_pd(
            static_cast<double>(k + 3),
            static_cast<double>(k + 2),
            static_cast<double>(k + 1),
            static_cast<double>(k)
        );
        
        // exp(k/n * log(phi)) = phi^(k/n)
        __m256d v_exponent = _mm256_mul_pd(_mm256_mul_pd(v_k, v_inv_n), v_log_phi);
        
        // Approximate exp using Taylor series (fast path)
        // For production, use Intel SVML or custom exp implementation
        __m256d v_phi_pow = _mm256_set_pd(
            std::exp(((k + 3) * inv_n) * log_phi),
            std::exp(((k + 2) * inv_n) * log_phi),
            std::exp(((k + 1) * inv_n) * log_phi),
            std::exp((k * inv_n) * log_phi)
        );
        
        __m256d v_phase = _mm256_mul_pd(v_two_pi, v_phi_pow);
        _mm256_storeu_pd(phases + k, v_phase);
    }
    
    // Handle remainder
    for (; k < n; ++k) {
        phases[k] = TWO_PI * std::pow(PHI, static_cast<double>(k) / n);
    }
}

/**
 * AVX2-accelerated complex multiplication with phase rotation.
 * result[k] = input[k] * exp(i * phase[k])
 */
inline void apply_phase_rotation_avx2(
    const Complex* input,
    Complex* output,
    const double* phases,
    size_t n
) {
    // Process 2 complex numbers at a time (4 doubles = 2 complex)
    for (size_t k = 0; k < n; k += 2) {
        if (k + 2 <= n) {
            // Load 2 complex numbers (4 doubles: r0, i0, r1, i1)
            __m256d v_in = _mm256_loadu_pd(reinterpret_cast<const double*>(input + k));
            
            // Compute cos/sin of phases
            double c0 = std::cos(phases[k]);
            double s0 = std::sin(phases[k]);
            double c1 = std::cos(phases[k + 1]);
            double s1 = std::sin(phases[k + 1]);
            
            // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            // input = [r0, i0, r1, i1]
            // rotation = [c0, s0, c1, s1]
            
            double r0 = input[k].real(), i0 = input[k].imag();
            double r1 = input[k+1].real(), i1 = input[k+1].imag();
            
            output[k] = Complex(r0*c0 - i0*s0, r0*s0 + i0*c0);
            output[k+1] = Complex(r1*c1 - i1*s1, r1*s1 + i1*c1);
        } else {
            // Handle last element if n is odd
            double c = std::cos(phases[k]);
            double s = std::sin(phases[k]);
            output[k] = Complex(
                input[k].real()*c - input[k].imag()*s,
                input[k].real()*s + input[k].imag()*c
            );
        }
    }
}

#endif // RFTMW_HAS_AVX2

// ============================================================================
// Scalar Fallback Implementations
// ============================================================================

inline void compute_golden_phases_scalar(double* phases, size_t n) {
    const double inv_n = 1.0 / static_cast<double>(n);
    for (size_t k = 0; k < n; ++k) {
        phases[k] = TWO_PI * std::pow(PHI, static_cast<double>(k) * inv_n);
    }
}

inline void apply_phase_rotation_scalar(
    const Complex* input,
    Complex* output,
    const double* phases,
    size_t n
) {
    for (size_t k = 0; k < n; ++k) {
        double c = std::cos(phases[k]);
        double s = std::sin(phases[k]);
        output[k] = Complex(
            input[k].real() * c - input[k].imag() * s,
            input[k].real() * s + input[k].imag() * c
        );
    }
}

// ============================================================================
// Cooley-Tukey FFT Implementation (for RFT core)
// ============================================================================

/**
 * In-place radix-2 Cooley-Tukey FFT.
 * n must be a power of 2.
 */
inline void fft_radix2_inplace(Complex* data, size_t n, bool inverse = false) {
    if (n <= 1) return;
    
    // Bit-reversal permutation
    size_t j = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        size_t m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    
    // Cooley-Tukey iterative FFT
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = (inverse ? TWO_PI : -TWO_PI) / static_cast<double>(len);
        Complex wlen(std::cos(angle), std::sin(angle));
        
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                Complex u = data[i + k];
                Complex t = w * data[i + k + len / 2];
                data[i + k] = u + t;
                data[i + k + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
    
    // Normalize for inverse FFT
    if (inverse) {
        double inv_n = 1.0 / static_cast<double>(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] *= inv_n;
        }
    }
}

/**
 * Mixed-radix FFT for non-power-of-2 sizes.
 * Falls back to DFT for prime factors.
 */
inline void fft_mixed_radix(Complex* data, size_t n, bool inverse = false) {
    // For power-of-2, use fast path
    if ((n & (n - 1)) == 0) {
        fft_radix2_inplace(data, n, inverse);
        return;
    }
    
    // General DFT for non-power-of-2 (O(n^2) fallback)
    ComplexVec result(n);
    double sign = inverse ? 1.0 : -1.0;
    
    for (size_t k = 0; k < n; ++k) {
        result[k] = Complex(0.0, 0.0);
        for (size_t j = 0; j < n; ++j) {
            double angle = sign * TWO_PI * static_cast<double>(k * j) / static_cast<double>(n);
            result[k] += data[j] * Complex(std::cos(angle), std::sin(angle));
        }
        if (inverse) {
            result[k] /= static_cast<double>(n);
        }
    }
    
    std::copy(result.begin(), result.end(), data);
}

// ============================================================================
// RFTMW Transform Class
// ============================================================================

class RFTMWEngine {
public:
    enum class Normalization {
        NONE,      // No normalization
        FORWARD,   // 1/n on forward
        ORTHO,     // 1/sqrt(n) on both
        BACKWARD   // 1/n on inverse (default FFT)
    };
    
    enum class Backend {
        AUTO,      // Auto-select best available
        ASM,       // Force assembly kernels
        SIMD,      // Force C++ SIMD (AVX2/SSE)
        SCALAR     // Force scalar fallback
    };
    
    // RFT Variant - matches C kernel variants
    enum class Variant {
        LEGACY,     // Original phase-modulated FFT (deprecated)
        CANONICAL,  // USPTO Patent 19/169,399 Claim 1: fₖ=(k+1)×φ, θₖ=2πk/φ
        BINARY_WAVE // USPTO Patent 19/169,399: BinaryRFT wave-domain logic
    };

private:
    size_t max_size_;
    Normalization norm_;
    Backend backend_;
    Variant variant_;
    RealVec phase_cache_;
    ComplexVec basis_cache_;  // For ASM kernel basis matrix
    bool use_simd_;
    bool use_asm_;
    
public:
    explicit RFTMWEngine(
        size_t max_size = 65536, 
        Normalization norm = Normalization::ORTHO,
        Backend backend = Backend::AUTO,
        Variant variant = Variant::CANONICAL  // Default to canonical
    )
        : max_size_(max_size)
        , norm_(norm)
        , backend_(backend)
        , variant_(variant)
        , phase_cache_(max_size)
        , use_simd_(RFTMW_HAS_AVX2)
        , use_asm_(RFTMW_ENABLE_ASM)
    {
        // Resolve AUTO backend
        if (backend_ == Backend::AUTO) {
#if RFTMW_ENABLE_ASM
            use_asm_ = true;
            use_simd_ = true;
#elif RFTMW_HAS_AVX2
            use_asm_ = false;
            use_simd_ = true;
#else
            use_asm_ = false;
            use_simd_ = false;
#endif
        } else if (backend_ == Backend::ASM) {
#if RFTMW_ENABLE_ASM
            use_asm_ = true;
#else
            throw std::runtime_error("ASM backend requested but not compiled with -DRFTMW_ENABLE_ASM=ON");
#endif
        } else if (backend_ == Backend::SIMD) {
            use_asm_ = false;
            use_simd_ = RFTMW_HAS_AVX2;
        } else {
            use_asm_ = false;
            use_simd_ = false;
        }
        
        // Pre-compute phases for max size
        precompute_phases(max_size);
        
        // Pre-compute basis matrix for ASM kernel if enabled
        if (use_asm_) {
            precompute_basis(max_size);
        }
    }
    
    /**
     * Pre-compute RFT basis matrix for ASM kernel.
     * 
     * For CANONICAL variant (USPTO Patent 19/169,399):
     *   Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
     *   where fₖ = (k+1) × φ, θₖ = 2πk/φ
     * 
     * For LEGACY variant:
     *   Basis[i,j] = exp(i * 2π * φ^(i*j/n))
     * 
     * Note: Only precompute for small sizes to avoid memory explosion
     * (n×n matrix would be 64GB for n=65536)
     */
    void precompute_basis(size_t n) {
        // Limit basis precomputation to avoid O(n²) memory explosion
        // For large transforms, compute on-the-fly or use ASM kernel's internal basis
        constexpr size_t MAX_BASIS_SIZE = 4096;
        
        if (n > MAX_BASIS_SIZE) {
            // Don't precompute giant basis matrices
            // ASM kernel will handle large transforms differently
            basis_cache_.clear();
            return;
        }
        
        basis_cache_.resize(n * n);
        const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
        
        if (variant_ == Variant::CANONICAL || variant_ == Variant::BINARY_WAVE) {
            // USPTO Patent 19/169,399 Claim 1: Canonical RFT
            // Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
            for (size_t k = 0; k < n; ++k) {
                double f_k = (static_cast<double>(k) + 1.0) * PHI;  // Resonant frequency
                double theta_k = TWO_PI * static_cast<double>(k) / PHI;  // Golden phase
                
                for (size_t t_idx = 0; t_idx < n; ++t_idx) {
                    double t = static_cast<double>(t_idx) / static_cast<double>(n);
                    double angle = TWO_PI * f_k * t + theta_k;
                    basis_cache_[k * n + t_idx] = Complex(
                        std::cos(angle) * inv_sqrt_n,
                        std::sin(angle) * inv_sqrt_n
                    );
                }
            }
        } else {
            // Legacy phase-modulated FFT
            const double inv_n2 = 1.0 / static_cast<double>(n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double exponent = static_cast<double>(i * j) * inv_n2;
                    double phase = TWO_PI * std::pow(PHI, exponent);
                    basis_cache_[i * n + j] = Complex(std::cos(phase), std::sin(phase));
                }
            }
        }
    }
    
    void precompute_phases(size_t n) {
        if (n > phase_cache_.size()) {
            phase_cache_.resize(n);
        }
        
#if RFTMW_HAS_AVX2
        if (use_simd_) {
            compute_golden_phases_avx2(phase_cache_.data(), n);
        } else
#endif
        {
            compute_golden_phases_scalar(phase_cache_.data(), n);
        }
    }
    
    /**
     * Forward Φ-RFT transform.
     * 
     * RFT(x) = FFT(x ⊙ exp(i·2π·φ^(k/n)))
     * 
     * Where ⊙ is element-wise multiplication.
     */
    ComplexVec forward(const RealVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        // Ensure phases are cached
        if (n > phase_cache_.size()) {
            precompute_phases(n);
        }
        
        // Convert real to complex and apply golden phase modulation
        ComplexVec data(n);
        for (size_t k = 0; k < n; ++k) {
            data[k] = Complex(input[k], 0.0);
        }
        
        ComplexVec modulated(n);
        
#if RFTMW_HAS_AVX2
        if (use_simd_) {
            apply_phase_rotation_avx2(data.data(), modulated.data(), phase_cache_.data(), n);
        } else
#endif
        {
            apply_phase_rotation_scalar(data.data(), modulated.data(), phase_cache_.data(), n);
        }
        
        // Apply FFT
        fft_mixed_radix(modulated.data(), n, false);
        
        // Apply normalization
        apply_normalization(modulated, false);
        
        return modulated;
    }
    
    ComplexVec forward_complex(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        if (n > phase_cache_.size()) {
            precompute_phases(n);
        }
        
        ComplexVec modulated(n);
        
#if RFTMW_HAS_AVX2
        if (use_simd_) {
            apply_phase_rotation_avx2(input.data(), modulated.data(), phase_cache_.data(), n);
        } else
#endif
        {
            apply_phase_rotation_scalar(input.data(), modulated.data(), phase_cache_.data(), n);
        }
        
        fft_mixed_radix(modulated.data(), n, false);
        apply_normalization(modulated, false);
        
        return modulated;
    }
    
    /**
     * Inverse Φ-RFT transform.
     * 
     * IRFT(X) = IFFT(X) ⊙ exp(-i·2π·φ^(k/n))
     */
    RealVec inverse(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        if (n > phase_cache_.size()) {
            precompute_phases(n);
        }
        
        // Copy and apply IFFT
        ComplexVec data = input;
        fft_mixed_radix(data.data(), n, true);
        
        // Apply inverse phase modulation (conjugate phases)
        RealVec phases_neg(n);
        for (size_t k = 0; k < n; ++k) {
            phases_neg[k] = -phase_cache_[k];
        }
        
        ComplexVec demodulated(n);
        apply_phase_rotation_scalar(data.data(), demodulated.data(), phases_neg.data(), n);
        
        // Apply normalization
        apply_normalization(demodulated, true);
        
        // Extract real part
        RealVec result(n);
        for (size_t k = 0; k < n; ++k) {
            result[k] = demodulated[k].real();
        }
        
        return result;
    }
    
    ComplexVec inverse_complex(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        if (n > phase_cache_.size()) {
            precompute_phases(n);
        }
        
        ComplexVec data = input;
        fft_mixed_radix(data.data(), n, true);
        
        RealVec phases_neg(n);
        for (size_t k = 0; k < n; ++k) {
            phases_neg[k] = -phase_cache_[k];
        }
        
        ComplexVec demodulated(n);
        apply_phase_rotation_scalar(data.data(), demodulated.data(), phases_neg.data(), n);
        
        apply_normalization(demodulated, true);
        
        return demodulated;
    }
    
    /**
     * Forward Canonical RFT (USPTO Patent 19/169,399 Claim 1)
     * 
     * X[k] = Σₙ x[n] × Ψₖ*(n/N)
     * where Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
     *       fₖ = (k+1) × φ
     *       θₖ = 2πk / φ
     */
    ComplexVec forward_canonical(const RealVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        ComplexVec result(n);
        const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
        
        for (size_t k = 0; k < n; ++k) {
            double f_k = (static_cast<double>(k) + 1.0) * PHI;
            double theta_k = TWO_PI * static_cast<double>(k) / PHI;
            
            Complex sum(0.0, 0.0);
            for (size_t t_idx = 0; t_idx < n; ++t_idx) {
                double t = static_cast<double>(t_idx) / static_cast<double>(n);
                double angle = TWO_PI * f_k * t + theta_k;
                // Conjugate for forward transform
                Complex basis_conj(std::cos(angle), -std::sin(angle));
                sum += input[t_idx] * basis_conj;
            }
            result[k] = sum * inv_sqrt_n;
        }
        
        return result;
    }
    
    /**
     * Inverse Canonical RFT (USPTO Patent 19/169,399 Claim 1)
     * 
     * x[n] = Σₖ X[k] × Ψₖ(n/N)
     */
    RealVec inverse_canonical(const ComplexVec& input) {
        size_t n = input.size();
        if (n == 0) return {};
        
        RealVec result(n);
        const double inv_sqrt_n = 1.0 / std::sqrt(static_cast<double>(n));
        
        for (size_t t_idx = 0; t_idx < n; ++t_idx) {
            double t = static_cast<double>(t_idx) / static_cast<double>(n);
            
            Complex sum(0.0, 0.0);
            for (size_t k = 0; k < n; ++k) {
                double f_k = (static_cast<double>(k) + 1.0) * PHI;
                double theta_k = TWO_PI * static_cast<double>(k) / PHI;
                double angle = TWO_PI * f_k * t + theta_k;
                
                Complex basis(std::cos(angle), std::sin(angle));
                sum += input[k] * basis;
            }
            result[t_idx] = sum.real() * inv_sqrt_n;
        }
        
        return result;
    }
    
    /**
     * Get the current variant.
     */
    Variant variant() const { return variant_; }
    
    /**
     * Set the variant (for runtime switching).
     */
    void set_variant(Variant v) { variant_ = v; }
    
    // Accessors
    bool has_simd() const { return use_simd_; }
    size_t max_size() const { return max_size_; }
    Normalization normalization() const { return norm_; }
    
private:
    void apply_normalization(ComplexVec& data, bool is_inverse) {
        size_t n = data.size();
        double factor = 1.0;
        
        switch (norm_) {
            case Normalization::ORTHO:
                factor = 1.0 / std::sqrt(static_cast<double>(n));
                break;
            case Normalization::FORWARD:
                factor = is_inverse ? 1.0 : (1.0 / static_cast<double>(n));
                break;
            case Normalization::BACKWARD:
                factor = is_inverse ? (1.0 / static_cast<double>(n)) : 1.0;
                break;
            case Normalization::NONE:
            default:
                return;
        }
        
        for (auto& c : data) {
            c *= factor;
        }
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

inline ComplexVec rft_forward(const RealVec& input) {
    static thread_local RFTMWEngine engine;
    return engine.forward(input);
}

inline RealVec rft_inverse(const ComplexVec& input) {
    static thread_local RFTMWEngine engine;
    return engine.inverse(input);
}

} // namespace rftmw
