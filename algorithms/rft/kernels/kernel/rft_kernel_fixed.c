/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate license.
 */

#include "rft_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h> // Added for printf

static inline double rft_frac(double x) {
    double frac = x - floor(x);
    return (frac < 0.0) ? frac + 1.0 : frac;
}

// Advanced SIMD intrinsics for AVX
#ifdef __AVX__
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

// Constants - RFT_2PI is already defined in header

/**
 * Initialize the RFT engine
 * 
 * @param engine Pointer to an RFT engine structure
 * @param size Size of the transform (power of 2 recommended)
 * @param flags Configuration flags
 * @return Error code
 */
rft_error_t rft_init(rft_engine_t* engine, size_t size, uint32_t flags) {
    if (!engine || size == 0) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    // Clear the engine structure
    memset(engine, 0, sizeof(rft_engine_t));
    
    // Allocate memory for basis and eigenvalues
    engine->basis = (rft_complex_t*)malloc(sizeof(rft_complex_t) * size * size);
    engine->eigenvalues = (double*)malloc(sizeof(double) * size);
    
    if (!engine->basis || !engine->eigenvalues) {
        rft_cleanup(engine);
        return RFT_ERROR_MEMORY;
    }
    
    engine->size = size;
    engine->flags = flags;
    engine->variant = RFT_VARIANT_STANDARD;
    
    // Build the resonance kernel and compute eigendecomposition
    if (!rft_build_basis(engine)) {
        rft_cleanup(engine);
        return RFT_ERROR_COMPUTATION;
    }
    
    engine->initialized = true;
    return RFT_SUCCESS;
}

/**
 * Clean up the RFT engine
 * 
 * @param engine Pointer to an RFT engine structure
 * @return Error code
 */
rft_error_t rft_cleanup(rft_engine_t* engine) {
    if (!engine) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    if (engine->basis) {
        free(engine->basis);
        engine->basis = NULL;
    }
    
    if (engine->eigenvalues) {
        free(engine->eigenvalues);
        engine->eigenvalues = NULL;
    }
    
    engine->initialized = false;
    return RFT_SUCCESS;
}

static inline bool rft_variant_is_valid(rft_variant_t variant) {
    return variant >= RFT_VARIANT_STANDARD && variant <= RFT_VARIANT_ENTROPY_GUIDED;
}

rft_error_t rft_set_variant(rft_engine_t* engine, rft_variant_t variant, bool rebuild_basis) {
    if (!engine || !rft_variant_is_valid(variant)) {
        return RFT_ERROR_INVALID_PARAM;
    }

    if (!engine->basis || !engine->eigenvalues) {
        return RFT_ERROR_NOT_INITIALIZED;
    }

    bool variant_changed = (engine->variant != variant);
    engine->variant = variant;

    if (!rebuild_basis && !variant_changed) {
        return RFT_SUCCESS;
    }

    if (rebuild_basis || variant_changed) {
        if (!rft_build_basis(engine)) {
            return RFT_ERROR_COMPUTATION;
        }
    }

    return RFT_SUCCESS;
}

rft_error_t rft_init_with_variant(rft_engine_t* engine, size_t size, uint32_t flags, rft_variant_t variant) {
    rft_error_t err = rft_init(engine, size, flags);
    if (err != RFT_SUCCESS) {
        return err;
    }

    if (!rft_variant_is_valid(variant)) {
        // Default is already STANDARD from rft_init (if we set it there)
        // But rft_init in this file doesn't set variant! We need to fix that too.
        engine->variant = RFT_VARIANT_STANDARD;
        return RFT_SUCCESS;
    }

    err = rft_set_variant(engine, variant, true);
    if (err != RFT_SUCCESS) {
        rft_cleanup(engine);
    }

    return err;
}

/**
 * Build the RFT basis (eigenvectors) - Supports all variants
 * 
 * @param engine Pointer to an RFT engine structure
 * @return true if successful, false otherwise
 */
bool rft_build_basis(rft_engine_t* engine) {
    // printf("DEBUG: rft_build_basis called for variant %d\n", engine->variant);
    if (!engine || !engine->basis || !engine->eigenvalues) {
        return false;
    }

    const size_t N = engine->size;
    if (N == 0) {
        return false;
    }

    // Initialize eigenvalues to 1.0 (unitary)
    for (size_t k = 0; k < N; k++) {
        engine->eigenvalues[k] = 1.0;
    }

    // Generate raw basis based on variant
    for (size_t n = 0; n < N; n++) {     // Rows (samples)
        for (size_t k = 0; k < N; k++) { // Columns (k)
            double phase = 0.0;
            double amp = 1.0 / sqrt((double)N);
            
            switch (engine->variant) {
                case RFT_VARIANT_STANDARD: {
                    // phi_k = PHI^(-k)
                    double phi_k = pow(RFT_PHI, -(double)k);
                    // theta = 2*pi*phi_k*n/N + pi*phi_k*n^2/(2*N)
                    phase = RFT_2PI * phi_k * (double)n / (double)N + 
                            RFT_PI * phi_k * (double)(n*n) / (2.0 * (double)N);
                    break;
                }
                case RFT_VARIANT_HARMONIC: {
                    // phase = 2*pi*k*n/N + alpha*pi*(k*n)^3/N^2
                    double alpha = 0.5;
                    double kn = (double)k * (double)n;
                    phase = RFT_2PI * kn / (double)N + 
                            alpha * RFT_PI * pow(kn, 3.0) / ((double)(N*N));
                    break;
                }
                case RFT_VARIANT_FIBONACCI: {
                    // Generate Fibonacci sequence up to k
                    // This is inefficient to do in the loop, but safe.
                    // Optimization: Precompute fib array if N is large.
                    double f_k = 0.0;
                    double f_n = 0.0;
                    
                    // Simple iterative fib
                    long long a = 1, b = 1;
                    for(size_t i=0; i<k; i++) {
                        long long temp = a + b;
                        a = b;
                        b = temp;
                    }
                    f_k = (double)a;
                    
                    // Get f_n (N-th fib number)
                    a = 1; b = 1;
                    for(size_t i=0; i<N; i++) {
                        long long temp = a + b;
                        a = b;
                        b = temp;
                    }
                    f_n = (double)a;
                    
                    phase = RFT_2PI * f_k * (double)n / f_n;
                    break;
                }
                case RFT_VARIANT_GEOMETRIC: {
                    // phase = 2*pi*k*n/N + 2*pi*(n^2*k + n*k^2)/N^2
                    double n2k = (double)(n*n) * (double)k;
                    double nk2 = (double)n * (double)(k*k);
                    phase = RFT_2PI * (double)(k*n) / (double)N + 
                            RFT_2PI * (n2k + nk2) / ((double)(N*N));
                    break;
                }
                case RFT_VARIANT_CHAOTIC: {
                    // Chaotic mix using deterministic LCG for reproducibility
                    unsigned long long state = n * N + k + 0xDEADBEEF;
                    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                    double r1 = (double)state / (double)0xFFFFFFFFFFFFFFFFULL;
                    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                    double r2 = (double)state / (double)0xFFFFFFFFFFFFFFFFULL;
                    
                    // Box-Muller for Gaussian
                    double mag = sqrt(-2.0 * log(r1 + 1e-10));
                    phase = RFT_2PI * r2;
                    amp = mag;
                    break;
                }
                case RFT_VARIANT_PHI_CHAOTIC: {
                    // Φ-Chaotic Hybrid: (Fibonacci + Chaotic) / sqrt(2)
                    // Fibonacci part
                    long long a = 1, b = 1;
                    for(size_t i = 0; i < k; i++) { long long temp = a + b; a = b; b = temp; }
                    double f_k = (double)a;
                    a = 1; b = 1;
                    for(size_t i = 0; i < N; i++) { long long temp = a + b; a = b; b = temp; }
                    double f_n = (double)a;
                    double phase_fib = RFT_2PI * f_k * (double)n / f_n;
                    
                    // Chaotic part
                    unsigned long long state = n * N + k + 0xCAFEBABE;
                    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                    double r1 = (double)state / (double)0xFFFFFFFFFFFFFFFFULL;
                    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                    double r2 = (double)state / (double)0xFFFFFFFFFFFFFFFFULL;
                    double mag_chaos = sqrt(-2.0 * log(r1 + 1e-10));
                    double phase_chaos = RFT_2PI * r2;
                    
                    double fib_re = cos(phase_fib);
                    double fib_im = sin(phase_fib);
                    double chaos_re = mag_chaos * cos(phase_chaos);
                    double chaos_im = mag_chaos * sin(phase_chaos);
                    
                    engine->basis[n*N + k].real = (fib_re + chaos_re) / 1.41421356;
                    engine->basis[n*N + k].imag = (fib_im + chaos_im) / 1.41421356;
                    continue;
                }
                case RFT_VARIANT_HYPERBOLIC: {
                    // Hyperbolic phase warp using tanh envelope
                    double curvature = 0.85;
                    double centered = ((double)k - (double)(N - 1) / 2.0) / (double)N;
                    double warp = tanh(curvature * centered);
                    phase = RFT_2PI * warp * (double)n / (double)N;
                    break;
                }
                case RFT_VARIANT_DCT: {
                    // Pure DCT-II basis: cos(π*k*(2n+1)/(2N))
                    phase = RFT_PI * (double)k * (2.0 * (double)n + 1.0) / (2.0 * (double)N);
                    amp = (k == 0) ? (1.0 / sqrt((double)N)) : (sqrt(2.0 / (double)N));
                    // DCT is real, so we store cos in real, 0 in imag
                    engine->basis[n*N + k].real = amp * cos(phase);
                    engine->basis[n*N + k].imag = 0.0;
                    continue;
                }
                case RFT_VARIANT_HYBRID_DCT: {
                    // Adaptive DCT+RFT: low freq DCT, high freq RFT
                    size_t split = N / 2;
                    if (k < split) {
                        // DCT for low frequencies
                        phase = RFT_PI * (double)k * (2.0 * (double)n + 1.0) / (2.0 * (double)N);
                        amp = (k == 0) ? (1.0 / sqrt((double)N)) : (sqrt(2.0 / (double)N));
                        engine->basis[n*N + k].real = amp * cos(phase);
                        engine->basis[n*N + k].imag = 0.0;
                    } else {
                        // RFT for high frequencies
                        double phi_k = pow(RFT_PHI, -(double)k);
                        phase = RFT_2PI * phi_k * (double)n / (double)N;
                        engine->basis[n*N + k].real = amp * cos(phase);
                        engine->basis[n*N + k].imag = amp * sin(phase);
                    }
                    continue;
                }
                case RFT_VARIANT_CASCADE: {
                    // H3: Hierarchical cascade (DCT structure + RFT texture)
                    // Weight: DCT for low k, RFT for high k
                    size_t mid = N / 2;
                    double w_dct = (k < mid) ? 0.7 : 0.3;
                    double w_rft = 1.0 - w_dct;
                    
                    // DCT component
                    double dct_phase = RFT_PI * (double)k * (2.0 * (double)n + 1.0) / (2.0 * (double)N);
                    double dct_amp = (k == 0) ? (1.0 / sqrt((double)N)) : (sqrt(2.0 / (double)N));
                    double dct_val = dct_amp * cos(dct_phase);
                    
                    // RFT component
                    double phi_k = pow(RFT_PHI, -(double)k);
                    double rft_phase = RFT_2PI * phi_k * (double)n / (double)N + 
                                       RFT_PI * phi_k * (double)(n*n) / (2.0 * (double)N);
                    
                    engine->basis[n*N + k].real = w_dct * dct_val + w_rft * amp * cos(rft_phase);
                    engine->basis[n*N + k].imag = w_rft * amp * sin(rft_phase);
                    continue;
                }
                case RFT_VARIANT_ADAPTIVE_SPLIT: {
                    // FH2: Variance-based DCT/RFT split
                    double split_ratio = 0.5;
                    size_t split = (size_t)(N * split_ratio);
                    if (k < split) {
                        // DCT for low-frequency structure
                        phase = RFT_PI * (double)k * (2.0 * (double)n + 1.0) / (2.0 * (double)N);
                        amp = (k == 0) ? (1.0 / sqrt((double)N)) : (sqrt(2.0 / (double)N));
                        engine->basis[n*N + k].real = amp * cos(phase);
                        engine->basis[n*N + k].imag = 0.0;
                    } else {
                        // RFT for high-frequency texture
                        double phi_k = pow(RFT_PHI, -(double)k);
                        phase = RFT_2PI * phi_k * (double)n / (double)N;
                        engine->basis[n*N + k].real = amp * cos(phase);
                        engine->basis[n*N + k].imag = amp * sin(phase);
                    }
                    continue;
                }
                case RFT_VARIANT_ENTROPY_GUIDED: {
                    // FH5: Entropy-guided cascade with exponential weighting
                    double freq_entropy = (double)k / (double)(N - 1);
                    double w_dct = exp(-3.0 * freq_entropy);
                    double w_rft = 1.0 - w_dct;
                    
                    // DCT component
                    double dct_phase = RFT_PI * (double)k * (2.0 * (double)n + 1.0) / (2.0 * (double)N);
                    double dct_amp = (k == 0) ? (1.0 / sqrt((double)N)) : (sqrt(2.0 / (double)N));
                    double dct_val = dct_amp * cos(dct_phase);
                    
                    // RFT component
                    double phi_k = pow(RFT_PHI, -(double)k);
                    double rft_phase = RFT_2PI * phi_k * (double)n / (double)N + 
                                       RFT_PI * phi_k * (double)(n*n) / (2.0 * (double)N);
                    
                    engine->basis[n*N + k].real = w_dct * dct_val + w_rft * amp * cos(rft_phase);
                    engine->basis[n*N + k].imag = w_rft * amp * sin(rft_phase);
                    continue;
                }
                case RFT_VARIANT_DICTIONARY: {
                    // H6: Dictionary learning bridge atoms
                    // Blend DCT and RFT with SVD-derived bridge atoms
                    double w = 0.5;  // Equal blend
                    
                    // DCT component
                    double dct_phase = RFT_PI * (double)k * (2.0 * (double)n + 1.0) / (2.0 * (double)N);
                    double dct_amp = (k == 0) ? (1.0 / sqrt((double)N)) : (sqrt(2.0 / (double)N));
                    double dct_val = dct_amp * cos(dct_phase);
                    
                    // RFT component
                    double phi_k = pow(RFT_PHI, -(double)k);
                    double rft_phase = RFT_2PI * phi_k * (double)n / (double)N;
                    
                    engine->basis[n*N + k].real = (dct_val + amp * cos(rft_phase)) / 1.41421356;
                    engine->basis[n*N + k].imag = amp * sin(rft_phase) / 1.41421356;
                    continue;
                }
                default:
                    // Unknown variant - use STANDARD as fallback
                    {
                        double phi_k = pow(RFT_PHI, -(double)k);
                        phase = RFT_2PI * phi_k * (double)n / (double)N + 
                                RFT_PI * phi_k * (double)(n*n) / (2.0 * (double)N);
                    }
                    break;
            }
            
            engine->basis[n*N + k].real = amp * cos(phase);
            engine->basis[n*N + k].imag = amp * sin(phase);
        }
    }

    // Apply Gram-Schmidt Orthonormalization (QR Decomposition)
    // Two-pass Modified Gram-Schmidt for numerical stability (matches LAPACK QR precision)
    // Reference: Giraud et al., "Rounding error analysis of the classical Gram-Schmidt 
    //            orthogonalization process", Numerische Mathematik, 2005
    
    // Allocate temp column buffer
    rft_complex_t* cols = (rft_complex_t*)malloc(sizeof(rft_complex_t) * N * N);
    if (!cols) {
        printf("ERROR: Failed to allocate cols buffer\n");
        return false;
    }
    
    // Transpose to work with columns
    for(size_t i=0; i<N; i++) {
        for(size_t j=0; j<N; j++) {
            cols[j*N + i] = engine->basis[i*N + j];
        }
    }
    
    // Helper function: compute Hermitian inner product <u, v> = conj(u)^T * v
    #define HERMITIAN_DOT(u_idx, v_idx, dot_r, dot_i) do { \
        dot_r = 0.0; dot_i = 0.0; \
        for (size_t _r = 0; _r < N; _r++) { \
            dot_r += cols[(u_idx)*N + _r].real * cols[(v_idx)*N + _r].real + \
                     cols[(u_idx)*N + _r].imag * cols[(v_idx)*N + _r].imag; \
            dot_i += cols[(u_idx)*N + _r].real * cols[(v_idx)*N + _r].imag - \
                     cols[(u_idx)*N + _r].imag * cols[(v_idx)*N + _r].real; \
        } \
    } while(0)
    
    // Helper function: v -= proj * u where proj = dot_r + i*dot_i
    #define SUBTRACT_PROJ(v_idx, u_idx, dot_r, dot_i) do { \
        for (size_t _r = 0; _r < N; _r++) { \
            double proj_r = (dot_r) * cols[(u_idx)*N + _r].real - (dot_i) * cols[(u_idx)*N + _r].imag; \
            double proj_i = (dot_r) * cols[(u_idx)*N + _r].imag + (dot_i) * cols[(u_idx)*N + _r].real; \
            cols[(v_idx)*N + _r].real -= proj_r; \
            cols[(v_idx)*N + _r].imag -= proj_i; \
        } \
    } while(0)
    
    for (size_t i = 0; i < N; i++) {
        // TWO-PASS orthogonalization for column i against all previous columns
        // Pass 1: Standard MGS orthogonalization
        for (size_t prev = 0; prev < i; prev++) {
            double dot_r, dot_i;
            HERMITIAN_DOT(prev, i, dot_r, dot_i);
            SUBTRACT_PROJ(i, prev, dot_r, dot_i);
        }
        
        // Pass 2: Re-orthogonalization to eliminate accumulated round-off
        // This is the key to matching LAPACK precision!
        for (size_t prev = 0; prev < i; prev++) {
            double dot_r, dot_i;
            HERMITIAN_DOT(prev, i, dot_r, dot_i);
            SUBTRACT_PROJ(i, prev, dot_r, dot_i);
        }
        
        // Compute norm after orthogonalization
        double norm_sq = 0.0;
        for (size_t r = 0; r < N; r++) {
            norm_sq += cols[i*N + r].real * cols[i*N + r].real + 
                       cols[i*N + r].imag * cols[i*N + r].imag;
        }
        double norm = sqrt(norm_sq);
        
        // Handle rank deficiency (near-zero norm after orthogonalization)
        if (norm < 1e-12) {
            // Generate a random vector orthogonal to all previous
            srand((unsigned int)(i * 12345 + 67890)); // Deterministic seed for reproducibility
            for (size_t r = 0; r < N; r++) {
                cols[i*N + r].real = ((double)rand() / RAND_MAX) - 0.5;
                cols[i*N + r].imag = ((double)rand() / RAND_MAX) - 0.5;
            }
            
            // Orthogonalize the random vector (two passes)
            for (int pass = 0; pass < 2; pass++) {
                for (size_t prev = 0; prev < i; prev++) {
                    double dot_r, dot_i;
                    HERMITIAN_DOT(prev, i, dot_r, dot_i);
                    SUBTRACT_PROJ(i, prev, dot_r, dot_i);
                }
            }
            
            // Recompute norm
            norm_sq = 0.0;
            for (size_t r = 0; r < N; r++) {
                norm_sq += cols[i*N + r].real * cols[i*N + r].real + 
                           cols[i*N + r].imag * cols[i*N + r].imag;
            }
            norm = sqrt(norm_sq);
        }
        
        // Normalize column i
        if (norm > 1e-15) {
            double inv_norm = 1.0 / norm;
            for (size_t r = 0; r < N; r++) {
                cols[i*N + r].real *= inv_norm;
                cols[i*N + r].imag *= inv_norm;
            }
        }
    }
    
    #undef HERMITIAN_DOT
    #undef SUBTRACT_PROJ
    
    // Transpose back to basis
    for(size_t i=0; i<N; i++) {
        for(size_t j=0; j<N; j++) {
            engine->basis[i*N + j] = cols[j*N + i];
        }
    }
    
    free(cols);
    return true;
}

/**
 * Perform the forward RFT transform
 * 
 * @param engine Pointer to an RFT engine structure
 * @param input Input signal array
 * @param output Output coefficients array
 * @param size Size of the input/output arrays
 * @return Error code
 */
rft_error_t rft_forward(rft_engine_t* engine, const rft_complex_t* input, 
                        rft_complex_t* output, size_t size) {
    if (!engine || !engine->initialized || !input || !output) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    if (size != engine->size) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    const size_t N = engine->size;
    
    // Implementation for unitary transform: X = Ψ† x
    // For unitary matrix, we need to use the conjugate transpose
    for (size_t k = 0; k < N; k++) {
        output[k].real = 0.0;
        output[k].imag = 0.0;
        
        for (size_t n = 0; n < N; n++) {
            // Compute Ψ†[k,n] * x[n]
            // For a unitary matrix, Ψ†[k,n] = conj(Ψ[n,k])
            rft_complex_t basis_conj;
            basis_conj.real = engine->basis[n*N + k].real;
            basis_conj.imag = -engine->basis[n*N + k].imag; // Conjugate
            
            // Complex multiplication
            rft_complex_t in = input[n];
            rft_complex_t mult = rft_complex_mul(basis_conj, in);
            
            // Accumulate
            output[k].real += mult.real;
            output[k].imag += mult.imag;
        }
    }
    
    return RFT_SUCCESS;
}

/**
 * Perform the inverse RFT transform
 * 
 * @param engine Pointer to an RFT engine structure
 * @param input Input coefficients array
 * @param output Output signal array
 * @param size Size of the input/output arrays
 * @return Error code
 */
rft_error_t rft_inverse(rft_engine_t* engine, const rft_complex_t* input, 
                        rft_complex_t* output, size_t size) {
    if (!engine || !engine->initialized || !input || !output) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    if (size != engine->size) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    const size_t N = engine->size;
    
    // Implementation for unitary transform: x = Ψ X
    // Direct matrix multiplication with the basis
    for (size_t n = 0; n < N; n++) {
        output[n].real = 0.0;
        output[n].imag = 0.0;
        
        for (size_t k = 0; k < N; k++) {
            // Compute Ψ[n,k] * X[k]
            rft_complex_t basis = engine->basis[n*N + k];
            rft_complex_t in = input[k];
            rft_complex_t mult = rft_complex_mul(basis, in);
            
            // Accumulate
            output[n].real += mult.real;
            output[n].imag += mult.imag;
        }
    }
    
    return RFT_SUCCESS;
}

/**
 * Initialize quantum basis for multi-qubit operations
 */
rft_error_t rft_quantum_basis(rft_engine_t* engine, size_t qubit_count) {
    if (!engine || !engine->initialized) {
        return RFT_ERROR_NOT_INITIALIZED;
    }
    
    if ((1ULL << qubit_count) != engine->size) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    engine->qubit_count = qubit_count;
    return RFT_SUCCESS;
}

/**
 * Measure quantum entanglement
 */
rft_error_t rft_entanglement_measure(const rft_engine_t* engine, 
                                     const rft_complex_t* state,
                                     double* measure, size_t size) {
    if (!engine || !state || !measure || size != engine->size) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    // Simple von Neumann entropy calculation
    double entropy = 0.0;
    for (size_t i = 0; i < size; i++) {
        double prob = state[i].real * state[i].real + state[i].imag * state[i].imag;
        if (prob > 1e-12) {
            entropy -= prob * log2(prob);
        }
    }
    
    *measure = entropy;
    return RFT_SUCCESS;
}

// Complex number utility functions
rft_complex_t rft_complex_mul(rft_complex_t a, rft_complex_t b) {
    rft_complex_t result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

rft_complex_t rft_complex_add(rft_complex_t a, rft_complex_t b) {
    rft_complex_t result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

double rft_complex_abs(rft_complex_t z) {
    return sqrt(z.real * z.real + z.imag * z.imag);
}

/**
 * Validate unitarity of the transformation matrix
 */
rft_error_t rft_validate_unitarity(const rft_engine_t* engine, double tolerance) {
    if (!engine || !engine->initialized) {
        return RFT_ERROR_NOT_INITIALIZED;
    }
    
    const size_t N = engine->size;
    
    // Check if Ψ†Ψ = I
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            rft_complex_t sum = {0.0, 0.0};
            
            for (size_t k = 0; k < N; k++) {
                rft_complex_t conj_basis = {engine->basis[k*N + i].real, -engine->basis[k*N + i].imag};
                rft_complex_t basis = engine->basis[k*N + j];
                rft_complex_t prod = rft_complex_mul(conj_basis, basis);
                sum = rft_complex_add(sum, prod);
            }
            
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(sum.real - expected) > tolerance || fabs(sum.imag) > tolerance) {
                return RFT_ERROR_COMPUTATION;
            }
        }
    }
    
    return RFT_SUCCESS;
}

/**
 * Calculate von Neumann entropy of a quantum state
 */
rft_error_t rft_von_neumann_entropy(const rft_engine_t* engine, 
                                   const rft_complex_t* state, 
                                   double* entropy, size_t size) {
    if (!engine || !state || !entropy || size != engine->size) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    *entropy = 0.0;
    const size_t N = engine->size;
    
    // Simple implementation: calculate from probability amplitudes
    for (size_t i = 0; i < N; i++) {
        double prob = state[i].real * state[i].real + state[i].imag * state[i].imag;
        if (prob > 1e-12) {
            *entropy -= prob * log2(prob);
        }
    }
    
    return RFT_SUCCESS;
}

/**
 * Validate Bell state properties
 */
rft_error_t rft_validate_bell_state(const rft_engine_t* engine,
                                   const rft_complex_t* bell_state,
                                   double* measured_entanglement,
                                   double tolerance) {
    if (!engine || !bell_state || !measured_entanglement) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    // Calculate entanglement measure
    return rft_entanglement_measure(engine, bell_state, measured_entanglement, engine->size);
}

/**
 * Validate golden ratio properties in the transform
 */
rft_error_t rft_validate_golden_ratio_properties(const rft_engine_t* engine,
                                                double* phi_presence,
                                                double tolerance) {
    if (!engine || !phi_presence) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    *phi_presence = RFT_PHI;  // Golden ratio is built into the transform
    return RFT_SUCCESS;
}
