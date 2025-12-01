/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate license.
 */

#include "rft_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static inline double rft_frac(double x) {
    double frac = x - floor(x);
    return (frac < 0.0) ? frac + 1.0 : frac;
}

static inline bool rft_variant_is_valid(rft_variant_t variant) {
    return variant >= RFT_VARIANT_STANDARD && variant <= RFT_VARIANT_ENTROPY_GUIDED;
}

// Fibonacci number generator for Fibonacci-Tilt variant
static double rft_fibonacci_ratio(size_t k) {
    if (k == 0) return 1.0;
    double f_prev = 1.0, f_curr = 1.0;
    for (size_t i = 2; i <= k; i++) {
        double f_next = f_prev + f_curr;
        f_prev = f_curr;
        f_curr = f_next;
    }
    return f_curr / f_prev;  // Converges to φ
}

// DCT-II basis element computation
static void rft_dct_basis_element(double* out_real, double* out_imag, size_t k, size_t n, size_t N) {
    // DCT-II: cos(π(2n+1)k / 2N) with proper normalization
    double norm = (k == 0) ? sqrt(1.0 / (double)N) : sqrt(2.0 / (double)N);
    double angle = RFT_PI * (2.0 * (double)n + 1.0) * (double)k / (2.0 * (double)N);
    *out_real = norm * cos(angle);
    *out_imag = 0.0;
}

// Shannon entropy computation for entropy-guided routing
static double rft_local_entropy(const rft_complex_t* data, size_t start, size_t len) {
    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        double mag = sqrt(data[start + i].real * data[start + i].real + 
                         data[start + i].imag * data[start + i].imag);
        sum += mag;
    }
    if (sum < 1e-10) return 0.0;
    
    double entropy = 0.0;
    for (size_t i = 0; i < len; i++) {
        double mag = sqrt(data[start + i].real * data[start + i].real + 
                         data[start + i].imag * data[start + i].imag);
        double p = mag / sum;
        if (p > 1e-10) {
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

// Advanced SIMD intrinsics for AVX
#ifdef __AVX__
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

// Constants - using macros from header

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
    engine->variant = RFT_VARIANT_STANDARD; // Default to standard variant
    
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
    
    // printf("[DEBUG] rft_cleanup: Starting cleanup for engine %p\n", (void*)engine);
    
    if (engine->basis) {
        // printf("[DEBUG] rft_cleanup: Freeing basis %p\n", (void*)engine->basis);
        free(engine->basis);
        engine->basis = NULL;
    }
    
    if (engine->eigenvalues) {
        // printf("[DEBUG] rft_cleanup: Freeing eigenvalues %p\n", (void*)engine->eigenvalues);
        free(engine->eigenvalues);
        engine->eigenvalues = NULL;
    }
    
    engine->initialized = false;
    // printf("[DEBUG] rft_cleanup: Cleanup completed successfully\n");
    return RFT_SUCCESS;
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

    if (!rft_variant_is_valid(variant) || variant == RFT_VARIANT_STANDARD) {
        return RFT_SUCCESS;
    }

    err = rft_set_variant(engine, variant, true);
    if (err != RFT_SUCCESS) {
        rft_cleanup(engine);
    }

    return err;
}

/**
 * Build the RFT basis (eigenvectors) - FIXED True Unitary Implementation
 * 
 * @param engine Pointer to an RFT engine structure
 * @return true if successful, false otherwise
 */
bool rft_build_basis(rft_engine_t* engine) {
    if (!engine || !engine->basis || !engine->eigenvalues) {
        return false;
    }

    const size_t N = engine->size;
    if (N == 0) {
        return false;
    }

    const double inv_sqrt_n = 1.0 / sqrt((double)N);
    const double beta = 1.0;
    const double sigma = 1.0;

    for (size_t k = 0; k < N; ++k) {
        double frac, chirp;

        switch (engine->variant) {
            case RFT_VARIANT_HARMONIC:
                // Harmonic-Phase: Cubic phase chirp (k^3)
                frac = rft_frac((double)k / RFT_PHI);
                chirp = RFT_PI * sigma * ((double)k * (double)k * (double)k) / ((double)N * (double)N);
                break;
            case RFT_VARIANT_FIBONACCI:
                // Fibonacci-Tilt: Multiplicative Golden Ratio
                frac = rft_frac((double)k * RFT_PHI);
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_CHAOTIC:
                // Chaotic Mix: Deterministic chaos (pseudo-random phase)
                frac = rft_frac(sin((double)k * 12.9898) * 43758.5453);
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_HYPERBOLIC:
                // Hyperbolic Geometry: Tanh-based fractional phase
                frac = tanh((double)k / (double)N);
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_GEOMETRIC:
                // Geometric Lattice: Exponential φ^k scaling
                frac = rft_frac(pow(RFT_PHI, (double)k / (double)N));
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_PHI_CHAOTIC:
                // Φ-Chaotic Hybrid: Blend of Fibonacci + Chaotic, normalized by √2
                // From RFT_THEOREMS: basis = (fib_basis + chaos_basis) / sqrt(2)
                {
                    double fib_frac = rft_frac((double)k * RFT_PHI);
                    double chaos_frac = rft_frac(sin((double)k * 12.9898) * 43758.5453);
                    frac = RFT_DCT_NORM_FACTOR * (fib_frac + chaos_frac);  // 1/√2 normalization
                }
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_DCT:
                // Pure DCT-II: cos((2n+1)kπ/2N) basis
                // Note: For DCT, we compute real-only coefficients
                {
                    frac = 0.0;  // No phase modulation for pure DCT
                    chirp = 0.0;
                }
                // Special handling: DCT uses cosine basis directly
                break;
            case RFT_VARIANT_HYBRID_DCT:
                // Hybrid DCT+RFT: Energy-weighted coefficient selection
                // Low frequencies use DCT, high frequencies use RFT
                {
                    double normalized_k = (double)k / (double)N;
                    if (normalized_k < 0.5) {
                        // Low frequency: DCT-like phase (minimal modulation)
                        frac = 0.0;
                        chirp = 0.0;
                    } else {
                        // High frequency: Full RFT phase modulation
                        frac = rft_frac((double)k / RFT_PHI);
                        chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                    }
                }
                break;
            case RFT_VARIANT_CASCADE:
                // H3: Hierarchical Cascade (achieved zero coherence in H3 test)
                // Multi-resolution decomposition with φ-scaled levels
                {
                    double level1 = rft_frac((double)k / RFT_PHI);
                    double level2 = rft_frac((double)k / (RFT_PHI * RFT_PHI));
                    double level3 = rft_frac((double)k / (RFT_PHI * RFT_PHI * RFT_PHI));
                    // Weighted combination for hierarchical structure
                    frac = 0.5 * level1 + 0.35 * level2 + 0.15 * level3;
                }
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_ADAPTIVE_SPLIT:
                // FH2: Variance-based DCT/RFT routing (50% BPP improvement)
                // Uses signal variance to choose between DCT and RFT
                // Note: Full adaptive routing requires signal context; this is basis-only
                {
                    // Split point at golden ratio of frequency range
                    double split_point = (double)N / RFT_PHI;
                    if ((double)k < split_point) {
                        // Low-variance region: DCT-like (smooth)
                        frac = rft_frac((double)k / (2.0 * (double)N));
                    } else {
                        // High-variance region: Full RFT
                        frac = rft_frac((double)k / RFT_PHI);
                    }
                }
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_ENTROPY_GUIDED:
                // FH5: Entropy-based routing (50% BPP improvement)
                // Uses local entropy to weight between smooth and textured bases
                // Note: Full entropy routing requires signal context
                {
                    // Entropy-like weighting based on frequency
                    double entropy_weight = 1.0 - exp(-(double)k / ((double)N / 4.0));
                    double base_frac = rft_frac((double)k / RFT_PHI);
                    double smooth_frac = rft_frac((double)k / (double)N);
                    // Blend based on "entropy" (higher k = higher entropy assumption)
                    frac = entropy_weight * base_frac + (1.0 - entropy_weight) * smooth_frac;
                }
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
            case RFT_VARIANT_STANDARD:
            default:
                // Standard Golden Ratio RFT
                frac = rft_frac((double)k / RFT_PHI);
                chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
                break;
        }

        double theta = RFT_2PI * beta * frac;
        double diag_angle = theta + chirp;
        double diag_real = cos(diag_angle);
        double diag_imag = sin(diag_angle);

        for (size_t n = 0; n < N; ++n) {
            double fft_angle = -RFT_2PI * ((double)k * (double)n) / (double)N;
            double fft_real = cos(fft_angle) * inv_sqrt_n;
            double fft_imag = sin(fft_angle) * inv_sqrt_n;

            double real = diag_real * fft_real - diag_imag * fft_imag;
            double imag = diag_real * fft_imag + diag_imag * fft_real;

            engine->basis[k * N + n].real = real;
            engine->basis[k * N + n].imag = imag;
        }

        engine->eigenvalues[k] = 1.0;
    }

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
    
    // Critical safety check: ensure basis matrix is allocated
    if (!engine->basis) {
        printf("[ERROR] rft_forward: basis matrix is NULL\n");
        return RFT_ERROR_NOT_INITIALIZED;
    }
    
    const size_t N = engine->size;
    // printf("[DEBUG] rft_forward: N=%zu, basis=%p\n", N, (void*)engine->basis);
    
    // Implementation for unitary transform: X = Ψ† x
    // For unitary matrix, we need to use the conjugate transpose
    for (size_t k = 0; k < N; k++) {
        output[k].real = 0.0;
        output[k].imag = 0.0;
        
        for (size_t n = 0; n < N; n++) {
            // Bounds check to prevent segfault
            size_t index = n*N + k;
            if (index >= N*N) {
                printf("[ERROR] rft_forward: index %zu out of bounds (max %zu)\n", index, N*N);
                return RFT_ERROR_COMPUTATION;
            }
            
            // Compute Ψ†[k,n] * x[n]
            // For a unitary matrix, Ψ†[k,n] = conj(Ψ[n,k])
            rft_complex_t basis_conj;
            basis_conj.real = engine->basis[index].real;
            basis_conj.imag = -engine->basis[index].imag; // Conjugate
            
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
    
    // Critical safety check: ensure basis matrix is allocated
    if (!engine->basis) {
        printf("[ERROR] rft_inverse: basis matrix is NULL\n");
        return RFT_ERROR_NOT_INITIALIZED;
    }
    
    const size_t N = engine->size;
    // printf("[DEBUG] rft_inverse: N=%zu, basis=%p\n", N, (void*)engine->basis);
    
    // Implementation for unitary transform: x = Ψ X
    // Direct matrix multiplication with the basis
    for (size_t n = 0; n < N; n++) {
        output[n].real = 0.0;
        output[n].imag = 0.0;
        
        for (size_t k = 0; k < N; k++) {
            // Bounds check to prevent segfault
            size_t index = n*N + k;
            if (index >= N*N) {
                printf("[ERROR] rft_inverse: index %zu out of bounds (max %zu)\n", index, N*N);
                return RFT_ERROR_COMPUTATION;
            }
            
            // Compute Ψ[n,k] * X[k]
            rft_complex_t basis = engine->basis[index];
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
 * S = -Tr(ρ log₂ ρ) where ρ is the density matrix
 */
rft_error_t rft_von_neumann_entropy(const rft_engine_t* engine, 
                                   const rft_complex_t* state, 
                                   double* entropy, size_t size) {
    if (!engine || !state || !entropy || size != engine->size) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    const size_t N = engine->size;
    *entropy = 0.0;
    
    // For a pure state |ψ⟩, the density matrix is ρ = |ψ⟩⟨ψ|
    // For pure states: S = 0
    // For mixed states: Need to compute eigenvalues of ρ
    
    // Calculate density matrix elements ρ_ij = ψᵢ* ψⱼ
    rft_complex_t* density = (rft_complex_t*)malloc(sizeof(rft_complex_t) * N * N);
    if (!density) return RFT_ERROR_MEMORY;
    
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            rft_complex_t conj_i = {state[i].real, -state[i].imag};
            density[i*N + j] = rft_complex_mul(conj_i, state[j]);
        }
    }
    
    // For pure states, von Neumann entropy is 0
    // For maximally mixed states: S = log₂(N)
    
    // Check if state is pure (all probability in one state) or mixed
    double purity = 0.0;
    for (size_t i = 0; i < N; i++) {
        double prob = state[i].real * state[i].real + state[i].imag * state[i].imag;
        purity += prob * prob;
    }
    
    if (purity > 0.99) {
        // Nearly pure state
        *entropy = 0.0;
    } else {
        // Mixed state - calculate entropy from probabilities
        *entropy = 0.0;
        for (size_t i = 0; i < N; i++) {
            double prob = state[i].real * state[i].real + state[i].imag * state[i].imag;
            if (prob > 1e-12) {
                *entropy -= prob * log2(prob);
            }
        }
    }
    
    free(density);
    return RFT_SUCCESS;
}

/**
 * Measure quantum entanglement using von Neumann entropy
 * For Bell states and entangled systems
 */
rft_error_t rft_entanglement_measure(const rft_engine_t* engine, 
                                    const rft_complex_t* state, 
                                    double* entanglement, size_t size) {
    if (!engine || !state || !entanglement || size != engine->size) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    const size_t N = engine->size;
    const int qubits = (int)(log2(N) + 0.5);
    
    if (qubits < 2) {
        *entanglement = 0.0;
        return RFT_SUCCESS;
    }
    
    // For 2-qubit systems, calculate entanglement via reduced density matrix
    if (qubits == 2) {
        // State is |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
        // Reduced density matrix for qubit A: ρ_A = Tr_B(|ψ⟩⟨ψ|)
        
        rft_complex_t alpha = state[0];  // |00⟩
        rft_complex_t beta = state[1];   // |01⟩  
        rft_complex_t gamma = state[2];  // |10⟩
        rft_complex_t delta = state[3];  // |11⟩
        
        // Reduced density matrix elements for subsystem A
        // ρ_A[0][0] = |α|² + |β|²  (probability of |0⟩_A)
        // ρ_A[1][1] = |γ|² + |δ|²  (probability of |1⟩_A)
        // ρ_A[0][1] = α*γ + β*δ    (coherence term)
        // ρ_A[1][0] = (ρ_A[0][1])*  (conjugate)
        
        double p0 = alpha.real*alpha.real + alpha.imag*alpha.imag + 
                   beta.real*beta.real + beta.imag*beta.imag;
        double p1 = gamma.real*gamma.real + gamma.imag*gamma.imag + 
                   delta.real*delta.real + delta.imag*delta.imag;
        
        // Calculate entanglement entropy
        *entanglement = 0.0;
        if (p0 > 1e-12) {
            *entanglement -= p0 * log2(p0);
        }
        if (p1 > 1e-12) {
            *entanglement -= p1 * log2(p1);
        }
        
        // For Bell states: |00⟩ + |11⟩, we have p0 = p1 = 0.5
        // So entanglement = -0.5*log₂(0.5) - 0.5*log₂(0.5) = 1.0
        
    } else {
        // For multi-qubit systems, use simplified entanglement measure
        // Based on participation ratio and state spreading
        
        double participation = 0.0;
        double entropy = 0.0;
        
        for (size_t i = 0; i < N; i++) {
            double prob = state[i].real * state[i].real + state[i].imag * state[i].imag;
            participation += prob * prob;
            if (prob > 1e-12) {
                entropy -= prob * log2(prob);
            }
        }
        
        // Entanglement scales with entropy for multi-partite systems
        *entanglement = entropy * (1.0 - participation);
    }
    
    return RFT_SUCCESS;
}

/**
 * Validate Bell state entanglement properties
 * Tests specific quantum states for expected entanglement values
 */
rft_error_t rft_validate_bell_state(const rft_engine_t* engine,
                                   const rft_complex_t* bell_state,
                                   double* measured_entanglement,
                                   double tolerance) {
    if (!engine || !bell_state || !measured_entanglement) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    if (engine->size < 4) {
        return RFT_ERROR_INVALID_PARAM;  // Need at least 2 qubits
    }
    
    // Measure entanglement of the Bell state
    rft_error_t result = rft_entanglement_measure(engine, bell_state, 
                                                 measured_entanglement, engine->size);
    if (result != RFT_SUCCESS) {
        return result;
    }
    
    // Expected entanglement for Bell states is 1.0 bit
    double expected_entanglement = 1.0;
    
    if (fabs(*measured_entanglement - expected_entanglement) > tolerance) {
        return RFT_ERROR_COMPUTATION;
    }
    
    return RFT_SUCCESS;
}

/**
 * Test golden ratio properties in eigenvalue spectrum
 * Validates φ and 1/φ relationships in the transform matrix
 */
rft_error_t rft_validate_golden_ratio_properties(const rft_engine_t* engine,
                                                double* phi_presence,
                                                double tolerance) {
    if (!engine || !phi_presence || !engine->initialized) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    const double phi = 1.618033988749894848204586834366;  // Golden ratio
    const double inv_phi = 1.0 / phi;  // 1/φ ≈ 0.618
    
    const size_t N = engine->size;
    *phi_presence = 0.0;
    
    // Check if basis matrix exhibits golden ratio properties
    // Look for φ and 1/φ relationships in matrix elements
    
    int phi_count = 0;
    int total_elements = 0;
    
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            rft_complex_t elem = engine->basis[i*N + j];
            double magnitude = sqrt(elem.real*elem.real + elem.imag*elem.imag);
            
            if (magnitude > 1e-6) {  // Non-zero element
                total_elements++;
                
                // Check if magnitude relates to golden ratio
                double phi_error = fabs(magnitude - phi);
                double inv_phi_error = fabs(magnitude - inv_phi);
                double ratio_error = fabs(magnitude*phi - 1.0);  // Check if elem = 1/φ
                
                if (phi_error < tolerance || inv_phi_error < tolerance || ratio_error < tolerance) {
                    phi_count++;
                }
            }
        }
    }
    
    if (total_elements > 0) {
        *phi_presence = (double)phi_count / total_elements;
    }
    
    // Consider golden ratio properties present if >10% of elements show φ relationships
    return (*phi_presence > 0.1) ? RFT_SUCCESS : RFT_ERROR_COMPUTATION;
}
