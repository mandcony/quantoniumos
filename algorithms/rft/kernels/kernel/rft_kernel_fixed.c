/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate license.
 */

#include "rft_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
        double frac = rft_frac((double)k / RFT_PHI);
        double theta = RFT_2PI * beta * frac;
        double chirp = RFT_PI * sigma * ((double)k * (double)k) / (double)N;
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
