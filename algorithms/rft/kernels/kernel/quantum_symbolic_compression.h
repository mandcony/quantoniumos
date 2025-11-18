/**
 * Quantum Symbolic Compression Engine - Header
 * High-performance C/Assembly implementation for million+ qubit simulation
 * 
 * This header defines the interface for the symbolic compression algorithm
 * that enables O(n) scaling for quantum state simulation beyond classical limits.
 */

/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate license.
 */

#ifndef QUANTUM_SYMBOLIC_COMPRESSION_H
#define QUANTUM_SYMBOLIC_COMPRESSION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mathematical constants
#define QSC_PHI 1.618033988749894848204586834366  // Golden ratio
#define QSC_PI 3.141592653589793238462643383279
#define QSC_2PI (2.0 * QSC_PI)
#define QSC_INV_SQRT_2 0.7071067811865475244008443621048490

// Error codes
typedef enum {
    QSC_SUCCESS = 0,
    QSC_ERROR_INVALID_PARAM = -1,
    QSC_ERROR_MEMORY = -2,
    QSC_ERROR_COMPUTATION = -3,
    QSC_ERROR_NOT_INITIALIZED = -4
} qsc_error_t;

// Complex number structure (aligned for SIMD)
typedef struct {
    double real;
    double imag;
} __attribute__((aligned(16))) qsc_complex_t;

// Quantum state compression parameters
typedef struct {
    size_t num_qubits;          // Number of logical qubits
    size_t compression_size;    // Compressed state vector size (typically 64)
    double phi;                 // Golden ratio parameter
    double normalization;       // Normalization factor
    bool use_simd;             // Enable SIMD optimizations
    bool use_assembly;         // Enable assembly optimizations
} qsc_params_t;

// Compressed quantum state
typedef struct {
    qsc_complex_t* amplitudes;  // Compressed amplitudes array
    size_t size;               // Size of amplitudes array
    size_t num_qubits;         // Number of logical qubits represented
    double norm;               // State norm
    bool initialized;          // Initialization flag
    void* metadata;            // Optional metadata
} qsc_state_t;

// Performance statistics
typedef struct {
    double compression_time_ms;
    double entanglement_time_ms;
    double total_time_ms;
    size_t operations_per_second;
    double memory_mb;
    double compression_ratio;
} qsc_perf_stats_t;

// Core API functions
qsc_error_t qsc_init_state(qsc_state_t* state, const qsc_params_t* params);
qsc_error_t qsc_cleanup_state(qsc_state_t* state);

// Symbolic compression engine
qsc_error_t qsc_compress_million_qubits(qsc_state_t* state, size_t num_qubits, size_t compression_size);
qsc_error_t qsc_compress_optimized_asm(qsc_state_t* state, size_t num_qubits, size_t compression_size);

// Entanglement measurement
qsc_error_t qsc_measure_entanglement(const qsc_state_t* state, double* entanglement);
qsc_error_t qsc_measure_bipartite_entanglement(const qsc_state_t* state, size_t partition_size, double* entanglement);

// Bell state generation
qsc_error_t qsc_create_bell_state(qsc_state_t* state, int bell_type);
qsc_error_t qsc_create_ghz_state(qsc_state_t* state, size_t num_qubits);

// Quantum operations
qsc_error_t qsc_apply_hadamard(qsc_state_t* state, size_t qubit_index);
qsc_error_t qsc_apply_cnot(qsc_state_t* state, size_t control, size_t target);
qsc_error_t qsc_apply_pauli_x(qsc_state_t* state, size_t qubit_index);
qsc_error_t qsc_apply_pauli_z(qsc_state_t* state, size_t qubit_index);

// Performance and validation
qsc_error_t qsc_validate_unitarity(const qsc_state_t* state, double tolerance, bool* is_unitary);
qsc_error_t qsc_get_performance_stats(const qsc_state_t* state, qsc_perf_stats_t* stats);
qsc_error_t qsc_benchmark_scaling(size_t max_qubits, qsc_perf_stats_t* results);

// Build information functions
qsc_error_t qsc_get_version(char* buffer, size_t buffer_size);
qsc_error_t qsc_get_build_info(char* buffer, size_t buffer_size);

// Utility functions
qsc_complex_t qsc_complex_mul(qsc_complex_t a, qsc_complex_t b);
qsc_complex_t qsc_complex_add(qsc_complex_t a, qsc_complex_t b);
double qsc_complex_abs(qsc_complex_t z);
double qsc_complex_norm_squared(qsc_complex_t z);

// Assembly-optimized functions (implemented in .asm file)
extern void qsc_symbolic_compression_asm(const double* params, qsc_complex_t* output, size_t num_qubits, size_t compression_size);
extern void qsc_entanglement_measure_asm(const qsc_complex_t* state, double* result, size_t size);
extern void qsc_complex_vector_norm_asm(qsc_complex_t* vector, size_t size);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_SYMBOLIC_COMPRESSION_H
