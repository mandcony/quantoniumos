/**
 * RFT Kernel - Resonance Field Theory Implementation
 * 
 * This header file defines the interface for the Resonance Field Theory (RFT)
 * kernel implementation. The RFT provides a unitary transform that preserves
 * quantum mechanical properties while enabling efficient signal processing.
 * 
 * Key Features:
 * - True unitary transforms (norm-preserving)
 * - Quantum-safe eigendecomposition
 * - SIMD optimization support
 * - Resonance-based basis functions
 * 
 * Author: QuantoniumOS Team
 * License: See LICENSE.md
 */

/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate license.
 */

#ifndef RFT_KERNEL_H
#define RFT_KERNEL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mathematical constants
#define RFT_PI 3.141592653589793238462643383279
#define RFT_2PI (2.0 * RFT_PI)
#define RFT_PHI 1.618033988749894848204586834366  // Golden ratio

// Error codes
typedef enum {
    RFT_SUCCESS = 0,
    RFT_ERROR_INVALID_PARAM = -1,
    RFT_ERROR_MEMORY = -2,
    RFT_ERROR_COMPUTATION = -3,
    RFT_ERROR_NOT_INITIALIZED = -4
} rft_error_t;

// Configuration flags
#define RFT_FLAG_DEFAULT            0x00000000
#define RFT_FLAG_OPTIMIZE_SIMD      0x00000001
#define RFT_FLAG_HIGH_PRECISION     0x00000002
#define RFT_FLAG_QUANTUM_SAFE       0x00000004
#define RFT_FLAG_DEBUG          0x0008
#define RFT_FLAG_TOPOLOGICAL    0x0010  // Enable topological protection

// RFT Variant Definitions - All Proven/Tested Variants
// Group A: Core 7 Unitary Variants (machine-precision validated)
// Group B: Hybrid DCT-RFT Variants (hypothesis-tested)
typedef enum {
    // === GROUP A: Core Unitary RFT Variants ===
    RFT_VARIANT_STANDARD = 0,       // Original Φ-RFT (k/φ fractional, k² chirp)
    RFT_VARIANT_HARMONIC = 1,       // Harmonic-Phase (k³ cubic chirp)
    RFT_VARIANT_FIBONACCI = 2,      // Fibonacci-Tilt Lattice (k*F_k, crypto-optimized)
    RFT_VARIANT_CHAOTIC = 3,        // Chaotic Mix (PRNG-based, max entropy)
    RFT_VARIANT_GEOMETRIC = 4,      // Geometric Lattice (φ^k, optical computing)
    RFT_VARIANT_PHI_CHAOTIC = 5,    // Φ-Chaotic Hybrid ((Fib + Chaos)/√2)
    RFT_VARIANT_HYPERBOLIC = 6,     // Hyperbolic (tanh-based fractional phase)
    
    // === GROUP B: Hybrid DCT-RFT Variants (H1-H12 tested) ===
    RFT_VARIANT_DCT = 7,            // Pure DCT-II basis
    RFT_VARIANT_HYBRID_DCT = 8,     // Adaptive DCT+RFT coefficient selection
    RFT_VARIANT_CASCADE = 9,        // H3: Hierarchical cascade (zero coherence)
    RFT_VARIANT_ADAPTIVE_SPLIT = 10,// FH2: Variance-based DCT/RFT routing (50% BPP win)
    RFT_VARIANT_ENTROPY_GUIDED = 11 // FH5: Entropy-based routing (50% BPP win)
} rft_variant_t;

// DCT/Hybrid computation constants
#define RFT_DCT_NORM_FACTOR  0.7071067811865476  // 1/sqrt(2) for DCT-II normalization

// Tetrahedral vertex coordinates (for geometric variants)
#define RFT_TETRA_V0_X  0.5773502691896258   // 1/√3
#define RFT_TETRA_V0_Y  0.5773502691896258
#define RFT_TETRA_V0_Z  0.5773502691896258
#define RFT_TETRA_V1_X  0.5773502691896258
#define RFT_TETRA_V1_Y -0.5773502691896258
#define RFT_TETRA_V1_Z -0.5773502691896258
#define RFT_FLAG_USE_RESONANCE      0x00000010

// Topological data structures for enhanced quantum computing
typedef struct {
    double real;
    double imag;
} topological_complex_t;

typedef struct {
    int vertex_id;
    double coordinates[3];          // 3D spatial coordinates
    topological_complex_t topological_charge;
    double local_curvature;
    double geometric_phase;
    int connections[10];            // Connected vertex IDs
    int connection_count;
    topological_complex_t local_state[2];  // Local qubit state
} vertex_manifold_t;

typedef struct {
    char edge_id[16];
    int vertex_pair[2];
    topological_complex_t edge_weight;
    topological_complex_t braiding_matrix[4];  // 2x2 complex matrix
    topological_complex_t holonomy;
    topological_complex_t wilson_loop;
    double gauge_field[3];
    int error_syndrome;
    bool has_stored_data;
} topological_edge_t;

typedef struct {
    double winding_number_real;
    double winding_number_imag;
    int chern_number;
    double berry_phase;
    int genus;
    int euler_characteristic;
} topological_invariant_t;

typedef struct {
    int qubit_id;
    int num_vertices;
    vertex_manifold_t* vertices;
    topological_edge_t* edges;
    int num_edges;
    int code_distance;
    topological_complex_t global_state[2];
    topological_invariant_t invariants;
} enhanced_topological_qubit_t;

// Complex number structure
typedef struct {
    double real;
    double imag;
} rft_complex_t;

// RFT engine structure with enhanced topological support
typedef struct {
    size_t size;                    // Transform size
    rft_complex_t* basis;          // Basis matrix (size x size)
    double* eigenvalues;           // Eigenvalues array
    bool initialized;              // Initialization flag
    uint32_t flags;               // Configuration flags
    rft_variant_t variant;        // Active RFT variant
    size_t qubit_count;           // Number of qubits (for quantum operations)
    void* quantum_context;        // Quantum-specific context
    
    // Enhanced topological support
    enhanced_topological_qubit_t* topological_qubits;
    size_t num_topological_qubits;
    bool topological_mode_enabled;
} rft_engine_t;

// Enhanced topological quantum operations
rft_error_t rft_init_topological_mode(rft_engine_t* engine, size_t num_topological_qubits);
rft_error_t rft_apply_braiding(rft_engine_t* engine, int qubit_id, int vertex_a, int vertex_b, bool clockwise);
rft_error_t rft_encode_topological_edge(rft_engine_t* engine, int qubit_id, const char* edge_id, 
                                       const rft_complex_t* data, size_t data_size);
rft_error_t rft_decode_topological_edge(rft_engine_t* engine, int qubit_id, const char* edge_id,
                                       rft_complex_t* output, size_t* output_size);
rft_error_t rft_surface_code_correction(rft_engine_t* engine, int qubit_id);
rft_error_t rft_measure_topological_invariant(rft_engine_t* engine, int qubit_id, 
                                             const char* invariant_type, double* result);

// Core API functions
rft_error_t rft_init(rft_engine_t* engine, size_t size, uint32_t flags);
rft_error_t rft_cleanup(rft_engine_t* engine);
rft_error_t rft_forward(rft_engine_t* engine, const rft_complex_t* input, 
                        rft_complex_t* output, size_t size);
rft_error_t rft_inverse(rft_engine_t* engine, const rft_complex_t* input, 
                        rft_complex_t* output, size_t size);
rft_error_t rft_set_variant(rft_engine_t* engine, rft_variant_t variant, bool rebuild_basis);
rft_error_t rft_init_with_variant(rft_engine_t* engine, size_t size, uint32_t flags, rft_variant_t variant);

// Quantum operations
rft_error_t rft_quantum_basis(rft_engine_t* engine, size_t qubit_count);

// Utility functions
rft_complex_t rft_complex_mul(rft_complex_t a, rft_complex_t b);
rft_complex_t rft_complex_add(rft_complex_t a, rft_complex_t b);
double rft_complex_abs(rft_complex_t z);

// Internal functions (for advanced users)
bool rft_build_basis(rft_engine_t* engine);
rft_error_t rft_validate_unitarity(const rft_engine_t* engine, double tolerance);

// Mathematical validation functions for proofs
rft_error_t rft_von_neumann_entropy(const rft_engine_t* engine, 
                                   const rft_complex_t* state, 
                                   double* entropy, size_t size);

rft_error_t rft_entanglement_measure(const rft_engine_t* engine, 
                                    const rft_complex_t* state, 
                                    double* entanglement, size_t size);

rft_error_t rft_validate_bell_state(const rft_engine_t* engine,
                                   const rft_complex_t* bell_state,
                                   double* measured_entanglement,
                                   double tolerance);

rft_error_t rft_validate_golden_ratio_properties(const rft_engine_t* engine,
                                                double* phi_presence,
                                                double tolerance);

#ifdef __cplusplus
}
#endif

#endif // RFT_KERNEL_H
