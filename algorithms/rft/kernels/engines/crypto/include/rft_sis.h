/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 */

/**
 * RFT-SIS: Post-Quantum Lattice-Based Cryptographic Hash
 * ======================================================
 * 
 * Combines the φ-RFT transform with the Short Integer Solution (SIS) 
 * lattice problem to achieve post-quantum security.
 * 
 * Security basis:
 * - SIS problem is NP-hard and believed quantum-resistant
 * - Best known quantum algorithms (e.g., Grover) provide only √n speedup
 * - NIST PQC standards (Kyber, Dilithium) use similar lattice assumptions
 * 
 * Parameters (128-bit post-quantum security):
 * - n = 512 (lattice dimension)
 * - m = 1024 (number of samples)  
 * - q = 3329 (modulus, same as Kyber)
 * - β = 100 (short vector bound)
 */

#ifndef QUANTONIUMOS_RFT_SIS_H
#define QUANTONIUMOS_RFT_SIS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "../../../include/rft_kernel.h"  // For rft_variant_t

#ifdef __cplusplus
extern "C" {
#endif

// SIS Parameters for 128-bit post-quantum security
#define RFT_SIS_N           512     // Lattice dimension
#define RFT_SIS_M           1024    // Number of samples
#define RFT_SIS_Q           3329    // Modulus (Kyber prime)
#define RFT_SIS_BETA        100     // Short vector bound
#define RFT_SIS_HASH_SIZE   32      // 256-bit output

// Golden ratio for RFT integration
#define RFT_SIS_PHI         1.6180339887498948482

// Error codes
typedef enum {
    RFT_SIS_SUCCESS = 0,
    RFT_SIS_ERROR_INVALID_PARAM = -1,
    RFT_SIS_ERROR_BUFFER_TOO_SMALL = -2,
    RFT_SIS_ERROR_INIT_FAILED = -3,
    RFT_SIS_ERROR_HASH_FAILED = -4
} rft_sis_error_t;

// RFT-SIS context structure
typedef struct {
    int32_t A[RFT_SIS_M][RFT_SIS_N];  // Public lattice matrix
    double rft_phases[RFT_SIS_N];      // Pre-computed RFT phases
    rft_variant_t variant;             // RFT variant (FIBONACCI recommended for lattice)
    bool initialized;
} rft_sis_ctx_t;

// Performance metrics
typedef struct {
    double hash_time_us;
    double avalanche_ratio;
    size_t bits_changed;
    size_t total_bits;
} rft_sis_metrics_t;

/**
 * Initialize RFT-SIS context
 * @param ctx Context to initialize
 * @param seed Random seed for lattice generation (use NULL for default)
 * @param variant RFT variant to use (RFT_VARIANT_FIBONACCI recommended for lattice crypto)
 * @return RFT_SIS_SUCCESS or error code
 */
rft_sis_error_t rft_sis_init(rft_sis_ctx_t* ctx, const uint8_t* seed, rft_variant_t variant);

/**
 * Cleanup RFT-SIS context
 */
rft_sis_error_t rft_sis_cleanup(rft_sis_ctx_t* ctx);

/**
 * Hash arbitrary data using RFT-SIS
 * @param ctx Initialized context
 * @param input Input data
 * @param input_len Length of input
 * @param output 32-byte output buffer
 * @return RFT_SIS_SUCCESS or error code
 */
rft_sis_error_t rft_sis_hash(const rft_sis_ctx_t* ctx,
                            const uint8_t* input, size_t input_len,
                            uint8_t* output);

/**
 * Derive key material using RFT-SIS (for post-quantum key derivation)
 * @param ctx Initialized context
 * @param master_key Master key input
 * @param master_key_len Length of master key
 * @param info Context/domain separation string
 * @param info_len Length of info
 * @param output Output buffer
 * @param output_len Desired output length
 * @return RFT_SIS_SUCCESS or error code
 */
rft_sis_error_t rft_sis_kdf(const rft_sis_ctx_t* ctx,
                           const uint8_t* master_key, size_t master_key_len,
                           const uint8_t* info, size_t info_len,
                           uint8_t* output, size_t output_len);

/**
 * Compute authentication tag using RFT-SIS (post-quantum MAC)
 * @param ctx Initialized context
 * @param key Authentication key
 * @param key_len Key length
 * @param data Data to authenticate
 * @param data_len Data length
 * @param tag 32-byte output tag
 * @return RFT_SIS_SUCCESS or error code
 */
rft_sis_error_t rft_sis_mac(const rft_sis_ctx_t* ctx,
                           const uint8_t* key, size_t key_len,
                           const uint8_t* data, size_t data_len,
                           uint8_t* tag);

/**
 * Verify authentication tag
 * @return RFT_SIS_SUCCESS if valid, RFT_SIS_ERROR_HASH_FAILED if invalid
 */
rft_sis_error_t rft_sis_mac_verify(const rft_sis_ctx_t* ctx,
                                  const uint8_t* key, size_t key_len,
                                  const uint8_t* data, size_t data_len,
                                  const uint8_t* tag);

/**
 * Run avalanche test
 */
rft_sis_error_t rft_sis_avalanche_test(const rft_sis_ctx_t* ctx,
                                       rft_sis_metrics_t* metrics);

/**
 * Get version string
 */
const char* rft_sis_version(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTONIUMOS_RFT_SIS_H
