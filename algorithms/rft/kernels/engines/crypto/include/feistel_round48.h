/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate license.
 */

/**
 * Enhanced RFT Crypto v2 - 48-Round Feistel Cipher
 * High-Performance C/Assembly Implementation
 * 
 * Target: 9.2 MB/s throughput as specified in QuantoniumOS paper
 * 
 * Features:
 * - 48-round Feistel network with 128-bit blocks
 * - AES S-box substitution with SIMD optimization
 * - MixColumns-like diffusion with AVX2
 * - ARX operations in assembly
 * - Domain-separated key derivation
 * - AEAD authenticated encryption
 */

#ifndef QUANTONIUMOS_FEISTEL_ROUND48_H
#define QUANTONIUMOS_FEISTEL_ROUND48_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "../../../include/rft_kernel.h"  // For rft_variant_t

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define FEISTEL_48_ROUNDS       48
#define FEISTEL_BLOCK_SIZE      16      // 128-bit blocks
#define FEISTEL_KEY_SIZE        32      // 256-bit keys
#define FEISTEL_ROUND_KEY_SIZE  16      // 128-bit round keys
#define FEISTEL_TAG_SIZE        16      // 128-bit authentication tag
#define FEISTEL_NONCE_SIZE      12      // 96-bit nonce

// Error codes
typedef enum {
    FEISTEL_SUCCESS = 0,
    FEISTEL_ERROR_INVALID_PARAM = -1,
    FEISTEL_ERROR_BUFFER_TOO_SMALL = -2,
    FEISTEL_ERROR_AUTH_FAILED = -3,
    FEISTEL_ERROR_INIT_FAILED = -4
} feistel_error_t;

// Performance flags
#define FEISTEL_FLAG_USE_AVX2    0x00000001
#define FEISTEL_FLAG_USE_AES_NI  0x00000002
#define FEISTEL_FLAG_PARALLEL    0x00000004

// Cipher context structure
typedef struct {
    uint8_t round_keys[FEISTEL_48_ROUNDS][FEISTEL_ROUND_KEY_SIZE];
    uint8_t pre_whiten_key[FEISTEL_ROUND_KEY_SIZE];
    uint8_t post_whiten_key[FEISTEL_ROUND_KEY_SIZE];
    uint8_t auth_key[FEISTEL_KEY_SIZE];
    rft_variant_t variant;  // RFT variant for round function (CHAOTIC recommended for diffusion)
    uint32_t flags;
    bool initialized;
} feistel_ctx_t;

// Metrics structure for performance analysis
typedef struct {
    double message_avalanche;
    double key_avalanche;
    double key_sensitivity;
    double throughput_mbps;
    uint64_t total_bytes_processed;
    uint64_t total_time_ns;
} feistel_metrics_t;

// Core API functions
feistel_error_t feistel_init(feistel_ctx_t* ctx, const uint8_t* master_key, 
                            size_t key_len, uint32_t flags, rft_variant_t variant);
feistel_error_t feistel_cleanup(feistel_ctx_t* ctx);

// Block cipher operations
feistel_error_t feistel_encrypt_block(const feistel_ctx_t* ctx,
                                     const uint8_t* plaintext,
                                     uint8_t* ciphertext);
feistel_error_t feistel_decrypt_block(const feistel_ctx_t* ctx,
                                     const uint8_t* ciphertext,
                                     uint8_t* plaintext);

// AEAD operations  
feistel_error_t feistel_aead_encrypt(const feistel_ctx_t* ctx,
                                    const uint8_t* nonce,
                                    const uint8_t* plaintext, size_t pt_len,
                                    const uint8_t* associated_data, size_t ad_len,
                                    uint8_t* ciphertext,
                                    uint8_t* tag);

feistel_error_t feistel_aead_decrypt(const feistel_ctx_t* ctx,
                                    const uint8_t* nonce,
                                    const uint8_t* ciphertext, size_t ct_len,
                                    const uint8_t* associated_data, size_t ad_len,
                                    const uint8_t* tag,
                                    uint8_t* plaintext);

// Performance and validation
feistel_error_t feistel_benchmark(feistel_ctx_t* ctx, size_t test_size,
                                 feistel_metrics_t* metrics);
feistel_error_t feistel_avalanche_test(feistel_ctx_t* ctx,
                                      feistel_metrics_t* metrics);

// Internal optimized functions (exposed for testing)
void feistel_round_function(const uint8_t* input, const uint8_t* round_key,
                           uint8_t* output);
void feistel_sbox_parallel(const uint8_t* input, uint8_t* output, size_t len);
void feistel_mixcolumns_avx2(const uint8_t* input, uint8_t* output);
void feistel_arx_operation(const uint8_t* a, const uint8_t* b, uint8_t* output);

// Key derivation functions
feistel_error_t feistel_hkdf(const uint8_t* master_key, size_t key_len,
                            const uint8_t* info, size_t info_len,
                            uint8_t* output, size_t output_len);

// Hardware capability detection
bool feistel_has_avx2(void);
bool feistel_has_aes_ni(void);
const char* feistel_get_cpu_info(void);

// Debugging and validation utilities
void feistel_print_block(const uint8_t* block, const char* label);
void feistel_print_metrics(const feistel_metrics_t* metrics);
feistel_error_t feistel_self_test(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTONIUMOS_FEISTEL_ROUND48_H
