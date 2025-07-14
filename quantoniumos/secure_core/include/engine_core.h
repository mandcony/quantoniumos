#ifndef ENGINE_CORE_H
#define ENGINE_CORE_H

#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// RFT Result structure
typedef struct {
    float* bins;
    int bin_count;
    float hr;  // Harmonic Resonance
} RFTResult;

// SA Vector structure
typedef struct {
    float* values;
    int count;
} SAVector;

/**
 * Initialize the engine
 * 
 * @return 0 on success, non-zero on failure
 */
EXPORT int engine_init(void);

/**
 * Clean up the engine
 */
EXPORT void engine_final(void);

/**
 * Run Resonance Fourier Transform on input data
 * 
 * @param data Input data buffer
 * @param length Length of input data
 * @return Pointer to RFTResult structure (must be freed with rft_free)
 */
EXPORT RFTResult* rft_run(const char* data, int length);

/**
 * Free RFT result memory
 * 
 * @param result RFTResult to free
 */
EXPORT void rft_free(RFTResult* result);

/**
 * Compute Symbolic Alignment vector
 * 
 * @param data Input data buffer
 * @param length Length of input data
 * @return Pointer to SAVector structure (must be freed with sa_free)
 */
EXPORT SAVector* sa_compute(const char* data, int length);

/**
 * Free SA vector memory
 * 
 * @param vector SAVector to free
 */
EXPORT void sa_free(SAVector* vector);

/**
 * Compute waveform hash
 * 
 * @param data Input data buffer
 * @param length Length of input data
 * @return Pointer to hash string (do not free - managed by engine)
 */
EXPORT const char* wave_hash(const char* data, int length);

/**
 * Encrypt data using symbolic XOR
 * 
 * @param plaintext Plaintext buffer
 * @param key Key buffer
 * @param length Length of buffers
 * @param result Buffer to store result (must be at least 'length' bytes)
 * @return 0 on success, non-zero on failure
 */
EXPORT int symbolic_xor(const uint8_t* plaintext, const uint8_t* key, int length, uint8_t* result);

/**
 * Generate quantum-inspired entropy
 * 
 * @param buffer Buffer to store entropy
 * @param length Number of bytes to generate
 * @return 0 on success, non-zero on failure
 */
EXPORT int generate_entropy(uint8_t* buffer, int length);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_CORE_H