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

/**
 * Spec-compliant RFT (basis path): forward transform X = Ψ^H x
 * Parameters may be null to use defaults: weights=[0.7,0.3], theta0=[0,pi/4], omega=[1,phi]
 * sequence_type: "golden_ratio" | "qpsk" | "const" | "circulant"
 * Returns 0 on success, non-zero on failure.
 */
EXPORT int rft_basis_forward(
    const double* x, int N,
    const double* weights, int M,
    const double* theta0,
    const double* omega,
    double sigma0,
    double gamma,
    const char* sequence_type,
    double* out_real,
    double* out_imag);

/**
 * Spec-compliant RFT (basis path): inverse transform x = Ψ X
 * X provided as real/imag arrays of length N. Returns 0 on success.
 */
EXPORT int rft_basis_inverse(
    const double* X_real,
    const double* X_imag,
    int N,
    const double* weights, int M,
    const double* theta0,
    const double* omega,
    double sigma0,
    double gamma,
    const char* sequence_type,
    double* out_x);

/**
 * Resonance operator apply (filter): x_hat = R x (not unitary unless R unitary)
 * Accepts complex input (real/imag arrays). Returns 0 on success.
 */
EXPORT int rft_operator_apply(
    const double* x_real,
    const double* x_imag,
    int N,
    const double* weights, int M,
    const double* theta0,
    const double* omega,
    double sigma0,
    double gamma,
    const char* sequence_type,
    double* out_real,
    double* out_imag);

/**
 * DEPRECATED: Fingerprint via Goertzel; prefer rft_basis_forward for the transform.
 * Alias for compatibility.
 */
EXPORT RFTResult* rft_fingerprint_goertzel(const char* data, int length);

/**
 * Genuine Resonance Fourier Transform with Coupling Matrix
 * RFT_k = Σ R[k,n] * F[k,n] * x[n], where K = R ⊙ F (NOT standard DFT!)
 * Uses exponential decay resonance coupling: R[k,n] = exp(-α|f_k - f_n|)
 * 
 * @param real_part Array of real parts (input/output)
 * @param imag_part Array of imaginary parts (input/output)  
 * @param size Size of the arrays
 */
EXPORT void forward_rft_run(double* real_part, double* imag_part, int size);

/**
 * Genuine Inverse Resonance Fourier Transform with Coupling Matrix
 * Solves: (R ⊙ F) x = y for x using iterative method (NOT standard IDFT!)
 * 
 * @param real_part Array of real parts (input/output)
 * @param imag_part Array of imaginary parts (input/output)
 * @param size Size of the arrays
 */
EXPORT void inverse_rft_run(double* real_part, double* imag_part, int size);

/**
 * Forward RFT with Configurable Resonance Coupling Parameter
 * 
 * @param real_part Array of real parts (input/output)
 * @param imag_part Array of imaginary parts (input/output)
 * @param size Size of the arrays
 * @param alpha Resonance coupling strength (0.0 = weak, 1.0 = strong)
 */
EXPORT void forward_rft_with_coupling(double* real_part, double* imag_part, int size, double alpha);

/**
 * Inverse RFT with Configurable Resonance Coupling Parameter
 * 
 * @param real_part Array of real parts (input/output)
 * @param imag_part Array of imaginary parts (input/output)
 * @param size Size of the arrays
 * @param alpha Resonance coupling strength (0.0 = weak, 1.0 = strong)
 */
EXPORT void inverse_rft_with_coupling(double* real_part, double* imag_part, int size, double alpha);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_CORE_H