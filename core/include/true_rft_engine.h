#ifndef TRUE_RFT_ENGINE_H
#define TRUE_RFT_ENGINE_H

#include <vector>
#include <complex>
#include <string>
#include <Eigen/Dense>

// Struct for returning RFT results from C-style functions
struct RFTResult {
    int bin_count;
    float* bins;
    float hr; // Harmonic Resonance
};

// Struct for returning Symbolic Alignment Vector
struct SAVector {
    int count;
    float* values;
};

extern "C" {
    // Engine lifecycle
    int engine_init(void);
    void engine_final(void);

    // Core RFT functions
    RFTResult* rft_run(const char* data, int length);
    void rft_free(RFTResult* result);

    // Symbolic Alignment
    SAVector* sa_compute(const char* data, int length);
    void sa_free(SAVector* vector);

    // Hashing and utilities
    const char* wave_hash(const char* data, int length);
    int symbolic_xor(const uint8_t* plaintext, const uint8_t* key, int length, uint8_t* result);
    int generate_entropy(uint8_t* buffer, int length);

    // True RFT Basis Transform API
    int rft_basis_forward(const double* x, int N, const double* w_arr, int M, const double* th_arr, const double* om_arr, double sigma0, double gamma, const char* seq, double* out_real, double* out_imag);
    int rft_basis_inverse(const double* Xr, const double* Xi, int N, const double* w_arr, int M, const double* th_arr, const double* om_arr, double sigma0, double gamma, const char* seq, double* out_x);
    int rft_operator_apply(const double* xr, const double* xi, int N, const double* w_arr, int M, const double* th_arr, const double* om_arr, double sigma0, double gamma, const char* seq, double* out_r, double* out_i);
    
    // Legacy resonance-coupled DFT (for compatibility)
    void inverse_rft_run(double* real_part, double* imag_part, int size);
    void forward_rft_run(double* real_part, double* imag_part, int size);
    void forward_rft_with_coupling(double* real_part, double* imag_part, int size, double alpha);
    void inverse_rft_with_coupling(double* real_part, double* imag_part, int size, double alpha);
}

#endif // TRUE_RFT_ENGINE_H
