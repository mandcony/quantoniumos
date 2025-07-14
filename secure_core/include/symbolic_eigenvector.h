#ifndef SYMBOLIC_EIGENVECTOR_H
#define SYMBOLIC_EIGENVECTOR_H

#include <cstdint>

#ifdef BUILDING_DLL
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

// --------- Symbolic Structures ---------
struct WaveNumber {
    double amplitude;
    double phase;
    WaveNumber(double amp, double ph) : amplitude(amp), phase(ph) {}
};

// --------- Symbolic State Operators ---------
EXPORT void U(double* state, double* derivative, int n, double* out, double dt);
EXPORT void T(double* state, double* transform, int n, double* out);
EXPORT void ComputeEigenvectors(double* state, int n, double* eigenvalues_out, double* eigenvectors_out);

// --------- Enhanced Basis Transformation ---------
EXPORT void transform_basis(const double* data, int data_size, const double* basis, int basis_rows, int basis_cols, double* output);
EXPORT void generate_eigenstate_entropy(int size, double* output);

// --------- Symbolic Resonance Encoding ---------
EXPORT void encode_resonance(const char* data, char* out, int* out_len);
EXPORT void decode_resonance(const char* encoded_data, char* out, int* out_len);
EXPORT double compute_similarity(const char* url1, const char* url2);

// --------- Post-Quantum Encryption ---------
EXPORT void ParallelXOREncrypt(const uint8_t* input, int input_len, const uint8_t* key, int key_len, uint8_t* output);

// --------- Vector Operations ---------
EXPORT double SumVector(const double* arr, int n);

// --------- Quantum Operations ---------
EXPORT void resonance_signature(const double* waveform, int size, char* output, int* output_size);
EXPORT void quantum_superposition(double* state1, double* state2, int size, double alpha, double beta, double* output);
EXPORT void hadamard_transform(double* data, int size, double* output);
EXPORT void generate_resonance_signature(double* data, int size, double* signature, int sig_size);
EXPORT void create_quantum_superposition(int size, double* output);

// --------- Optional: Symbolic Eigenvector Reduction ---------
EXPORT int symbolic_eigenvector_reduction(WaveNumber* wave_list, int wave_count, double threshold, WaveNumber* output_buffer);

#endif // SYMBOLIC_EIGENVECTOR_H