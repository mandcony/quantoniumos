#ifdef BUILDING_DLL
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unordered_map>
#include <mutex>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include "../secure_core/include/symbolic_eigenvector.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;

// Input validation helper function
inline void validate_inputs(const double* ptr, int size, const char* name) {
    if (!ptr) throw std::invalid_argument(std::string(name) + " pointer is null");
    if (size <= 0) throw std::invalid_argument(std::string(name) + " size must be positive");
}

// --------------------- Caching ---------------------

struct CacheEntry {
    VectorXd result;
    MatrixXd eigenvectors;
    double timestamp;
    CacheEntry() : timestamp(0.0) {}
    CacheEntry(const VectorXd& r, const MatrixXd& evecs, double t) 
        : result(r), eigenvectors(evecs), timestamp(t) {}
};

// Additional functionality for C++ implementation

// Transform data using the eigenvector basis
EXPORT void transform_basis(const double* data, int data_size, const double* basis, int basis_rows, int basis_cols, double* output) {
    validate_inputs(data, data_size, "data");
    validate_inputs(basis, basis_rows * basis_cols, "basis");
    validate_inputs(output, data_size, "output");
    
    Map<const VectorXd> vec_data(data, data_size);
    Map<const MatrixXd> mat_basis(basis, basis_rows, basis_cols);
    Map<VectorXd> vec_output(output, data_size);
    
    // Transform the data using the basis
    vec_output = mat_basis * vec_data;
}

// Generate entropy values based on eigenstate distribution
EXPORT void generate_eigenstate_entropy(int size, double* output) {
    validate_inputs(output, size, "output");
    
    Map<VectorXd> vec_output(output, size);
    
    // Generate a quantum-like probability distribution
    VectorXd probabilities = VectorXd::Zero(size);
    double theta = M_PI / 4.0; // π/4 gives balanced superposition
    
    // Create a simple superposition state
    for (int i = 0; i < size; i++) {
        // Amplitude for basis state |i⟩ = cos(θ) for i=0, sin(θ) for i=1, etc.
        double amplitude = 0.0;
        if (i == 0) {
            amplitude = cos(theta);
        } else if (i == 1 && size > 1) {
            amplitude = sin(theta);
        } else {
            // For larger spaces, distribute remaining probability
            amplitude = 0.1 * sin(theta * i) / sqrt(size);
        }
        // Probability is |amplitude|²
        probabilities[i] = amplitude * amplitude;
    }
    
    // Normalize probabilities
    double sum = probabilities.sum();
    if (sum > 0) {
        probabilities /= sum;
    } else {
        // Fallback to uniform distribution
        probabilities = VectorXd::Constant(size, 1.0/size);
    }
    
    // Calculate von Neumann entropy: -∑ p_i log(p_i)
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        if (probabilities[i] > 1e-10) { // Avoid log(0)
            vec_output[i] = -probabilities[i] * log2(probabilities[i]);
        } else {
            vec_output[i] = 0.0;
        }
    }
}

class TransformationCache {
private:
    std::unordered_map<std::string, CacheEntry> cache;
    size_t max_size;
    std::mutex mutex;
    double current_time;

    std::string generateKey(const VectorXd& state, double dt = 0.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        for (int i = 0; i < state.size(); ++i) {
            oss << state[i] << ",";
        }
        if (dt != 0.0) {
            oss << dt;
        }
        return oss.str();
    }

    void evict() {
        if (cache.empty()) return;
        auto oldest = cache.begin();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->second.timestamp < oldest->second.timestamp) {
                oldest = it;
            }
        }
        cache.erase(oldest);
    }

public:
    TransformationCache(size_t maxSize = 1000) : max_size(maxSize), current_time(0.0) {}

    std::pair<VectorXd, MatrixXd> getOrCompute(const VectorXd& state, double dt,
        std::function<std::pair<VectorXd, MatrixXd>(const VectorXd&, double)> computeFunc) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string key = generateKey(state, dt);
        current_time += 1.0;

        auto it = cache.find(key);
        if (it != cache.end()) {
            it->second.timestamp = current_time;
            return {it->second.result, it->second.eigenvectors};
        }

        auto [result, eigenvectors] = computeFunc(state, dt);
        if (cache.size() >= max_size) {
            evict();
        }
        cache[key] = CacheEntry(result, eigenvectors, current_time);
        return {result, eigenvectors};
    }
};

TransformationCache globalCache(1000);

// --------------------- Core Exports ---------------------

EXPORT void U(double* state, double* derivative, int n, double* out, double dt) {
    validate_inputs(state, n, "state");
    validate_inputs(derivative, n, "derivative");
    validate_inputs(out, n, "out");

    Map<VectorXd> vec_state(state, n);
    Map<VectorXd> vec_derivative(derivative, n);
    Map<VectorXd> vec_out(out, n);

    vec_out = vec_state + dt * vec_derivative;
}

EXPORT void T(double* state, double* transform, int n, double* out) {
    validate_inputs(state, n, "state");
    validate_inputs(transform, n, "transform");
    validate_inputs(out, n, "out");

    Map<VectorXd> vec_state(state, n);
    Map<VectorXd> vec_transform(transform, n);
    Map<VectorXd> vec_out(out, n);

    vec_out = vec_state.array() * vec_transform.array();
}

EXPORT void ComputeEigenvectors(double* state, int n, double* eigenvalues_out, double* eigenvectors_out) {
    validate_inputs(state, n, "state");
    validate_inputs(eigenvalues_out, n, "eigenvalues_out");
    validate_inputs(eigenvectors_out, n * n, "eigenvectors_out");

    Map<VectorXd> vec_state(state, n);
    Map<VectorXd> vec_eigenvalues(eigenvalues_out, n);
    Map<MatrixXd> mat_eigenvectors(eigenvectors_out, n, n);

    auto computeEigen = [](const VectorXd& s, double) {
        MatrixXd mat = s.asDiagonal();
        EigenSolver<MatrixXd> solver(mat);
        return std::make_pair(solver.eigenvalues().real(), solver.eigenvectors().real());
    };

    auto [eigenvalues, eigenvectors] = globalCache.getOrCompute(vec_state, 0.0, computeEigen);
    vec_eigenvalues = eigenvalues;
    mat_eigenvectors = eigenvectors;
}

// --------------------- Symbolic Resonance Encoding ---------------------

// Helper function: Calculate an amplitude-based hash for resonance encoding
double calculateResonanceHash(const std::string& input) {
    double A = 1.5; // Base amplitude
    double phi = 0.0; // Phase accumulator
    
    for (char c : input) {
        phi += (c * 0.01);
        A += std::sin(phi) * 0.1;
    }
    
    // Ensure amplitude stays in reasonable bounds (1.0-3.0)
    return 1.0 + std::fmod(std::abs(A), 2.0);
}

// Helper function: Calculate symbolic entropy of a byte sequence
double calculateSymbolicEntropy(const std::vector<uint8_t>& values) {
    // Count frequencies
    std::vector<int> counts(256, 0);
    for (uint8_t val : values) {
        counts[val]++;
    }
    
    // Calculate entropy
    double entropy = 0.0;
    for (int count : counts) {
        if (count > 0) {
            double prob = static_cast<double>(count) / values.size();
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}

EXPORT void encode_resonance(const char* data, char* out, int* out_len) {
    if (!data || !out || !out_len) throw std::invalid_argument("Input or output pointers are null");
    std::string input(data);
    
    // Calculate resonance parameters from input
    double A = calculateResonanceHash(input);
    
    // Create a symbolic waveform signature from the input
    std::vector<uint8_t> waveform;
    for (size_t i = 0; i < input.length(); i++) {
        double phase = static_cast<double>(i) / input.length() * 2.0 * M_PI;
        double signal = A * std::cos(phase);
        waveform.push_back(static_cast<uint8_t>((signal + 2.0) * 50));
    }
    
    // Apply a simple XOR encoding with the waveform
    std::vector<uint8_t> encoded(input.length());
    for (size_t i = 0; i < input.length(); i++) {
        encoded[i] = input[i] ^ waveform[i % waveform.size()];
    }
    
    // Convert to Base64-like encoding
    std::stringstream ss;
    ss << "QWV" << std::fixed << std::setprecision(3) << A << "_";
    for (uint8_t byte : encoded) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    
    std::string result = ss.str();
    *out_len = result.length() + 1;
    strncpy(out, result.c_str(), *out_len);
    out[*out_len - 1] = '\0';
}

EXPORT void decode_resonance(const char* encoded_data, char* out, int* out_len) {
    if (!encoded_data || !out || !out_len) throw std::invalid_argument("Input or output pointers are null");
    std::string input(encoded_data);
    
    // Parse the amplitude and encoded hex data
    if (input.substr(0, 3) != "QWV") {
        throw std::invalid_argument("Invalid resonance encoding format");
    }
    
    size_t sep_pos = input.find('_');
    if (sep_pos == std::string::npos) {
        throw std::invalid_argument("Invalid resonance encoding format");
    }
    
    double A = std::stod(input.substr(3, sep_pos - 3));
    std::string hex_data = input.substr(sep_pos + 1);
    
    // Convert hex to bytes
    std::vector<uint8_t> encoded;
    for (size_t i = 0; i < hex_data.length(); i += 2) {
        std::string byte_str = hex_data.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byte_str, nullptr, 16));
        encoded.push_back(byte);
    }
    
    // Generate the same waveform
    std::vector<uint8_t> waveform;
    for (size_t i = 0; i < encoded.size(); i++) {
        double phase = static_cast<double>(i) / encoded.size() * 2.0 * M_PI;
        double signal = A * std::cos(phase);
        waveform.push_back(static_cast<uint8_t>((signal + 2.0) * 50));
    }
    
    // Decode with XOR
    std::string decoded;
    for (size_t i = 0; i < encoded.size(); i++) {
        char c = encoded[i] ^ waveform[i % waveform.size()];
        decoded.push_back(c);
    }
    
    *out_len = decoded.length() + 1;
    strncpy(out, decoded.c_str(), *out_len);
    out[*out_len - 1] = '\0';
}

EXPORT double compute_similarity(const char* str1, const char* str2) {
    if (!str1 || !str2) throw std::invalid_argument("String pointers are null");
    std::string s1(str1), s2(str2);
    
    // Convert strings to resonance waveforms
    std::vector<double> wave1, wave2;
    double A1 = calculateResonanceHash(s1);
    double A2 = calculateResonanceHash(s2);
    
    for (size_t i = 0; i < std::max(s1.length(), s2.length()); i++) {
        double phase = static_cast<double>(i) * 0.1;
        
        // Generate wave point for s1
        if (i < s1.length()) {
            wave1.push_back(A1 * std::cos(phase + static_cast<double>(s1[i]) * 0.01));
        } else {
            wave1.push_back(0.0);
        }
        
        // Generate wave point for s2
        if (i < s2.length()) {
            wave2.push_back(A2 * std::cos(phase + static_cast<double>(s2[i]) * 0.01));
        } else {
            wave2.push_back(0.0);
        }
    }
    
    // Calculate cosine similarity between the two waveforms
    double dot_product = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (size_t i = 0; i < wave1.size(); i++) {
        dot_product += wave1[i] * wave2[i];
        norm1 += wave1[i] * wave1[i];
        norm2 += wave2[i] * wave2[i];
    }
    
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

// --------------------- Quantum-Inspired Algorithms ---------------------

// Perform Hadamard-like transformation on a vector
EXPORT void hadamard_transform(double* data, int size, double* output) {
    validate_inputs(data, size, "data");
    validate_inputs(output, size, "output");
    
    Map<const VectorXd> vec_data(data, size);
    Map<VectorXd> vec_output(output, size);
    
    // Construct a Hadamard-like matrix
    MatrixXd H = MatrixXd::Zero(size, size);
    double norm_factor = 1.0 / sqrt(size);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            H(i, j) = ((i & j) % 2 == 0) ? norm_factor : -norm_factor;
        }
    }
    
    // Apply the transformation
    vec_output = H * vec_data;
}

// Generate a symbolic resonance signature from data
EXPORT void generate_resonance_signature(double* data, int size, double* signature, int sig_size) {
    validate_inputs(data, size, "data");
    validate_inputs(signature, sig_size, "signature");
    
    Map<const VectorXd> vec_data(data, size);
    Map<VectorXd> vec_signature(signature, sig_size);
    
    // Create a symbolic resonance signature
    for (int i = 0; i < sig_size; i++) {
        double phase = 2.0 * M_PI * i / sig_size;
        double sum = 0.0;
        
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < size; j++) {
            double j_norm = static_cast<double>(j) / size;
            sum += vec_data[j] * cos(phase + 2.0 * M_PI * j_norm);
        }
        
        vec_signature[i] = sum / size;
    }
}

// Quantum superposition-inspired function
EXPORT void create_quantum_superposition(int size, double* output) {
    validate_inputs(output, size, "output");
    Map<VectorXd> vec_output(output, size);
    
    // Equal superposition of all states
    double norm_factor = 1.0 / sqrt(size);
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        vec_output[i] = norm_factor;
    }
}

// --------------------- XOR Encryption ---------------------

EXPORT void ParallelXOREncrypt(const uint8_t* input, int input_len, const uint8_t* key, int key_len, uint8_t* output) {
    if (!input || !key || !output) throw std::invalid_argument("Input, key, or output pointer is null");
    if (input_len <= 0 || key_len <= 0) throw std::invalid_argument("Input length or key length must be positive");
    #pragma omp parallel for
    for (int i = 0; i < input_len; ++i) {
        output[i] = input[i] ^ key[i % key_len];
    }
}

// --------------------- New: SumVector ---------------------
EXPORT double SumVector(const double* arr, int n) {
    if (!arr || n <= 0) throw std::invalid_argument("Invalid array or size");
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return sum;
}

// --------- Implementation of resonance signature ---------
EXPORT void resonance_signature(const double* waveform, int size, char* output, int* output_size) {
    if (!waveform || !output || !output_size) {
        throw std::invalid_argument("Input pointers cannot be null");
    }
    
    // Calculate the resonance signature
    double amplitude_sum = 0.0;
    double phase_product = 1.0;
    
    for (int i = 0; i < size; i++) {
        amplitude_sum += std::abs(waveform[i]);
        phase_product *= (1.0 + std::sin(waveform[i]));
    }
    
    double avg_amplitude = amplitude_sum / size;
    double resonance_factor = phase_product / size;
    
    // Format the signature string
    std::stringstream ss;
    ss << "RES" << std::fixed << std::setprecision(4) << avg_amplitude << "_" << resonance_factor;
    
    std::string result = ss.str();
    *output_size = result.length() + 1;
    
    if (*output_size > 1024) { // Add a reasonable limit
        *output_size = 1024;
    }
    
    strncpy(output, result.c_str(), *output_size);
    output[*output_size - 1] = '\0';
}

// --------- Implementation of quantum superposition ---------
EXPORT void quantum_superposition(double* state1, double* state2, int size, double alpha, double beta, double* output) {
    if (!state1 || !state2 || !output) {
        throw std::invalid_argument("State pointers cannot be null");
    }
    
    Map<const VectorXd> vec_state1(state1, size);
    Map<const VectorXd> vec_state2(state2, size);
    Map<VectorXd> vec_output(output, size);
    
    // Ensure alpha and beta satisfy |α|² + |β|² = 1
    double norm = std::sqrt(alpha*alpha + beta*beta);
    if (norm == 0.0) {
        throw std::invalid_argument("Alpha and beta cannot both be zero");
    }
    
    alpha /= norm;
    beta /= norm;
    
    // Create superposition: |ψ⟩ = α|state1⟩ + β|state2⟩
    vec_output = alpha * vec_state1 + beta * vec_state2;
}

// --------- Implementation of symbolic eigenvector reduction ---------
EXPORT int symbolic_eigenvector_reduction(WaveNumber* wave_list, int wave_count, double threshold, WaveNumber* output_buffer) {
    if (!wave_list || !output_buffer) {
        throw std::invalid_argument("Wave list or output buffer cannot be null");
    }
    
    if (wave_count <= 0) {
        return 0;
    }
    
    // Group similar waves based on their amplitude and phase
    std::vector<std::vector<int>> groups;
    std::vector<bool> processed(wave_count, false);
    
    for (int i = 0; i < wave_count; i++) {
        if (processed[i]) continue;
        
        std::vector<int> group;
        group.push_back(i);
        processed[i] = true;
        
        for (int j = i + 1; j < wave_count; j++) {
            if (processed[j]) continue;
            
            double amp_diff = std::abs(wave_list[i].amplitude - wave_list[j].amplitude);
            double phase_diff = std::abs(wave_list[i].phase - wave_list[j].phase);
            
            // Normalize phase difference to [0, π]
            if (phase_diff > M_PI) {
                phase_diff = 2 * M_PI - phase_diff;
            }
            
            if (amp_diff < threshold && phase_diff < threshold) {
                group.push_back(j);
                processed[j] = true;
            }
        }
        
        groups.push_back(group);
    }
    
    // Create reduced set of wave numbers by averaging each group
    int output_count = 0;
    for (const auto& group : groups) {
        if (output_count >= wave_count) break; // Safety check
        
        double avg_amp = 0.0;
        double avg_phase = 0.0;
        
        for (int idx : group) {
            avg_amp += wave_list[idx].amplitude;
            avg_phase += wave_list[idx].phase;
        }
        
        avg_amp /= group.size();
        avg_phase /= group.size();
        
        output_buffer[output_count] = WaveNumber(avg_amp, avg_phase);
        output_count++;
    }
    
    return output_count;
}