#ifdef BUILDING_DLL
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <stdexcept>
#include <cmath>

using namespace Eigen;

// ------- Basic U and T -------

EXPORT void U(double* state, double* derivative, int n, double* out, double dt) {
    Map<VectorXd> vec_state(state, n);
    Map<VectorXd> vec_derivative(derivative, n);
    Map<VectorXd> vec_out(out, n);
    vec_out = vec_state + dt * vec_derivative;
}

EXPORT void T(double* state, double* transform, int n, double* out) {
    Map<VectorXd> vec_state(state, n);
    Map<VectorXd> vec_transform(transform, n);
    Map<VectorXd> vec_out(out, n);
    vec_out = vec_state.array() * vec_transform.array();
}

// ------- Basic Eigenvector Solver -------

EXPORT void ComputeEigenvectors(double* state, int n, double* eigenvalues_out, double* eigenvectors_out) {
    if (!state || !eigenvalues_out || !eigenvectors_out || n <= 0) {
        std::cerr << "❌ Invalid arguments to ComputeEigenvectors" << std::endl;
        return;
    }

    Map<VectorXd> vec_state(state, n);
    Map<VectorXd> vec_eigenvalues(eigenvalues_out, n);
    Map<MatrixXd> mat_eigenvectors(eigenvectors_out, n, n);

    // Create diagonal matrix and compute eigen
    MatrixXd diag = vec_state.asDiagonal();
    EigenSolver<MatrixXd> solver(diag);
    if (solver.info() != Success) {
        std::cerr << "❌ Eigen decomposition failed." << std::endl;
        return;
    }

    vec_eigenvalues = solver.eigenvalues().real();
    mat_eigenvectors = solver.eigenvectors().real();
}

// Additional functionality for the Quantum OS module

// Compute resonance signature for a waveform
EXPORT void resonance_signature(const double* waveform, int size, char* output, int* output_size) {
    if (!waveform || !output || !output_size || size <= 0) {
        std::cerr << "❌ Invalid arguments to resonance_signature" << std::endl;
        return;
    }
    
    Map<const VectorXd> vec_waveform(waveform, size);
    
    // Calculate frequency domain representation
    VectorXd frequencies = VectorXd::Zero(size);
    for (int k = 0; k < size; k++) {
        for (int n = 0; n < size; n++) {
            double angle = 2 * M_PI * k * n / size;
            frequencies[k] += vec_waveform[n] * cos(angle);
        }
    }
    
    // Find dominant frequencies
    std::vector<std::pair<double, int>> freq_power;
    for (int i = 0; i < size; i++) {
        freq_power.push_back({std::abs(frequencies[i]), i});
    }
    
    // Sort by power
    std::sort(freq_power.begin(), freq_power.end(), 
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first;
        });
    
    // Create signature string
    std::stringstream signature;
    int top_count = std::min(5, size);  // Take top 5 frequencies or less
    
    for (int i = 0; i < top_count; i++) {
        if (i > 0) signature << ":";
        signature << freq_power[i].second << "=" << std::fixed << std::setprecision(3) << freq_power[i].first;
    }
    
    std::string result = signature.str();
    if (result.length() + 1 > *output_size) {
        *output_size = result.length() + 1;
        std::cerr << "❌ Output buffer too small for resonance signature" << std::endl;
        return;
    }
    
    strcpy(output, result.c_str());
    *output_size = result.length() + 1;
    strcpy(output, result.c_str());
    *output_size = result.length() + 1;
}

// Additional quantum operations

EXPORT void quantum_superposition(double* state1, double* state2, int size, double alpha, double beta, double* output) {
    validate_inputs(state1, size, "state1");
    validate_inputs(state2, size, "state2");
    validate_inputs(output, size, "output");
    
    Map<VectorXd> vec_state1(state1, size);
    Map<VectorXd> vec_state2(state2, size);
    Map<VectorXd> vec_output(output, size);
    
    // Create a superposition |ψ⟩ = α|state1⟩ + β|state2⟩
    vec_output = alpha * vec_state1 + beta * vec_state2;
    
    // Normalize the result
    double norm = vec_output.norm();
    if (norm > 1e-10) {
        vec_output /= norm;
    }
}