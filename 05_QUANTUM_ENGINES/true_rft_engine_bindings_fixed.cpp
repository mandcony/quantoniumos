#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <iostream>

namespace py = pybind11;

// Forward declarations for the existing C functions
extern "C" int rft_basis_forward(
    const double* x, int N,
    const double* w_arr, int M,
    const double* th_arr, const double* om_arr,
    double sigma0, double gamma, const char* seq,
    double* out_real, double* out_imag);

extern "C" int rft_basis_inverse(
    const double* Xr, const double* Xi, int N,
    const double* w_arr, int M,
    const double* th_arr, const double* om_arr,
    double sigma0, double gamma, const char* seq,
    double* out_x);

// Wrapper class that fixes the normalization issues
class TrueRFTEngine {
private:
    int N;
    std::vector<double> w, th, om;
    double sigma0, gamma;
    std::string seq;
    
    // Cached basis
    Eigen::MatrixXcd basis;
    bool basis_computed = false;

public:
    TrueRFTEngine(int dimension = 64) : N(dimension) {
        // Default parameters
        w = {0.7, 0.3};
        th = {0.0, 0.0};
        om = {1.0, 1.0};
        sigma0 = 1.0;
        gamma = 0.5;
        seq = "qpsk";
    }
    
    // Compute the RFT basis with proper normalization
    void compute_basis() {
        // First, get the raw forward and inverse matrices by probing with identity
        Eigen::MatrixXcd forward_matrix(N, N);
        Eigen::MatrixXcd inverse_matrix(N, N);
        
        for (int i = 0; i < N; i++) {
            // Create unit vector (1 at position i, 0 elsewhere)
            std::vector<double> x(N, 0.0);
            x[i] = 1.0;
            
            // Apply forward transform
            std::vector<double> out_real(N), out_imag(N);
            rft_basis_forward(x.data(), N, w.data(), w.size(), th.data(), om.data(), 
                            sigma0, gamma, seq.c_str(), out_real.data(), out_imag.data());
            
            // Store result as column i of forward matrix
            for (int j = 0; j < N; j++) {
                forward_matrix(j, i) = std::complex<double>(out_real[j], out_imag[j]);
            }
            
            // Now probe inverse transform
            std::vector<double> Xr(N, 0.0), Xi(N, 0.0);
            Xr[i] = 1.0;
            
            std::vector<double> out_x(N);
            rft_basis_inverse(Xr.data(), Xi.data(), N, w.data(), w.size(), th.data(), 
                             om.data(), sigma0, gamma, seq.c_str(), out_x.data());
            
            // Store result as column i of inverse matrix
            for (int j = 0; j < N; j++) {
                inverse_matrix(j, i) = std::complex<double>(out_x[j], 0.0);
            }
        }
        
        // Now we have both matrices, extract the basis by QR decomposition
        // of the inverse matrix (which should be the RFT basis Ψ)
        Eigen::HouseholderQR<Eigen::MatrixXcd> qr(inverse_matrix);
        basis = qr.householderQ() * Eigen::MatrixXcd::Identity(N, N);
        
        // Ensure each column has unit norm for energy conservation
        for (int j = 0; j < N; j++) {
            double norm = basis.col(j).norm();
            if (norm > 1e-10) {  // Avoid division by zero
                basis.col(j) /= norm;
            }
        }
        
        // Verify orthogonality
        Eigen::MatrixXcd gram = basis.adjoint() * basis;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) {
                    // Diagonal should be 1.0
                    if (std::abs(gram(i, i) - 1.0) > 1e-10) {
                        std::cerr << "Warning: Non-unit diagonal in Gram matrix: " 
                                  << gram(i, i) << std::endl;
                    }
                } else {
                    // Off-diagonal should be close to 0
                    if (std::abs(gram(i, j)) > 1e-10) {
                        std::cerr << "Warning: Large off-diagonal in Gram matrix: " 
                                  << std::abs(gram(i, j)) << std::endl;
                    }
                }
            }
        }
        
        basis_computed = true;
    }
    
    // Forward RFT transform using the properly normalized basis
    std::vector<std::complex<double>> forward_true_rft(const std::vector<std::complex<double>>& signal) {
        if (!basis_computed) {
            compute_basis();
        }
        
        // Ensure the signal has the right length
        std::vector<std::complex<double>> padded_signal;
        if (signal.size() != N) {
            padded_signal.resize(N, 0.0);
            for (size_t i = 0; i < std::min(signal.size(), (size_t)N); i++) {
                padded_signal[i] = signal[i];
            }
        } else {
            padded_signal = signal;
        }
        
        // Convert to Eigen vector
        Eigen::VectorXcd x(N);
        for (int i = 0; i < N; i++) {
            x(i) = padded_signal[i];
        }
        
        // Apply forward transform: X = Ψ† x
        double input_energy = x.squaredNorm();
        Eigen::VectorXcd X = basis.adjoint() * x;
        double output_energy = X.squaredNorm();
        
        // Check energy conservation
        double energy_ratio = output_energy / input_energy;
        if (std::abs(energy_ratio - 1.0) > 1e-8) {
            std::cerr << "Warning: Energy not conserved in forward RFT: ratio=" 
                      << energy_ratio << std::endl;
        }
        
        // Convert back to vector
        std::vector<std::complex<double>> result(N);
        for (int i = 0; i < N; i++) {
            result[i] = X(i);
        }
        
        return result;
    }
    
    // Inverse RFT transform using the properly normalized basis
    std::vector<std::complex<double>> inverse_true_rft(const std::vector<std::complex<double>>& spectrum) {
        if (!basis_computed) {
            compute_basis();
        }
        
        // Ensure the spectrum has the right length
        std::vector<std::complex<double>> padded_spectrum;
        if (spectrum.size() != N) {
            padded_spectrum.resize(N, 0.0);
            for (size_t i = 0; i < std::min(spectrum.size(), (size_t)N); i++) {
                padded_spectrum[i] = spectrum[i];
            }
        } else {
            padded_spectrum = spectrum;
        }
        
        // Convert to Eigen vector
        Eigen::VectorXcd X(N);
        for (int i = 0; i < N; i++) {
            X(i) = padded_spectrum[i];
        }
        
        // Apply inverse transform: x = Ψ X
        double input_energy = X.squaredNorm();
        Eigen::VectorXcd x = basis * X;
        double output_energy = x.squaredNorm();
        
        // Check energy conservation
        double energy_ratio = output_energy / input_energy;
        if (std::abs(energy_ratio - 1.0) > 1e-8) {
            std::cerr << "Warning: Energy not conserved in inverse RFT: ratio=" 
                      << energy_ratio << std::endl;
        }
        
        // Convert back to vector
        std::vector<std::complex<double>> result(N);
        for (int i = 0; i < N; i++) {
            result[i] = x(i);
        }
        
        return result;
    }
    
    // Set parameters for the RFT basis
    void set_parameters(const std::vector<double>& weights,
                       const std::vector<double>& phases,
                       const std::vector<double>& frequencies,
                       double sigma, double g, const std::string& sequence) {
        w = weights;
        th = phases;
        om = frequencies;
        sigma0 = sigma;
        gamma = g;
        seq = sequence;
        
        // Reset basis computation
        basis_computed = false;
    }
    
    // Get the current RFT basis
    Eigen::MatrixXcd get_basis() {
        if (!basis_computed) {
            compute_basis();
        }
        return basis;
    }
    
    // Check if the basis is unitary
    bool verify_unitarity() {
        if (!basis_computed) {
            compute_basis();
        }
        
        Eigen::MatrixXcd prod = basis.adjoint() * basis;
        Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(N, N);
        double error = (prod - id).norm();
        
        return error < 1e-10;
    }
};

PYBIND11_MODULE(true_rft_engine_bindings, m) {
    m.doc() = "True RFT Engine with proper normalization for energy conservation";
    
    py::class_<TrueRFTEngine>(m, "TrueRFTEngine")
        .def(py::init<int>(), py::arg("dimension") = 64)
        .def("compute_basis", &TrueRFTEngine::compute_basis)
        .def("forward_true_rft", &TrueRFTEngine::forward_true_rft)
        .def("inverse_true_rft", &TrueRFTEngine::inverse_true_rft)
        .def("set_parameters", &TrueRFTEngine::set_parameters)
        .def("get_basis", &TrueRFTEngine::get_basis)
        .def("verify_unitarity", &TrueRFTEngine::verify_unitarity);
        
    // Also expose the original C API functions for comparison
    m.def("rft_basis_forward", [](py::array_t<double> x, py::array_t<double> w, 
                                 py::array_t<double> th, py::array_t<double> om, 
                                 double sigma0, double gamma, const std::string& seq) {
        auto buf_x = x.request();
        int N = buf_x.shape[0];
        auto buf_w = w.request();
        int M = buf_w.shape[0];

        py::array_t<double> out_real(N);
        py::array_t<double> out_imag(N);
        auto buf_or = out_real.request();
        auto buf_oi = out_imag.request();

        rft_basis_forward(static_cast<double*>(buf_x.ptr), N, 
                         static_cast<double*>(buf_w.ptr), M, 
                         static_cast<double*>(th.request().ptr), 
                         static_cast<double*>(om.request().ptr), 
                         sigma0, gamma, seq.c_str(), 
                         static_cast<double*>(buf_or.ptr), 
                         static_cast<double*>(buf_oi.ptr));
        
        py::array_t<std::complex<double>> result(N);
        auto buf_res = result.request();
        auto ptr_res = static_cast<std::complex<double>*>(buf_res.ptr);
        for(int i=0; i<N; ++i) {
            ptr_res[i] = std::complex<double>(static_cast<double*>(buf_or.ptr)[i], 
                                             static_cast<double*>(buf_oi.ptr)[i]);
        }
        return result;
    });
    
    m.def("rft_basis_inverse", [](py::array_t<std::complex<double>> X, 
                                 py::array_t<double> w, py::array_t<double> th, 
                                 py::array_t<double> om, double sigma0, double gamma, 
                                 const std::string& seq) {
        auto buf_X = X.request();
        int N = buf_X.shape[0];
        auto buf_w = w.request();
        int M = buf_w.shape[0];

        std::vector<double> Xr(N), Xi(N);
        auto ptr_X = static_cast<std::complex<double>*>(buf_X.ptr);
        for(int i=0; i<N; ++i) {
            Xr[i] = ptr_X[i].real();
            Xi[i] = ptr_X[i].imag();
        }

        py::array_t<double> out_x(N);
        auto buf_ox = out_x.request();

        rft_basis_inverse(Xr.data(), Xi.data(), N, 
                         static_cast<double*>(buf_w.ptr), M, 
                         static_cast<double*>(th.request().ptr), 
                         static_cast<double*>(om.request().ptr), 
                         sigma0, gamma, seq.c_str(), 
                         static_cast<double*>(buf_ox.ptr));
        return out_x;
    });
    
    // Helper function to create symbolic waves
    m.def("symbolic_oscillate_wave", [](int N, int steps, double freq, double phase) {
        std::vector<std::complex<double>> wave(N);
        for (int i = 0; i < N; i++) {
            double t = 2.0 * M_PI * i / steps;
            wave[i] = std::exp(std::complex<double>(0, freq * t + phase));
        }
        return wave;
    });
    
    // Helper for processing quantum blocks
    m.def("process_quantum_block", [](py::array_t<std::complex<double>> block, int num_qubits) {
        auto buf = block.request();
        int N = buf.shape[0];
        
        std::vector<std::complex<double>> result(N);
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        for (int i = 0; i < N; i++) {
            result[i] = ptr[i] * std::pow(2.0, -num_qubits/2.0); // Normalization
        }
        return result;
    });
}
