#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "true_rft_engine.h"

namespace py = pybind11;

// This file provides the Python bindings for the C++ True RFT engine.
// It exposes the C-style interface from true_rft_engine.h to Python.

// Helper function to convert C-style RFTResult to a Python dictionary
py::dict rft_result_to_dict(const RFTResult* result) {
    if (!result) {
        return py::dict();
    }
    py::list bins;
    for (int i = 0; i < result->bin_count; ++i) {
        bins.append(result->bins[i]);
    }
    return py::dict(
        "bin_count"_a=result->bin_count,
        "bins"_a=bins,
        "hr"_a=result->hr
    );
}

// Helper function to convert C-style SAVector to a Python list
py::list sa_vector_to_list(const SAVector* sa_vector) {
    if (!sa_vector) {
        return py::list();
    }
    py::list values;
    for (int i = 0; i < sa_vector->count; ++i) {
        values.append(sa_vector->values[i]);
    }
    return values;
}


PYBIND11_MODULE(true_rft_engine_bindings, m) {
    m.doc() = "True RFT Engine - Symbolic Resonance Kernel for Quantum Simulation";

    m.def("engine_init", &engine_init, "Initialize the engine");
    m.def("engine_final", &engine_final, "Finalize the engine");

    m.def("rft_run", [](py::bytes data) {
        std::string s = data;
        RFTResult* result = rft_run(s.c_str(), s.length());
        py::dict res_dict = rft_result_to_dict(result);
        rft_free(result);
        return res_dict;
    }, "Run Resonance Fourier Transform on input data");

    m.def("sa_compute", [](py::bytes data) {
        std::string s = data;
        SAVector* sa_vec = sa_compute(s.c_str(), s.length());
        py::list res_list = sa_vector_to_list(sa_vec);
        sa_free(sa_vec);
        return res_list;
    }, "Compute Symbolic Alignment vector");

    m.def("wave_hash", [](py::bytes data) {
        std::string s = data;
        return std::string(wave_hash(s.c_str(), s.length()));
    }, "Compute a deterministic 64-char hex hash");

    m.def("generate_entropy", [](int length) {
        std::vector<uint8_t> buffer(length);
        if (generate_entropy(buffer.data(), length) == 0) {
            return py::bytes(reinterpret_cast<const char*>(buffer.data()), length);
        }
        return py::bytes();
    }, "Generate entropy using engine RNG");

    m.def("rft_basis_forward", [](py::array_t<double> x, py::array_t<double> w, py::array_t<double> th, py::array_t<double> om, double sigma0, double gamma, const std::string& seq) {
        auto buf_x = x.request();
        int N = buf_x.shape[0];
        auto buf_w = w.request();
        int M = buf_w.shape[0];

        py::array_t<double> out_real(N);
        py::array_t<double> out_imag(N);
        auto buf_or = out_real.request();
        auto buf_oi = out_imag.request();

        rft_basis_forward(static_cast<double*>(buf_x.ptr), N, static_cast<double*>(buf_w.ptr), M, static_cast<double*>(th.request().ptr), static_cast<double*>(om.request().ptr), sigma0, gamma, seq.c_str(), static_cast<double*>(buf_or.ptr), static_cast<double*>(buf_oi.ptr));
        
        py::array_t<std::complex<double>> result(N);
        auto buf_res = result.request();
        auto ptr_res = static_cast<std::complex<double>*>(buf_res.ptr);
        for(int i=0; i<N; ++i) {
            ptr_res[i] = std::complex<double>(static_cast<double*>(buf_or.ptr)[i], static_cast<double*>(buf_oi.ptr)[i]);
        }
        return result;
    }, "RFT basis forward transform");

    m.def("rft_basis_inverse", [](py::array_t<std::complex<double>> X, py::array_t<double> w, py::array_t<double> th, py::array_t<double> om, double sigma0, double gamma, const std::string& seq) {
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

        rft_basis_inverse(Xr.data(), Xi.data(), N, static_cast<double*>(buf_w.ptr), M, static_cast<double*>(th.request().ptr), static_cast<double*>(om.request().ptr), sigma0, gamma, seq.c_str(), static_cast<double*>(buf_ox.ptr));
        return out_x;
    }, "RFT basis inverse transform");
}
