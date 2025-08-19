#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engine_core.h"
#include <string> // Required for std::string

namespace py = pybind11;

PYBIND11_MODULE(engine_core_pybind, m) {
    m.doc() = "pybind11 bindings for Quantonium EngineCore C-API";

    py::class_<RFTResult>(m, "RFTResult")
        .def_readonly("bin_count", &RFTResult::bin_count)
        .def_readonly("hr", &RFTResult::hr)
        .def_property_readonly("bins", [](const RFTResult& r) {
            return py::array_t<float>(r.bin_count, r.bins, py::cast(r, py::return_value_policy::reference));
        });

    py::class_<SAVector>(m, "SAVector")
        .def_readonly("count", &SAVector::count)
        .def_property_readonly("values", [](const SAVector& s) {
            return py::array_t<float>(s.count, s.values, py::cast(s, py::return_value_policy::reference));
        });

    m.def("engine_init", &engine_init, "Initialize the engine");
    m.def("engine_final", &engine_final, "Clean up the engine");

    m.def("rft_run", [](const py::bytes& data) {
        py::buffer_info info = py::buffer(data).request();
        return rft_run(static_cast<const char*>(info.ptr), static_cast<int>(info.size));
    }, py::return_value_policy::take_ownership, "Run Resonance Fourier Transform on input data");

    m.def("rft_free", &rft_free, "Free RFT result memory");

    m.def("sa_compute", [](const py::bytes& data) {
        py::buffer_info info = py::buffer(data).request();
        return sa_compute(static_cast<const char*>(info.ptr), static_cast<int>(info.size));
    }, py::return_value_policy::take_ownership, "Compute Symbolic Alignment vector");

    m.def("sa_free", &sa_free, "Free SA vector memory");

    m.def("wave_hash", [](const py::bytes& data) {
        py::buffer_info info = py::buffer(data).request();
        return std::string(wave_hash(static_cast<const char*>(info.ptr), static_cast<int>(info.size)));
    }, "Compute waveform hash");

    m.def("symbolic_xor", [](const py::bytes& plaintext, const py::bytes& key) {
        if (py::len(plaintext) != py::len(key)) {
            throw std::runtime_error("Plaintext and key must have the same length");
        }
        py::buffer_info p_info = py::buffer(plaintext).request();
        py::buffer_info k_info = py::buffer(key).request();
        int len = static_cast<int>(p_info.size);
        
        py::bytearray result(std::string(len, '\0'));
        py::buffer_info r_info = py::buffer(result).request();

        int ret = symbolic_xor(
            static_cast<const uint8_t*>(p_info.ptr),
            static_cast<const uint8_t*>(k_info.ptr),
            len,
            static_cast<uint8_t*>(r_info.ptr)
        );

        if (ret != 0) throw std::runtime_error("symbolic_xor failed");
        return py::bytes(result);
    }, "Encrypt data using symbolic XOR");

    m.def("generate_entropy", [](int length) {
        py::bytearray buffer(std::string(length, '\0'));
        py::buffer_info info = py::buffer(buffer).request();
        int ret = generate_entropy(static_cast<uint8_t*>(info.ptr), length);
        if (ret != 0) throw std::runtime_error("generate_entropy failed");
        return py::bytes(buffer);
    }, "Generate quantum-inspired entropy");

    m.def("rft_basis_forward", [](py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                   py::object weights_obj, py::object theta0_obj, py::object omega_obj,
                                   double sigma0, double gamma, const std::string& sequence_type) {
        int N = static_cast<int>(x.shape(0));
        auto out_real = py::array_t<double>(N);
        auto out_imag = py::array_t<double>(N);

        const double* w_ptr = weights_obj.is_none() ? nullptr : py::array_t<double, py::array::c_style | py::array::forcecast>(weights_obj).data();
        int M = weights_obj.is_none() ? 0 : static_cast<int>(py::array_t<double>(weights_obj).shape(0));
        const double* th_ptr = theta0_obj.is_none() ? nullptr : py::array_t<double, py::array::c_style | py::array::forcecast>(theta0_obj).data();
        const double* om_ptr = omega_obj.is_none() ? nullptr : py::array_t<double, py::array::c_style | py::array::forcecast>(omega_obj).data();

        int ret = rft_basis_forward(x.data(), N, w_ptr, M, th_ptr, om_ptr, sigma0, gamma, sequence_type.c_str(), out_real.mutable_data(), out_imag.mutable_data());
        if (ret != 0) throw std::runtime_error("rft_basis_forward failed with code " + std::to_string(ret));
        return std::make_pair(out_real, out_imag);
    }, py::arg("x"), py::arg("weights"), py::arg("theta0"), py::arg("omega"), py::arg("sigma0"), py::arg("gamma"), py::arg("sequence_type"), "Spec-compliant RFT (basis path): forward transform");

    m.def("rft_basis_inverse", [](py::array_t<double, py::array::c_style | py::array::forcecast> X_real,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> X_imag,
                                   py::object weights_obj, py::object theta0_obj, py::object omega_obj,
                                   double sigma0, double gamma, const std::string& sequence_type) {
        int N = static_cast<int>(X_real.shape(0));
        auto out_x = py::array_t<double>(N);

        const double* w_ptr = weights_obj.is_none() ? nullptr : py::array_t<double, py::array::c_style | py::array::forcecast>(weights_obj).data();
        int M = weights_obj.is_none() ? 0 : static_cast<int>(py::array_t<double>(weights_obj).shape(0));
        const double* th_ptr = theta0_obj.is_none() ? nullptr : py::array_t<double, py::array::c_style | py::array::forcecast>(theta0_obj).data();
        const double* om_ptr = omega_obj.is_none() ? nullptr : py::array_t<double, py::array::c_style | py::array::forcecast>(omega_obj).data();

        int ret = rft_basis_inverse(X_real.data(), X_imag.data(), N, w_ptr, M, th_ptr, om_ptr, sigma0, gamma, sequence_type.c_str(), out_x.mutable_data());
        if (ret != 0) throw std::runtime_error("rft_basis_inverse failed with code " + std::to_string(ret));
        return out_x;
    }, py::arg("X_real"), py::arg("X_imag"), py::arg("weights"), py::arg("theta0"), py::arg("omega"), py::arg("sigma0"), py::arg("gamma"), py::arg("sequence_type"), "Spec-compliant RFT (basis path): inverse transform");
}
