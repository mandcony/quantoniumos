#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "engine_core.h"
#include "resonance_fourier_engine.h"

namespace py = pybind11;

// Helper: Convert RFTResult to Python dict
py::dict rft_result_to_dict(const RFTResult* result) {
	if (!result) return py::dict();
	py::list bins;
	for (int i = 0; i < result->bin_count; ++i) bins.append(result->bins[i]);
	py::dict d;
	d["bin_count"] = result->bin_count;
	d["bins"] = bins;
	d["hr"] = result->hr;
	return d;
}

// Helper: Convert SAVector to Python list
py::list sa_vector_to_list(const SAVector* vec) {
	if (!vec) return py::list();
	py::list values;
	for (int i = 0; i < vec->count; ++i) values.append(vec->values[i]);
	return values;
}

PYBIND11_MODULE(resonance_engine, m) {
	m.doc() = "Production-grade Resonance Engine Python bindings";
	m.def("engine_init", &engine_init, "Initialize the engine");
	m.def("engine_final", &engine_final, "Finalize the engine");
	m.def("rft_run", [](py::bytes data) {
		std::string s = data;
		RFTResult* result = rft_run(s.c_str(), s.length());
		py::dict res = rft_result_to_dict(result);
		rft_free(result);
		return res;
	}, "Run Resonance Fourier Transform on input data");
	m.def("sa_compute", [](py::bytes data) {
		std::string s = data;
		SAVector* vec = sa_compute(s.c_str(), s.length());
		py::list res = sa_vector_to_list(vec);
		sa_free(vec);
		return res;
	}, "Compute Symbolic Alignment vector");
	m.def("wave_hash", [](py::bytes data) {
		std::string s = data;
		return std::string(wave_hash(s.c_str(), s.length()));
	}, "Compute waveform hash");
	m.def("symbolic_xor", [](py::bytes plaintext, py::bytes key) {
		std::string pt = plaintext, k = key;
		if (pt.size() != k.size()) throw std::runtime_error("Length mismatch");
		std::vector<uint8_t> result(pt.size());
		int status = symbolic_xor(reinterpret_cast<const uint8_t*>(pt.data()), reinterpret_cast<const uint8_t*>(k.data()), pt.size(), result.data());
		if (status != 0) throw std::runtime_error("symbolic_xor failed");
		return py::bytes(reinterpret_cast<const char*>(result.data()), result.size());
	}, "Symbolic XOR encryption");
	m.def("generate_entropy", [](int length) {
		std::vector<uint8_t> buffer(length);
		int status = generate_entropy(buffer.data(), length);
		if (status != 0) throw std::runtime_error("generate_entropy failed");
		return py::bytes(reinterpret_cast<const char*>(buffer.data()), length);
	}, "Generate quantum-inspired entropy");
	m.def("rft_basis_forward", [](py::array_t<double> x, py::array_t<double> weights, py::array_t<double> theta0, py::array_t<double> omega, double sigma0, double gamma, std::string seq) {
		auto buf_x = x.request();
		int N = buf_x.shape[0];
		auto buf_w = weights.request();
		int M = buf_w.shape[0];
		std::vector<double> out_real(N), out_imag(N);
		int status = rft_basis_forward(static_cast<const double*>(buf_x.ptr), N, static_cast<const double*>(buf_w.ptr), M, static_cast<const double*>(theta0.request().ptr), static_cast<const double*>(omega.request().ptr), sigma0, gamma, seq.c_str(), out_real.data(), out_imag.data());
		if (status != 0) throw std::runtime_error("rft_basis_forward failed");
		return py::make_tuple(py::array_t<double>(N, out_real.data()), py::array_t<double>(N, out_imag.data()));
	}, "Forward RFT basis transform");
	m.def("rft_basis_inverse", [](py::array_t<double> X_real, py::array_t<double> X_imag, py::array_t<double> weights, py::array_t<double> theta0, py::array_t<double> omega, double sigma0, double gamma, std::string seq) {
		int N = X_real.request().shape[0];
		int M = weights.request().shape[0];
		std::vector<double> out_x(N);
		int status = rft_basis_inverse(static_cast<const double*>(X_real.request().ptr), static_cast<const double*>(X_imag.request().ptr), N, static_cast<const double*>(weights.request().ptr), M, static_cast<const double*>(theta0.request().ptr), static_cast<const double*>(omega.request().ptr), sigma0, gamma, seq.c_str(), out_x.data());
		if (status != 0) throw std::runtime_error("rft_basis_inverse failed");
		return py::array_t<double>(N, out_x.data());
	}, "Inverse RFT basis transform");
	m.def("rft_operator_apply", [](py::array_t<double> x_real, py::array_t<double> x_imag, py::array_t<double> weights, py::array_t<double> theta0, py::array_t<double> omega, double sigma0, double gamma, std::string seq) {
		int N = x_real.request().shape[0];
		int M = weights.request().shape[0];
		std::vector<double> out_real(N), out_imag(N);
		int status = rft_operator_apply(static_cast<const double*>(x_real.request().ptr), static_cast<const double*>(x_imag.request().ptr), N, static_cast<const double*>(weights.request().ptr), M, static_cast<const double*>(theta0.request().ptr), static_cast<const double*>(omega.request().ptr), sigma0, gamma, seq.c_str(), out_real.data(), out_imag.data());
		if (status != 0) throw std::runtime_error("rft_operator_apply failed");
		return py::make_tuple(py::array_t<double>(N, out_real.data()), py::array_t<double>(N, out_imag.data()));
	}, "Apply resonance operator");
	m.def("forward_rft_run", [](py::array_t<double> real, py::array_t<double> imag) {
		int N = real.request().shape[0];
		std::vector<double> r(real.size()), i(imag.size());
		std::memcpy(r.data(), real.request().ptr, N * sizeof(double));
		std::memcpy(i.data(), imag.request().ptr, N * sizeof(double));
		forward_rft_run(r.data(), i.data(), N);
		return py::make_tuple(py::array_t<double>(N, r.data()), py::array_t<double>(N, i.data()));
	}, "Forward RFT with coupling matrix");
	m.def("inverse_rft_run", [](py::array_t<double> real, py::array_t<double> imag) {
		int N = real.request().shape[0];
		std::vector<double> r(real.size()), i(imag.size());
		std::memcpy(r.data(), real.request().ptr, N * sizeof(double));
		std::memcpy(i.data(), imag.request().ptr, N * sizeof(double));
		inverse_rft_run(r.data(), i.data(), N);
		return py::make_tuple(py::array_t<double>(N, r.data()), py::array_t<double>(N, i.data()));
	}, "Inverse RFT with coupling matrix");
	m.def("forward_rft_with_coupling", [](py::array_t<double> real, py::array_t<double> imag, double alpha) {
		int N = real.request().shape[0];
		std::vector<double> r(real.size()), i(imag.size());
		std::memcpy(r.data(), real.request().ptr, N * sizeof(double));
		std::memcpy(i.data(), imag.request().ptr, N * sizeof(double));
		forward_rft_with_coupling(r.data(), i.data(), N, alpha);
		return py::make_tuple(py::array_t<double>(N, r.data()), py::array_t<double>(N, i.data()));
	}, "Forward RFT with configurable coupling");
	m.def("inverse_rft_with_coupling", [](py::array_t<double> real, py::array_t<double> imag, double alpha) {
		int N = real.request().shape[0];
		std::vector<double> r(real.size()), i(imag.size());
		std::memcpy(r.data(), real.request().ptr, N * sizeof(double));
		std::memcpy(i.data(), imag.request().ptr, N * sizeof(double));
		inverse_rft_with_coupling(r.data(), i.data(), N, alpha);
		return py::make_tuple(py::array_t<double>(N, r.data()), py::array_t<double>(N, i.data()));
	}, "Inverse RFT with configurable coupling");
	m.def("rft_fingerprint_goertzel", [](py::bytes data) {
		std::string s = data;
		RFTResult* result = rft_fingerprint_goertzel(s.c_str(), s.length());
		py::dict res = rft_result_to_dict(result);
		rft_free(result);
		return res;
	}, "Goertzel fingerprint (deprecated)");
	
	// ResonanceFourierEngine class binding
	py::class_<ResonanceFourierEngine>(m, "ResonanceFourierEngine")
		.def(py::init<>(), "Initialize ResonanceFourierEngine")
		.def("forward_true_rft", &ResonanceFourierEngine::forward_true_rft, 
			 "Forward True RFT transformation",
			 py::arg("input_data"))
		.def("inverse_true_rft", &ResonanceFourierEngine::inverse_true_rft,
			 "Inverse True RFT transformation", 
			 py::arg("spectrum_data"))
		.def("validate_roundtrip_accuracy", &ResonanceFourierEngine::validate_roundtrip_accuracy,
			 "Validate roundtrip accuracy",
			 py::arg("original"), py::arg("tolerance") = 1e-10)
		.def("get_quantum_amplitudes", &ResonanceFourierEngine::get_quantum_amplitudes,
			 "Get quantum amplitudes from cached frequencies")
		.def("status", &ResonanceFourierEngine::status,
			 "Get engine status");
}
