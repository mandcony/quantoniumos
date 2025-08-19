#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "enhanced_rft_crypto.cpp"

namespace py = pybind11;

// Python wrapper class for enhanced performance
class PyEnhancedRFTCrypto {
private:
    EnhancedRFTCrypto crypto;

public:
    py::bytes encrypt(py::bytes plaintext_bytes, py::bytes key_bytes) {
        std::string plaintext_str = plaintext_bytes;
        std::string key_str = key_bytes;
        
        std::vector<uint8_t> plaintext(plaintext_str.begin(), plaintext_str.end());
        std::vector<uint8_t> key(key_str.begin(), key_str.end());
        
        std::vector<uint8_t> result = crypto.encrypt(plaintext, key);
        return py::bytes(std::string(result.begin(), result.end()));
    }

    py::bytes decrypt(py::bytes ciphertext_bytes, py::bytes key_bytes) {
        std::string ciphertext_str = ciphertext_bytes;
        std::string key_str = key_bytes;
        
        std::vector<uint8_t> ciphertext(ciphertext_str.begin(), ciphertext_str.end());
        std::vector<uint8_t> key(key_str.begin(), key_str.end());
        
        std::vector<uint8_t> result = crypto.decrypt(ciphertext, key);
        return py::bytes(std::string(result.begin(), result.end()));
    }

    std::vector<py::bytes> encrypt_batch(const std::vector<py::bytes>& plaintexts, py::bytes key_bytes) {
        std::string key_str = key_bytes;
        std::vector<uint8_t> key(key_str.begin(), key_str.end());
        
        std::vector<std::vector<uint8_t>> plaintext_vectors;
        for (const auto& pt : plaintexts) {
            std::string pt_str = pt;
            plaintext_vectors.emplace_back(pt_str.begin(), pt_str.end());
        }
        
        auto results = crypto.encrypt_batch(plaintext_vectors, key);
        
        std::vector<py::bytes> py_results;
        for (const auto& result : results) {
            py_results.push_back(py::bytes(std::string(result.begin(), result.end())));
        }
        
        return py_results;
    }
};

PYBIND11_MODULE(enhanced_rft_crypto, m) {
    m.doc() = "Enhanced RFT Feistel Cipher - Optimized C++ Implementation";
    
    py::class_<PyEnhancedRFTCrypto>(m, "PyEnhancedRFTCrypto")
        .def(py::init<>())
        .def("encrypt", &PyEnhancedRFTCrypto::encrypt, "Encrypt data using enhanced RFT")
        .def("decrypt", &PyEnhancedRFTCrypto::decrypt, "Decrypt data using enhanced RFT")
        .def("encrypt_batch", &PyEnhancedRFTCrypto::encrypt_batch, "Batch encrypt multiple plaintexts");
}
