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

        // Pad to even length if needed (Feistel requirement)
        if (plaintext.size() % 2 != 0) {
            plaintext.push_back(0x00);
        }

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

    // Batch processing for better throughput
    std::vector<py::bytes> encrypt_batch(const std::vector<py::bytes>& plaintexts, py::bytes key_bytes) {
        std::string key_str = key_bytes;
        std::vector<uint8_t> key(key_str.begin(), key_str.end());
        std::vector<py::bytes> results;
        results.reserve(plaintexts.size());

        for (const auto& pt_bytes : plaintexts) {
            std::string pt_str = pt_bytes;
            std::vector<uint8_t> plaintext(pt_str.begin(), pt_str.end());

            // Pad to even length if needed
            if (plaintext.size() % 2 != 0) {
                plaintext.push_back(0x00);
            }
            std::vector<uint8_t> result = crypto.encrypt(plaintext, key);
            results.push_back(py::bytes(std::string(result.begin(), result.end())));
        }
        return results;
    }
};

PYBIND11_MODULE(enhanced_rft_crypto_bindings, m) {
    m.doc() = "Enhanced RFT Feistel Crypto - High-performance C++ implementation";

    py::class_<PyEnhancedRFTCrypto>(m, "PyEnhancedRFTCrypto")
        .def(py::init<>())
        .def("encrypt", &PyEnhancedRFTCrypto::encrypt, "Encrypt data using Enhanced RFT Feistel cipher")
        .def("decrypt", &PyEnhancedRFTCrypto::decrypt, "Decrypt data using Enhanced RFT Feistel cipher")
        .def("encrypt_batch", &PyEnhancedRFTCrypto::encrypt_batch, "Batch encrypt multiple plaintexts");
} 