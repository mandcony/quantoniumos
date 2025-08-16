#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>

// Include our enhanced RFT crypto implementation
#include "enhanced_rft_crypto.cpp"

namespace py = pybind11;

// High-performance Python wrapper
class PyEnhancedRFTCrypto {
private:
    EnhancedRFTCrypto engine;
    
public:
    PyEnhancedRFTCrypto() = default;
    
    py::bytes encrypt(py::bytes plaintext, py::bytes key) {
        std::string pt_str = plaintext;
        std::string key_str = key;
        
        std::vector<uint8_t> pt_vec(pt_str.begin(), pt_str.end());
        std::vector<uint8_t> key_vec(key_str.begin(), key_str.end());
        
        // Ensure even length for Feistel
        if (pt_vec.size() % 2 != 0) {
            pt_vec.push_back(0);
        }
        
        auto result = engine.encrypt(pt_vec, key_vec);
        return py::bytes(std::string(result.begin(), result.end()));
    }
    
    py::bytes decrypt(py::bytes ciphertext, py::bytes key) {
        std::string ct_str = ciphertext;
        std::string key_str = key;
        
        std::vector<uint8_t> ct_vec(ct_str.begin(), ct_str.end());
        std::vector<uint8_t> key_vec(key_str.begin(), key_str.end());
        
        auto result = engine.decrypt(ct_vec, key_vec);
        return py::bytes(std::string(result.begin(), result.end()));
    }
    
    // Batch encryption for better performance
    std::vector<py::bytes> encrypt_batch(const std::vector<py::bytes>& plaintexts, py::bytes key) {
        std::string key_str = key;
        std::vector<uint8_t> key_vec(key_str.begin(), key_str.end());
        
        std::vector<py::bytes> results;
        results.reserve(plaintexts.size());
        
        for (const auto& pt : plaintexts) {
            std::string pt_str = pt;
            std::vector<uint8_t> pt_vec(pt_str.begin(), pt_str.end());
            
            if (pt_vec.size() % 2 != 0) {
                pt_vec.push_back(0);
            }
            
            auto result = engine.encrypt(pt_vec, key_vec);
            results.emplace_back(std::string(result.begin(), result.end()));
        }
        
        return results;
    }
};

PYBIND11_MODULE(enhanced_rft_crypto, m) {
    m.doc() = "Enhanced RFT Crypto Engine with optimized C++ implementation";
    
    py::class_<PyEnhancedRFTCrypto>(m, "EnhancedRFTCrypto")
        .def(py::init<>())
        .def("encrypt", &PyEnhancedRFTCrypto::encrypt, "Encrypt data with RFT Feistel cipher")
        .def("decrypt", &PyEnhancedRFTCrypto::decrypt, "Decrypt data with RFT Feistel cipher")
        .def("encrypt_batch", &PyEnhancedRFTCrypto::encrypt_batch, "Batch encrypt for performance");
}
