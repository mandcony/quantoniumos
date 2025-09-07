/**
 * Python bindings for Enhanced RFT Crypto v2
 * Using pybind11 for high-performance integration
 * 
 * Exposes the 48-round Feistel cipher to Python with minimal overhead
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "feistel_48.h"

namespace py = pybind11;

class PyFeistelCipher {
private:
    feistel_ctx_t ctx;
    bool initialized;

public:
    PyFeistelCipher() : initialized(false) {}
    
    ~PyFeistelCipher() {
        if (initialized) {
            feistel_cleanup(&ctx);
        }
    }
    
    bool init(py::bytes master_key, uint32_t flags = 0) {
        std::string key_str = master_key;
        
        if (feistel_init(&ctx, (const uint8_t*)key_str.data(), 
                        key_str.size(), flags) == FEISTEL_SUCCESS) {
            initialized = true;
            return true;
        }
        return false;
    }
    
    py::bytes encrypt_block(py::bytes plaintext) {
        if (!initialized) {
            throw std::runtime_error("Cipher not initialized");
        }
        
        std::string pt_str = plaintext;
        if (pt_str.size() != FEISTEL_BLOCK_SIZE) {
            throw std::invalid_argument("Block must be exactly 16 bytes");
        }
        
        std::string ciphertext(FEISTEL_BLOCK_SIZE, 0);
        
        if (feistel_encrypt_block(&ctx, 
                                 (const uint8_t*)pt_str.data(),
                                 (uint8_t*)ciphertext.data()) != FEISTEL_SUCCESS) {
            throw std::runtime_error("Encryption failed");
        }
        
        return py::bytes(ciphertext);
    }
    
    py::bytes decrypt_block(py::bytes ciphertext) {
        if (!initialized) {
            throw std::runtime_error("Cipher not initialized");
        }
        
        std::string ct_str = ciphertext;
        if (ct_str.size() != FEISTEL_BLOCK_SIZE) {
            throw std::invalid_argument("Block must be exactly 16 bytes");
        }
        
        std::string plaintext(FEISTEL_BLOCK_SIZE, 0);
        
        if (feistel_decrypt_block(&ctx,
                                 (const uint8_t*)ct_str.data(),
                                 (uint8_t*)plaintext.data()) != FEISTEL_SUCCESS) {
            throw std::runtime_error("Decryption failed");
        }
        
        return py::bytes(plaintext);
    }
    
    py::dict benchmark(size_t test_size = 1048576) {  // 1MB default
        if (!initialized) {
            throw std::runtime_error("Cipher not initialized");
        }
        
        feistel_metrics_t metrics;
        if (feistel_benchmark(&ctx, test_size, &metrics) != FEISTEL_SUCCESS) {
            throw std::runtime_error("Benchmark failed");
        }
        
        py::dict result;
        result["throughput_mbps"] = metrics.throughput_mbps;
        result["total_bytes"] = metrics.total_bytes_processed;
        result["total_time_ns"] = metrics.total_time_ns;
        
        return result;
    }
    
    py::dict avalanche_test() {
        if (!initialized) {
            throw std::runtime_error("Cipher not initialized");
        }
        
        feistel_metrics_t metrics;
        if (feistel_avalanche_test(&ctx, &metrics) != FEISTEL_SUCCESS) {
            throw std::runtime_error("Avalanche test failed");
        }
        
        py::dict result;
        result["message_avalanche"] = metrics.message_avalanche;
        result["key_avalanche"] = metrics.key_avalanche;
        result["key_sensitivity"] = metrics.key_sensitivity;
        
        return result;
    }
    
    py::bytes encrypt_bytes(py::bytes data) {
        if (!initialized) {
            throw std::runtime_error("Cipher not initialized");
        }
        
        std::string data_str = data;
        size_t original_size = data_str.size();
        
        // Pad to block boundary
        size_t padded_size = ((original_size + FEISTEL_BLOCK_SIZE - 1) / FEISTEL_BLOCK_SIZE) * FEISTEL_BLOCK_SIZE;
        data_str.resize(padded_size, 0);
        
        std::string ciphertext(padded_size, 0);
        
        // Encrypt block by block
        for (size_t i = 0; i < padded_size; i += FEISTEL_BLOCK_SIZE) {
            if (feistel_encrypt_block(&ctx,
                                     (const uint8_t*)data_str.data() + i,
                                     (uint8_t*)ciphertext.data() + i) != FEISTEL_SUCCESS) {
                throw std::runtime_error("Encryption failed");
            }
        }
        
        return py::bytes(ciphertext);
    }
    
    py::bytes decrypt_bytes(py::bytes data) {
        if (!initialized) {
            throw std::runtime_error("Cipher not initialized");
        }
        
        std::string data_str = data;
        if (data_str.size() % FEISTEL_BLOCK_SIZE != 0) {
            throw std::invalid_argument("Data size must be multiple of block size");
        }
        
        std::string plaintext(data_str.size(), 0);
        
        // Decrypt block by block
        for (size_t i = 0; i < data_str.size(); i += FEISTEL_BLOCK_SIZE) {
            if (feistel_decrypt_block(&ctx,
                                     (const uint8_t*)data_str.data() + i,
                                     (uint8_t*)plaintext.data() + i) != FEISTEL_SUCCESS) {
                throw std::runtime_error("Decryption failed");
            }
        }
        
        return py::bytes(plaintext);
    }
    
    static py::dict get_cpu_capabilities() {
        py::dict caps;
        caps["avx2"] = feistel_has_avx2();
        caps["aes_ni"] = feistel_has_aes_ni();
        caps["cpu_info"] = feistel_get_cpu_info();
        return caps;
    }
    
    static bool self_test() {
        return feistel_self_test() == FEISTEL_SUCCESS;
    }
};

// High-level convenience functions matching your original Python API
py::dict validate_paper_metrics() {
    // Return the exact metrics from your paper for validation
    py::dict metrics;
    metrics["message_avalanche"] = 0.438;
    metrics["key_avalanche"] = 0.527;
    metrics["throughput_mbps"] = 9.2;
    return metrics;
}

py::bytes enhanced_rft_encrypt(py::bytes data, py::bytes key) {
    PyFeistelCipher cipher;
    if (!cipher.init(key)) {
        throw std::runtime_error("Failed to initialize cipher");
    }
    return cipher.encrypt_bytes(data);
}

py::bytes enhanced_rft_decrypt(py::bytes data, py::bytes key) {
    PyFeistelCipher cipher;
    if (!cipher.init(key)) {
        throw std::runtime_error("Failed to initialize cipher");
    }
    return cipher.decrypt_bytes(data);
}

PYBIND11_MODULE(feistel_crypto, m) {
    m.doc() = "Enhanced RFT Crypto v2 - 48-Round Feistel Cipher";
    
    // Main cipher class
    py::class_<PyFeistelCipher>(m, "FeistelCipher")
        .def(py::init<>())
        .def("init", &PyFeistelCipher::init, 
             "Initialize cipher with master key",
             py::arg("master_key"), py::arg("flags") = 0)
        .def("encrypt_block", &PyFeistelCipher::encrypt_block,
             "Encrypt a single 16-byte block")
        .def("decrypt_block", &PyFeistelCipher::decrypt_block,
             "Decrypt a single 16-byte block")
        .def("encrypt_bytes", &PyFeistelCipher::encrypt_bytes,
             "Encrypt arbitrary length data")
        .def("decrypt_bytes", &PyFeistelCipher::decrypt_bytes,
             "Decrypt arbitrary length data")
        .def("benchmark", &PyFeistelCipher::benchmark,
             "Run performance benchmark",
             py::arg("test_size") = 1048576)
        .def("avalanche_test", &PyFeistelCipher::avalanche_test,
             "Run avalanche effect test")
        .def_static("get_cpu_capabilities", &PyFeistelCipher::get_cpu_capabilities,
                   "Get CPU capabilities for optimization")
        .def_static("self_test", &PyFeistelCipher::self_test,
                   "Run internal self-tests");
    
    // Convenience functions
    m.def("enhanced_rft_encrypt", &enhanced_rft_encrypt,
          "High-level encryption function");
    m.def("enhanced_rft_decrypt", &enhanced_rft_decrypt,
          "High-level decryption function");
    m.def("validate_paper_metrics", &validate_paper_metrics,
          "Get expected metrics from QuantoniumOS paper");
    
    // Constants
    m.attr("BLOCK_SIZE") = FEISTEL_BLOCK_SIZE;
    m.attr("KEY_SIZE") = FEISTEL_KEY_SIZE;
    m.attr("ROUNDS") = FEISTEL_48_ROUNDS;
    m.attr("TARGET_THROUGHPUT") = 9.2;  // MB/s from paper
    
    // Flags
    m.attr("FLAG_USE_AVX2") = FEISTEL_FLAG_USE_AVX2;
    m.attr("FLAG_USE_AES_NI") = FEISTEL_FLAG_USE_AES_NI;
    m.attr("FLAG_PARALLEL") = FEISTEL_FLAG_PARALLEL;
}
