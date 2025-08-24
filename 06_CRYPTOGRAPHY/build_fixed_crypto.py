#!/usr/bin/env python3
"""
Manual build for enhanced_rft_crypto_bindings with corrected C++ code
"""

import os
import subprocess
import sys


def build_enhanced_crypto():
    """Build the enhanced crypto bindings manually."""

    print("🔧 Building Enhanced RFT Crypto with Fixed Decryption...")

    # Create the pybind11 wrapper
    wrapper_code = """
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

// Include the fixed enhanced_rft_crypto.cpp directly
#include "core/cpp/cryptography/enhanced_rft_crypto.cpp"

namespace py = pybind11;

// Global crypto engine instance
static EnhancedRFTCrypto g_crypto_engine;

PYBIND11_MODULE(enhanced_rft_crypto_bindings, m) {
    m.doc() = "Enhanced RFT Crypto Engine with Fixed Decryption";
    
    m.def("init_engine", []() {
        // Engine is already initialized as global instance
        return 0;
    }, "Initialize the crypto engine");
    
    m.def("generate_key_material", [](const std::vector<uint8_t>& password, 
                                     const std::vector<uint8_t>& salt, 
                                     int key_length) {
        // Simple key derivation using SHA256
        std::vector<uint8_t> combined;
        combined.reserve(password.size() + salt.size() + 4);
        combined.insert(combined.end(), password.begin(), password.end());
        combined.insert(combined.end(), salt.begin(), salt.end());
        
        // Add key length as bytes
        combined.push_back((key_length >> 24) & 0xFF);
        combined.push_back((key_length >> 16) & 0xFF);
        combined.push_back((key_length >> 8) & 0xFF);
        combined.push_back(key_length & 0xFF);
        
        auto hash = SHA256::hash(combined);
        
        // Extend if needed
        std::vector<uint8_t> key;
        while (key.size() < static_cast<size_t>(key_length)) {
            key.insert(key.end(), hash.begin(), hash.end());
            if (key.size() < static_cast<size_t>(key_length)) {
                hash = SHA256::hash(hash); // rehash for more bytes
            }
        }
        key.resize(key_length);
        return key;
    }, "Generate key material from password and salt");
    
    m.def("encrypt_block", [](const std::vector<uint8_t>& plaintext, 
                             const std::vector<uint8_t>& key) {
        if (plaintext.size() != 16) {
            throw std::runtime_error("Plaintext must be exactly 16 bytes");
        }
        return g_crypto_engine.encrypt(plaintext, key);
    }, "Encrypt a 16-byte block");
    
    m.def("decrypt_block", [](const std::vector<uint8_t>& ciphertext, 
                             const std::vector<uint8_t>& key) {
        if (ciphertext.size() != 16) {
            throw std::runtime_error("Ciphertext must be exactly 16 bytes");
        }
        return g_crypto_engine.decrypt(ciphertext, key);
    }, "Decrypt a 16-byte block");
    
    m.def("avalanche_test", [](const std::vector<uint8_t>& data1, 
                              const std::vector<uint8_t>& data2) {
        if (data1.size() != data2.size()) {
            throw std::runtime_error("Data sizes must match for avalanche test");
        }
        
        int diff_bits = 0;
        for (size_t i = 0; i < data1.size(); ++i) {
            uint8_t xor_result = data1[i] ^ data2[i];
            // Count bits that are different
            while (xor_result) {
                diff_bits += xor_result & 1;
                xor_result >>= 1;
            }
        }
        
        return static_cast<double>(diff_bits) / (data1.size() * 8.0);
    }, "Test avalanche effect between two data blocks");
}
"""

    # Write wrapper to file
    with open("enhanced_rft_crypto_wrapper.cpp", "w") as f:
        f.write(wrapper_code)

    # Build command
    build_cmd = [sys.executable, "-m", "pybind11", "--includes"]

    try:
        # Get pybind11 includes
        result = subprocess.run(build_cmd, capture_output=True, text=True, check=True)
        includes = result.stdout.strip()

        # Compile command
        compile_cmd = [
            "cl",  # MSVC compiler
            "/O2",  # Optimize for speed
            "/std:c++17",  # C++17 standard
            "/EHsc",  # Exception handling
            includes,  # pybind11 includes
            f"/I{sys.prefix}\\include",  # Python includes
            '/DVERSION_INFO="dev"',
            "/c",  # Compile only
            "enhanced_rft_crypto_wrapper.cpp",
            "/Fo:enhanced_rft_crypto_wrapper.obj",
        ]

        print(f"🔨 Compiling: {' '.join(compile_cmd)}")
        subprocess.run(compile_cmd, check=True)

        # Link command
        link_cmd = [
            "link",
            "/DLL",  # Build DLL
            "/OUT:enhanced_rft_crypto_bindings.cp312-win_amd64.pyd",
            "enhanced_rft_crypto_wrapper.obj",
            f"{sys.prefix}\\libs\\python312.lib",
        ]

        print(f"🔗 Linking: {' '.join(link_cmd)}")
        subprocess.run(link_cmd, check=True)

        print("✅ Build successful!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ MSVC compiler not found. Using fallback method...")
        return build_with_setuptools()


def build_with_setuptools():
    """Fallback: build using setuptools directly."""

    print("🔄 Trying setuptools build method...")

    setup_code = """
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension

ext_modules = [
    Pybind11Extension(
        "enhanced_rft_crypto_bindings",
        [
            "enhanced_rft_crypto_wrapper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
"""

    with open("temp_setup.py", "w") as f:
        f.write(setup_code)

    try:
        subprocess.run(
            [sys.executable, "temp_setup.py", "build_ext", "--inplace"], check=True
        )
        os.remove("temp_setup.py")
        print("✅ Setuptools build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Setuptools build failed: {e}")
        return False


if __name__ == "__main__":
    success = build_enhanced_crypto()
    if success:
        print("🎉 Enhanced RFT Crypto with Fixed Decryption built successfully!")
        print("   You can now test with: python fix_decryption_debug.py")
    else:
        print("💥 Build failed. Check the error messages above.")
        sys.exit(1)
