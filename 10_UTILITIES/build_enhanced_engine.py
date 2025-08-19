#!/usr/bin/env python3
"""
Build script for enhanced RFT C++ engine
"""

import os
import sys
import subprocess
import platform from pathlib
import Path
def find_pybind11(): """
        Find pybind11 installation
"""

        try:
import pybind11
        return pybind11.get_cmake_dir()
        except ImportError:
        print("pybind11 not found. Installing...") subprocess.run([sys.executable, "-m", "pip", "install", "pybind11"], check=True)
import pybind11
        return pybind11.get_cmake_dir()
def build_enhanced_engine(): """
        Build the enhanced RFT C++ engine
"""

        print("Building Enhanced RFT C++ Engine...")

        # Get current directory current_dir = Path(__file__).parent core_dir = current_dir / "core"

        # Find pybind11 pybind11_dir = find_pybind11()
        print(f"Found pybind11 at: {pybind11_dir}")

        # Source files sources = [ str(core_dir / "enhanced_rft_crypto.cpp"), str(core_dir / "enhanced_rft_crypto_bindings.cpp") ]

        # Include directories
import pybind11 includes = [ str(core_dir / "include"), pybind11.get_include(), pybind11.get_include(user=True) ]

        # Output file
        if platform.system() == "Windows": output_file = "enhanced_rft_crypto_bindings.pyd" extra_flags = ["/std:c++17"]
        else: output_file = "enhanced_rft_crypto_bindings.so" extra_flags = ["-std=c++17"]

        # Build command - Use g++ for all platforms cmd = ["g++"] + extra_flags + ["-shared", "-fPIC", "-O3"] + \ [f"-I{inc}"
        for inc in includes] + \ sources + ["-o", output_file]
        print(f"Build command: {' '.join(cmd)}")
        try: result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        return True except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
def build_with_cmake(): """
        Alternative build using CMake
"""

        print("Trying CMake build...")

        # Create CMakeLists.txt cmake_content = ''' cmake_minimum_required(VERSION 3.12) project(enhanced_rft_crypto) set(CMAKE_CXX_STANDARD 17) find_package(pybind11 REQUIRED) pybind11_add_module(enhanced_rft_crypto_bindings core/enhanced_rft_crypto.cpp core/enhanced_rft_crypto_bindings.cpp ) target_include_directories(enhanced_rft_crypto_bindings PRIVATE core/include) target_compile_definitions(enhanced_rft_crypto_bindings PRIVATE VERSION_INFO="1.0.0") ''' with open("CMakeLists.txt", "w") as f: f.write(cmake_content)

        # Create build directory os.makedirs("build_enhanced", exist_ok=True)
        try:

        # Configure subprocess.run([ "cmake", "-S", ".", "-B", "build_enhanced", "-DCMAKE_BUILD_TYPE=Release" ], check=True)

        # Build subprocess.run([ "cmake", "--build", "build_enhanced", "--config", "Release" ], check=True)

        # Copy the built module
import shutil
import glob

        # Find the built module built_files = glob.glob("build_enhanced/**/*enhanced_rft_crypto_bindings*", recursive=True)
        if built_files: shutil.copy2(built_files[0], ".")
        print("CMake build successful!")
        return True
        else:
        print("CMake build completed but module not found")
        return False except subprocess.CalledProcessError as e:
        print(f"CMake build failed: {e}")
        return False
def build_with_setup_py(): """
        Build using setup.py
"""

        print("Trying setup.py build...") setup_content = ''' from setuptools
import setup, Extension from pybind11.setup_helpers
import Pybind11Extension, build_ext from pybind11
import get_cmake_dir
import pybind11 ext_modules = [ Pybind11Extension( "enhanced_rft_crypto_bindings", [ "core/enhanced_rft_crypto.cpp", "core/enhanced_rft_crypto_bindings.cpp", ], include_dirs=[ "core/include", pybind11.get_include(), ], cxx_std=17, ), ] setup( name="enhanced_rft_crypto", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False, ) ''' with open("setup_enhanced.py", "w") as f: f.write(setup_content)
        try: result = subprocess.run([ sys.executable, "setup_enhanced.py", "build_ext", "--inplace" ], check=True, capture_output=True, text=True)
        print("setup.py build successful!")
        return True except subprocess.CalledProcessError as e:
        print(f"setup.py build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
def main():
        print("Enhanced RFT C++ Engine Build System")
        print("=" * 40)

        # Try different build methods methods = [ ("Direct compilation", build_enhanced_engine), ("setup.py", build_with_setup_py), ("CMake", build_with_cmake), ] for method_name, method_func in methods:
        print(f"\\nTrying {method_name}...")
        if method_func():
        print(f"{method_name} succeeded!") break
        else:
        print(f"{method_name} failed.")
        else:
        print("\\nAll build methods failed. Please check dependencies:")
        print("- C++ compiler (g++, clang++, or MSVC)")
        print("- pybind11 (pip install pybind11)")
        print("- CMake (optional)")
        return False

        # Test the built module
        print("\\nTesting built module...")
        try:
import enhanced_rft_crypto_bindings enhanced_rft_crypto_bindings.init_engine()
        print("Module
import and initialization successful!")
        return True except ImportError as e:
        print(f"Module
import failed: {e}")
        return False

if __name__ == "__main__": success = main() sys.exit(0
if success else 1)