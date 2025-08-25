#!/usr/bin/env python3
"""
Build and install the symbiotic RFT bridge.
This connects the Python implementation with C++ for optimal performance with energy conservation.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_symbiotic_rft():
    """Build the symbiotic RFT bridge."""
    print("Building symbiotic RFT bridge...")

    # Ensure the target directory exists
    target_dir = Path("build_symbiotic")
    target_dir.mkdir(exist_ok=True)

    # Create the C++ implementation file
    cpp_file = target_dir / "symbiotic_rft_engine.cpp"

    with open(cpp_file, "w") as f:
        f.write(
            """
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <vector>
#include <complex>
#include <iostream>
#include <cmath>

namespace py = pybind11;
using namespace pybind11::literals;

// Ensure energy conservation with proper normalization
class SymbioticRFTEngine {
private:
    int dimension;
    std::vector<std::vector<std::complex<double>>> basis;
    bool has_basis;
    
public:
    SymbioticRFTEngine(int dim) : dimension(dim), has_basis(false) {
        basis.resize(dim, std::vector<std::complex<double>>(dim));
    }
    
    void set_basis(const py::array_t<std::complex<double>>& py_basis) {
        // Copy basis from Python to C++
        auto buf = py_basis.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Basis must be a 2D array");
        }
        
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        
        if (rows != dimension || cols != dimension) {
            throw std::runtime_error("Basis dimensions don't match");
        }
        
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                basis[i][j] = ptr[i * cols + j];
            }
        }
        
        has_basis = true;
    }
    
    py::array_t<std::complex<double>> forward_rft(const py::array_t<std::complex<double>>& signal) {
        if (!has_basis) {
            throw std::runtime_error("Basis not set");
        }
        
        auto buf = signal.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Signal must be a 1D array");
        }
        
        int signal_size = buf.shape[0];
        if (signal_size != dimension) {
            throw std::runtime_error("Signal size doesn't match dimension");
        }
        
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        
        // Compute forward transform: spectrum = basis† * signal
        std::vector<std::complex<double>> spectrum(dimension);
        for (int i = 0; i < dimension; i++) {
            spectrum[i] = 0;
            for (int j = 0; j < dimension; j++) {
                spectrum[i] += std::conj(basis[j][i]) * ptr[j];
            }
        }
        
        // Check energy conservation
        double input_energy = 0.0;
        for (int i = 0; i < dimension; i++) {
            input_energy += std::norm(ptr[i]);
        }
        
        double output_energy = 0.0;
        for (int i = 0; i < dimension; i++) {
            output_energy += std::norm(spectrum[i]);
        }
        
        double energy_ratio = input_energy > 0 ? output_energy / input_energy : 1.0;
        if (std::abs(energy_ratio - 1.0) > 0.01) {
            std::cerr << "⚠️ Energy not conserved in C++ forward RFT: ratio=" 
                      << energy_ratio << std::endl;
        }
        
        // Create and return result array
        py::array_t<std::complex<double>> result(dimension);
        auto result_buf = result.request();
        auto result_ptr = static_cast<std::complex<double>*>(result_buf.ptr);
        
        for (int i = 0; i < dimension; i++) {
            result_ptr[i] = spectrum[i];
        }
        
        return result;
    }
    
    py::array_t<std::complex<double>> inverse_rft(const py::array_t<std::complex<double>>& spectrum) {
        if (!has_basis) {
            throw std::runtime_error("Basis not set");
        }
        
        auto buf = spectrum.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Spectrum must be a 1D array");
        }
        
        int spectrum_size = buf.shape[0];
        if (spectrum_size != dimension) {
            throw std::runtime_error("Spectrum size doesn't match dimension");
        }
        
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        
        // Compute inverse transform: signal = basis * spectrum
        std::vector<std::complex<double>> signal(dimension);
        for (int i = 0; i < dimension; i++) {
            signal[i] = 0;
            for (int j = 0; j < dimension; j++) {
                signal[i] += basis[i][j] * ptr[j];
            }
        }
        
        // Check energy conservation
        double input_energy = 0.0;
        for (int i = 0; i < dimension; i++) {
            input_energy += std::norm(ptr[i]);
        }
        
        double output_energy = 0.0;
        for (int i = 0; i < dimension; i++) {
            output_energy += std::norm(signal[i]);
        }
        
        double energy_ratio = input_energy > 0 ? output_energy / input_energy : 1.0;
        if (std::abs(energy_ratio - 1.0) > 0.01) {
            std::cerr << "⚠️ Energy not conserved in C++ inverse RFT: ratio=" 
                      << energy_ratio << std::endl;
        }
        
        // Create and return result array
        py::array_t<std::complex<double>> result(dimension);
        auto result_buf = result.request();
        auto result_ptr = static_cast<std::complex<double>*>(result_buf.ptr);
        
        for (int i = 0; i < dimension; i++) {
            result_ptr[i] = signal[i];
        }
        
        return result;
    }
};

PYBIND11_MODULE(symbiotic_rft_engine, m) {
    m.doc() = "Symbiotic RFT Engine with energy conservation";
    
    py::class_<SymbioticRFTEngine>(m, "SymbioticRFTEngine")
        .def(py::init<int>())
        .def("set_basis", &SymbioticRFTEngine::set_basis)
        .def("forward_rft", &SymbioticRFTEngine::forward_rft)
        .def("inverse_rft", &SymbioticRFTEngine::inverse_rft);
}
"""
        )

    # Create the Python wrapper
    py_file = Path("symbiotic_rft_wrapper.py")

    with open(py_file, "w") as f:
        f.write(
            """#!/usr/bin/env python3
\"\"\"
Symbiotic RFT Wrapper - Ensures Python and C++ implementations work together.
This adapter ensures energy conservation in all RFT transformations.
\"\"\"

import numpy as np
import importlib.util
import os
import sys
from pathlib import Path

# Try to import the C++ engine
try:
    import symbiotic_rft_engine
    cpp_available = True
    print("Successfully imported Symbiotic RFT Engine")
except ImportError:
    cpp_available = False
    print("Warning: Symbiotic RFT Engine not available. Using Python fallback.")

# Import canonical True RFT implementation
try:
    import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
from canonical_true_rft import get_rft_basis as py_get_rft_basis
except ImportError:
    # Try relative import
    sys.path.append(str(Path(__file__).parent))
    try:
        import sys, os
from canonical_true_rft import get_rft_basis as py_get_rft_basis
    except ImportError:
        print("Error: Could not import get_rft_basis from canonical_true_rft.py")
        print("Please ensure canonical_true_rft.py is in the correct location.")
        sys.exit(1)

# Cache for basis matrices to avoid recomputation
basis_cache = {}

class SymbioticRFTEngine:
    \"\"\"
    Symbiotic RFT Engine that ensures Python and C++ implementations work together.
    This adapter ensures energy conservation in all RFT transformations.
    \"\"\"
    
    def __init__(self, dimension=16):
        \"\"\"
        Initialize the symbiotic RFT engine.
        
        Args:
            dimension: Size of the RFT basis (default: 16)
        \"\"\"
        self.dimension = dimension
        self.cpp_available = cpp_available
        
        # Precompute and cache the properly normalized basis
        if dimension not in basis_cache:
            basis_cache[dimension] = py_get_rft_basis(dimension)
        self.basis = basis_cache[dimension]
        
        # Initialize C++ engine if available
        if self.cpp_available:
            self.cpp_engine = symbiotic_rft_engine.SymbioticRFTEngine(dimension)
            self.cpp_engine.set_basis(self.basis)
        else:
            print("Using Python fallback for RFT transforms")
    
    def forward_true_rft(self, signal):
        \"\"\"
        Apply forward True RFT with energy conservation.
        
        Args:
            signal: Input signal to transform
            
        Returns:
            RFT domain representation of the signal
        \"\"\"
        # Ensure signal has the right size
        if len(signal) != self.dimension:
            if len(signal) < self.dimension:
                padded_signal = np.zeros(self.dimension, dtype=complex)
                padded_signal[:len(signal)] = signal
                signal = padded_signal
            else:
                signal = signal[:self.dimension]
        
        # Convert to numpy array if needed
        if not isinstance(signal, np.ndarray):
            signal = np.array([complex(x.real, x.imag) for x in signal])
        
        # Store original energy for verification
        original_energy = np.linalg.norm(signal)**2
        
        # Apply transform using C++ or Python
        if self.cpp_available:
            try:
                spectrum = self.cpp_engine.forward_rft(signal)
            except Exception as e:
                print(f"Warning: C++ forward RFT failed, falling back to Python: {e}")
                spectrum = self.basis.conj().T @ signal
        else:
            spectrum = self.basis.conj().T @ signal
        
        # Verify energy conservation
        spectrum_energy = np.linalg.norm(spectrum)**2
        energy_ratio = spectrum_energy / original_energy if original_energy > 0 else 1.0
        
        if abs(energy_ratio - 1.0) > 0.01:
            print(f"⚠️ Energy not conserved in forward RFT: ratio={energy_ratio:.4f}")
        
        return spectrum
    
    def inverse_true_rft(self, spectrum):
        \"\"\"
        Apply inverse True RFT with proper basis.
        
        Args:
            spectrum: RFT domain representation to inverse transform
            
        Returns:
            Time domain reconstruction of the signal
        \"\"\"
        # Ensure spectrum has the right size
        if len(spectrum) != self.dimension:
            if len(spectrum) < self.dimension:
                padded_spectrum = np.zeros(self.dimension, dtype=complex)
                padded_spectrum[:len(spectrum)] = spectrum
                spectrum = padded_spectrum
            else:
                spectrum = spectrum[:self.dimension]
        
        # Convert to numpy array if needed
        if not isinstance(spectrum, np.ndarray):
            spectrum = np.array([complex(x.real, x.imag) for x in spectrum])
        
        # Store spectrum energy for verification
        spectrum_energy = np.linalg.norm(spectrum)**2
        
        # Apply transform using C++ or Python
        if self.cpp_available:
            try:
                reconstructed = self.cpp_engine.inverse_rft(spectrum)
            except Exception as e:
                print(f"Warning: C++ inverse RFT failed, falling back to Python: {e}")
                reconstructed = self.basis @ spectrum
        else:
            reconstructed = self.basis @ spectrum
        
        # Verify energy conservation
        reconstructed_energy = np.linalg.norm(reconstructed)**2
        energy_ratio = reconstructed_energy / spectrum_energy if spectrum_energy > 0 else 1.0
        
        if abs(energy_ratio - 1.0) > 0.01:
            print(f"⚠️ Energy not conserved in inverse RFT: ratio={energy_ratio:.4f}")
        
        return reconstructed

# Simple test if run directly
if __name__ == "__main__":
    print("Symbiotic RFT Engine Test")
    print("=" * 40)
    
    # Create engine and test it
    dimensions = [16, 32, 64, 128]
    
    for N in dimensions:
        print(f"\\nTesting dimension N={N}")
        print("-" * 30)
        
        engine = SymbioticRFTEngine(dimension=N)
        
        # Generate a random signal
        signal = np.random.normal(size=N) + 1j * np.random.normal(size=N)
        signal_energy = np.linalg.norm(signal)**2
        
        # Apply forward transform
        spectrum = engine.forward_true_rft(signal)
        spectrum_energy = np.linalg.norm(spectrum)**2
        
        # Check energy conservation
        forward_energy_ratio = spectrum_energy / signal_energy
        print(f"Forward energy ratio: {forward_energy_ratio:.6f}")
        
        # Apply inverse transform
        reconstructed = engine.inverse_true_rft(spectrum)
        reconstructed_energy = np.linalg.norm(reconstructed)**2
        
        # Check energy conservation
        inverse_energy_ratio = reconstructed_energy / spectrum_energy
        print(f"Inverse energy ratio: {inverse_energy_ratio:.6f}")
        
        # Check round-trip error
        roundtrip_error = np.linalg.norm(signal - reconstructed)
        print(f"Round-trip error: {roundtrip_error:.6e}")
        
        print(f"All tests pass: {forward_energy_ratio > 0.99 and forward_energy_ratio < 1.01 and roundtrip_error < 1e-8}")
"""
        )

    # Create the build script
    build_script = Path("build_symbiotic_rft.py")

    with open(build_script, "w") as f:
        f.write(
            """#!/usr/bin/env python3
\"\"\"
Build script for the Symbiotic RFT Engine.
\"\"\"

import os
import sys
import platform
import subprocess
from pathlib import Path
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Find pybind11 headers
class get_pybind_include:
    def __init__(self, user=False):
        self.user = user
    
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# Build the extension
ext_modules = [
    Extension(
        'symbiotic_rft_engine',
        ['build_symbiotic/symbiotic_rft_engine.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    )
]

# Build configuration
def has_flag(compiler, flagname):
    import tempfile
import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True

def cpp_flag(compiler):
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    
    for flag in flags:
        if has_flag(compiler, flag):
            return flag
    
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc', '/std:c++14', '/O2'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }
    
    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts
    
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        
        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '\\"{}\\"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        
        build_ext.build_extensions(self)

setup(
    name='symbiotic_rft_engine',
    version='0.1.0',
    author='QuantoniumOS Team',
    author_email='info@quantoniumos.org',
    description='Symbiotic RFT Engine with energy conservation',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
"""
        )

    # Try to build the extension
    try:
        print("Building symbiotic_rft_engine extension...")
        subprocess.check_call(
            [sys.executable, "build_symbiotic_rft.py", "build_ext", "--inplace"]
        )
        print("Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False


if __name__ == "__main__":
    if build_symbiotic_rft():
        print("\nTo use the symbiotic RFT engine, import it like this:")
        print("from symbiotic_rft_wrapper import SymbioticRFTEngine")
        print("\nThen create an instance and use it:")
        print("engine = SymbioticRFTEngine(dimension=16)")
        print("spectrum = engine.forward_true_rft(signal)")
        print("reconstructed = engine.inverse_true_rft(spectrum)")
    else:
        print("\nBuild failed. Please check the error messages and try again.")
