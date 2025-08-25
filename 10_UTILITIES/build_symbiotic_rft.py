#!/usr/bin/env python3
"""
Build script for the symbiotic True RFT engine.
This builds a C++ engine that works with the Python-generated basis.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Configure paths
SOURCE_FILE = Path("symbiotic_true_rft_engine.cpp")
BINDINGS_FILE = Path("symbiotic_true_rft_engine_bindings.cpp")
OUTPUT_MODULE = "symbiotic_true_rft_engine"


def create_bindings_file():
    """Create the Python bindings file for the symbiotic RFT engine."""
    print("Creating Python bindings file...")

    bindings_code = """
    /*
     * Python bindings for the symbiotic True RFT engine.
     * This module provides Python interfaces to the C++ implementation
     * that uses the Python-generated basis for perfect compatibility.
     */
    
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/stl.h>
    #include <complex>
    #include <vector>
    #include "include/true_rft_engine.h"
    
    namespace py = pybind11;
    
    // Helper functions for NumPy array conversion
    std::vector<double> np_array_to_vector(py::array_t<double> array) {
        py::buffer_info buf = array.request();
        double* ptr = static_cast<double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + buf.size);
    }
    
    // Python wrapper for engine_init
    int py_engine_init() {
        return engine_init();
    }
    
    // Python wrapper for engine_final
    void py_engine_final() {
        engine_final();
    }
    
    // Python wrapper for load_rft_basis_from_file
    bool py_load_rft_basis(const std::string& filename, int dimension) {
        return load_rft_basis_from_file(filename.c_str(), dimension);
    }
    
    // Python wrapper for verify_basis_orthonormality
    bool py_verify_basis_orthonormality() {
        return verify_basis_orthonormality();
    }
    
    // Python wrapper for forward_true_rft
    py::tuple py_forward_true_rft(py::array_t<double> x) {
        py::buffer_info buf = x.request();
        double* x_ptr = static_cast<double*>(buf.ptr);
        int N = buf.size;
        
        // Allocate output arrays
        std::vector<double> out_real(N);
        std::vector<double> out_imag(N);
        
        // Call C++ implementation
        int result = rft_basis_forward(x_ptr, N, nullptr, 0, nullptr, nullptr, 0, 0, nullptr,
                                      out_real.data(), out_imag.data());
        
        // Convert results to NumPy arrays
        py::array_t<double> np_real = py::array_t<double>(N);
        py::array_t<double> np_imag = py::array_t<double>(N);
        
        py::buffer_info buf_real = np_real.request();
        py::buffer_info buf_imag = np_imag.request();
        
        double* real_ptr = static_cast<double*>(buf_real.ptr);
        double* imag_ptr = static_cast<double*>(buf_imag.ptr);
        
        std::copy(out_real.begin(), out_real.end(), real_ptr);
        std::copy(out_imag.begin(), out_imag.end(), imag_ptr);
        
        return py::make_tuple(np_real, np_imag, result);
    }
    
    // Python wrapper for inverse_true_rft
    py::tuple py_inverse_true_rft(py::array_t<double> real, py::array_t<double> imag) {
        py::buffer_info buf_real = real.request();
        py::buffer_info buf_imag = imag.request();
        
        if (buf_real.size != buf_imag.size) {
            throw std::runtime_error("Real and imaginary parts must have the same size");
        }
        
        int N = buf_real.size;
        double* real_ptr = static_cast<double*>(buf_real.ptr);
        double* imag_ptr = static_cast<double*>(buf_imag.ptr);
        
        // Allocate output array
        std::vector<double> out_x(N);
        
        // Call C++ implementation
        int result = rft_basis_inverse(real_ptr, imag_ptr, N, nullptr, 0, nullptr, nullptr,
                                      0, 0, nullptr, out_x.data());
        
        // Convert results to NumPy array
        py::array_t<double> np_out = py::array_t<double>(N);
        py::buffer_info buf_out = np_out.request();
        double* out_ptr = static_cast<double*>(buf_out.ptr);
        
        std::copy(out_x.begin(), out_x.end(), out_ptr);
        
        return py::make_tuple(np_out, result);
    }
    
    // Python wrapper for verify_rft_energy_conservation
    int py_verify_energy_conservation(int N) {
        return verify_rft_energy_conservation(N);
    }
    
    // Python wrapper for export_rft_basis
    int py_export_rft_basis(const std::string& filename) {
        return export_rft_basis(filename.c_str());
    }
    
    PYBIND11_MODULE(symbiotic_true_rft_engine, m) {
        m.doc() = "Symbiotic True RFT Engine that works with Python-generated basis";
        
        m.def("engine_init", &py_engine_init, "Initialize the RFT engine");
        m.def("engine_final", &py_engine_final, "Finalize the RFT engine");
        
        m.def("load_rft_basis", &py_load_rft_basis, "Load RFT basis from file",
              py::arg("filename"), py::arg("dimension"));
              
        m.def("verify_basis_orthonormality", &py_verify_basis_orthonormality,
              "Verify orthonormality of the loaded basis");
              
        m.def("forward_true_rft", &py_forward_true_rft,
              "Apply forward RFT transform", py::arg("x"));
              
        m.def("inverse_true_rft", &py_inverse_true_rft,
              "Apply inverse RFT transform", py::arg("real"), py::arg("imag"));
              
        m.def("verify_energy_conservation", &py_verify_energy_conservation,
              "Verify energy conservation", py::arg("N"));
              
        m.def("export_rft_basis", &py_export_rft_basis,
              "Export RFT basis to file", py::arg("filename"));
    }
    """

    # Write the bindings file
    with open(BINDINGS_FILE, "w") as f:
        f.write(bindings_code.strip())

    return True


def create_header_file():
    """Create the header file for the symbiotic RFT engine."""
    print("Creating header file...")

    # Make sure include directory exists
    Path("include").mkdir(exist_ok=True)

    header_code = """
    /*
     * Header file for the symbiotic True RFT engine.
     */
    
    #ifndef TRUE_RFT_ENGINE_H
    #define TRUE_RFT_ENGINE_H
    
    #ifdef __cplusplus
    extern "C" {
    #endif
    
    // Engine initialization and cleanup
    int engine_init(void);
    void engine_final(void);
    
    // RFT basis forward transform
    int rft_basis_forward(const double* x, int N, const double* w_arr, int M, 
                         const double* th_arr, const double* om_arr, 
                         double sigma0, double gamma, const char* seq, 
                         double* out_real, double* out_imag);
    
    // RFT basis inverse transform
    int rft_basis_inverse(const double* Xr, const double* Xi, int N, 
                         const double* w_arr, int M, const double* th_arr, 
                         const double* om_arr, double sigma0, double gamma, 
                         const char* seq, double* out_x);
    
    #ifdef __cplusplus
    }
    
    // C++ only functions
    bool load_rft_basis_from_file(const char* filename, int expected_dimension);
    bool verify_basis_orthonormality();
    int export_rft_basis(const char* filename);
    int verify_rft_energy_conservation(int N);
    
    #endif
    
    #endif // TRUE_RFT_ENGINE_H
    """

    # Write the header file
    with open("include/true_rft_engine.h", "w") as f:
        f.write(header_code.strip())

    return True


def create_numpy_basis_exporter():
    """Create a Python utility to export NumPy arrays in a format C++ can read."""
    print("Creating NumPy basis exporter...")

    exporter_code = """
    #!/usr/bin/env python3
    \"\"\"
    Utility to export RFT basis matrix in a format C++ can read.
    \"\"\"
    
    import numpy as np
import struct
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent))
    
    try:
        import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
from canonical_true_rft import get_rft_basis
    except ImportError:
        # Try relative import
        sys.path.append(str(Path(__file__).parent))
        try:
            import sys, os
from canonical_true_rft import get_rft_basis
        except ImportError:
            print("Error: Could not import get_rft_basis from canonical_true_rft.py")
            print("Please ensure canonical_true_rft.py is in the correct location.")
            sys.exit(1)
    
    def export_basis_to_binary(dimension, output_file="rft_basis_export.bin"):
        \"\"\"
        Export RFT basis matrix to a binary file that C++ can read.
        
        Args:
            dimension: Size of the RFT basis
            output_file: Path to save the exported basis
        \"\"\"
        print(f"Generating RFT basis for dimension {dimension}...")
        basis = get_rft_basis(dimension)
        
        # Verify orthonormality
        for j in range(dimension):
            col_norm = np.linalg.norm(basis[:, j])
            if abs(col_norm - 1.0) > 1e-10:
                print(f"Warning: Column {j} has norm {col_norm} (should be 1.0)")
        
        # Verify energy conservation with test signal
        signal = np.random.normal(size=dimension) + 1j * np.random.normal(size=dimension)
        signal_energy = np.linalg.norm(signal)**2
        
        # Forward transform
        spectrum = basis.conj().T @ signal
        spectrum_energy = np.linalg.norm(spectrum)**2
        
        energy_ratio = spectrum_energy / signal_energy
        print(f"Energy conservation test: {energy_ratio:.8f} (should be 1.0)")
        
        # Write to binary file
        with open(output_file, 'wb') as f:
            # Write dimension as int
            f.write(struct.pack('i', dimension))
            
            # Write real parts as doubles
            real_parts = basis.real.flatten()
            f.write(struct.pack(f'{len(real_parts)}d', *real_parts))
            
            # Write imaginary parts as doubles
            imag_parts = basis.imag.flatten()
            f.write(struct.pack(f'{len(imag_parts)}d', *imag_parts))
        
        print(f"Basis exported to {output_file}")
        print(f"- Dimension: {dimension}")
        print(f"- Matrix shape: {basis.shape}")
        print(f"- File size: {Path(output_file).stat().st_size} bytes")
        
        return output_file
    
    if __name__ == "__main__":
        import argparse
        
        parser = argparse.ArgumentParser(description="Export RFT basis matrix for C++ integration")
        parser.add_argument("dimension", type=int, help="Size of the RFT basis")
        parser.add_argument("--output", "-o", default="rft_basis_export.bin",
                           help="Output file path (default: rft_basis_export.bin)")
        
        args = parser.parse_args()
        export_basis_to_binary(args.dimension, args.output)
    """

    # Write the exporter utility
    with open("export_rft_basis.py", "w") as f:
        f.write(exporter_code.strip())

    # Make it executable
    os.chmod("export_rft_basis.py", 0o755)

    return True


def generate_example_basis():
    """Generate example basis files for common dimensions."""
    print("Generating example basis files...")

    # Create the export script
    create_numpy_basis_exporter()

    # Run the exporter for common dimensions
    dimensions = [16, 32, 64, 128]
    for dim in dimensions:
        cmd = [
            sys.executable,
            "export_rft_basis.py",
            str(dim),
            "--output",
            f"rft_basis_{dim}.bin",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating basis for dimension {dim}: {e}")
            continue

    return True


def build_windows():
    """Build the symbiotic RFT engine on Windows."""
    print("Building on Windows...")

    # Create necessary files
    create_header_file()
    create_bindings_file()

    try:
        # Setup environment
        import setuptools
        from setuptools import Extension, setup
        from setuptools.command.build_ext import build_ext

        # Configure the extension
        ext_modules = [
            Extension(
                name=OUTPUT_MODULE,
                sources=[str(SOURCE_FILE), str(BINDINGS_FILE)],
                include_dirs=["include"],
                language="c++",
            )
        ]

        # Configure setup
        setup(
            name=OUTPUT_MODULE,
            ext_modules=ext_modules,
            cmdclass={"build_ext": build_ext},
            script_args=["build_ext", "--inplace"],
        )

        print(f"Successfully built {OUTPUT_MODULE}")
        return True

    except Exception as e:
        print(f"Error building on Windows: {e}")
        return False


def build_unix():
    """Build the symbiotic RFT engine on Unix-like systems."""
    print("Building on Unix-like system...")

    # Create necessary files
    create_header_file()
    create_bindings_file()

    try:
        # Set up build command
        cmd = [
            "c++",
            "-O3",
            "-Wall",
            "-shared",
            "-std=c++11",
            "-fPIC",
            f"-I{sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}",
            "-Iinclude",
            str(SOURCE_FILE),
            str(BINDINGS_FILE),
            "-o",
            f"{OUTPUT_MODULE}{'.so' if os.name == 'posix' else '.pyd'}",
        ]

        # Execute build
        subprocess.run(cmd, check=True)

        print(f"Successfully built {OUTPUT_MODULE}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error building on Unix: {e}")
        return False


if __name__ == "__main__":
    print("Building symbiotic True RFT engine...")

    # Generate example basis files
    generate_example_basis()

    # Build based on platform
    if platform.system() == "Windows":
        success = build_windows()
    else:
        success = build_unix()

    if success:
        print("\nBuild completed successfully!")
        print("The symbiotic True RFT engine is now ready to use.")
        print("This engine uses the Python-generated basis for perfect compatibility.")
    else:
        print("\nBuild failed.")
        print("Please check the error messages above and try again.")
