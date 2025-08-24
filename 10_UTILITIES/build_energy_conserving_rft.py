#!/usr/bin/env python3
"""
Build script for the fixed True RFT engine with proper normalization.
This script compiles the fixed C++ implementation with energy-conserving normalization.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def get_eigen_include_path():
    """Get the Eigen include path, downloading if necessary."""
    eigen_dir = Path("./third_party/eigen")

    if not eigen_dir.exists():
        print("Eigen not found, downloading...")
        os.makedirs(eigen_dir.parent, exist_ok=True)

        # Download Eigen
        import urllib.request
        import zipfile

        eigen_zip = "eigen-3.4.0.zip"
        eigen_url = f"https://gitlab.com/libeigen/eigen/-/archive/3.4.0/{eigen_zip}"

        print(f"Downloading Eigen from {eigen_url}")
        urllib.request.urlretrieve(eigen_url, eigen_zip)

        # Extract
        with zipfile.ZipFile(eigen_zip, "r") as zip_ref:
            zip_ref.extractall("./third_party")

        # Rename directory
        os.rename(f"./third_party/eigen-3.4.0", eigen_dir)

        # Cleanup
        os.remove(eigen_zip)

    return str(eigen_dir)


def build_fixed_engine():
    """Build the fixed True RFT engine."""
    print("Building fixed True RFT engine...")

    # Get include paths
    python_include = Path(sys.executable).parent.parent / "include"
    if not python_include.exists():
        # Try alternate location for Python includes
        python_include = Path(sys.executable).parent / "include"

    python_include = str(python_include)

    # Get pybind11 include path
    try:
        import pybind11

        pybind11_include = pybind11.get_include()
    except ImportError:
        print("pybind11 not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
        import pybind11

        pybind11_include = pybind11.get_include()

    eigen_include = get_eigen_include_path()

    # Create include directory if it doesn't exist
    os.makedirs("include", exist_ok=True)

    # Source and output files
    source_file = "fixed_true_rft_engine.cpp"
    output_file = "fixed_true_rft_engine_bindings"

    # Platform-specific settings
    if platform.system() == "Windows":
        output_extension = ".pyd"
        compiler_cmd = "cl"
        compile_flags = [
            "/EHsc",
            "/std:c++17",
            "/O2",
            "/I",
            python_include,
            "/I",
            pybind11_include,
            "/I",
            eigen_include,
            "/I",
            ".",
            "/LD",
            source_file,
            f"/Fe{output_file}{output_extension}",
        ]
    else:
        output_extension = ".so"
        compiler_cmd = "g++"
        compile_flags = [
            "-O3",
            "-Wall",
            "-shared",
            "-std=c++17",
            "-fPIC",
            f"-I{python_include}",
            f"-I{pybind11_include}",
            f"-I{eigen_include}",
            "-I.",
            source_file,
            "-o",
            f"{output_file}{output_extension}",
        ]

    # Add Python lib directory for Windows
    if platform.system() == "Windows":
        python_lib = Path(sys.executable).parent / "libs"
        if python_lib.exists():
            compile_flags.extend(["/link", f"/LIBPATH:{python_lib}"])

    # Print the command for debugging
    print(f"Compiler command: {compiler_cmd} {' '.join(compile_flags)}")

    # Run the compiler
    try:
        result = subprocess.run(
            [compiler_cmd] + compile_flags,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        print(f"Successfully built {output_file}{output_extension}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    build_fixed_engine()
