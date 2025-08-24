#!/usr/bin/env python3
"""
Direct build script for the fixed True RFT engine.
This uses a simpler approach for Windows compatibility.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# Configure paths
EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
EIGEN_DIR = Path("./third_party/eigen")
SOURCE_FILE = Path("fixed_true_rft_engine.cpp")
BINDINGS_FILE = Path("fixed_true_rft_engine_bindings.cpp")
OUTPUT_FILE = "fixed_true_rft_engine_bindings"


def download_eigen():
    """Download and extract the Eigen library."""
    print("Downloading Eigen library...")
    if EIGEN_DIR.exists():
        print("Eigen already exists, skipping download.")
        return True

    try:
        # Create directory
        os.makedirs(EIGEN_DIR.parent, exist_ok=True)

        # Download Eigen
        eigen_zip = "eigen-3.4.0.zip"
        print(f"Downloading from {EIGEN_URL}")
        urllib.request.urlretrieve(EIGEN_URL, eigen_zip)

        # Extract
        with zipfile.ZipFile(eigen_zip, "r") as zip_ref:
            zip_ref.extractall("./third_party")

        # Rename directory
        extracted_dir = Path("./third_party/eigen-3.4.0")
        if extracted_dir.exists():
            shutil.move(str(extracted_dir), str(EIGEN_DIR))

        # Cleanup
        if Path(eigen_zip).exists():
            os.remove(eigen_zip)

        return True
    except Exception as e:
        print(f"Error downloading Eigen: {e}")
        return False


def install_dependencies():
    """Install required Python dependencies."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pybind11", "numpy"]
        )
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False


def build_engine_windows():
    """Build the engine using MSVC on Windows."""
    print("Building with MSVC...")

    # Get Python paths
    python_dir = Path(sys.executable).parent
    python_include = python_dir.parent / "include"
    if not python_include.exists():
        python_include = python_dir / "include"

    python_libs = python_dir / "libs"

    # Get pybind11 include path
    import pybind11

    pybind11_include = pybind11.get_include()

    # Ensure include directory exists
    os.makedirs("include", exist_ok=True)

    # Build command
    cmd = [
        "cl",
        "/EHsc",
        "/std:c++17",
        "/O2",
        f"/I{python_include}",
        f"/I{pybind11_include}",
        f"/I{EIGEN_DIR}",
        "/I.",
        "/LD",
        str(BINDINGS_FILE),
        str(SOURCE_FILE),
        f"/Fe{OUTPUT_FILE}.pyd",
        "/link",
        f"/LIBPATH:{python_libs}",
    ]

    # Run the build
    try:
        print(f"Running: {' '.join(cmd)}")
        # Security fix: Remove shell=True to prevent command injection
        subprocess.run(cmd, check=True)
        return os.path.exists(f"{OUTPUT_FILE}.pyd")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False


def build_engine_unix():
    """Build the engine on Unix systems."""
    print("Building with g++/clang...")

    # Get Python paths
    python_dir = Path(sys.executable).parent
    python_include = (
        python_dir.parent
        / "include"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    if not python_include.exists():
        python_include = (
            Path(sys.exec_prefix)
            / "include"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
        )

    # Get pybind11 include path
    import pybind11

    pybind11_include = pybind11.get_include()

    # Build command
    cmd = [
        "g++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        f"-I{python_include}",
        f"-I{pybind11_include}",
        f"-I{EIGEN_DIR}",
        "-I.",
        str(BINDINGS_FILE),
        str(SOURCE_FILE),
        "-o",
        f"{OUTPUT_FILE}.so",
    ]

    # Run the build
    try:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return os.path.exists(f"{OUTPUT_FILE}.so")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False


def main():
    """Main build function."""
    print("=" * 60)
    print("Building Fixed True RFT Engine with Energy Conservation")
    print("=" * 60)

    # Step 1: Install dependencies
    print("\nStep 1: Installing dependencies...")
    if not install_dependencies():
        print("Failed to install dependencies. Aborting.")
        return False

    # Step 2: Download Eigen
    print("\nStep 2: Setting up Eigen library...")
    if not download_eigen():
        print("Failed to download Eigen. Aborting.")
        return False

    # Step 3: Build the engine
    print("\nStep 3: Building the engine...")
    if platform.system() == "Windows":
        success = build_engine_windows()
    else:
        success = build_engine_unix()

    if success:
        print("\nBuild successful!")
        extension = ".pyd" if platform.system() == "Windows" else ".so"
        print(f"Output file: {OUTPUT_FILE}{extension}")
        return True
    else:
        print("\nBuild failed.")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build the Fixed RFT Direct Engine")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--help", action="help", help="Show this help message and exit")

    args = parser.parse_args()

    # Build the engine
    success = main()
    sys.exit(0 if success else 1)
