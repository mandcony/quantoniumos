#!/usr/bin/env python3
"""
QuantoniumOS Setup Configuration
===============================
Package build configuration for the world's first quantum operating system.
"""

import os
import sys

import numpy
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, find_packages, setup

# Version information
VERSION = "1.0.0"
DESCRIPTION = "QuantoniumOS - Complete Quantum Operating System"
LONG_DESCRIPTION = """
QuantoniumOS: The World's First Quantum Operating System
========================================================

A complete quantum operating system featuring:

🚀 Core Features:
- 1000-qubit quantum vertex network with 32x32 grid topology
- True Recursive Frequency Transform (RFT) engine with patent protection
- Enhanced quantum cryptography with hardware acceleration
- Real-time quantum process management and scheduling
- Unified desktop and web interface

🔬 Quantum Technology:
- Canonical True RFT implementation (Patent US 19/169,399)
- Topological and bulletproof quantum simulation kernels
- Quantum harmonic oscillators for state evolution
- Multi-engine quantum processing support

🔐 Cryptography:
- Enhanced RFT-based encryption with C++ acceleration
- Quantum-resistant cryptographic algorithms
- Wave-based hash functions and entropy generation
- Patent-protected security implementations

🎯 Applications:
- Quantum circuit simulation and visualization
- RFT transform analysis and visualization
- Quantum cryptography playground
- Patent validation dashboard
- Real-time system monitoring

Installation:
    pip install quantoniumos

Quick Start:
    python -c "from core.quantonium_os_unified import QuantoniumOSUnified; QuantoniumOSUnified().run()"

Repository: https://github.com/mandcony/quantoniumos
Documentation: See README.md and UNIFIED_README.md
"""

# C++ Extensions for performance-critical components
cpp_extensions = []

# True RFT Engine C++ Extension
if os.path.exists("04_RFT_ALGORITHMS/true_rft_engine.cpp"):
    true_rft_ext = Pybind11Extension(
        "quantoniumos.true_rft_engine",
        sources=["04_RFT_ALGORITHMS/true_rft_engine.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir(),
            numpy.get_include(),
            "04_RFT_ALGORITHMS",
        ],
        language="c++",
        cxx_std=17,
    )
    cpp_extensions.append(true_rft_ext)

# Enhanced RFT Crypto C++ Extension
if os.path.exists("06_CRYPTOGRAPHY/enhanced_rft_crypto.cpp"):
    crypto_ext = Pybind11Extension(
        "quantoniumos.enhanced_rft_crypto",
        sources=["06_CRYPTOGRAPHY/enhanced_rft_crypto.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir(),
            numpy.get_include(),
            "06_CRYPTOGRAPHY",
        ],
        language="c++",
        cxx_std=17,
    )
    cpp_extensions.append(crypto_ext)

# Core quantum engine extension
if os.path.exists("core/engine_core.cpp"):
    core_ext = Pybind11Extension(
        "quantoniumos.engine_core",
        sources=["core/engine_core.cpp"],
        include_dirs=[
            pybind11.get_cmake_dir(),
            numpy.get_include(),
            "core",
        ],
        language="c++",
        cxx_std=17,
    )
    cpp_extensions.append(core_ext)

# Package data
package_data = {
    "quantoniumos": [
        "*.py",
        "*.md",
        "*.txt",
        "*.json",
        "11_QUANTONIUMOS/**/*",
        "04_RFT_ALGORITHMS/**/*",
        "05_QUANTUM_ENGINES/**/*",
        "06_CRYPTOGRAPHY/**/*",
        "web/**/*",
        "wave_ui/**/*",
        "phase3/**/*",
        "phase4/**/*",
    ]
}

# Entry points for command-line access
entry_points = {
    "console_scripts": [
        "quantoniumos=quantonium_os_unified:main",
        "quantonium-launcher=start_quantoniumos:main",
        "qos=quantonium_os_unified:main",
    ]
}

# Dependencies
install_requires = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "networkx>=2.6",
    "psutil>=5.8.0",
    "pybind11>=2.6.0",
]

# Optional dependencies for enhanced features
extras_require = {
    "gui": ["tkinter"],
    "web": ["flask>=2.0.0", "flask-cors>=4.0.0"],
    "crypto": ["cryptography>=3.4.8", "pycryptodome>=3.15.0"],
    "jwt": ["pyjwt>=2.0.0"],
    "validation": ["pydantic>=2.0.0"],
    "qt": ["PyQt5>=5.15.0"],
    "all": [
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        "cryptography>=3.4.8",
        "pycryptodome>=3.15.0",
        "pyjwt>=2.0.0",
        "pydantic>=2.0.0",
        "PyQt5>=5.15.0",
    ],
}

# Classifiers for PyPI
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: System :: Operating System",
    "Topic :: Security :: Cryptography",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Setup configuration
setup(
    name="quantoniumos",
    version=VERSION,
    author="QuantoniumOS Development Team",
    author_email="dev@quantoniumos.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/mandcony/quantoniumos",
    project_urls={
        "Bug Tracker": "https://github.com/mandcony/quantoniumos/issues",
        "Documentation": "https://github.com/mandcony/quantoniumos/blob/main/README.md",
        "Source Code": "https://github.com/mandcony/quantoniumos",
    },
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    # ext_modules=cpp_extensions,  # Temporarily disabled for wheel build
    # cmdclass={"build_ext": build_ext},  # Temporarily disabled for wheel build
    entry_points=entry_points,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
    classifiers=classifiers,
    keywords="quantum computing, operating system, rft, cryptography, physics, simulation",
    zip_safe=False,
    platforms=["any"],
    license="Proprietary",
)
