"""
QuantoniumOS Setup Script
Builds the complete package including C++ extensions
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Define the C++ extension
ext_modules = [
    Pybind11Extension(
        "quantonium_core",
        ["core/pybind_interface.cpp"],
        cxx_std=17,
        include_dirs=[
            pybind11.get_include(),
        ],
        define_macros=[
            ("VERSION_INFO", '"0.3.0-rc1"'),
            ("PATENT_APPLICATION", '"USPTO #19/169,399"'),
        ],
        language="c++",
    ),
]

setup(
    name="quantonium-os",
    version="0.3.0-rc1",
    description="Quantum-inspired computational framework with patent-protected algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="QuantoniumOS Development Team",
    author_email="dev@quantonium.io",
    url="https://github.com/quantonium/quantonium-os",
    packages=[
        "core",
        "core.encryption",
        "core.protected", 
        "core.HPC",
        "tests",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    install_requires=[
        "flask>=3.1.0",
        "gunicorn>=23.0.0",
        "numpy>=2.2.0",
        "cryptography>=44.0.0",
        "pybind11>=2.13.0",
        "psycopg2-binary>=2.9.0",
        "pytest>=8.3.0",
        "pydantic>=2.11.0",
        "flask-cors>=5.0.0",
        "flask-limiter>=3.10.0",
        "PyJWT>=2.10.0",
        "redis>=5.2.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=6.0.0",
            "pytest-xdist>=3.6.0",
            "bandit>=1.7.0",
            "safety>=3.0.0",
        ],
        "gui": [
            "PyQt5>=5.15.0",
            "qtawesome>=1.3.0",
        ],
        "ai": [
            "anthropic>=0.43.0",
            "notion-client>=2.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="Dual License: Academic Free / Commercial",
    keywords="quantum computing, cryptography, fourier transform, patent-protected",
    project_urls={
        "Bug Reports": "https://github.com/quantonium/quantonium-os/issues",
        "Source": "https://github.com/quantonium/quantonium-os",
        "Documentation": "https://quantonium.io/docs",
        "Patent": "https://patents.uspto.gov/search?q=19/169,399",
    },
)