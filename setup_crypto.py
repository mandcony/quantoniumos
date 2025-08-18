from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "enhanced_rft_crypto_bindings",
        ["paper_compliant_crypto_bindings.cpp"],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="enhanced_rft_crypto_bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
