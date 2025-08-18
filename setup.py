# setup.py for QuantoniumOS C++ Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "quantoniumos.engine_core_pybind",
        [
            "core/engine_core_pybind.cpp",
            "core/engine_core.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_cmake_dir() + "/../../../include",
            "core/include",
            "third_party/eigen",
        ],
        language='c++',
        cxx_std=17,
        define_macros=[
            ("PYBIND11_DETAILED_ERROR_MESSAGES", None),
        ],
    ),
]

setup(
    name="quantoniumos",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
) 