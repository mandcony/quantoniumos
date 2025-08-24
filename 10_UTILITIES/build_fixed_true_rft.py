"""
Build the fixed true_rft_engine bindings with proper normalization.
"""

import os
import subprocess
import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# Determine the eigen3 include path for the current environment
def find_eigen_include():
    try:
        # Try to find Eigen3 using pkg-config
        import pkgconfig

        if pkgconfig.exists("eigen3"):
            return pkgconfig.parse("eigen3")["include_dirs"][0]
    except:
        pass

    # Standard locations to check for Eigen3
    standard_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "C:/Program Files/Eigen3/include/eigen3",
        "C:/Eigen3/include/eigen3",
        # Add user-specific paths for your environment if needed
    ]

    for path in standard_paths:
        if os.path.exists(path):
            return path

    # If not found, fall back to using the vendored Eigen in the current directory
    if os.path.exists("eigen3"):
        return "eigen3"

    print("Warning: Could not find Eigen3. Downloading...")
    # Download Eigen3 if not found
    try:
        import urllib.request
        import zipfile

        eigen_url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
        eigen_zip = "eigen-3.4.0.zip"

        urllib.request.urlretrieve(eigen_url, eigen_zip)
        with zipfile.ZipFile(eigen_zip, "r") as zip_ref:
            zip_ref.extractall(".")

        os.rename("eigen-3.4.0", "eigen3")
        os.remove(eigen_zip)

        return "eigen3"
    except Exception as e:
        print(f"Error downloading Eigen: {e}")
        print("Please install Eigen3 manually and specify the path.")
        sys.exit(1)


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


# Define the extension module
ext_modules = [
    Extension(
        "true_rft_engine_bindings",
        ["true_rft_engine_bindings_fixed.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            # Path to Eigen3 headers
            find_eigen_include(),
            # Path to core engine headers
            "./core/include",
            np.get_include(),
        ],
        libraries=["engine_core"],  # Link to the engine_core library
        library_dirs=["./core/lib"],  # Look for the library here
        extra_compile_args=["-std=c++14"],  # Use C++14
        language="c++",
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except:
            return False
    return True


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc", "/bigobj"],  # /MT for static runtime
        "unix": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++14")
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args = opts

        build_ext.build_extensions(self)


# Set up the extension
setup(
    name="true_rft_engine_bindings",
    version="0.1",
    author="Fixed RFT Team",
    author_email="info@example.com",
    description="Improved bindings for True RFT Engine with proper normalization",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)

print(
    "Build completed. Run 'python -c \"import true_rft_engine_bindings; print(true_rft_engine_bindings.__file__)\"' to verify installation."
)
