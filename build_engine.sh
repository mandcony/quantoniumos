#!/bin/bash
set -e

# === CONFIGURATION ===
ROOT_DIR="$(pwd)"
BUILD_DIR="$ROOT_DIR/bin"
CORE_DIR="$ROOT_DIR/core"
EIGEN_DIR="$ROOT_DIR/Eigen/eigen-3.4.0"
PYTHON_INCLUDE=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PYTHON_LIBDIR=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_LIB="-L$PYTHON_LIBDIR -lpython$PYTHON_VERSION"
PYBIND_INCLUDE=$(python -m pybind11 --includes)
CXX="g++"

COMMON_FLAGS="-O3 -std=c++17 -fPIC $PYBIND_INCLUDE -I$EIGEN_DIR -I$PYTHON_INCLUDE"
OPENMP_FLAGS="-fopenmp"
LINK_FLAGS="-shared $PYTHON_LIB -lstdc++ $OPENMP_FLAGS"

# === CREATE BUILD DIR ===
mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR"/*.so "$BUILD_DIR"/*.o

# === BUILD FUNCTION ===
build_module () {
    local source_file="$1"
    local output_name="$2"
    local use_openmp="${3:-false}"
    local object_file="${source_file%.cpp}.o"
    local output_path="$BUILD_DIR/${output_name}.so"

    echo "Compiling: $source_file"
    local compile_flags="$COMMON_FLAGS"
    [[ "$use_openmp" == "true" ]] && compile_flags="$compile_flags $OPENMP_FLAGS"
    $CXX -c "$source_file" -o "$object_file" $compile_flags

    echo "Linking: $output_path"
    $CXX -shared "$object_file" -o "$output_path" $LINK_FLAGS
    echo "✅ Built $output_path"
}

# === COMPILE MODULES ===
build_module "$CORE_DIR/quantum_os.cpp" "quantum_os"
build_module "$CORE_DIR/symbolic_eigenvector.cpp" "engine_core" true

echo "✅ All Pybind11 modules built into: $BUILD_DIR"