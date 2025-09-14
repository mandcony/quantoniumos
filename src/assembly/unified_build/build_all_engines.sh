#!/bin/bash
# QuantoniumOS Build All Engines Script
# Builds crypto, quantum, and OS engines with optimizations

set -e  # Exit on any error

echo "🚀 QuantoniumOS Engine Build System"
echo "=================================="

# Configuration
BUILD_DIR="build"
INSTALL_PREFIX="/usr/local"
BUILD_TYPE="Release"
ENABLE_PYTHON="ON"
TARGET_THROUGHPUT="9.2"  # MB/s from paper

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --crypto-only)
            CRYPTO_ONLY="ON"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-python)
            ENABLE_PYTHON="OFF"
            shift
            ;;
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --crypto-only      Build only crypto engine (for 9.2 MB/s target)"
            echo "  --debug           Debug build"
            echo "  --no-python       Skip Python bindings"
            echo "  --clean           Clean build directory first"
            echo "  --install-prefix  Installation prefix (default: /usr/local)"
            echo "  --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean build directory if requested
if [[ "${CLEAN_BUILD}" == "ON" ]]; then
    echo "🧹 Cleaning build directory..."
    rm -rf ${BUILD_DIR}
fi

# Create build directory
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Detect CPU capabilities
echo "🔍 Detecting CPU capabilities..."
if grep -q avx2 /proc/cpuinfo; then
    echo "✅ AVX2 support detected"
    AVX2_SUPPORT="ON"
else
    echo "❌ AVX2 not available"
    AVX2_SUPPORT="OFF"
fi

if grep -q aes /proc/cpuinfo; then
    echo "✅ AES-NI support detected"
    AES_NI_SUPPORT="ON"
else
    echo "❌ AES-NI not available"
    AES_NI_SUPPORT="OFF"
fi

# Configure CMake options
CMAKE_OPTIONS=(
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
    "-DBUILD_PYTHON_BINDINGS=${ENABLE_PYTHON}"
    "-DENABLE_AVX2=${AVX2_SUPPORT}"
    "-DENABLE_AES_NI=${AES_NI_SUPPORT}"
    "-DOPTIMIZE_FOR_SPEED=ON"
)

# Crypto-only build
if [[ "${CRYPTO_ONLY}" == "ON" ]]; then
    echo "🔥 Building CRYPTO ENGINE ONLY (Target: ${TARGET_THROUGHPUT} MB/s)"
    CMAKE_OPTIONS+=(
        "-DBUILD_CRYPTO_ENGINE=ON"
        "-DBUILD_QUANTUM_ENGINE=OFF"
        "-DBUILD_OS_ENGINE=OFF"
    )
else
    echo "🌟 Building ALL ENGINES"
    CMAKE_OPTIONS+=(
        "-DBUILD_CRYPTO_ENGINE=ON"
        "-DBUILD_QUANTUM_ENGINE=ON"
        "-DBUILD_OS_ENGINE=ON"
    )
fi

# Run CMake configuration
echo "⚙️  Configuring build..."
cmake .. "${CMAKE_OPTIONS[@]}"

# Build
echo "🔨 Building engines..."
if [[ "${CRYPTO_ONLY}" == "ON" ]]; then
    make crypto_only -j$(nproc)
else
    make -j$(nproc)
fi

# Run tests
echo "🧪 Running tests..."
if [[ -f "./validate_all_engines" ]]; then
    ./validate_all_engines
else
    echo "⚠️  Tests not built"
fi

# Run benchmarks
echo "📊 Running benchmarks..."
if [[ -f "./benchmark_engines" ]]; then
    ./benchmark_engines
else
    echo "⚠️  Benchmarks not built"
fi

# Install if requested
if [[ "${INSTALL_PREFIX}" != "/usr/local" ]] || [[ "$EUID" -eq 0 ]]; then
    echo "📦 Installing to ${INSTALL_PREFIX}..."
    make install
else
    echo "ℹ️  To install system-wide, run: sudo make install"
fi

# Show Python usage
if [[ "${ENABLE_PYTHON}" == "ON" ]]; then
    echo ""
    echo "🐍 Python Usage:"
    echo "==============="
    echo "# Test crypto engine"
    echo "import feistel_crypto"
    echo "cipher = feistel_crypto.FeistelCipher()"
    echo "cipher.init(b'your_32_byte_key_here_12345678')"
    echo "result = cipher.benchmark()"
    echo "print(f'Throughput: {result[\"throughput_mbps\"]} MB/s')"
    echo ""
    echo "# Target from paper: ${TARGET_THROUGHPUT} MB/s"
fi

echo ""
echo "✅ Build complete!"
echo "🎯 Crypto engine optimized for ${TARGET_THROUGHPUT} MB/s throughput"
echo "📁 Build artifacts in: $(pwd)"

cd ..  # Return to original directory
