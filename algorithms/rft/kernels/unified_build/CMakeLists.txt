# QuantoniumOS Unified Build System
# Builds all 3 engines: OS + Crypto + Quantum

cmake_minimum_required(VERSION 3.15)
project(QuantoniumOS_Engines VERSION 1.0 LANGUAGES C CXX ASM_NASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Configuration options
option(BUILD_CRYPTO_ENGINE "Build crypto engine" ON)
option(BUILD_QUANTUM_ENGINE "Build quantum engine" ON)
option(BUILD_OS_ENGINE "Build OS engine" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AES_NI "Enable AES-NI instructions" ON)
option(OPTIMIZE_FOR_SPEED "Optimize for maximum speed" ON)

# Compiler flags
if(OPTIMIZE_FOR_SPEED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
endif()

if(ENABLE_AVX2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

if(ENABLE_AES_NI)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")
endif()

# Enable NASM
enable_language(ASM_NASM)
set(CMAKE_ASM_NASM_FLAGS "-f elf64")

# Find required packages
find_package(OpenSSL)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../engines)

# =============================================================================
# CRYPTO ENGINE (Priority 1 - Target: 9.2 MB/s)
# =============================================================================

if(BUILD_CRYPTO_ENGINE)
    message(STATUS "Building Crypto Engine")
    
    # Crypto engine sources
    set(CRYPTO_SOURCES
        ../engines/crypto/src/feistel_round48.c
        ../engines/crypto/src/sha256_portable.c
        ../engines/crypto/asm/feistel_round48.asm
    )
    
    # Crypto engine library
    add_library(crypto_engine SHARED ${CRYPTO_SOURCES})
    target_include_directories(crypto_engine PUBLIC ../engines/crypto/include)
    target_compile_definitions(crypto_engine PRIVATE 
        -DTARGET_THROUGHPUT=9.2
        -DQUANTONIUMOS_CRYPTO_ENGINE=1
    )
    
    if(OpenSSL_FOUND)
        target_link_libraries(crypto_engine OpenSSL::SSL OpenSSL::Crypto)
    endif()
    
    # Python bindings for crypto engine
    if(BUILD_PYTHON_BINDINGS)
        pybind11_add_module(feistel_crypto ../engines/crypto/bindings/feistel_pybind.cpp)
        target_include_directories(feistel_crypto PRIVATE ../engines/crypto/include)
        target_link_libraries(feistel_crypto PRIVATE crypto_engine)
        target_compile_definitions(feistel_crypto PRIVATE VERSION_INFO=${PROJECT_VERSION})
    endif()
    
    # Crypto engine tests
    set(CRYPTO_TEST_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../engines/crypto/tests/test_crypto.c)
    if(EXISTS ${CRYPTO_TEST_SOURCE})
        add_executable(crypto_test ${CRYPTO_TEST_SOURCE})
        target_link_libraries(crypto_test crypto_engine)
    endif()
endif()

# =============================================================================
# QUANTUM ENGINE (Priority 2)
# =============================================================================

if(BUILD_QUANTUM_ENGINE)
    message(STATUS "Building Quantum Engine")
    
    # Port canonical_true_rft.py to C
    set(QUANTUM_SOURCES
        ../engines/quantum_engine/unitary_core.c
        ../engines/quantum_engine/vertex_rft.c
        ../engines/quantum_engine/quantum_asm.asm
    )
    
    add_library(quantum_engine SHARED ${QUANTUM_SOURCES})
    target_include_directories(quantum_engine PUBLIC ../engines/quantum_engine)
    target_compile_definitions(quantum_engine PRIVATE
        -DGOLDEN_RATIO_PHI=1.618033988749894848204586834366
        -DMAX_VERTEX_QUBITS=1000
    )
    target_link_libraries(quantum_engine m)  # Math library
    
    # Python bindings for quantum engine
    if(BUILD_PYTHON_BINDINGS)
        pybind11_add_module(quantum_rft ../engines/quantum_engine/python_bindings.cpp)
        target_link_libraries(quantum_rft PRIVATE quantum_engine)
    endif()
endif()

# =============================================================================
# OS ENGINE (Priority 3)
# =============================================================================

if(BUILD_OS_ENGINE)
    message(STATUS "Building OS Engine")
    
    # Port existing RFT kernel to OS engine
    set(OS_SOURCES
        ../engines/os_engine/kernel_core.c
        ../engines/os_engine/kernel_asm.asm
    )
    
    add_library(os_engine SHARED ${OS_SOURCES})
    target_include_directories(os_engine PUBLIC ../engines/os_engine)
    target_compile_definitions(os_engine PRIVATE
        -DQUANTONIUMOS_KERNEL=1
        -DREAL_TIME_CAPABLE=1
    )
endif()

# =============================================================================
# UNIFIED API
# =============================================================================

if(BUILD_PYTHON_BINDINGS)
    # Create unified Python module
    pybind11_add_module(quantonium_engines ../unified_build/unified_bindings.cpp)
    
    if(BUILD_CRYPTO_ENGINE)
        target_link_libraries(quantonium_engines PRIVATE crypto_engine)
    endif()
    
    if(BUILD_QUANTUM_ENGINE)
        target_link_libraries(quantonium_engines PRIVATE quantum_engine)
    endif()
    
    if(BUILD_OS_ENGINE)
        target_link_libraries(quantonium_engines PRIVATE os_engine)
    endif()
endif()

# =============================================================================
# VALIDATION AND BENCHMARKS
# =============================================================================

# Comprehensive test suite
add_executable(validate_all_engines ../unified_build/validate_engines.c)

if(BUILD_CRYPTO_ENGINE)
    target_link_libraries(validate_all_engines crypto_engine)
endif()

if(BUILD_QUANTUM_ENGINE)
    target_link_libraries(validate_all_engines quantum_engine)
endif()

if(BUILD_OS_ENGINE)
    target_link_libraries(validate_all_engines os_engine)
endif()

# Performance benchmark
add_executable(benchmark_engines ../unified_build/benchmark_engines.c)
target_link_libraries(benchmark_engines 
    $<$<BOOL:${BUILD_CRYPTO_ENGINE}>:crypto_engine>
    $<$<BOOL:${BUILD_QUANTUM_ENGINE}>:quantum_engine>
    $<$<BOOL:${BUILD_OS_ENGINE}>:os_engine>
)

# =============================================================================
# INSTALLATION
# =============================================================================

# Install libraries
if(BUILD_CRYPTO_ENGINE)
    install(TARGETS crypto_engine DESTINATION lib)
    install(FILES
        ../engines/crypto/include/feistel_round48.h
        ../engines/crypto/include/sha256_portable.h
    DESTINATION include)
endif()

if(BUILD_QUANTUM_ENGINE)
    install(TARGETS quantum_engine DESTINATION lib)
    install(FILES ../engines/quantum_engine/vertex_rft.h DESTINATION include)
endif()

if(BUILD_OS_ENGINE)
    install(TARGETS os_engine DESTINATION lib)
    install(FILES ../engines/os_engine/kernel_core.h DESTINATION include)
endif()

# Install Python modules
if(BUILD_PYTHON_BINDINGS)
    if(BUILD_CRYPTO_ENGINE)
        install(TARGETS feistel_crypto DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3/site-packages)
    endif()
    if(BUILD_QUANTUM_ENGINE)
        install(TARGETS quantum_rft DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3/site-packages)
    endif()
    install(TARGETS quantonium_engines DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3/site-packages)
endif()

# =============================================================================
# CUSTOM TARGETS
# =============================================================================

# Quick build target for crypto engine only
add_custom_target(crypto_only
    COMMAND ${CMAKE_COMMAND} --build . --target crypto_engine --target feistel_crypto
    COMMENT "Building crypto engine only for 9.2 MB/s target"
)

# Test target
add_custom_target(test_all
    COMMAND ./validate_all_engines
    DEPENDS validate_all_engines
    COMMENT "Running all engine tests"
)

# Benchmark target
add_custom_target(benchmark
    COMMAND ./benchmark_engines
    DEPENDS benchmark_engines
    COMMENT "Running performance benchmarks"
)

# Clean redundant files target
add_custom_target(clean_bloat
    COMMAND ${CMAKE_COMMAND} -E echo "Removing redundant ASSEMBLY files..."
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_SOURCE_DIR}/../test_suite.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../benchmark_suite.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../performance_analysis.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../performance_analyzer.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../hardware_validation_tests.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../build_simple.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/../build_integrated_os.bat
    COMMENT "Cleaning up redundant ASSEMBLY files"
)

# Print configuration
message(STATUS "QuantoniumOS Engine Configuration:")
message(STATUS "  Crypto Engine: ${BUILD_CRYPTO_ENGINE}")
message(STATUS "  Quantum Engine: ${BUILD_QUANTUM_ENGINE}")
message(STATUS "  OS Engine: ${BUILD_OS_ENGINE}")
message(STATUS "  Python Bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  AVX2 Support: ${ENABLE_AVX2}")
message(STATUS "  AES-NI Support: ${ENABLE_AES_NI}")
message(STATUS "  Target Crypto Throughput: 9.2 MB/s")
message(STATUS "  Compiler Optimizations: ${OPTIMIZE_FOR_SPEED}")
