cmake_minimum_required(VERSION 3.18)
project(rftmw_native VERSION 1.0.0 LANGUAGES CXX C)

# ============================================================================
# Build Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")

# SIMD flags - auto-detect and enable
include(CheckCXXCompilerFlag)

# Check for AVX2
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 support detected")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
endif()

# Check for AVX-512
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    message(STATUS "AVX-512 support detected")
endif()

# FMA (fused multiply-add)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_FMA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
endif()

# Fast math (use carefully - may affect precision)
option(ENABLE_FAST_MATH "Enable fast-math optimizations" OFF)
if(ENABLE_FAST_MATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
endif()

# Sanitizers (debug/verification)
set(RFTMW_SANITIZER "none" CACHE STRING "Sanitizer: none,address,thread,undefined")
set_property(CACHE RFTMW_SANITIZER PROPERTY STRINGS none address thread undefined)
if(NOT RFTMW_SANITIZER STREQUAL "none")
    message(STATUS "Sanitizer enabled: ${RFTMW_SANITIZER}")
    set(SAN_FLAGS "-fno-omit-frame-pointer")
    if(RFTMW_SANITIZER STREQUAL "address")
        set(SAN_FLAGS "${SAN_FLAGS} -fsanitize=address")
    elseif(RFTMW_SANITIZER STREQUAL "thread")
        set(SAN_FLAGS "${SAN_FLAGS} -fsanitize=thread")
    elseif(RFTMW_SANITIZER STREQUAL "undefined")
        set(SAN_FLAGS "${SAN_FLAGS} -fsanitize=undefined")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_FLAGS}")
    set(ENABLE_LTO OFF CACHE BOOL "Enable Link-Time Optimization" FORCE)
endif()

# Link-time optimization
option(ENABLE_LTO "Enable Link-Time Optimization" ON)
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_OUTPUT)
    if(LTO_SUPPORTED)
        message(STATUS "LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "LTO not supported: ${LTO_OUTPUT}")
    endif()
endif()

# ============================================================================
# Check for NASM Assembler (Optional)
# ============================================================================

find_program(NASM_EXECUTABLE nasm)
if(NASM_EXECUTABLE)
    message(STATUS "Found NASM: ${NASM_EXECUTABLE}")
    enable_language(ASM_NASM)
    set(HAVE_NASM TRUE)
else()
    message(STATUS "NASM not found - ASM kernels will use C fallbacks")
    set(HAVE_NASM FALSE)
endif()

# ============================================================================
# ASM Kernel Integration (from algorithms/rft/kernels/)
# ============================================================================

option(RFTMW_ENABLE_ASM "Enable assembly kernel integration" ON)

# Path to existing kernel infrastructure
set(KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../algorithms/rft/kernels")

if(RFTMW_ENABLE_ASM AND EXISTS "${KERNELS_DIR}")
    message(STATUS "ASM kernel integration ENABLED")
    message(STATUS "  Kernels directory: ${KERNELS_DIR}")
    
    add_definitions(-DRFTMW_ENABLE_ASM=1)
    
    # Include directories from existing kernel infrastructure
    set(KERNEL_INCLUDE_DIRS
        ${KERNELS_DIR}/include
        ${KERNELS_DIR}/kernel
        ${KERNELS_DIR}/engines/crypto/include
        ${KERNELS_DIR}/engines/orchestrator/include
    )
    
    # C source files from kernel infrastructure (always available)
    set(KERNEL_C_SOURCES_CANDIDATE
        ${KERNELS_DIR}/kernel/rft_kernel.c
        ${KERNELS_DIR}/kernel/rft_kernel_ui.c
        ${KERNELS_DIR}/kernel/quantum_symbolic_compression.c
        ${KERNELS_DIR}/engines/crypto/src/feistel_round48.c
        ${KERNELS_DIR}/engines/crypto/src/sha256_portable.c
        ${KERNELS_DIR}/engines/crypto/src/rft_sis.c
    )
    
    # Check which C sources exist
    set(KERNEL_C_SOURCES "")
    foreach(src ${KERNEL_C_SOURCES_CANDIDATE})
        if(EXISTS ${src})
            message(STATUS "  Found C source: ${src}")
            list(APPEND KERNEL_C_SOURCES ${src})
        else()
            message(STATUS "  Missing C source: ${src}")
        endif()
    endforeach()
    
    # Assembly source files (only if NASM available)
    # NOTE: Some ASM files have syntax issues, only include validated ones
    set(KERNEL_ASM_SOURCES "")
    if(HAVE_NASM)
        set(KERNEL_ASM_SOURCES_CANDIDATE
            ${KERNELS_DIR}/kernel/rft_kernel_asm.asm
            ${KERNELS_DIR}/kernel/quantum_symbolic_compression.asm
            # These have addressing mode issues, skip for now:
            # ${KERNELS_DIR}/engines/crypto/asm/feistel_round48.asm
            # ${KERNELS_DIR}/engines/orchestrator/asm/rft_transform.asm
        )
        
        foreach(src ${KERNEL_ASM_SOURCES_CANDIDATE})
            if(EXISTS ${src})
                message(STATUS "  Found ASM source: ${src}")
                list(APPEND KERNEL_ASM_SOURCES ${src})
            else()
                message(STATUS "  Missing ASM source: ${src}")
            endif()
        endforeach()
    else()
        message(STATUS "  Skipping ASM sources (NASM not available)")
    endif()
    
else()
    message(STATUS "ASM kernel integration DISABLED")
    add_definitions(-DRFTMW_ENABLE_ASM=0)
    set(KERNEL_INCLUDE_DIRS "")
    set(KERNEL_C_SOURCES "")
    set(KERNEL_ASM_SOURCES "")
endif()

# ============================================================================
# Find Dependencies
# ============================================================================

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Try to find pybind11 via different methods
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_cmake_dir)
        list(APPEND CMAKE_PREFIX_PATH ${pybind11_cmake_dir})
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endif()

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")

# ============================================================================
# RFTMW Native Library with Kernel Integration
# ============================================================================

# Header-only library target
add_library(rftmw_headers INTERFACE)
target_include_directories(rftmw_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${KERNEL_INCLUDE_DIRS}
)

# Static library with kernel sources
if(KERNEL_C_SOURCES OR KERNEL_ASM_SOURCES)
    add_library(rftmw_kernels STATIC
        ${KERNEL_C_SOURCES}
        ${KERNEL_ASM_SOURCES}
    )
    target_include_directories(rftmw_kernels PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${KERNEL_INCLUDE_DIRS}
    )
    if(UNIX)
        target_link_libraries(rftmw_kernels PUBLIC m)
    endif()
    set_target_properties(rftmw_kernels PROPERTIES
        OUTPUT_NAME rftmw_kernels
    )
else()
    # Create empty library if no kernel sources
    add_library(rftmw_kernels INTERFACE)
endif()

# ============================================================================
# Python Extension Module
# ============================================================================

pybind11_add_module(rftmw_native MODULE
    rftmw_python.cpp
)

target_link_libraries(rftmw_native PRIVATE
    rftmw_headers
    rftmw_kernels
)

target_include_directories(rftmw_native PRIVATE
    ${Python3_NumPy_INCLUDE_DIRS}
    ${KERNEL_INCLUDE_DIRS}
)

if(RFTMW_ENABLE_ASM)
    target_compile_definitions(rftmw_native PRIVATE RFTMW_ENABLE_ASM=1)
endif()

# Install the Python module to site-packages
install(TARGETS rftmw_native
    LIBRARY DESTINATION ${Python3_SITEARCH}
)

# ============================================================================
# Tests (optional)
# ============================================================================

option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_rftmw_core tests/test_core.cpp)
    target_link_libraries(test_rftmw_core PRIVATE rftmw_headers rftmw_kernels)
    add_test(NAME test_rftmw_core COMMAND test_rftmw_core)
    
    add_executable(test_rftmw_compression tests/test_compression.cpp)
    target_link_libraries(test_rftmw_compression PRIVATE rftmw_headers rftmw_kernels)
    add_test(NAME test_rftmw_compression COMMAND test_rftmw_compression)
endif()

# ============================================================================
# Benchmarks (optional)
# ============================================================================

option(BUILD_BENCHMARKS "Build benchmark executables" OFF)

if(BUILD_BENCHMARKS)
    add_executable(bench_rftmw bench/benchmark.cpp)
    target_link_libraries(bench_rftmw PRIVATE rftmw_headers rftmw_kernels)
endif()

# ============================================================================
# Installation
# ============================================================================

# Install headers
install(FILES
    rftmw_core.hpp
    rftmw_compression.hpp
    rftmw_asm_kernels.hpp
    DESTINATION include/rftmw
)

# Generate pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/rftmw.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/rftmw.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rftmw.pc
    DESTINATION lib/pkgconfig
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "RFTMW Native Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  AVX2: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "  AVX-512: ${COMPILER_SUPPORTS_AVX512}")
message(STATUS "  FMA: ${COMPILER_SUPPORTS_FMA}")
message(STATUS "  LTO: ${LTO_SUPPORTED}")
message(STATUS "  Fast math: ${ENABLE_FAST_MATH}")
message(STATUS "  ASM kernels: ${RFTMW_ENABLE_ASM}")
if(RFTMW_ENABLE_ASM)
    message(STATUS "    C sources: ${KERNEL_C_SOURCES}")
    message(STATUS "    ASM sources: ${KERNEL_ASM_SOURCES}")
endif()
message(STATUS "")
