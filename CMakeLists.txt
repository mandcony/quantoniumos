cmake_minimum_required(VERSION 3.15)
project(QuantoniumOS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force static linking to avoid path issues
set(BUILD_SHARED_LIBS OFF)

# Pin Eigen to the local copy to avoid path flakiness
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Eigen/eigen-3.4.0")
set(CMAKE_PREFIX_PATH "${EIGEN3_INCLUDE_DIR}")

# Ensure we find our pinned Eigen
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # Fallback: use our bundled Eigen directly
    message(STATUS "Using bundled Eigen from: ${EIGEN3_INCLUDE_DIR}")
    add_library(eigen INTERFACE)
    target_include_directories(eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
    add_library(Eigen3::Eigen ALIAS eigen)
endif()

# Find OpenMP with fallback
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found - continuing without parallel support")
    # Create dummy OpenMP target
    add_library(openmp_dummy INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_dummy)
endif()

# Include directories for our custom headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/secure_core/include")

# Core library for symbolic operations
add_library(symbolic_core
    secure_core/src/symbolic_eigenvector.cpp
)

target_include_directories(symbolic_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/secure_core/include"
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(symbolic_core PUBLIC 
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(symbolic_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Test executable
add_executable(test_geowave_kat
    tests/test_geowave_kat.cpp
)

target_link_libraries(test_geowave_kat PRIVATE
    symbolic_core
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_geowave_kat PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compiler-specific flags for robustness
if(MSVC)
    target_compile_options(test_geowave_kat PRIVATE /W4 /EHsc)
    # Disable specific warnings that Eigen triggers
    target_compile_options(test_geowave_kat PRIVATE /wd4127 /wd4996)
else()
    target_compile_options(test_geowave_kat PRIVATE -Wall -Wextra -O2)
    # Disable Eigen-specific warnings
    target_compile_options(test_geowave_kat PRIVATE -Wno-deprecated-declarations)
endif()

# Test target
enable_testing()
add_test(NAME symbolic_tests COMMAND test_geowave_kat)

# Custom target for quick build verification
add_custom_target(verify_build
    COMMAND ${CMAKE_COMMAND} --build . --target test_geowave_kat
    COMMENT "Verifying C++ build can complete"
)

# Install rules (optional)
install(TARGETS test_geowave_kat
    RUNTIME DESTINATION bin
)

install(DIRECTORY secure_core/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
