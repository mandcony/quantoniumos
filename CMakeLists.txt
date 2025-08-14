cmake_minimum_required(VERSION 3.12)
project(QuantoniumOS VERSION 0.3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)

# Set Eigen3 target manually since we're using the bundled version
set(EIGEN3_FOUND TRUE)
set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL -DEIGEN_VECTORIZE_SSE4_2")

# Core engine source files
set(CORE_SOURCES
    core/engine_core.cpp
    core/symbolic_eigenvector.cpp
)

# Create shared library for C bindings
add_library(quantonium_engine SHARED ${CORE_SOURCES})
target_link_libraries(quantonium_engine Eigen3::Eigen)
set_target_properties(quantonium_engine PROPERTIES
    PREFIX ""
    OUTPUT_NAME "quantonium_engine"
    SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

# Create pybind11 module for Python bindings
pybind11_add_module(quantonium_core
    core/pybind_interface.cpp
    core/engine_core.cpp
    core/symbolic_eigenvector.cpp
)

target_link_libraries(quantonium_core PRIVATE Eigen3::Eigen)
target_compile_definitions(quantonium_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Set properties for Python module
set_target_properties(quantonium_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME "quantonium_core"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Create additional specialized modules
pybind11_add_module(resonance_engine
    core/resonance_engine_bindings.cpp
    core/engine_core.cpp
    core/symbolic_eigenvector.cpp
)

target_link_libraries(resonance_engine PRIVATE Eigen3::Eigen)
target_compile_definitions(resonance_engine PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Quantum module
pybind11_add_module(quantum_engine
    core/quantum_engine_bindings.cpp
    core/engine_core.cpp
    core/symbolic_eigenvector.cpp
)

target_link_libraries(quantum_engine PRIVATE Eigen3::Eigen)
target_compile_definitions(quantum_engine PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Install targets
install(TARGETS quantonium_engine DESTINATION lib)
install(TARGETS quantonium_core DESTINATION ${Python3_SITEARCH})
install(TARGETS resonance_engine DESTINATION ${Python3_SITEARCH})
install(TARGETS quantum_engine DESTINATION ${Python3_SITEARCH})

# Copy Python modules to build directory for development
add_custom_command(TARGET quantonium_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quantonium_core> ${CMAKE_SOURCE_DIR}/core/
)

add_custom_command(TARGET resonance_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:resonance_engine> ${CMAKE_SOURCE_DIR}/core/
)

add_custom_command(TARGET quantum_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quantum_engine> ${CMAKE_SOURCE_DIR}/core/
)
