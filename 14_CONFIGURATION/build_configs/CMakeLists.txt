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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_VECTORIZE_SSE4_2")

# Core engine source files (check if these exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/engine_core.cpp")
    set(CORE_SOURCES core/engine_core.cpp)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/symbolic_eigenvector.cpp")
        list(APPEND CORE_SOURCES core/symbolic_eigenvector.cpp)
    endif()
    
    # Create shared library for C bindings
    add_library(quantonium_engine SHARED ${CORE_SOURCES})
    target_link_libraries(quantonium_engine Eigen3::Eigen)
    set_target_properties(quantonium_engine PROPERTIES
        PREFIX ""
        OUTPUT_NAME "quantonium_engine"
        SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
endif()

# Enhanced RFT Crypto module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/enhanced_rft_crypto.cpp")
    pybind11_add_module(enhanced_rft_crypto 
        core/enhanced_rft_crypto.cpp
    )
    target_link_libraries(enhanced_rft_crypto PRIVATE Eigen3::Eigen)
    target_compile_definitions(enhanced_rft_crypto PRIVATE VERSION_INFO="${PROJECT_VERSION}")
    
    # Set additional optimization flags for the crypto module
    target_compile_options(enhanced_rft_crypto PRIVATE 
        -O3 -march=native -funroll-loops -fomit-frame-pointer
    )
    
    # Copy to source directory for development
    add_custom_command(TARGET enhanced_rft_crypto POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:enhanced_rft_crypto> ${CMAKE_SOURCE_DIR}/
    )
endif()

# True RFT Engine module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/04_RFT_ALGORITHMS/true_rft_engine.cpp" AND 
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/04_RFT_ALGORITHMS/true_rft_engine_bindings.cpp")
    pybind11_add_module(true_rft_engine 
        04_RFT_ALGORITHMS/true_rft_engine_bindings.cpp
        04_RFT_ALGORITHMS/true_rft_engine.cpp
    )
    target_include_directories(true_rft_engine PRIVATE 04_RFT_ALGORITHMS core/include)
    target_link_libraries(true_rft_engine PRIVATE Eigen3::Eigen)
    target_compile_definitions(true_rft_engine PRIVATE VERSION_INFO="${PROJECT_VERSION}")
    
    # Copy to source directory for development
    add_custom_command(TARGET true_rft_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:true_rft_engine> ${CMAKE_SOURCE_DIR}/
    )
endif()

# Resonance engine module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/resonance_engine_bindings.cpp")
    pybind11_add_module(resonance_engine 
        core/resonance_engine_bindings.cpp
        core/resonance_engine_simple.cpp
    )
    target_include_directories(resonance_engine PRIVATE core/include)
    target_link_libraries(resonance_engine PRIVATE Eigen3::Eigen)
    target_compile_definitions(resonance_engine PRIVATE VERSION_INFO="${PROJECT_VERSION}")
    
    # Copy to source directory for development
    add_custom_command(TARGET resonance_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:resonance_engine> ${CMAKE_SOURCE_DIR}/
    )
endif()

# Quantonium Core module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/cpp/bindings/pybind_interface.cpp")
    pybind11_add_module(quantonium_core 
        core/cpp/bindings/pybind_interface.cpp
        core/resonance_engine_simple.cpp
    )
    target_include_directories(quantonium_core PRIVATE core/include)
    target_link_libraries(quantonium_core PRIVATE Eigen3::Eigen)
    target_compile_definitions(quantonium_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")
    
    # Copy to source directory for development
    add_custom_command(TARGET quantonium_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quantonium_core> ${CMAKE_SOURCE_DIR}/
    )
endif()

# Quantum Engine module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/cpp/bindings/quantum_engine_bindings.cpp")
    pybind11_add_module(quantum_engine 
        core/cpp/bindings/quantum_engine_bindings.cpp
        core/resonance_engine_simple.cpp
    )
    target_include_directories(quantum_engine PRIVATE core/include)
    target_link_libraries(quantum_engine PRIVATE Eigen3::Eigen)
    target_compile_definitions(quantum_engine PRIVATE VERSION_INFO="${PROJECT_VERSION}")
    
    # Copy to source directory for development
    add_custom_command(TARGET quantum_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quantum_engine> ${CMAKE_SOURCE_DIR}/
    )
endif()

# Engine Core Pybind module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/cpp/bindings/engine_core_pybind.cpp")
    pybind11_add_module(engine_core_pybind 
        core/cpp/bindings/engine_core_pybind.cpp
        core/resonance_engine_simple.cpp
    )
    target_include_directories(engine_core_pybind PRIVATE core/include)
    target_link_libraries(engine_core_pybind PRIVATE Eigen3::Eigen)
    target_compile_definitions(engine_core_pybind PRIVATE VERSION_INFO="${PROJECT_VERSION}")
    
    # Copy to source directory for development
    add_custom_command(TARGET engine_core_pybind POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:engine_core_pybind> ${CMAKE_SOURCE_DIR}/
    )
endif()

# Vertex Engine module - NEW ADDITION FOR QUANTUM VERTEX OPERATIONS
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/vertex_engine_bindings.cpp")
    pybind11_add_module(vertex_engine 
        core/vertex_engine_bindings.cpp
    )
    target_include_directories(vertex_engine PRIVATE core/include)
    target_link_libraries(vertex_engine PRIVATE Eigen3::Eigen)
    target_compile_definitions(vertex_engine PRIVATE VERSION_INFO="${PROJECT_VERSION}")
    
    # Copy to source directory for development
    add_custom_command(TARGET vertex_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vertex_engine> ${CMAKE_SOURCE_DIR}/
    )
endif()
