cmake_minimum_required(VERSION 3.15)
project(QuantoniumOS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force static linking to avoid path issues
set(BUILD_SHARED_LIBS OFF)

# Try to find system Eigen first
find_package(Eigen3 QUIET)

# If system Eigen not found, look for local copy
if(NOT Eigen3_FOUND)
    # Try local copy
    set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Eigen/eigen-3.4.0")
    if(EXISTS "${EIGEN3_INCLUDE_DIR}")
        message(STATUS "Using local Eigen from: ${EIGEN3_INCLUDE_DIR}")
        set(CMAKE_PREFIX_PATH "${EIGEN3_INCLUDE_DIR}")
        find_package(Eigen3 QUIET)
    endif()
    
    # If still not found, create a stub Eigen interface
    if(NOT Eigen3_FOUND)
        message(STATUS "Eigen not found in system or locally, creating stub interface library")
        add_library(eigen INTERFACE)
        
        # For Windows, we'll just use a minimal interface with our headers
        # This avoids download issues that can cause CI failures
        if(WIN32 OR DEFINED ENV{SKIP_EIGEN_DOWNLOAD})
            message(STATUS "Windows build or SKIP_EIGEN_DOWNLOAD set: Using minimal Eigen stub")
            target_include_directories(eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/secure_core/include)
        # For other platforms in CI, try downloading but with fallback
        elseif(DEFINED ENV{CI})
            message(STATUS "Attempting to download Eigen (non-Windows CI)")
            include(FetchContent)
            set(FETCHCONTENT_QUIET OFF)
            
            # Set a timeout for the download to prevent hanging
            set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
            set(FETCHCONTENT_QUIET OFF)
            
            # Try to download but catch any failures
            try_compile(EIGEN_DOWNLOAD_SUCCESS ${CMAKE_BINARY_DIR}/eigen_test 
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_eigen_download.cmake
                OUTPUT_VARIABLE EIGEN_DOWNLOAD_OUTPUT)
                
            if(EIGEN_DOWNLOAD_SUCCESS)
                FetchContent_Declare(
                    eigen
                    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
                    TIMEOUT 60  # 60 second timeout
                )
                FetchContent_MakeAvailable(eigen)
                target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})
                message(STATUS "Successfully downloaded Eigen")
            else()
                message(WARNING "Failed to download Eigen, using minimal stub instead")
                target_include_directories(eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/secure_core/include)
            endif()
        else()
            # For local build, just create an interface
            target_include_directories(eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/secure_core/include)
        endif()
        
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
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
    # Use separate calls to avoid issues with flag parsing
    target_compile_options(symbolic_core PRIVATE /W3)
    target_compile_options(symbolic_core PRIVATE /EHsc)
    target_compile_options(test_geowave_kat PRIVATE /W3)
    target_compile_options(test_geowave_kat PRIVATE /EHsc)
    # Disable specific warnings that Eigen triggers
    target_compile_options(test_geowave_kat PRIVATE /wd4127)
    target_compile_options(test_geowave_kat PRIVATE /wd4996)
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
