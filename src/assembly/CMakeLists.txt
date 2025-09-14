cmake_minimum_required(VERSION 3.12)
project(QuantoniumOS_RFT_Kernel LANGUAGES C ASM_NASM)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Platform-specific settings
if(WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    set(CMAKE_STATIC_LIBRARY_SUFFIX ".lib")
else()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Include directories
include_directories(include)

# Source files
set(C_SOURCES
    kernel/rft_kernel.c
    kernel/rft_kernel_ui.c
)

set(ASM_SOURCES
    kernel/rft_kernel_asm.asm
)

set(HEADERS
    include/rft_kernel.h
    include/rft_kernel_ui.h
)

# Create shared library
add_library(rftkernel SHARED ${C_SOURCES} ${ASM_SOURCES})

# Create static library
add_library(rftkernel_static STATIC ${C_SOURCES} ${ASM_SOURCES})

# Link math library on Unix
if(UNIX)
    target_link_libraries(rftkernel m)
    target_link_libraries(rftkernel_static m)
endif()

# Set target properties
set_target_properties(rftkernel PROPERTIES
    OUTPUT_NAME "rftkernel"
    VERSION 1.0.0
    SOVERSION 1
)

set_target_properties(rftkernel_static PROPERTIES
    OUTPUT_NAME "rftkernel"
    VERSION 1.0.0
)

# Install targets
install(TARGETS rftkernel rftkernel_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include/quantoniumos
)

# Copy to compiled directory for development
add_custom_command(TARGET rftkernel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/compiled
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rftkernel> ${CMAKE_SOURCE_DIR}/compiled/
)

add_custom_command(TARGET rftkernel_static POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rftkernel_static> ${CMAKE_SOURCE_DIR}/compiled/
)

# Test target
enable_testing()
add_test(NAME rft_python_test
    COMMAND python3 unitary_rft.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python_bindings
)

# Documentation target (if Doxygen is available)
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
    )
endif()

# Print build information
message(STATUS "QuantoniumOS RFT Kernel Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Shared library: ${CMAKE_SHARED_LIBRARY_PREFIX}rftkernel${CMAKE_SHARED_LIBRARY_SUFFIX}")
message(STATUS "  Static library: ${CMAKE_STATIC_LIBRARY_PREFIX}rftkernel${CMAKE_STATIC_LIBRARY_SUFFIX}")
