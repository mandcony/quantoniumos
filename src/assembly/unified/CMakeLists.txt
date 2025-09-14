cmake_minimum_required(VERSION 3.15)
project(UnifiedAssembly VERSION 1.0.0 LANGUAGES C)

# Configuration
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(UNIFIED_SOURCES
    kernel/unified_orchestrator.c
)

# Create shared library
add_library(unifiedorchestrator SHARED ${UNIFIED_SOURCES})

# Link libraries
target_link_libraries(unifiedorchestrator 
    Threads::Threads
)

# Compiler flags
if(MSVC)
    target_compile_options(unifiedorchestrator PRIVATE /W4)
else()
    target_compile_options(unifiedorchestrator PRIVATE -Wall -Wextra -O3 -fPIC)
endif()

# Install targets
install(TARGETS unifiedorchestrator
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Copy to build directory for Python bindings
add_custom_command(TARGET unifiedorchestrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:unifiedorchestrator> ${CMAKE_CURRENT_SOURCE_DIR}/python_bindings/
    COMMENT "Copying unified orchestrator library to python_bindings"
)

# Status message
message(STATUS "Unified Assembly Orchestrator - 4th Component")
message(STATUS "  - Eliminates bottlenecks between 3 RFT engines")
message(STATUS "  - Patent-aligned unification architecture")
message(STATUS "  - Intelligent routing and load balancing")
