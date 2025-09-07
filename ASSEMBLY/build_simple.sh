#!/bin/bash
# Simplified build script for RFT optimized libraries

echo "Building RFT assembly libraries..."

# Set directories
BUILD_DIR="/workspaces/quantoniumos/ASSEMBLY/optimized"
OUTPUT_DIR="/workspaces/quantoniumos/ASSEMBLY/compiled"
KERNEL_DIR="/workspaces/quantoniumos/ASSEMBLY/kernel"

# Create directories if they don't exist
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy source files
cp "$KERNEL_DIR/rft_kernel_asm.asm" "$BUILD_DIR/simd_rft_core.asm"
cp "$KERNEL_DIR/rft_kernel.c" "$BUILD_DIR/rft_wrapper.c"

# Create header file
cat > "$BUILD_DIR/rft_optimized.h" << EOF
/**
 * RFT Optimized Header
 */
#ifndef RFT_OPTIMIZED_H
#define RFT_OPTIMIZED_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Complex number structure
typedef struct {
    float real;
    float imag;
} rft_complex_t;

// Engine structure
typedef struct {
    size_t size;
    void* internal;
} rft_optimized_engine_t;

// Function declarations
int rft_init(rft_optimized_engine_t* engine, size_t size);
int rft_forward(rft_optimized_engine_t* engine, const rft_complex_t* input, rft_complex_t* output);

#ifdef __cplusplus
}
#endif

#endif /* RFT_OPTIMIZED_H */
EOF

# Assemble
echo "Assembling SIMD core..."
nasm -f elf64 -o "$BUILD_DIR/simd_rft_core.o" "$BUILD_DIR/simd_rft_core.asm"

# Compile C wrapper
echo "Compiling C wrapper..."
gcc -c -fPIC -O3 -I"$BUILD_DIR" -o "$BUILD_DIR/rft_wrapper.o" "$KERNEL_DIR/rft_kernel.c" 

# Link shared library
echo "Creating shared library..."
gcc -shared -fPIC -o "$OUTPUT_DIR/librftoptimized.so" "$BUILD_DIR/simd_rft_core.o" "$BUILD_DIR/rft_wrapper.o" -lm

# Copy header
cp "$BUILD_DIR/rft_optimized.h" "$OUTPUT_DIR/"

echo "Build complete! Library available at: $OUTPUT_DIR/librftoptimized.so"
