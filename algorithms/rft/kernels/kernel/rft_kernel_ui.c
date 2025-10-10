#include "rft_kernel_ui.h"
#include "rft_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Initialize RFT UI engine with UI-specific configuration
 */
rft_error_t rft_ui_init(rft_ui_engine_t* ui_engine, size_t size, 
                        uint32_t flags, const rft_ui_config_t* ui_config) {
    if (!ui_engine || !ui_config) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    // Initialize base RFT engine
    rft_error_t result = rft_init(&ui_engine->base, size, flags);
    if (result != RFT_SUCCESS) {
        return result;
    }
    
    // Copy UI configuration
    ui_engine->ui_config = *ui_config;
    ui_engine->visualization_context = NULL;
    
    // Initialize visualization context if needed
    if (ui_config->enable_visualization) {
        // Allocate visualization buffers for real-time RFT visualization
        size_t buffer_size = size * sizeof(double);

        // Allocate buffers for: magnitude, phase, real part, imaginary part
        ui_engine->visualization_context = calloc(4, buffer_size);
        if (!ui_engine->visualization_context) {
            rft_cleanup(&ui_engine->base);
            return RFT_ERROR_MEMORY;
        }

        // Initialize buffers to zero
        memset(ui_engine->visualization_context, 0, 4 * buffer_size);
    }
    
    return RFT_SUCCESS;
}

/**
 * Clean up RFT UI engine
 */
rft_error_t rft_ui_cleanup(rft_ui_engine_t* ui_engine) {
    if (!ui_engine) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    // Clean up visualization context
    if (ui_engine->visualization_context) {
        free(ui_engine->visualization_context);
        ui_engine->visualization_context = NULL;
    }
    
    // Clean up base engine
    return rft_cleanup(&ui_engine->base);
}

/**
 * Process real-time data with UI updates
 */
rft_error_t rft_ui_process_realtime(rft_ui_engine_t* ui_engine,
                                    const rft_complex_t* input,
                                    rft_complex_t* output,
                                    size_t size) {
    if (!ui_engine || !input || !output) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    // Perform forward transform
    rft_error_t result = rft_forward(&ui_engine->base, input, output, size);
    if (result != RFT_SUCCESS) {
        return result;
    }
    
    // Update visualization data if enabled
    if (ui_engine->ui_config.enable_visualization && ui_engine->visualization_context) {
        double* viz_data = (double*)ui_engine->visualization_context;
        
        // Store magnitude and phase data for visualization
        for (size_t i = 0; i < size; i++) {
            viz_data[i*2] = sqrt(output[i].real * output[i].real + output[i].imag * output[i].imag); // Magnitude
            viz_data[i*2 + 1] = atan2(output[i].imag, output[i].real); // Phase
        }
    }
    
    return RFT_SUCCESS;
}

/**
 * Get spectrum data for visualization
 */
rft_error_t rft_ui_get_spectrum_data(const rft_ui_engine_t* ui_engine,
                                     double* magnitudes, double* phases,
                                     size_t max_points) {
    if (!ui_engine || !magnitudes || !phases) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    if (!ui_engine->visualization_context) {
        return RFT_ERROR_NOT_INITIALIZED;
    }
    
    double* viz_data = (double*)ui_engine->visualization_context;
    size_t copy_size = (max_points < ui_engine->base.size) ? max_points : ui_engine->base.size;
    
    // Copy magnitude and phase data
    for (size_t i = 0; i < copy_size; i++) {
        magnitudes[i] = viz_data[i*2];     // Magnitude
        phases[i] = viz_data[i*2 + 1];     // Phase
    }
    
    return RFT_SUCCESS;
}

/**
 * Get basis visualization data
 */
rft_error_t rft_ui_get_basis_visualization(const rft_ui_engine_t* ui_engine,
                                           double* basis_data, size_t max_size) {
    if (!ui_engine || !basis_data) {
        return RFT_ERROR_INVALID_PARAM;
    }
    
    if (!ui_engine->base.initialized) {
        return RFT_ERROR_NOT_INITIALIZED;
    }
    
    const size_t N = ui_engine->base.size;
    size_t copy_size = (max_size < N * N) ? max_size : N * N;
    
    // Copy basis matrix magnitudes for visualization
    for (size_t i = 0; i < copy_size; i++) {
        rft_complex_t basis_elem = ui_engine->base.basis[i];
        basis_data[i] = sqrt(basis_elem.real * basis_elem.real + basis_elem.imag * basis_elem.imag);
    }
    
    return RFT_SUCCESS;
}