/**
 * RFT Kernel UI - User Interface Extensions for RFT
 * 
 * This header provides UI-specific extensions and convenience functions
 * for the RFT kernel, making it easier to integrate with desktop applications.
 * 
 * Author: QuantoniumOS Team
 * License: See LICENSE.md
 */

/* SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate license.
 */

#ifndef RFT_KERNEL_UI_H
#define RFT_KERNEL_UI_H

#include "rft_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

// UI-specific configuration
typedef struct {
    bool enable_visualization;
    bool real_time_processing;
    double ui_update_rate;
    size_t max_display_size;
} rft_ui_config_t;

// Extended engine with UI capabilities
typedef struct {
    rft_engine_t base;
    rft_ui_config_t ui_config;
    void* visualization_context;
} rft_ui_engine_t;

// UI API functions
rft_error_t rft_ui_init(rft_ui_engine_t* ui_engine, size_t size, 
                        uint32_t flags, const rft_ui_config_t* ui_config);
rft_error_t rft_ui_cleanup(rft_ui_engine_t* ui_engine);
rft_error_t rft_ui_process_realtime(rft_ui_engine_t* ui_engine,
                                    const rft_complex_t* input,
                                    rft_complex_t* output,
                                    size_t size);

// Visualization helpers
rft_error_t rft_ui_get_spectrum_data(const rft_ui_engine_t* ui_engine,
                                     double* magnitudes, double* phases,
                                     size_t max_points);
rft_error_t rft_ui_get_basis_visualization(const rft_ui_engine_t* ui_engine,
                                           double* basis_data, size_t max_size);

#ifdef __cplusplus
}
#endif

#endif // RFT_KERNEL_UI_H