// Simplified DLL wrapper for engine_core functions
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../include/engine_core.h"

// DLL entry point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// Export the main functions with simple C linkage
extern "C" __declspec(dllexport) int engine_init() {
    return ::engine_init();
}

extern "C" __declspec(dllexport) int rft_basis_forward(
    const double* input_real, int input_size,
    double* output_real, double* output_imag) {
    return ::rft_basis_forward(input_real, input_size, output_real, output_imag);
}

extern "C" __declspec(dllexport) int rft_basis_inverse(
    const double* input_real, const double* input_imag, int input_size,
    double* output_real) {
    return ::rft_basis_inverse(input_real, input_imag, input_size, output_real);
}

extern "C" __declspec(dllexport) int rft_operator_apply(
    const double* input_real, const double* input_imag, int input_size,
    double* output_real, double* output_imag) {
    return ::rft_operator_apply(input_real, input_imag, input_size, output_real, output_imag);
}
