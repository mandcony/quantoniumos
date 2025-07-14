#include "../include/symbolic_eigenvector.h"

// Implements the symbolic eigenvector reduction.
// It filters out waves with amplitude above a threshold and merges the rest.
EXPORT int symbolic_eigenvector_reduction(
    WaveNumber* wave_list, 
    int wave_count, 
    double threshold, 
    WaveNumber* output_buffer
) {
    std::vector<WaveNumber> filtered;
    double merged_amp = 0.0;
    double merged_phase = 0.0;
    int merged_count = 0;

    for (int i = 0; i < wave_count; ++i) {
        if (wave_list[i].amplitude >= threshold)
            filtered.push_back(wave_list[i]);
        else {
            merged_amp += wave_list[i].amplitude;
            merged_phase += wave_list[i].phase;
            merged_count++;
        }
    }

    if (merged_count > 0)
        filtered.emplace_back(merged_amp, merged_phase / merged_count);

    // Copy the filtered results into the output buffer.
    for (size_t i = 0; i < filtered.size(); ++i)
        output_buffer[i] = filtered[i];

    return static_cast<int>(filtered.size());
}