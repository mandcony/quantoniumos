#ifndef SYMBOLIC_EIGENVECTOR_H
#define SYMBOLIC_EIGENVECTOR_H

#include <vector>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Wave number structure representing amplitude and phase
struct WaveNumber {
    double amplitude;
    double phase;
    
    WaveNumber() : amplitude(0.0), phase(0.0) {}
    WaveNumber(double a, double p) : amplitude(a), phase(p) {}
};

// Symbolic eigenvector reduction function for wave filtering
EXPORT int symbolic_eigenvector_reduction(
    WaveNumber* wave_list, 
    int wave_count, 
    double threshold, 
    WaveNumber* output_buffer
);

#endif // SYMBOLIC_EIGENVECTOR_H