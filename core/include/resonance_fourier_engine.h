#ifndef RESONANCE_FOURIER_ENGINE_H
#define RESONANCE_FOURIER_ENGINE_H

#include <vector>
#include <complex>
#include <string>

class ResonanceFourierEngine {
private:
    std::vector<double> signal_cache;
    std::vector<std::complex<double>> frequency_cache;

public:
    ResonanceFourierEngine() = default;
    ~ResonanceFourierEngine() = default;

    // Forward True RFT implementation
    std::vector<std::complex<double>> forward_true_rft(const std::vector<double>& input_data);

    // Inverse True RFT implementation
    std::vector<double> inverse_true_rft(const std::vector<std::complex<double>>& spectrum_data);

    // Validate roundtrip accuracy
    double validate_roundtrip_accuracy(const std::vector<double>& original, double tolerance = 1e-10);

    // Get quantum amplitudes
    std::vector<double> get_quantum_amplitudes();

    // Status check
    std::string status();
};

#endif // RESONANCE_FOURIER_ENGINE_H
