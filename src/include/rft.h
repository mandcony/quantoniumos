#ifndef RFT_H
#define RFT_H

#include <vector>
#include <complex>

// Forward declarations for functions in rft.cpp
std::vector<std::complex<double>> forward_rft(const std::vector<std::complex<double>>& signal);
std::vector<std::complex<double>> inverse_rft(const std::vector<std::complex<double>>& spectrum);

#endif // RFT_H
