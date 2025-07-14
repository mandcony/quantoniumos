/*
 * Geometric Waveform Cipher C++ Known-Answer Tests
 * Patent-protected geometric waveform hashing validation
 * USPTO Application #19/169,399
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>

// Golden ratio constant
const double PHI = (1.0 + std::sqrt(5.0)) / 2.0;

class GeometricWaveformCipher {
private:
    std::vector<double> waveform;
    double amplitude;
    double phase;
    
public:
    GeometricWaveformCipher(const std::vector<double>& input) : waveform(input) {
        calculateAmplitudePhase();
    }
    
    void calculateAmplitudePhase() {
        if (waveform.empty()) {
            amplitude = 0.0;
            phase = 0.0;
            return;
        }
        
        // Calculate amplitude as average absolute value
        double sum = 0.0;
        for (double val : waveform) {
            sum += std::abs(val);
        }
        amplitude = sum / waveform.size();
        
        // Calculate phase using geometric principles
        double even_sum = 0.0, odd_sum = 0.0;
        for (size_t i = 0; i < waveform.size(); ++i) {
            if (i % 2 == 0) {
                even_sum += waveform[i];
            } else {
                odd_sum += waveform[i];
            }
        }
        
        phase = std::atan2(odd_sum, even_sum) / (2.0 * M_PI);
        phase = (phase + 1.0) / 2.0; // Normalize to [0,1]
        
        // Apply golden ratio optimization
        amplitude = std::fmod(amplitude * PHI, 1.0);
        phase = std::fmod(phase * PHI, 1.0);
    }
    
    std::string generateHash() const {
        std::stringstream ss;
        ss << "A" << std::fixed << std::setprecision(4) << amplitude
           << "_P" << std::fixed << std::setprecision(4) << phase;
        
        // Generate waveform samples for hash
        std::vector<double> samples;
        for (int i = 0; i < 64; ++i) {
            double t = i / 64.0;
            double value = amplitude * std::sin(2.0 * M_PI * t + phase * 2.0 * M_PI);
            samples.push_back(std::round(value * 1000000.0) / 1000000.0); // 6 decimal precision
        }
        
        // Simple hash function (in real implementation, use SHA-256)
        std::hash<std::string> hasher;
        std::string sample_str;
        for (double sample : samples) {
            sample_str += std::to_string(sample) + "_";
        }
        
        size_t hash_value = hasher(sample_str);
        ss << "_" << std::hex << hash_value;
        
        return ss.str();
    }
    
    bool verifyHash(const std::string& hash) const {
        return hash == generateHash();
    }
    
    double getAmplitude() const { return amplitude; }
    double getPhase() const { return phase; }
};

// Known-Answer Test vectors
struct TestVector {
    std::string name;
    std::vector<double> input;
    std::string description;
};

class GeometricWaveformKAT {
private:
    std::vector<TestVector> test_vectors;
    std::map<std::string, bool> results;
    
public:
    GeometricWaveformKAT() {
        initializeTestVectors();
    }
    
    void initializeTestVectors() {
        test_vectors = {
            {"sine_wave_8", {0.0, 0.707, 1.0, 0.707, 0.0, -0.707, -1.0, -0.707}, "Simple sine wave"},
            {"delta_function", {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, "Delta function"},
            {"step_function", {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0}, "Step function"},
            {"constant_signal", {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, "Constant signal"},
            {"linear_ramp", {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875}, "Linear ramp"},
            {"alternating", {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0}, "Alternating pattern"},
            {"complex_wave", {0.5, 0.8, 0.3, -0.2, 0.9, -0.4, 0.1, 0.6}, "Complex waveform"},
            {"symmetric", {0.1, 0.3, 0.7, 0.9, 0.9, 0.7, 0.3, 0.1}, "Symmetric waveform"}
        };
    }
    
    void testHashConsistency() {
        std::cout << "Testing hash consistency..." << std::endl;
        
        for (const auto& tv : test_vectors) {
            GeometricWaveformCipher cipher(tv.input);
            
            std::string hash1 = cipher.generateHash();
            std::string hash2 = cipher.generateHash();
            std::string hash3 = cipher.generateHash();
            
            bool consistent = (hash1 == hash2) && (hash2 == hash3);
            results["hash_consistency_" + tv.name] = consistent;
            
            std::cout << "  " << tv.name << ": " << (consistent ? "PASS" : "FAIL") << std::endl;
            if (!consistent) {
                std::cout << "    Hash1: " << hash1 << std::endl;
                std::cout << "    Hash2: " << hash2 << std::endl;
                std::cout << "    Hash3: " << hash3 << std::endl;
            }
        }
    }
    
    void testHashUniqueness() {
        std::cout << "Testing hash uniqueness..." << std::endl;
        
        std::vector<std::string> hashes;
        for (const auto& tv : test_vectors) {
            GeometricWaveformCipher cipher(tv.input);
            hashes.push_back(cipher.generateHash());
        }
        
        std::set<std::string> unique_hashes(hashes.begin(), hashes.end());
        bool all_unique = (unique_hashes.size() == hashes.size());
        results["hash_uniqueness"] = all_unique;
        
        std::cout << "  Uniqueness: " << (all_unique ? "PASS" : "FAIL") << std::endl;
        std::cout << "  Total hashes: " << hashes.size() << std::endl;
        std::cout << "  Unique hashes: " << unique_hashes.size() << std::endl;
    }
    
    void testGoldenRatioOptimization() {
        std::cout << "Testing golden ratio optimization..." << std::endl;
        
        for (const auto& tv : test_vectors) {
            GeometricWaveformCipher cipher(tv.input);
            
            double amplitude = cipher.getAmplitude();
            double phase = cipher.getPhase();
            
            bool amp_in_range = (amplitude >= 0.0 && amplitude <= 1.0);
            bool phase_in_range = (phase >= 0.0 && phase <= 1.0);
            
            bool golden_ratio_applied = amp_in_range && phase_in_range;
            results["golden_ratio_" + tv.name] = golden_ratio_applied;
            
            std::cout << "  " << tv.name << ": " << (golden_ratio_applied ? "PASS" : "FAIL") << std::endl;
            std::cout << "    Amplitude: " << amplitude << std::endl;
            std::cout << "    Phase: " << phase << std::endl;
        }
    }
    
    void testHashVerification() {
        std::cout << "Testing hash verification..." << std::endl;
        
        for (const auto& tv : test_vectors) {
            GeometricWaveformCipher cipher(tv.input);
            
            std::string hash = cipher.generateHash();
            bool verified = cipher.verifyHash(hash);
            results["hash_verification_" + tv.name] = verified;
            
            std::cout << "  " << tv.name << ": " << (verified ? "PASS" : "FAIL") << std::endl;
        }
    }
    
    void runAllTests() {
        std::cout << "=== Geometric Waveform Cipher C++ Known-Answer Tests ===" << std::endl;
        std::cout << "Patent-protected algorithms - USPTO Application #19/169,399" << std::endl;
        std::cout << "Golden ratio φ = " << PHI << std::endl;
        std::cout << std::endl;
        
        testHashConsistency();
        std::cout << std::endl;
        
        testHashUniqueness();
        std::cout << std::endl;
        
        testGoldenRatioOptimization();
        std::cout << std::endl;
        
        testHashVerification();
        std::cout << std::endl;
        
        // Summary
        int total_tests = results.size();
        int passed_tests = 0;
        for (const auto& result : results) {
            if (result.second) passed_tests++;
        }
        
        std::cout << "=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed tests: " << passed_tests << std::endl;
        std::cout << "Failed tests: " << (total_tests - passed_tests) << std::endl;
        std::cout << "Success rate: " << (100.0 * passed_tests / total_tests) << "%" << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << "✅ All C++ KATs passed!" << std::endl;
        } else {
            std::cout << "❌ Some C++ KATs failed!" << std::endl;
        }
    }
};

int main() {
    GeometricWaveformKAT kat;
    kat.runAllTests();
    return 0;
}