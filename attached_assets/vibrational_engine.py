import numpy as np
import pywt
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

class VibrationalEngine:
    def __init__(self, sample_rate=50):
        self.sample_rate = sample_rate  # Hz
        self.data_buffer = []  # Store vibration readings
        self.threshold = 2.0  # Default threshold for anomaly detection

    def apply_highpass_filter(self, signal, cutoff=0.1, order=4):
        """Applies a high-pass Butterworth filter to remove low-frequency noise."""
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, signal)

    def apply_wavelet_transform(self, signal):
        """Decomposes signal using wavelet transform to detect anomalies."""
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        details = coeffs[-1]  # Get high-frequency components
        return np.abs(details)

    def detect_anomalies(self, signal):
        """Detects anomalies using wavelet decomposition and thresholding."""
        processed_signal = self.apply_highpass_filter(signal)
        wavelet_details = self.apply_wavelet_transform(processed_signal)

        # Anomaly when wavelet coefficient is above the dynamic threshold
        anomalies = wavelet_details > self.threshold
        return anomalies, wavelet_details

    def add_vibration_sample(self, sample):
        """Adds new vibration data and detects anomalies in real-time."""
        self.data_buffer.append(sample)

        if len(self.data_buffer) >= self.sample_rate:
            anomalies, wavelet_details = self.detect_anomalies(self.data_buffer[-self.sample_rate:])
            
            if any(anomalies):
                print("⚠️ Structural Anomaly Detected!")

            return anomalies, wavelet_details
        return None, None

    def visualize_wavelet_analysis(self, signal):
        """Plots wavelet transformation details for analysis."""
        _, wavelet_details = self.detect_anomalies(signal)
        plt.figure(figsize=(10, 4))
        plt.plot(wavelet_details, label="Wavelet Detail Coefficients")
        plt.axhline(y=self.threshold, color='r', linestyle='--', label="Anomaly Threshold")
        plt.title("Wavelet Analysis for Vibrational Anomaly Detection")
        plt.xlabel("Time (samples)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.show()

# TESTING THE ENHANCED VIBRATION ENGINE
if __name__ == "__main__":
    engine = VibrationalEngine(sample_rate=50)

    # Simulate normal vibration data
    normal_signal = np.sin(np.linspace(0, 10, 50)) * 0.5

    # Introduce an anomaly at sample 30
    anomaly_signal = normal_signal.copy()
    anomaly_signal[30] += 3.0  # Sudden spike

    # Test anomaly detection
    anomalies, _ = engine.detect_anomalies(anomaly_signal)
    if any(anomalies):
        print("✅ Test Passed: Anomalies detected successfully!")
    else:
        print("❌ Test Failed: No anomalies detected.")

    # Visualize the wavelet analysis
    engine.visualize_wavelet_analysis(anomaly_signal)
