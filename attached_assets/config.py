"""
QuantoniumOS Configuration Module

This module provides configuration settings for the QuantoniumOS desktop applications.
"""

class Config:
    """Configuration class for QuantoniumOS applications."""
    
    # General application settings
    APP_NAME = "QuantoniumOS"
    VERSION = "1.0.0"
    
    # Wave composer settings
    WAVE_COMPOSER = {
        "min_frequency": 0.1,  # Hz
        "max_frequency": 5.0,  # Hz
        "default_frequency": 1.0,  # Hz
        "sample_rate": 44100,  # Hz
        "buffer_size": 1024,  # samples
    }
    
    # Q-Mail settings
    MAIL = {
        "encryption_enabled": True,
        "default_signature": "Sent with QuantoniumOS Secure Mail",
        "max_attachment_size": 10 * 1024 * 1024,  # 10 MB
    }
    
    # Wave debugger settings
    WAVE_DEBUGGER = {
        "max_channels": 8,
        "default_channels": 4,
        "view_duration": 10,  # seconds
    }
    
    # Resonance analyzer settings
    RESONANCE_ANALYZER = {
        "fft_resolution": 2048,
        "window_function": "hann",
        "default_resonance_threshold": 0.75,
    }
    
    # Security settings
    SECURITY = {
        "encryption_algorithm": "resonance",
        "key_length": 32,  # bytes
        "entropy_source": "quantum",
    }