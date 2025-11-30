"""
Test the hardened audio backend.
"""
import time
from src.apps.quantsounddesign.audio_backend import (
    AudioSettings, LatencyMode, AudioBackend, 
    get_audio_devices, get_default_device_info, PerformanceStats
)


def test_audio_backend():
    print("=" * 70)
    print("  AUDIO BACKEND HARDENING TEST")
    print("=" * 70)

    # Test AudioSettings
    print("\n📋 Audio Settings Presets:")
    for mode in LatencyMode:
        settings = AudioSettings.from_latency_mode(mode, sample_rate=48000)
        latency = settings.get_latency_ms()
        print(f"   {mode.value:12} -> {settings.buffer_size:4} samples, {latency:.1f}ms latency")

    # Test device enumeration
    print("\n🎧 Audio Devices:")
    info = get_default_device_info()
    if info.get("available"):
        print(f"   Default input:  {info.get('default_input')}")
        print(f"   Default output: {info.get('default_output')}")
        inputs, outputs = get_audio_devices()
        print(f"   Available inputs:  {len(inputs)}")
        print(f"   Available outputs: {len(outputs)}")
    else:
        print("   Audio devices not available")

    # Test serialization
    print("\n💾 Settings Serialization:")
    settings = AudioSettings(
        sample_rate=96000,
        buffer_size=128,
        latency_mode=LatencyMode.LOW
    )
    data = settings.to_dict()
    restored = AudioSettings.from_dict(data)
    print(f"   Original:  {settings.sample_rate}Hz, {settings.buffer_size} samples")
    print(f"   Restored:  {restored.sample_rate}Hz, {restored.buffer_size} samples")
    match = settings.sample_rate == restored.sample_rate and settings.buffer_size == restored.buffer_size
    print(f"   Match: {match}")

    # Test PerformanceStats
    print("\n📊 Performance Stats:")
    stats = PerformanceStats()
    for i in range(10):
        stats.record_callback(100 + i * 10)
    stats.buffer_time_us = 5333
    print(f"   Avg callback time: {stats.callback_time_us:.1f} µs")
    print(f"   Max callback time: {stats.max_callback_time_us:.1f} µs")
    print(f"   CPU load: {stats.cpu_load_percent:.1f}%")

    # Test AudioBackend initialization
    print("\n🔊 AudioBackend Test:")
    settings = AudioSettings.from_latency_mode(LatencyMode.BALANCED)
    backend = AudioBackend(settings=settings)
    print(f"   Settings: {backend.settings.sample_rate}Hz, {backend.settings.buffer_size} samples")
    print(f"   Latency: {backend.settings.get_latency_ms():.1f}ms")
    print(f"   Round-trip: {backend.settings.get_roundtrip_latency_ms():.1f}ms")

    # Start the backend briefly
    if backend.start():
        time.sleep(0.5)
        stats = backend.get_stats()
        print(f"   Callbacks: {stats.callbacks_total}")
        print(f"   XRuns: {stats.underruns} underruns, {stats.overruns} overruns")
        backend.stop()
    else:
        print("   Could not start audio backend")

    print("\n" + "=" * 70)
    print("  ✅ AUDIO BACKEND TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_audio_backend()
