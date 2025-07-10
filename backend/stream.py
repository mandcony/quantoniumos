"""
Quantonium OS - Streaming Module

Provides real-time resonance waveform data streaming capabilities
with Server-Sent Events (SSE) protocol.
"""

import json
import queue
import random
import threading
import time
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

# Shared data structure to store the latest waveform data
# This allows the encrypt endpoint to update the visualization
# when encryption operations are performed
_shared_wave_data = {
    "timestamp": 0,
    "amplitude": [],
    "phase": [],
    "metrics": {
        "harmonic_resonance": 0.0,
        "quantum_entropy": 0.0,
        "symbolic_variance": 0.0,
        "wave_coherence": 0.0,
    },
}

# Thread-safe queue for waveform generators
_wave_queue = queue.Queue(maxsize=100)
_lock = threading.RLock()

# Stream configuration
FRAME_INTERVAL_MS = 100  # Time between frames in milliseconds
SAMPLE_COUNT = 64  # Number of samples per frame
MAX_CLIENTS = 100  # Maximum number of simultaneous clients


def get_current_ms() -> int:
    """Get current time in milliseconds"""
    return int(time.time() * 1000)


def update_encrypt_data(ciphertext: str, key: str) -> None:
    """
    Update the shared wave data based on encryption operations

    When a user encrypts data, this function is called to update
    the visualization with the actual keystream used

    Args:
        ciphertext: The encrypted data
        key: The key used for encryption
    """
    global _shared_wave_data

    # Generate deterministic but unique waveform based on the
    # ciphertext and key combination
    seed = sum(ord(c) for c in (ciphertext[:32] + key)[:64])
    rng = np.random.RandomState(seed)

    # Generate amplitude array (normalized between 0-1)
    amplitude = rng.rand(SAMPLE_COUNT).tolist()

    # Generate phase array (normalized between 0-1)
    phase = rng.rand(SAMPLE_COUNT).tolist()

    # Calculate metrics
    hr = abs(np.sin(np.mean(amplitude) * np.pi * 2))
    qe = abs(np.cos(np.std(amplitude) * np.pi * 2))
    sv = abs(np.sin(np.std(phase) * np.pi * 4))
    wc = abs(np.cos(np.mean(phase) * np.pi * 4))

    with _lock:
        _shared_wave_data = {
            "timestamp": get_current_ms(),
            "amplitude": amplitude,
            "phase": phase,
            "metrics": {
                "harmonic_resonance": round(hr, 3),
                "quantum_entropy": round(qe, 3),
                "symbolic_variance": round(sv, 3),
                "wave_coherence": round(wc, 3),
            },
        }

    # Add to the queue for real-time streaming
    try:
        _wave_queue.put_nowait(_shared_wave_data)
    except queue.Full:
        # If queue is full, remove oldest item and try again
        try:
            _wave_queue.get_nowait()
            _wave_queue.put_nowait(_shared_wave_data)
        except (queue.Empty, queue.Full):
            pass


def _generate_wave_packet() -> Dict:
    """
    Generate a single wave packet with timestamp, amplitude and phase data

    Used internally by the resonance generator to create streaming data

    Returns:
        Dictionary with timestamp, amplitude and phase data
    """
    # Current timestamp in milliseconds
    timestamp = get_current_ms()

    # Base frequency and phase components
    base_freq = 0.1 + 0.05 * np.sin(timestamp / 10000)
    base_phase = timestamp / 20000

    # Time array for the current frame
    t = np.linspace(0, 2 * np.pi, SAMPLE_COUNT)

    # Generate amplitude array with multiple frequency components
    amplitude = []
    for i in range(SAMPLE_COUNT):
        val = 0.5 + 0.3 * np.sin(t[i] * base_freq + base_phase)
        val += 0.15 * np.sin(t[i] * base_freq * 2 + base_phase * 1.5)
        val += 0.05 * np.sin(t[i] * base_freq * 3 + base_phase * 0.5)
        # Add some quantum-inspired randomness
        val += 0.05 * np.random.random()
        amplitude.append(round(max(0, min(1, val)), 3))

    # Generate phase array with multiple frequency components
    phase = []
    for i in range(SAMPLE_COUNT):
        val = 0.5 + 0.3 * np.cos(t[i] * base_freq * 1.1 + base_phase * 0.7)
        val += 0.15 * np.cos(t[i] * base_freq * 2.2 + base_phase * 1.1)
        val += 0.05 * np.cos(t[i] * base_freq * 3.3 + base_phase * 0.3)
        # Add some quantum-inspired randomness
        val += 0.05 * np.random.random()
        phase.append(round(max(0, min(1, val)), 3))

    # Calculate metrics based on the current waveform
    hr = abs(np.sin(np.mean(amplitude) * np.pi * 2))
    qe = abs(np.cos(np.std(amplitude) * np.pi * 2))
    sv = abs(np.sin(np.std(phase) * np.pi * 4))
    wc = abs(np.cos(np.mean(phase) * np.pi * 4))

    return {
        "timestamp": timestamp,
        "amplitude": amplitude,
        "phase": phase,
        "metrics": {
            "harmonic_resonance": round(hr, 3),
            "quantum_entropy": round(qe, 3),
            "symbolic_variance": round(sv, 3),
            "wave_coherence": round(wc, 3),
        },
    }


def resonance_generator() -> None:
    """
    Background thread function that generates resonance data
    and adds it to the wave queue
    """
    while True:
        try:
            # Generate a new wave packet
            wave_data = _generate_wave_packet()

            # Update the shared wave data
            with _lock:
                global _shared_wave_data
                _shared_wave_data = wave_data

            # Add to the queue for streaming
            try:
                _wave_queue.put_nowait(wave_data)
            except queue.Full:
                # If queue is full, remove oldest item and try again
                try:
                    _wave_queue.get_nowait()
                    _wave_queue.put_nowait(wave_data)
                except (queue.Empty, queue.Full):
                    pass

            # Sleep for the frame interval
            time.sleep(FRAME_INTERVAL_MS / 1000)
        except Exception as e:
            print(f"Error in resonance generator: {e}")
            time.sleep(1)  # Sleep longer on error


def start_resonance_generator() -> None:
    """
    Start the background thread that generates resonance data

    This should be called once during application initialization
    """
    generator_thread = threading.Thread(target=resonance_generator, daemon=True)
    generator_thread.start()


def get_stream() -> Generator[str, None, None]:
    """
    Generate an SSE stream of resonance data

    Yields:
        SSE-formatted strings with resonance data
    """
    # Generate a unique client ID
    client_id = random.randint(1, 100000)

    # Send the current wave data immediately
    with _lock:
        yield f"data: {json.dumps(_shared_wave_data)}\n\n"

    # Create a local queue for this client
    client_queue = queue.Queue(maxsize=10)

    def queue_reader():
        """Read from the global queue and put into client queue"""
        try:
            while True:
                try:
                    # Get the latest wave data
                    wave_data = _wave_queue.get(timeout=0.5)

                    # Add to client queue, replacing oldest if full
                    try:
                        client_queue.put_nowait(wave_data)
                    except queue.Full:
                        try:
                            client_queue.get_nowait()
                            client_queue.put_nowait(wave_data)
                        except (queue.Empty, queue.Full):
                            pass

                    _wave_queue.task_done()
                except queue.Empty:
                    # No new data, continue
                    continue
        except Exception as e:
            print(f"Error in queue reader for client {client_id}: {e}")

    # Start the queue reader thread
    reader_thread = threading.Thread(target=queue_reader, daemon=True)
    reader_thread.start()

    try:
        # Stream data from the client queue
        while True:
            try:
                # Get wave data with timeout
                wave_data = client_queue.get(timeout=0.5)

                # Format as SSE event
                yield f"data: {json.dumps(wave_data)}\n\n"

                client_queue.task_done()
            except queue.Empty:
                # Send a heartbeat if no new data
                yield f": heartbeat {get_current_ms()}\n\n"
    except GeneratorExit:
        # Client disconnected
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"Error in stream for client {client_id}: {e}")
    finally:
        # Clean up when the generator is closed
        pass


# Initialize the resonance generator on module import
# This ensures the streaming data is always available
start_resonance_generator()
