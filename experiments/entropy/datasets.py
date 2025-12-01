#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Dataset Loaders for Shannon Entropy & Compression Benchmarks
=============================================================

Provides consistent, reusable data loading for entropy gap experiments.
Each loader returns numpy arrays suitable for transform/compression testing.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Tuple, Generator
import numpy as np

# Project root detection
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_DATA_ROOT = _PROJECT_ROOT / "data" / "entropy"


def _ensure_data_dir(subdir: str) -> Path:
    """Ensure data directory exists and return path."""
    path = _DATA_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# ASCII Text Corpus
# =============================================================================

def load_ascii_corpus(max_bytes: int = 100_000) -> np.ndarray:
    """
    Load ASCII text corpus as byte array.
    
    Returns:
        np.ndarray: uint8 array of ASCII bytes (0-127 range typical)
    """
    data_dir = _ensure_data_dir("ascii")
    all_bytes = bytearray()
    
    # Try to load existing files
    for f in sorted(data_dir.glob("*.txt")):
        with open(f, "rb") as fp:
            all_bytes.extend(fp.read())
        if len(all_bytes) >= max_bytes:
            break
    
    # Generate synthetic if no files found
    if len(all_bytes) < 1000:
        # Generate English-like text with realistic character distribution
        import string
        rng = np.random.default_rng(42)
        # Approximate English letter frequencies
        letters = list(string.ascii_lowercase)
        weights = [0.082, 0.015, 0.028, 0.043, 0.127, 0.022, 0.020, 0.061,  # a-h
                   0.070, 0.002, 0.008, 0.040, 0.024, 0.067, 0.075, 0.019,  # i-p
                   0.001, 0.060, 0.063, 0.091, 0.028, 0.010, 0.024, 0.002,  # q-x
                   0.020, 0.001]  # y-z
        weights = np.array(weights)
        weights /= weights.sum()
        
        # Generate words and sentences
        words = []
        for _ in range(max_bytes // 5):
            word_len = rng.integers(2, 10)
            word = ''.join(rng.choice(letters, size=word_len, p=weights))
            words.append(word)
        
        text = ' '.join(words)
        # Add punctuation
        text = text.replace('  ', '. ').replace('   ', '! ')
        all_bytes = text.encode('ascii', errors='ignore')[:max_bytes]
    
    return np.frombuffer(bytes(all_bytes[:max_bytes]), dtype=np.uint8)


def load_source_code_corpus(max_bytes: int = 100_000) -> np.ndarray:
    """
    Load source code corpus (Python/C/JSON).
    
    Returns:
        np.ndarray: uint8 array of source code bytes
    """
    data_dir = _ensure_data_dir("source_code")
    all_bytes = bytearray()
    
    # Try to load from data dir first
    for pattern in ["*.py", "*.c", "*.json", "*.js"]:
        for f in sorted(data_dir.glob(pattern)):
            with open(f, "rb") as fp:
                all_bytes.extend(fp.read())
            if len(all_bytes) >= max_bytes:
                break
    
    # Fall back to actual project source files
    if len(all_bytes) < 1000:
        src_dirs = [
            _PROJECT_ROOT / "algorithms",
            _PROJECT_ROOT / "quantonium_os_src",
            _PROJECT_ROOT / "src",
        ]
        for src_dir in src_dirs:
            if src_dir.exists():
                for f in sorted(src_dir.rglob("*.py"))[:20]:
                    try:
                        with open(f, "rb") as fp:
                            all_bytes.extend(fp.read())
                    except:
                        pass
                    if len(all_bytes) >= max_bytes:
                        break
    
    return np.frombuffer(bytes(all_bytes[:max_bytes]), dtype=np.uint8)


# =============================================================================
# Audio Data
# =============================================================================

def load_audio_frames(
    block_size: int = 1024,
    sample_rate: int = 44100,
    max_blocks: int = 100
) -> np.ndarray:
    """
    Load audio data as float64 PCM frames.
    
    Returns:
        np.ndarray: Shape (n_blocks, block_size), float64 in [-1, 1]
    """
    data_dir = _ensure_data_dir("audio")
    frames = []
    
    # Try to load WAV files
    for f in sorted(data_dir.glob("*.wav")):
        try:
            import wave
            with wave.open(str(f), 'rb') as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                width = wf.getsampwidth()
                if width == 2:
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
                elif width == 4:
                    audio = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
                else:
                    continue
                
                # Mono conversion
                channels = wf.getnchannels()
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                
                # Split into blocks
                n_full = len(audio) // block_size
                for i in range(min(n_full, max_blocks - len(frames))):
                    frames.append(audio[i*block_size:(i+1)*block_size])
        except Exception:
            pass
        
        if len(frames) >= max_blocks:
            break
    
    # Generate synthetic audio if needed
    if len(frames) < max_blocks:
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, block_size)
        
        while len(frames) < max_blocks:
            # Mix of sine waves with some noise
            freq = rng.uniform(100, 2000)
            harmonics = rng.integers(1, 6)
            signal = np.zeros(block_size)
            for h in range(1, harmonics + 1):
                amp = 1.0 / h
                signal += amp * np.sin(2 * np.pi * freq * h * t + rng.uniform(0, 2*np.pi))
            signal /= np.max(np.abs(signal)) + 1e-10
            signal += rng.normal(0, 0.01, block_size)
            signal = np.clip(signal, -1, 1)
            frames.append(signal)
    
    return np.array(frames[:max_blocks], dtype=np.float64)


# =============================================================================
# Image Data
# =============================================================================

def load_image_blocks(
    block_size: int = 16,
    max_blocks: int = 1000
) -> np.ndarray:
    """
    Load image data as grayscale blocks.
    
    Returns:
        np.ndarray: Shape (n_blocks, block_size, block_size), float64 in [0, 1]
    """
    data_dir = _ensure_data_dir("images")
    blocks = []
    
    # Try to load PNG/JPG files
    for pattern in ["*.png", "*.jpg", "*.jpeg"]:
        for f in sorted(data_dir.glob(pattern)):
            try:
                from PIL import Image
                img = Image.open(f).convert('L')
                arr = np.array(img, dtype=np.float64) / 255.0
                
                # Extract non-overlapping blocks
                h, w = arr.shape
                for i in range(0, h - block_size + 1, block_size):
                    for j in range(0, w - block_size + 1, block_size):
                        blocks.append(arr[i:i+block_size, j:j+block_size])
                        if len(blocks) >= max_blocks:
                            break
                    if len(blocks) >= max_blocks:
                        break
            except Exception:
                pass
            
            if len(blocks) >= max_blocks:
                break
    
    # Generate synthetic if needed
    if len(blocks) < max_blocks:
        rng = np.random.default_rng(42)
        while len(blocks) < max_blocks:
            # Create structured patterns
            pattern_type = rng.integers(0, 4)
            if pattern_type == 0:
                # Gradient
                x = np.linspace(0, 1, block_size)
                y = np.linspace(0, 1, block_size)
                block = np.outer(x, y)
            elif pattern_type == 1:
                # Checkerboard
                x = np.arange(block_size)
                y = np.arange(block_size)
                block = ((x[:, None] + y[None, :]) % 2).astype(np.float64)
            elif pattern_type == 2:
                # Sine pattern
                freq = rng.uniform(1, 4)
                x = np.linspace(0, 2*np.pi*freq, block_size)
                y = np.linspace(0, 2*np.pi*freq, block_size)
                block = 0.5 + 0.5 * np.sin(x[:, None] + y[None, :])
            else:
                # Random noise
                block = rng.random((block_size, block_size))
            
            blocks.append(block)
    
    return np.array(blocks[:max_blocks], dtype=np.float64)


# =============================================================================
# Texture Data
# =============================================================================

def load_texture_blocks(
    block_size: int = 64,
    max_blocks: int = 100
) -> np.ndarray:
    """
    Load or generate texture patterns.
    
    Returns:
        np.ndarray: Shape (n_blocks, block_size), float64
    """
    data_dir = _ensure_data_dir("textures")
    blocks = []
    
    # Generate synthetic textures with various patterns
    rng = np.random.default_rng(42)
    PHI = (1 + np.sqrt(5)) / 2
    
    while len(blocks) < max_blocks:
        pattern_type = rng.integers(0, 6)
        
        if pattern_type == 0:
            # Pink noise (1/f)
            freqs = np.fft.rfftfreq(block_size)
            spectrum = 1.0 / (freqs + 0.01)
            phases = rng.uniform(0, 2*np.pi, len(spectrum))
            block = np.fft.irfft(spectrum * np.exp(1j * phases), block_size)
        
        elif pattern_type == 1:
            # Fractal Brownian motion
            block = np.cumsum(rng.normal(0, 1, block_size))
        
        elif pattern_type == 2:
            # Periodic with noise
            t = np.linspace(0, 4*np.pi, block_size)
            block = np.sin(t) + 0.3 * np.sin(3*t) + 0.1 * rng.normal(0, 1, block_size)
        
        elif pattern_type == 3:
            # Step function with noise
            block = np.zeros(block_size)
            n_steps = rng.integers(3, 10)
            for _ in range(n_steps):
                pos = rng.integers(0, block_size)
                block[pos:] += rng.uniform(-1, 1)
            block += 0.1 * rng.normal(0, 1, block_size)
        
        elif pattern_type == 4:
            # Sawtooth
            t = np.linspace(0, 4, block_size)
            block = 2 * (t - np.floor(t + 0.5))
        
        else:
            # White noise
            block = rng.normal(0, 1, block_size)
        
        # Normalize
        block = (block - block.mean()) / (block.std() + 1e-10)
        blocks.append(block)
    
    return np.array(blocks[:max_blocks], dtype=np.float64)


# =============================================================================
# Golden Ratio Signals (RFT's "home turf")
# =============================================================================

def load_golden_signals(
    block_size: int = 256,
    max_blocks: int = 100
) -> np.ndarray:
    """
    Generate golden-ratio and Fibonacci-based signals.
    These are designed to be maximally sparse in the RFT domain.
    
    Returns:
        np.ndarray: Shape (n_blocks, block_size), float64
    """
    blocks = []
    rng = np.random.default_rng(42)
    PHI = (1 + np.sqrt(5)) / 2
    
    while len(blocks) < max_blocks:
        pattern_type = rng.integers(0, 8)
        t = np.arange(block_size, dtype=np.float64)
        
        if pattern_type == 0:
            # Golden ratio chirp
            block = np.cos(2 * np.pi * PHI ** (t / block_size * 4))
        
        elif pattern_type == 1:
            # Fibonacci sequence modulated
            fib = [1, 1]
            while len(fib) < block_size:
                fib.append(fib[-1] + fib[-2])
            fib = np.array(fib[:block_size], dtype=np.float64)
            fib = fib % block_size
            block = np.sin(2 * np.pi * fib / block_size)
        
        elif pattern_type == 2:
            # Golden ratio phase modulation
            phase = 2 * np.pi * (t / PHI - np.floor(t / PHI))
            block = np.cos(phase)
        
        elif pattern_type == 3:
            # Sum of golden-ratio harmonics
            block = np.zeros(block_size)
            for k in range(1, 6):
                freq = PHI ** k
                block += np.cos(2 * np.pi * freq * t / block_size) / k
        
        elif pattern_type == 4:
            # Quasi-periodic golden
            block = np.cos(2 * np.pi * t / PHI) + np.cos(2 * np.pi * t / (PHI ** 2))
        
        elif pattern_type == 5:
            # Golden ratio decay
            block = np.exp(-t / (block_size / PHI)) * np.cos(2 * np.pi * t / 10)
        
        elif pattern_type == 6:
            # Fibonacci-tilt (fractional part of k/phi)
            frac_phi = (t / PHI) - np.floor(t / PHI)
            block = np.sin(2 * np.pi * frac_phi * 5)
        
        else:
            # Log-periodic golden
            block = np.cos(2 * np.pi * np.log(t + 1) / np.log(PHI))
        
        # Normalize
        block = (block - block.mean()) / (block.std() + 1e-10)
        blocks.append(block)
    
    return np.array(blocks[:max_blocks], dtype=np.float64)


# =============================================================================
# Unified Dataset Interface
# =============================================================================

DATASET_LOADERS = {
    'ascii': lambda bs, n: load_ascii_corpus(n * bs).reshape(-1, bs)[:n] if bs else load_ascii_corpus(n),
    'source_code': lambda bs, n: load_source_code_corpus(n * bs).reshape(-1, bs)[:n] if bs else load_source_code_corpus(n),
    'audio': lambda bs, n: load_audio_frames(bs, max_blocks=n),
    'images': lambda bs, n: load_image_blocks(int(np.sqrt(bs)), max_blocks=n).reshape(n, -1)[:, :bs],
    'textures': lambda bs, n: load_texture_blocks(bs, max_blocks=n),
    'golden': lambda bs, n: load_golden_signals(bs, max_blocks=n),
}


def load_dataset(
    name: str,
    block_size: int = 256,
    max_blocks: int = 100
) -> np.ndarray:
    """
    Unified dataset loader.
    
    Args:
        name: One of 'ascii', 'source_code', 'audio', 'images', 'textures', 'golden'
        block_size: Size of each block
        max_blocks: Maximum number of blocks to load
    
    Returns:
        np.ndarray: Shape (n_blocks, block_size) or similar
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_LOADERS.keys())}")
    
    return DATASET_LOADERS[name](block_size, max_blocks)


def list_datasets() -> List[str]:
    """Return list of available dataset names."""
    return list(DATASET_LOADERS.keys())


if __name__ == "__main__":
    # Quick test
    print("Testing dataset loaders...")
    for name in list_datasets():
        try:
            data = load_dataset(name, block_size=64, max_blocks=10)
            print(f"  {name}: shape={data.shape}, dtype={data.dtype}, "
                  f"range=[{data.min():.3f}, {data.max():.3f}]")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    print("Done.")
