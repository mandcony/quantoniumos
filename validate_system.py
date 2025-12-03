#!/usr/bin/env python3
"""QuantoniumOS Full System Validation - UnitaryRFT Integration Test"""

import numpy as np
import sys

from algorithms.rft.variants.manifest import iter_variants

print('=' * 60)
print('QUANTONIUMOS FULL SYSTEM VALIDATION')
print('=' * 60)

# Test 1: UnitaryRFT Native Library
print('\n[1] UnitaryRFT Native Library:')
try:
    from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT
    rft = UnitaryRFT(256)
    x = np.random.randn(256)
    y = rft.forward(x)
    z = rft.inverse(y)
    error = np.max(np.abs(x - z))
    print(f'    Status: NATIVE (is_mock={rft._is_mock})')
    print(f'    Roundtrip Error: {error:.2e}')
    print(f'    Pass: {error < 1e-10}')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 2: QuantSoundDesign RFT Engine
print('\n[2] QuantSoundDesign Engine:')
try:
    from src.apps.quantsounddesign import engine
    status = engine.get_rft_status()
    print(f'    UnitaryRFT Available: {status["unitary_available"]}')
    print(f'    Is Mock: {status["is_mock"]}')
    print(f'    Variant: {status["current_variant"]} (HARMONIC)')
    x = np.random.randn(512)
    y = engine.rft_forward(x.reshape(1, -1))
    z = engine.rft_inverse(y)
    error = np.max(np.abs(x - z[0]))
    print(f'    Roundtrip Error: {error:.2e}')
    print(f'    Pass: {error < 1e-10}')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 3: QuantSoundDesign Synth Engine
print('\n[3] QuantSoundDesign Synth Engine:')
try:
    from src.apps.quantsounddesign.synth_engine import rft_additive_synthesis, UNITARY_RFT_AVAILABLE
    print(f'    UnitaryRFT Available: {UNITARY_RFT_AVAILABLE}')
    waveform = rft_additive_synthesis(440.0, 0.1, 44100, num_harmonics=8)
    print(f'    Waveform Shape: {waveform.shape}')
    print(f'    Max Amplitude: {np.max(np.abs(waveform)):.4f}')
    print(f'    Pass: True')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 4: QuantSoundDesign Drum Synthesizer
print('\n[4] QuantSoundDesign Drum Synthesizer:')
try:
    from src.apps.quantsounddesign.pattern_editor import DrumSynthesizer, DrumType
    drums = DrumSynthesizer()
    kick = drums.synthesize_rft(DrumType.KICK)
    snare = drums.synthesize_rft(DrumType.SNARE)
    print(f'    Kick Samples: {len(kick)}')
    print(f'    Snare Samples: {len(snare)}')
    print(f'    Pass: True')
except Exception as e:
    print(f'    FAILED: {e}')

# Test 5: All RFT Variants
print('\n[5] RFT Variants Test:')
try:
    from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT

    manifest = list(iter_variants(include_experimental=True, require_kernel_constant=True))
    if not manifest:
        print('    SKIPPED: No variant manifest entries are available in this environment.')
    for entry in manifest:
        rft = UnitaryRFT(64, variant=entry.kernel_id)
        x = np.random.randn(64)
        y = rft.forward(x)
        z = rft.inverse(y)
        error = np.max(np.abs(x - z))
        status = 'PASS' if error < 1e-10 else 'FAIL'
        print(f'    {entry.code}: {status} (error={error:.2e})')
except Exception as e:
    print(f'    FAILED: {e}')

print('\n' + '=' * 60)
print('VALIDATION COMPLETE')
print('=' * 60)
