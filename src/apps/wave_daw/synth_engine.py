"""
Wave DAW Synthesizer Engine - Φ-RFT Native Synthesis

Provides:
- Polyphonic synth with multiple oscillators
- RFT-based wave shaping via UnitaryRFT
- Computer keyboard input (ASDFGHJK = piano)
- MIDI note playback
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY RFT INTEGRATION FOR SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from algorithms.rft.kernels.python_bindings.unitary_rft import (
        UnitaryRFT,
        RFT_FLAG_UNITARY,
        RFT_FLAG_USE_RESONANCE,
        RFT_VARIANT_HARMONIC,
        RFT_VARIANT_FIBONACCI,
    )
    UNITARY_RFT_AVAILABLE = True
    print("[OK] UnitaryRFT connected to Synth Engine")
except ImportError as e:
    UNITARY_RFT_AVAILABLE = False
    print(f"[WARN] UnitaryRFT not available for synth: {e}")

# RFT engine cache for synthesis
_synth_rft_engines: Dict[int, 'UnitaryRFT'] = {}


def get_synth_rft_engine(size: int, variant: int = None):
    """Get or create RFT engine for synthesis."""
    if variant is None:
        variant = RFT_VARIANT_HARMONIC if UNITARY_RFT_AVAILABLE else 1
    
    key = (size, variant)
    if key not in _synth_rft_engines:
        if UNITARY_RFT_AVAILABLE:
            _synth_rft_engines[key] = UnitaryRFT(
                size=size,
                flags=RFT_FLAG_UNITARY | RFT_FLAG_USE_RESONANCE,
                variant=variant
            )
        else:
            _synth_rft_engines[key] = None
    return _synth_rft_engines[key]


def rft_additive_synthesis(freq: float, duration: float, sample_rate: int,
                           num_harmonics: int = 8, variant: int = None) -> np.ndarray:
    """
    Generate waveform using UnitaryRFT additive synthesis.
    Creates harmonics in RFT domain and transforms back.
    """
    samples = int(duration * sample_rate)
    
    # Pad to power of 2 for RFT
    n = 2 ** int(np.ceil(np.log2(max(samples, 64))))
    
    engine = get_synth_rft_engine(n, variant)
    
    if engine is not None and UNITARY_RFT_AVAILABLE:
        # Build spectrum in RFT domain with Φ-spaced harmonics
        spectrum = np.zeros(n, dtype=np.complex128)
        
        for h in range(1, num_harmonics + 1):
            # Place harmonic at Φ-scaled positions
            harmonic_freq = freq * (PHI ** (h - 1))
            bin_idx = int(harmonic_freq * n / sample_rate) % n
            amplitude = 1.0 / h
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase for richness
            spectrum[bin_idx] = amplitude * np.exp(1j * phase)
            # Mirror for real signal
            if bin_idx > 0:
                spectrum[n - bin_idx] = np.conj(spectrum[bin_idx])
        
        # Inverse RFT to get time domain
        wave = np.real(engine.inverse(spectrum))
        
        # Trim to requested length and normalize
        wave = wave[:samples]
        max_val = np.max(np.abs(wave))
        if max_val > 1e-10:
            wave = wave / max_val
        return wave.astype(np.float32)
    else:
        # Fallback: simple additive synthesis
        t = np.arange(samples) / sample_rate
        wave = np.zeros(samples)
        for h in range(1, num_harmonics + 1):
            harmonic_freq = freq * (PHI ** (h - 1))
            amplitude = 1.0 / h
            wave += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
        return (wave / np.max(np.abs(wave) + 1e-10)).astype(np.float32)


def rft_spectral_morph(wave1: np.ndarray, wave2: np.ndarray, 
                       morph_pos: float) -> np.ndarray:
    """
    Morph between two waveforms in RFT spectral domain.
    morph_pos: 0.0 = wave1, 1.0 = wave2
    """
    n = max(len(wave1), len(wave2))
    n = 2 ** int(np.ceil(np.log2(n)))
    
    # Pad waves
    w1 = np.zeros(n)
    w2 = np.zeros(n)
    w1[:len(wave1)] = wave1
    w2[:len(wave2)] = wave2
    
    engine = get_synth_rft_engine(n)
    
    if engine is not None and UNITARY_RFT_AVAILABLE:
        # Transform both to RFT domain
        spec1 = engine.forward(w1.astype(np.float64))
        spec2 = engine.forward(w2.astype(np.float64))
        
        # Interpolate magnitude and phase separately
        mag1, phase1 = np.abs(spec1), np.angle(spec1)
        mag2, phase2 = np.abs(spec2), np.angle(spec2)
        
        # Morph
        mag = (1 - morph_pos) * mag1 + morph_pos * mag2
        phase = (1 - morph_pos) * phase1 + morph_pos * phase2
        
        # Reconstruct spectrum
        morphed_spec = mag * np.exp(1j * phase)
        
        # Inverse RFT
        result = np.real(engine.inverse(morphed_spec))
        max_val = np.max(np.abs(result))
        if max_val > 1e-10:
            result = result / max_val
        return result[:len(wave1)].astype(np.float32)
    else:
        # Fallback: simple crossfade
        min_len = min(len(wave1), len(wave2))
        return ((1 - morph_pos) * wave1[:min_len] + morph_pos * wave2[:min_len]).astype(np.float32)


def rft_filter(signal: np.ndarray, cutoff_normalized: float, 
               resonance: float = 0.5, filter_type: str = "lowpass") -> np.ndarray:
    """
    Apply filter in RFT domain with Φ-resonance.
    cutoff_normalized: 0.0-1.0 (fraction of Nyquist)
    """
    n = len(signal)
    n_padded = 2 ** int(np.ceil(np.log2(n)))
    
    padded = np.zeros(n_padded)
    padded[:n] = signal
    
    engine = get_synth_rft_engine(n_padded)
    
    if engine is not None and UNITARY_RFT_AVAILABLE:
        # Transform to RFT domain
        spectrum = engine.forward(padded.astype(np.float64))
        
        # Create filter curve with Φ-resonance
        freqs = np.fft.fftfreq(n_padded)
        cutoff_bin = int(cutoff_normalized * n_padded / 2)
        
        if filter_type == "lowpass":
            # Smooth rolloff with Φ-scaled resonance peak
            filter_curve = np.ones(n_padded)
            for i in range(n_padded):
                freq_idx = abs(i if i < n_padded // 2 else n_padded - i)
                if freq_idx > cutoff_bin:
                    # Rolloff after cutoff
                    rolloff = (freq_idx - cutoff_bin) / (n_padded / 2 - cutoff_bin + 1)
                    filter_curve[i] = max(0, 1 - rolloff * 2)
                elif freq_idx > cutoff_bin * (1 / PHI):
                    # Resonance peak near cutoff
                    filter_curve[i] = 1 + resonance * 0.5
        else:
            filter_curve = np.ones(n_padded)
        
        # Apply filter
        filtered_spec = spectrum * filter_curve
        
        # Inverse RFT
        result = np.real(engine.inverse(filtered_spec))
        return result[:n].astype(np.float32)
    else:
        # Fallback: simple filter (already in apply_filter function)
        return signal


class WaveShape(Enum):
    SINE = "sine"
    SAW = "saw"
    SQUARE = "square"
    TRIANGLE = "triangle"
    PHI_WAVE = "phi_wave"      # Golden ratio harmonic series
    RFT_MORPH = "rft_morph"    # Morphable RFT waveform


class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    PHI_RESONANT = "phi_resonant"  # Resonance at φ intervals


@dataclass
class ADSREnvelope:
    """Attack-Decay-Sustain-Release envelope"""
    attack: float = 0.01      # seconds
    decay: float = 0.1        # seconds
    sustain: float = 0.7      # level 0-1
    release: float = 0.3      # seconds
    
    def generate(self, duration: float, sample_rate: int, 
                 note_off_time: Optional[float] = None) -> np.ndarray:
        """Generate envelope for given duration"""
        samples = int(duration * sample_rate)
        env = np.zeros(samples)
        
        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        
        # Attack
        if attack_samples > 0:
            env[:min(attack_samples, samples)] = np.linspace(0, 1, attack_samples)[:samples]
        
        # Decay to sustain
        decay_end = attack_samples + decay_samples
        if decay_end < samples and decay_samples > 0:
            env[attack_samples:decay_end] = np.linspace(1, self.sustain, decay_samples)
        
        # Sustain
        if decay_end < samples:
            env[decay_end:] = self.sustain
        
        # Release (if note_off specified)
        if note_off_time is not None:
            release_start = int(note_off_time * sample_rate)
            release_samples = int(self.release * sample_rate)
            if release_start < samples:
                release_end = min(release_start + release_samples, samples)
                current_level = env[release_start] if release_start < samples else self.sustain
                release_len = release_end - release_start
                env[release_start:release_end] = np.linspace(current_level, 0, release_len)
                env[release_end:] = 0
        
        return env


@dataclass
class OscillatorConfig:
    """Single oscillator configuration"""
    shape: WaveShape = WaveShape.SAW
    detune_cents: float = 0.0
    octave_shift: int = 0
    level: float = 1.0
    pan: float = 0.0  # -1 to 1
    
    # For PHI_WAVE
    phi_harmonics: int = 5
    
    # For RFT_MORPH  
    morph_position: float = 0.5  # 0-1 between two shapes


@dataclass
class InstrumentPreset:
    """Complete instrument preset"""
    name: str = "Init"
    category: str = "Lead"
    
    # Oscillators (up to 3)
    osc1: OscillatorConfig = field(default_factory=OscillatorConfig)
    osc2: OscillatorConfig = field(default_factory=lambda: OscillatorConfig(
        shape=WaveShape.SQUARE, detune_cents=7, level=0.5
    ))
    osc3: OscillatorConfig = field(default_factory=lambda: OscillatorConfig(
        shape=WaveShape.SINE, octave_shift=1, level=0.3
    ))
    
    # Amp envelope
    amp_env: ADSREnvelope = field(default_factory=ADSREnvelope)
    
    # Filter
    filter_type: FilterType = FilterType.LOWPASS
    filter_cutoff: float = 5000.0  # Hz
    filter_resonance: float = 0.3  # 0-1
    filter_env_amount: float = 0.0  # How much envelope affects cutoff
    filter_env: ADSREnvelope = field(default_factory=lambda: ADSREnvelope(
        attack=0.0, decay=0.5, sustain=0.3, release=0.5
    ))
    
    # Master
    master_volume: float = 0.7
    glide_time: float = 0.0  # Portamento in seconds


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

PRESET_LIBRARY: Dict[str, Dict[str, InstrumentPreset]] = {
    "Lead": {
        "Φ-Lead": InstrumentPreset(
            name="Φ-Lead",
            category="Lead",
            osc1=OscillatorConfig(shape=WaveShape.PHI_WAVE, phi_harmonics=6),
            osc2=OscillatorConfig(shape=WaveShape.SAW, detune_cents=12, level=0.4),
            amp_env=ADSREnvelope(attack=0.01, decay=0.2, sustain=0.6, release=0.4),
            filter_cutoff=3000,
            filter_resonance=0.5,
        ),
        "Super Saw": InstrumentPreset(
            name="Super Saw",
            category="Lead",
            osc1=OscillatorConfig(shape=WaveShape.SAW),
            osc2=OscillatorConfig(shape=WaveShape.SAW, detune_cents=15, level=0.8),
            osc3=OscillatorConfig(shape=WaveShape.SAW, detune_cents=-12, level=0.8),
            amp_env=ADSREnvelope(attack=0.005, decay=0.1, sustain=0.8, release=0.3),
            filter_cutoff=8000,
        ),
        "Sine Lead": InstrumentPreset(
            name="Sine Lead",
            category="Lead",
            osc1=OscillatorConfig(shape=WaveShape.SINE),
            osc2=OscillatorConfig(shape=WaveShape.SINE, octave_shift=1, level=0.3),
            amp_env=ADSREnvelope(attack=0.02, decay=0.1, sustain=0.7, release=0.5),
        ),
    },
    "Bass": {
        "Sub Bass": InstrumentPreset(
            name="Sub Bass",
            category="Bass",
            osc1=OscillatorConfig(shape=WaveShape.SINE, octave_shift=-1),
            osc2=OscillatorConfig(shape=WaveShape.SAW, level=0.3),
            amp_env=ADSREnvelope(attack=0.005, decay=0.3, sustain=0.5, release=0.2),
            filter_cutoff=500,
            filter_resonance=0.2,
        ),
        "Φ-Bass": InstrumentPreset(
            name="Φ-Bass",
            category="Bass",
            osc1=OscillatorConfig(shape=WaveShape.PHI_WAVE, octave_shift=-1, phi_harmonics=4),
            amp_env=ADSREnvelope(attack=0.01, decay=0.4, sustain=0.4, release=0.3),
            filter_cutoff=800,
            filter_resonance=0.6,
        ),
        "Reese Bass": InstrumentPreset(
            name="Reese Bass",
            category="Bass",
            osc1=OscillatorConfig(shape=WaveShape.SAW, octave_shift=-1),
            osc2=OscillatorConfig(shape=WaveShape.SAW, octave_shift=-1, detune_cents=10),
            amp_env=ADSREnvelope(attack=0.01, decay=0.2, sustain=0.7, release=0.3),
            filter_cutoff=1200,
        ),
    },
    "Pad": {
        "Warm Pad": InstrumentPreset(
            name="Warm Pad",
            category="Pad",
            osc1=OscillatorConfig(shape=WaveShape.SAW),
            osc2=OscillatorConfig(shape=WaveShape.SQUARE, detune_cents=5, level=0.5),
            amp_env=ADSREnvelope(attack=0.5, decay=0.3, sustain=0.8, release=1.0),
            filter_cutoff=2000,
            filter_resonance=0.3,
        ),
        "Φ-Pad": InstrumentPreset(
            name="Φ-Pad",
            category="Pad",
            osc1=OscillatorConfig(shape=WaveShape.PHI_WAVE, phi_harmonics=8),
            osc2=OscillatorConfig(shape=WaveShape.TRIANGLE, detune_cents=3, level=0.6),
            amp_env=ADSREnvelope(attack=0.8, decay=0.5, sustain=0.7, release=1.5),
            filter_cutoff=3000,
        ),
        "Strings": InstrumentPreset(
            name="Strings",
            category="Pad",
            osc1=OscillatorConfig(shape=WaveShape.SAW),
            osc2=OscillatorConfig(shape=WaveShape.SAW, detune_cents=8, level=0.8),
            osc3=OscillatorConfig(shape=WaveShape.SAW, detune_cents=-6, level=0.7),
            amp_env=ADSREnvelope(attack=0.3, decay=0.2, sustain=0.9, release=0.8),
            filter_cutoff=4000,
        ),
    },
    "Keys": {
        "Electric Piano": InstrumentPreset(
            name="Electric Piano",
            category="Keys",
            osc1=OscillatorConfig(shape=WaveShape.SINE),
            osc2=OscillatorConfig(shape=WaveShape.SINE, octave_shift=2, level=0.2),
            amp_env=ADSREnvelope(attack=0.001, decay=1.0, sustain=0.0, release=0.5),
        ),
        "Organ": InstrumentPreset(
            name="Organ",
            category="Keys",
            osc1=OscillatorConfig(shape=WaveShape.SINE),
            osc2=OscillatorConfig(shape=WaveShape.SINE, octave_shift=1, level=0.5),
            osc3=OscillatorConfig(shape=WaveShape.SINE, octave_shift=2, level=0.25),
            amp_env=ADSREnvelope(attack=0.01, decay=0.0, sustain=1.0, release=0.1),
        ),
    },
    "FX": {
        "Rise": InstrumentPreset(
            name="Rise",
            category="FX",
            osc1=OscillatorConfig(shape=WaveShape.SAW),
            amp_env=ADSREnvelope(attack=2.0, decay=0.1, sustain=0.8, release=0.5),
            filter_cutoff=500,
            filter_env_amount=5000,
            filter_env=ADSREnvelope(attack=2.0, decay=0.0, sustain=1.0, release=0.5),
        ),
        "Impact": InstrumentPreset(
            name="Impact",
            category="FX",
            osc1=OscillatorConfig(shape=WaveShape.SAW),
            osc2=OscillatorConfig(shape=WaveShape.SQUARE, octave_shift=-2, level=0.8),
            amp_env=ADSREnvelope(attack=0.001, decay=0.5, sustain=0.0, release=1.0),
            filter_cutoff=2000,
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# OSCILLATOR GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_waveform(shape: WaveShape, freq: float, duration: float, 
                      sample_rate: int, config: OscillatorConfig) -> np.ndarray:
    """Generate a waveform for given parameters"""
    samples = int(duration * sample_rate)
    t = np.arange(samples) / sample_rate
    
    # Apply detune
    detune_factor = 2 ** (config.detune_cents / 1200)
    freq = freq * detune_factor
    
    # Apply octave shift
    freq = freq * (2 ** config.octave_shift)
    
    phase = 2 * np.pi * freq * t
    
    if shape == WaveShape.SINE:
        return np.sin(phase)
    
    elif shape == WaveShape.SAW:
        return 2 * (t * freq % 1) - 1
    
    elif shape == WaveShape.SQUARE:
        return np.sign(np.sin(phase))
    
    elif shape == WaveShape.TRIANGLE:
        return 2 * np.abs(2 * (t * freq % 1) - 1) - 1
    
    elif shape == WaveShape.PHI_WAVE:
        # Golden ratio harmonic series via UnitaryRFT
        if UNITARY_RFT_AVAILABLE:
            return rft_additive_synthesis(
                freq, duration, sample_rate,
                num_harmonics=config.phi_harmonics,
                variant=RFT_VARIANT_HARMONIC if UNITARY_RFT_AVAILABLE else None
            )
        else:
            # Fallback: simple additive
            wave = np.zeros(samples)
            for h in range(1, config.phi_harmonics + 1):
                harmonic_freq = freq * (PHI ** (h - 1))
                amplitude = 1.0 / h
                wave += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
            return wave / np.max(np.abs(wave) + 1e-10)
    
    elif shape == WaveShape.RFT_MORPH:
        # Morph between saw and sine in RFT spectral domain
        saw = 2 * (t * freq % 1) - 1
        sine = np.sin(phase)
        if UNITARY_RFT_AVAILABLE:
            return rft_spectral_morph(saw, sine, config.morph_position)
        else:
            return (1 - config.morph_position) * saw + config.morph_position * sine
    
    return np.sin(phase)


def apply_filter(signal: np.ndarray, filter_type: FilterType, 
                 cutoff: float, resonance: float, sample_rate: int) -> np.ndarray:
    """Apply simple filter to signal"""
    # Simple one-pole filter approximation
    dt = 1.0 / sample_rate
    rc = 1.0 / (2 * np.pi * cutoff)
    alpha = dt / (rc + dt)
    
    if filter_type in [FilterType.LOWPASS, FilterType.PHI_RESONANT]:
        # Low pass
        output = np.zeros_like(signal)
        output[0] = signal[0]
        for i in range(1, len(signal)):
            output[i] = output[i-1] + alpha * (signal[i] - output[i-1])
        
        # Add resonance
        if resonance > 0:
            output += resonance * 0.5 * (signal - output)
        
        return output
    
    elif filter_type == FilterType.HIGHPASS:
        lp = apply_filter(signal, FilterType.LOWPASS, cutoff, 0, sample_rate)
        return signal - lp
    
    return signal


# ═══════════════════════════════════════════════════════════════════════════════
# POLYPHONIC SYNTHESIZER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Voice:
    """Single voice in polyphonic synth"""
    note: int = 60  # MIDI note number
    velocity: float = 1.0
    start_time: float = 0.0
    release_time: Optional[float] = None
    is_active: bool = True


class PolySynth:
    """Polyphonic synthesizer with preset support"""
    
    def __init__(self, sample_rate: int = 44100, max_voices: int = 16):
        self.sample_rate = sample_rate
        self.max_voices = max_voices
        self.preset = PRESET_LIBRARY["Lead"]["Φ-Lead"]
        self.voices: List[Voice] = []
        self.current_time = 0.0
        self.lock = threading.Lock()
    
    def note_on(self, note: int, velocity: float = 1.0):
        """Start a note"""
        with self.lock:
            # Check if note already playing
            for v in self.voices:
                if v.note == note and v.is_active and v.release_time is None:
                    return  # Already playing
            
            # Remove oldest voice if at max
            if len(self.voices) >= self.max_voices:
                self.voices.pop(0)
            
            self.voices.append(Voice(
                note=note,
                velocity=velocity,
                start_time=self.current_time,
            ))
    
    def note_off(self, note: int):
        """Release a note"""
        with self.lock:
            for v in self.voices:
                if v.note == note and v.release_time is None:
                    v.release_time = self.current_time
    
    def set_preset(self, preset: InstrumentPreset):
        """Set current preset"""
        self.preset = preset
    
    def render(self, num_samples: int) -> np.ndarray:
        """Render audio for given number of samples"""
        output = np.zeros(num_samples, dtype=np.float32)
        duration = num_samples / self.sample_rate
        
        with self.lock:
            voices_to_remove = []
            
            for i, voice in enumerate(self.voices):
                if not voice.is_active:
                    voices_to_remove.append(i)
                    continue
                
                # Calculate frequency from MIDI note
                freq = 440.0 * (2 ** ((voice.note - 69) / 12))
                
                # Voice age
                age = self.current_time - voice.start_time
                voice_duration = duration
                
                # Calculate note_off time relative to voice start
                note_off_rel = None
                if voice.release_time is not None:
                    note_off_rel = voice.release_time - voice.start_time
                    # Check if voice is done
                    if age > note_off_rel + self.preset.amp_env.release:
                        voice.is_active = False
                        voices_to_remove.append(i)
                        continue
                
                # Generate oscillators
                voice_signal = np.zeros(num_samples)
                
                for osc in [self.preset.osc1, self.preset.osc2, self.preset.osc3]:
                    if osc.level > 0:
                        wave = generate_waveform(
                            osc.shape, freq, voice_duration, 
                            self.sample_rate, osc
                        )
                        voice_signal += wave * osc.level
                
                # Normalize
                if len(self.preset.osc2.shape.value) > 0:
                    voice_signal /= 3.0
                
                # Apply filter
                voice_signal = apply_filter(
                    voice_signal,
                    self.preset.filter_type,
                    self.preset.filter_cutoff,
                    self.preset.filter_resonance,
                    self.sample_rate
                )
                
                # Apply envelope (simplified - just use age-based position)
                env = self.preset.amp_env.generate(
                    age + voice_duration, 
                    self.sample_rate,
                    note_off_rel
                )
                
                # Get envelope segment for this render
                start_sample = int(age * self.sample_rate)
                end_sample = start_sample + num_samples
                if end_sample <= len(env):
                    env_segment = env[start_sample:end_sample]
                else:
                    env_segment = np.ones(num_samples) * self.preset.amp_env.sustain
                
                voice_signal *= env_segment * voice.velocity
                output += voice_signal
            
            # Remove dead voices
            for i in reversed(voices_to_remove):
                if i < len(self.voices):
                    self.voices.pop(i)
            
            self.current_time += duration
        
        # Apply master volume and clip
        output *= self.preset.master_volume
        return np.clip(output, -1.0, 1.0)
    
    def get_active_notes(self) -> List[int]:
        """Get list of currently active note numbers"""
        with self.lock:
            return [v.note for v in self.voices if v.is_active and v.release_time is None]


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTER KEYBOARD MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# QWERTY keyboard to MIDI notes (C4 = 60)
KEYBOARD_MAP = {
    # Bottom row - C3 to B3
    'z': 48, 's': 49, 'x': 50, 'd': 51, 'c': 52, 'v': 53, 
    'g': 54, 'b': 55, 'h': 56, 'n': 57, 'j': 58, 'm': 59,
    
    # Middle row - C4 to B4
    'q': 60, '2': 61, 'w': 62, '3': 63, 'e': 64, 'r': 65,
    '5': 66, 't': 67, '6': 68, 'y': 69, '7': 70, 'u': 71,
    
    # Top extends
    'i': 72, '9': 73, 'o': 74, '0': 75, 'p': 76,
}

# Alternative simpler mapping (A-K = C to C)
SIMPLE_KEYBOARD_MAP = {
    'a': 60, 'w': 61, 's': 62, 'e': 63, 'd': 64,
    'f': 65, 't': 66, 'g': 67, 'y': 68, 'h': 69,
    'u': 70, 'j': 71, 'k': 72,
}


def key_to_note(key: str, use_simple: bool = True) -> Optional[int]:
    """Convert keyboard key to MIDI note"""
    key = key.lower()
    if use_simple:
        return SIMPLE_KEYBOARD_MAP.get(key)
    return KEYBOARD_MAP.get(key)


def note_to_name(note: int) -> str:
    """Convert MIDI note to name (e.g., 60 -> 'C4')"""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note // 12) - 1
    name = names[note % 12]
    return f"{name}{octave}"


def name_to_note(name: str) -> int:
    """Convert note name to MIDI note (e.g., 'C4' -> 60)"""
    names = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    name = name.upper().strip()
    
    # Parse note name
    note_char = name[0]
    rest = name[1:]
    
    # Check for sharp/flat
    offset = 0
    if rest.startswith('#'):
        offset = 1
        rest = rest[1:]
    elif rest.startswith('B'):
        offset = -1
        rest = rest[1:]
    
    # Parse octave
    octave = int(rest) if rest else 4
    
    return (octave + 1) * 12 + names[note_char] + offset
