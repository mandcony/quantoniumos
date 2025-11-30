#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""QuantSoundDesign - Core Engine

Φ-RFT native sound design studio core data models and processing engine.
Now connected to the full UnitaryRFT system for true unitary transforms.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
import numpy as np
import uuid

# ═══════════════════════════════════════════════════════════════════════════════
# UNITARY RFT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Try to import the full UnitaryRFT system
try:
    from algorithms.rft.kernels.python_bindings.unitary_rft import (
        UnitaryRFT, 
        RFT_FLAG_UNITARY, 
        RFT_FLAG_QUANTUM_SAFE,
        RFT_FLAG_USE_RESONANCE,
        RFT_FLAG_HIGH_PRECISION,
        RFT_VARIANT_STANDARD,
        RFT_VARIANT_HARMONIC,
        RFT_VARIANT_FIBONACCI,
        RFT_VARIANT_CHAOTIC,
        RFT_VARIANT_GEOMETRIC,
        RFT_VARIANT_HYBRID,
        RFT_VARIANT_ADAPTIVE,
    )
    UNITARY_RFT_AVAILABLE = True
    print("[OK] UnitaryRFT system connected to QuantSoundDesign")
except ImportError as e:
    UNITARY_RFT_AVAILABLE = False
    print(f"⚠ UnitaryRFT not available, using fallback: {e}")
    # Define fallback constants
    RFT_FLAG_UNITARY = 0x00000008
    RFT_FLAG_QUANTUM_SAFE = 0x00000004
    RFT_FLAG_USE_RESONANCE = 0x00000010
    RFT_FLAG_HIGH_PRECISION = 0x00000002
    RFT_VARIANT_STANDARD = 0
    RFT_VARIANT_HARMONIC = 1
    RFT_VARIANT_FIBONACCI = 2
    RFT_VARIANT_CHAOTIC = 3
    RFT_VARIANT_GEOMETRIC = 4
    RFT_VARIANT_HYBRID = 5
    RFT_VARIANT_ADAPTIVE = 6

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = PHI - 1  # 0.618...


class Domain(Enum):
    TIME = "time"
    FREQUENCY = "frequency"
    RFT_PHI = "rft_phi"
    RFT_DCT_HYBRID = "rft_dct_hybrid"


class TrackKind(Enum):
    AUDIO = "audio"
    INSTRUMENT = "instrument"
    GROUP = "group"
    RETURN = "return"
    MASTER = "master"


class ClipKind(Enum):
    AUDIO = "audio"
    MIDI = "midi"


class DeviceKind(Enum):
    INSTRUMENT = "instrument"
    EFFECT = "effect"
    UTILITY = "utility"
    METER = "meter"
    SEND = "send"


# ═══════════════════════════════════════════════════════════════════════════════
# WAVEFIELD - Core signal abstraction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WaveField:
    """
    Core signal abstraction for QuantSoundDesign.
    Can represent audio in time domain or wave domain (RFT).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    domain: Domain = Domain.TIME
    length: int = 0
    sample_rate: float = 48000.0
    channels: int = 2
    
    # Data storage - only one should be populated at a time
    data_time: Optional[np.ndarray] = None      # Shape: (channels, samples)
    data_wave: Optional[np.ndarray] = None      # Shape: (channels, coefficients) - complex
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.data_time is None and self.data_wave is None:
            # Initialize empty time-domain buffer
            self.data_time = np.zeros((self.channels, self.length), dtype=np.float32)
    
    @classmethod
    def from_buffer(cls, buffer: np.ndarray, sample_rate: float = 48000.0) -> "WaveField":
        """Create WaveField from numpy buffer"""
        if buffer.ndim == 1:
            buffer = buffer.reshape(1, -1)
        return cls(
            domain=Domain.TIME,
            length=buffer.shape[1],
            sample_rate=sample_rate,
            channels=buffer.shape[0],
            data_time=buffer.astype(np.float32)
        )
    
    def to_time(self) -> np.ndarray:
        """Get time-domain data, converting if necessary"""
        if self.data_time is not None:
            return self.data_time
        elif self.data_wave is not None:
            # Inverse RFT
            self.data_time = rft_inverse(self.data_wave)
            return self.data_time
        return np.zeros((self.channels, self.length), dtype=np.float32)
    
    def to_wave(self) -> np.ndarray:
        """Get wave-domain data, converting if necessary"""
        if self.data_wave is not None:
            return self.data_wave
        elif self.data_time is not None:
            # Forward RFT
            self.data_wave = rft_forward(self.data_time)
            self.domain = Domain.RFT_PHI
            return self.data_wave
        return np.zeros((self.channels, self.length), dtype=np.complex64)
    
    def copy(self) -> "WaveField":
        """Create a copy of this WaveField"""
        wf = WaveField(
            domain=self.domain,
            length=self.length,
            sample_rate=self.sample_rate,
            channels=self.channels,
            metadata=self.metadata.copy()
        )
        if self.data_time is not None:
            wf.data_time = self.data_time.copy()
        if self.data_wave is not None:
            wf.data_wave = self.data_wave.copy()
        return wf


# ═══════════════════════════════════════════════════════════════════════════════
# RFT TRANSFORMS - Φ-weighted spectral transforms with UnitaryRFT integration
# ═══════════════════════════════════════════════════════════════════════════════

# Global RFT engine cache for performance
_rft_engines: Dict[int, Any] = {}
_current_rft_variant = RFT_VARIANT_HARMONIC  # Default to Harmonic for audio
_current_rft_flags = RFT_FLAG_UNITARY | RFT_FLAG_USE_RESONANCE


def set_rft_variant(variant: int) -> None:
    """Set the RFT variant for all audio processing."""
    global _current_rft_variant, _rft_engines
    _current_rft_variant = variant
    _rft_engines.clear()  # Clear cache to use new variant
    print(f"✓ RFT variant set to: {variant}")


def set_rft_flags(flags: int) -> None:
    """Set the RFT flags for all audio processing."""
    global _current_rft_flags, _rft_engines
    _current_rft_flags = flags
    _rft_engines.clear()  # Clear cache to use new flags
    print(f"✓ RFT flags set to: 0x{flags:08x}")


def get_rft_engine(size: int) -> Any:
    """Get or create an RFT engine for the given size."""
    global _rft_engines
    
    if size not in _rft_engines:
        if UNITARY_RFT_AVAILABLE:
            engine = UnitaryRFT(
                size=size, 
                flags=_current_rft_flags,
                variant=_current_rft_variant
            )
            _rft_engines[size] = engine
        else:
            _rft_engines[size] = None
    
    return _rft_engines[size]


def rft_forward(signal: np.ndarray) -> np.ndarray:
    """
    Φ-RFT Forward Transform using UnitaryRFT.
    Converts time-domain signal to wave-domain with golden-ratio weighting.
    
    Args:
        signal: Shape (channels, samples) or (samples,)
    
    Returns:
        Complex coefficients with Φ-weighting applied
    """
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    channels, n = signal.shape
    result = np.zeros((channels, n), dtype=np.complex128)
    
    engine = get_rft_engine(n)
    
    for ch in range(channels):
        if engine is not None and UNITARY_RFT_AVAILABLE:
            # Use full UnitaryRFT system
            result[ch] = engine.forward(signal[ch].astype(np.float64))
        else:
            # Fallback: Standard FFT with Φ-weighting
            spectrum = np.fft.fft(signal[ch])
            freqs = np.fft.fftfreq(n)
            phi_weight = PHI ** (-np.abs(freqs) * n / 10)
            result[ch] = spectrum * phi_weight
    
    return result  # Keep full precision


def rft_inverse(coeffs: np.ndarray) -> np.ndarray:
    """
    Φ-RFT Inverse Transform using UnitaryRFT.
    Converts wave-domain back to time-domain, undoing Φ-weighting.
    
    Args:
        coeffs: Shape (channels, coefficients) complex
    
    Returns:
        Time-domain signal
    """
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(1, -1)
    
    channels, n = coeffs.shape
    result = np.zeros((channels, n), dtype=np.float64)
    
    engine = get_rft_engine(n)
    
    for ch in range(channels):
        if engine is not None and UNITARY_RFT_AVAILABLE:
            # Use full UnitaryRFT system - take real part of inverse
            inverse_result = engine.inverse(coeffs[ch].astype(np.complex128))
            result[ch] = np.real(inverse_result)
        else:
            # Fallback: Undo Φ-weighting + IFFT
            freqs = np.fft.fftfreq(n)
            phi_weight = PHI ** (-np.abs(freqs) * n / 10)
            phi_weight = np.where(phi_weight > 1e-10, phi_weight, 1e-10)
            unweighted = coeffs[ch] / phi_weight
            result[ch] = np.real(np.fft.ifft(unweighted))
    
    return result  # Keep full precision for accuracy


def rft_overlap_add(process_func: Callable, signal: np.ndarray, 
                    window_size: int = 2048, hop_size: int = 1024) -> np.ndarray:
    """
    Process signal through RFT with overlap-add for real-time.
    Uses UnitaryRFT for perfect reconstruction.
    
    Args:
        process_func: Function that takes wave-domain coeffs and returns processed coeffs
        signal: Time-domain input
        window_size: Analysis window size
        hop_size: Hop between windows (typically window_size // 2)
    
    Returns:
        Processed time-domain signal
    """
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    channels, n = signal.shape
    output = np.zeros_like(signal)
    
    # Hann window for smooth overlap
    window = np.hanning(window_size).astype(np.float32)
    
    # Pre-create engine for this window size
    _ = get_rft_engine(window_size)
    
    for start in range(0, n - window_size + 1, hop_size):
        end = start + window_size
        
        # Extract windowed segment
        segment = signal[:, start:end] * window
        
        # Forward RFT (now uses UnitaryRFT if available)
        coeffs = rft_forward(segment)
        
        # Process in wave domain
        processed_coeffs = process_func(coeffs)
        
        # Inverse RFT (now uses UnitaryRFT if available)
        processed = rft_inverse(processed_coeffs)
        
        # Overlap-add
        output[:, start:end] += processed * window
    
    return output


def rft_analyze_spectrum(signal: np.ndarray, variant: int = None) -> Dict[str, Any]:
    """
    Analyze a signal's spectral content using RFT.
    
    Args:
        signal: Input signal
        variant: RFT variant to use (None = current default)
    
    Returns:
        Dictionary with spectral analysis results
    """
    if variant is not None:
        old_variant = _current_rft_variant
        set_rft_variant(variant)
    
    coeffs = rft_forward(signal)
    
    if variant is not None:
        set_rft_variant(old_variant)
    
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    n = signal.shape[1]
    
    return {
        "coefficients": coeffs,
        "magnitude": np.abs(coeffs),
        "phase": np.angle(coeffs),
        "energy": np.sum(np.abs(coeffs) ** 2),
        "peak_freq_bin": np.argmax(np.abs(coeffs[0])),
        "phi_weighted": True,
        "unitary": UNITARY_RFT_AVAILABLE,
        "size": n,
    }


def rft_synthesize(coeffs: np.ndarray, variant: int = None) -> np.ndarray:
    """
    Synthesize a signal from RFT coefficients.
    
    Args:
        coeffs: RFT coefficients
        variant: RFT variant to use (None = current default)
    
    Returns:
        Time-domain signal
    """
    if variant is not None:
        old_variant = _current_rft_variant
        set_rft_variant(variant)
    
    signal = rft_inverse(coeffs)
    
    if variant is not None:
        set_rft_variant(old_variant)
    
    return signal


def get_rft_status() -> Dict[str, Any]:
    """Get the current RFT engine status."""
    engine = get_rft_engine(64) if len(_rft_engines) == 0 else list(_rft_engines.values())[0]
    
    return {
        "unitary_available": UNITARY_RFT_AVAILABLE,
        "is_mock": engine._is_mock if engine else True,
        "current_variant": _current_rft_variant,
        "current_flags": _current_rft_flags,
        "cached_sizes": list(_rft_engines.keys()),
        "variants": {
            "STANDARD": RFT_VARIANT_STANDARD,
            "HARMONIC": RFT_VARIANT_HARMONIC,
            "FIBONACCI": RFT_VARIANT_FIBONACCI,
            "CHAOTIC": RFT_VARIANT_CHAOTIC,
            "GEOMETRIC": RFT_VARIANT_GEOMETRIC,
            "HYBRID": RFT_VARIANT_HYBRID,
            "ADAPTIVE": RFT_VARIANT_ADAPTIVE,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Note:
    """MIDI note event"""
    pitch: int          # 0-127
    velocity: int       # 0-127
    start_beat: float
    length_beats: float
    channel: int = 0


@dataclass
class CCEvent:
    """MIDI CC event"""
    cc_number: int
    value: int
    beat: float
    channel: int = 0


@dataclass
class AudioPayload:
    """Audio clip data"""
    file_path: str = ""
    samples: Optional[np.ndarray] = None  # Cached audio data
    start_sample: int = 0
    length_samples: int = 0
    warp_mode: str = "none"  # "none", "rft_resonant", "elastic"
    gain_db: float = 0.0


@dataclass
class MidiPayload:
    """MIDI clip data"""
    notes: List[Note] = field(default_factory=list)
    cc_events: List[CCEvent] = field(default_factory=list)


@dataclass
class Clip:
    """Audio or MIDI clip on a track"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    track_id: str = ""
    kind: ClipKind = ClipKind.AUDIO
    name: str = "Clip"
    color: str = "#00aaff"
    
    start_beat: float = 0.0
    length_beats: float = 4.0
    loop: bool = False
    
    # Payload
    audio: Optional[AudioPayload] = None
    midi: Optional[MidiPayload] = None
    
    def get_samples_at(self, beat: float, beats_per_block: float, 
                       sample_rate: float, tempo: float) -> Optional[np.ndarray]:
        """Get audio samples for a specific beat range"""
        if self.kind != ClipKind.AUDIO or self.audio is None:
            return None
        
        if self.audio.samples is None:
            return None
        
        # Convert beats to samples
        samples_per_beat = (sample_rate * 60) / tempo
        clip_start_sample = int(self.start_beat * samples_per_beat)
        
        block_start_sample = int(beat * samples_per_beat)
        block_samples = int(beats_per_block * samples_per_beat)
        
        # Calculate offset into clip
        offset = block_start_sample - clip_start_sample
        
        if offset < 0 or offset >= len(self.audio.samples[0]):
            return None
        
        end = min(offset + block_samples, len(self.audio.samples[0]))
        return self.audio.samples[:, offset:end]


@dataclass  
class AutomationPoint:
    """Single automation point"""
    beat: float
    value: float
    curve: str = "linear"  # "linear", "smooth", "step"


@dataclass
class AutomationLane:
    """Automation for a parameter"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_track_id: str = ""
    target_device_id: str = ""
    target_param: str = ""
    points: List[AutomationPoint] = field(default_factory=list)
    
    def get_value_at(self, beat: float) -> float:
        """Interpolate automation value at beat"""
        if not self.points:
            return 0.0
        
        # Find surrounding points
        before = None
        after = None
        
        for p in self.points:
            if p.beat <= beat:
                before = p
            elif after is None:
                after = p
                break
        
        if before is None:
            return self.points[0].value
        if after is None:
            return before.value
        
        # Linear interpolation
        t = (beat - before.beat) / (after.beat - before.beat)
        return before.value + t * (after.value - before.value)


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DevicePort:
    """Input/output port on a device"""
    name: str
    kind: str = "audio"  # "audio", "midi", "control"
    channels: int = 2


@dataclass
class DeviceConnection:
    """Connection between device ports"""
    source_device_id: str
    source_port: str
    dest_device_id: str
    dest_port: str


@dataclass
class DeviceNode:
    """
    A single device in the processing chain.
    Can be instrument, effect, utility, etc.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    kind: DeviceKind = DeviceKind.EFFECT
    subtype: str = "utility"  # "rft_eq", "rft_filter", "compressor", "gain", etc.
    name: str = "Device"
    
    # Parameters
    params: Dict[str, float] = field(default_factory=dict)
    param_specs: Dict[str, Dict] = field(default_factory=dict)  # min, max, default, unit
    
    # Ports
    inputs: List[DevicePort] = field(default_factory=list)
    outputs: List[DevicePort] = field(default_factory=list)
    
    # State
    bypass: bool = False
    _state: Dict[str, Any] = field(default_factory=dict)
    _processor: Optional[Callable] = None
    
    def process_block(self, input_buffer: np.ndarray, 
                      sample_rate: float, block_size: int) -> np.ndarray:
        """Process audio block through this device"""
        if self.bypass:
            return input_buffer
        
        if self._processor is not None:
            return self._processor(input_buffer, self.params, self._state, 
                                   sample_rate, block_size)
        
        # Default passthrough
        return input_buffer


@dataclass
class DeviceChain:
    """Chain of devices on a track"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    devices: List[DeviceNode] = field(default_factory=list)
    connections: List[DeviceConnection] = field(default_factory=list)
    
    def process_block(self, input_buffer: np.ndarray,
                      sample_rate: float, block_size: int) -> np.ndarray:
        """Process through all devices in order"""
        buffer = input_buffer
        
        for device in self.devices:
            buffer = device.process_block(buffer, sample_rate, block_size)
        
        return buffer
    
    def add_device(self, device: DeviceNode, index: int = -1):
        """Add device to chain"""
        if index < 0:
            self.devices.append(device)
        else:
            self.devices.insert(index, device)
    
    def remove_device(self, device_id: str):
        """Remove device from chain"""
        self.devices = [d for d in self.devices if d.id != device_id]


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Track:
    """Audio/MIDI/Group track"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Track"
    color: str = "#00aaff"
    kind: TrackKind = TrackKind.AUDIO
    
    # Device chain
    devices: DeviceChain = field(default_factory=DeviceChain)
    
    # Clips
    clips: List[Clip] = field(default_factory=list)
    
    # Routing
    input_routing: Dict[str, str] = field(default_factory=dict)
    output_routing: Dict[str, str] = field(default_factory=dict)
    sends: Dict[str, float] = field(default_factory=dict)  # return_track_id -> level_db
    
    # Mix controls
    mute: bool = False
    solo: bool = False
    arm: bool = False
    volume_db: float = 0.0
    pan: float = 0.0  # -1 (L) to +1 (R)
    
    def get_clips_at(self, beat: float) -> List[Clip]:
        """Get all clips that overlap with given beat"""
        result = []
        for clip in self.clips:
            if clip.start_beat <= beat < clip.start_beat + clip.length_beats:
                result.append(clip)
        return result
    
    def apply_pan(self, buffer: np.ndarray) -> np.ndarray:
        """Apply panning to stereo buffer"""
        if buffer.shape[0] < 2:
            return buffer
        
        # Constant power panning
        angle = (self.pan + 1) * np.pi / 4  # 0 to π/2
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)
        
        buffer[0] *= left_gain
        buffer[1] *= right_gain
        
        return buffer
    
    def apply_volume(self, buffer: np.ndarray) -> np.ndarray:
        """Apply volume in dB"""
        gain = 10 ** (self.volume_db / 20)
        return buffer * gain


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION
# ═══════════════════════════════════════════════════════════════════════════════

class RecordingMode(Enum):
    """Recording behavior modes"""
    REPLACE = "replace"      # Overwrite existing data
    OVERDUB = "overdub"      # Layer on top
    PUNCH_IN = "punch_in"    # Record only in loop region


class AutomationMode(Enum):
    """Automation recording modes"""
    READ = "read"       # Playback only
    WRITE = "write"     # Record automation, replacing existing
    LATCH = "latch"     # Start recording on first touch, continue
    TOUCH = "touch"     # Record while touching, return to existing on release


@dataclass
class TransportState:
    """Playback transport state"""
    playing: bool = False
    recording: bool = False
    position_beats: float = 0.0
    position_samples: int = 0
    loop_enabled: bool = False
    loop_start_beats: float = 0.0
    loop_end_beats: float = 8.0
    
    # Recording modes
    recording_mode: RecordingMode = RecordingMode.REPLACE
    automation_mode: AutomationMode = AutomationMode.READ
    
    # Punch in/out
    punch_in_enabled: bool = False
    punch_in_beats: float = 0.0
    punch_out_beats: float = 4.0
    
    # Count-in
    count_in_bars: int = 0  # 0=off, 1, 2
    
    # Pre-roll
    pre_roll_bars: float = 0.0


@dataclass
class Session:
    """
    Complete DAW session/project.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Untitled Project"
    
    # Audio settings
    sample_rate: float = 48000.0
    block_size: int = 256
    
    # Tempo & time
    tempo_bpm: float = 120.0
    time_sig_num: int = 4
    time_sig_den: int = 4
    
    # Tracks
    tracks: List[Track] = field(default_factory=list)
    master_track: Track = field(default_factory=lambda: Track(
        name="Master", 
        kind=TrackKind.MASTER,
        color="#00ffaa"
    ))
    
    # Automation
    automation_lanes: List[AutomationLane] = field(default_factory=list)
    
    # Transport
    transport: TransportState = field(default_factory=TransportState)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def samples_per_beat(self) -> float:
        """Samples per beat at current tempo"""
        return (self.sample_rate * 60) / self.tempo_bpm
    
    @property
    def beats_per_block(self) -> float:
        """Beats per audio block"""
        return self.block_size / self.samples_per_beat
    
    def add_track(self, kind: TrackKind = TrackKind.AUDIO, name: str = None) -> Track:
        """Add a new track to the session"""
        colors = ["#00aaff", "#00ffaa", "#ffaa00", "#ff00aa", "#aa00ff", "#00ffff"]
        color = colors[len(self.tracks) % len(colors)]
        
        track = Track(
            name=name or f"Track {len(self.tracks) + 1}",
            kind=kind,
            color=color
        )
        self.tracks.append(track)
        return track
    
    def remove_track(self, track_id: str):
        """Remove track from session"""
        self.tracks = [t for t in self.tracks if t.id != track_id]
    
    def get_track(self, track_id: str) -> Optional[Track]:
        """Get track by ID"""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary for JSON storage"""
        return {
            "format_version": "1.0",
            "id": self.id,
            "name": self.name,
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "tempo_bpm": self.tempo_bpm,
            "time_sig_num": self.time_sig_num,
            "time_sig_den": self.time_sig_den,
            "tracks": [self._track_to_dict(t) for t in self.tracks],
            "master_track": self._track_to_dict(self.master_track),
            "automation_lanes": [self._automation_to_dict(a) for a in self.automation_lanes],
            "transport": {
                "loop_enabled": self.transport.loop_enabled,
                "loop_start_beats": self.transport.loop_start_beats,
                "loop_end_beats": self.transport.loop_end_beats,
                "recording_mode": self.transport.recording_mode.value,
                "automation_mode": self.transport.automation_mode.value,
            },
            "metadata": self.metadata
        }
    
    def _track_to_dict(self, track: 'Track') -> Dict[str, Any]:
        """Serialize a track"""
        return {
            "id": track.id,
            "name": track.name,
            "kind": track.kind.value,
            "color": track.color,
            "volume": track.volume,
            "pan": track.pan,
            "mute": track.mute,
            "solo": track.solo,
            "armed": track.armed,
            "clips": [self._clip_to_dict(c) for c in track.clips],
        }
    
    def _clip_to_dict(self, clip: 'Clip') -> Dict[str, Any]:
        """Serialize a clip"""
        return {
            "id": clip.id,
            "name": clip.name,
            "kind": clip.kind.value,
            "color": clip.color,
            "start_beat": clip.start_beat,
            "length_beats": clip.length_beats,
            "offset_beats": clip.offset_beats,
            "loop_enabled": clip.loop_enabled,
            "muted": clip.muted,
        }
    
    def _automation_to_dict(self, lane: 'AutomationLane') -> Dict[str, Any]:
        """Serialize an automation lane"""
        return {
            "id": lane.id,
            "target_track_id": lane.target_track_id,
            "target_device_id": lane.target_device_id,
            "target_param": lane.target_param,
            "points": [{"beat": p.beat, "value": p.value, "curve": p.curve} for p in lane.points]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Deserialize session from dictionary"""
        session = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Untitled"),
            sample_rate=data.get("sample_rate", 48000.0),
            block_size=data.get("block_size", 256),
            tempo_bpm=data.get("tempo_bpm", 120.0),
            time_sig_num=data.get("time_sig_num", 4),
            time_sig_den=data.get("time_sig_den", 4),
            metadata=data.get("metadata", {})
        )
        
        # Restore tracks
        for track_data in data.get("tracks", []):
            track = cls._track_from_dict(track_data)
            session.tracks.append(track)
        
        # Restore master
        if "master_track" in data:
            session.master_track = cls._track_from_dict(data["master_track"])
        
        # Restore automation
        for lane_data in data.get("automation_lanes", []):
            lane = cls._automation_from_dict(lane_data)
            session.automation_lanes.append(lane)
        
        # Restore transport settings
        if "transport" in data:
            t = data["transport"]
            session.transport.loop_enabled = t.get("loop_enabled", False)
            session.transport.loop_start_beats = t.get("loop_start_beats", 0.0)
            session.transport.loop_end_beats = t.get("loop_end_beats", 8.0)
            if "recording_mode" in t:
                session.transport.recording_mode = RecordingMode(t["recording_mode"])
            if "automation_mode" in t:
                session.transport.automation_mode = AutomationMode(t["automation_mode"])
        
        return session
    
    @staticmethod
    def _track_from_dict(data: Dict[str, Any]) -> 'Track':
        """Deserialize a track"""
        track = Track(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Track"),
            kind=TrackKind(data.get("kind", "audio")),
            color=data.get("color", "#00aaff"),
            volume=data.get("volume", 1.0),
            pan=data.get("pan", 0.0),
            mute=data.get("mute", False),
            solo=data.get("solo", False),
            armed=data.get("armed", False),
        )
        
        for clip_data in data.get("clips", []):
            clip = Session._clip_from_dict(clip_data)
            track.clips.append(clip)
        
        return track
    
    @staticmethod
    def _clip_from_dict(data: Dict[str, Any]) -> 'Clip':
        """Deserialize a clip"""
        return Clip(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Clip"),
            kind=ClipKind(data.get("kind", "audio")),
            color=data.get("color", "#00aaff"),
            start_beat=data.get("start_beat", 0.0),
            length_beats=data.get("length_beats", 4.0),
            offset_beats=data.get("offset_beats", 0.0),
            loop_enabled=data.get("loop_enabled", False),
            muted=data.get("muted", False),
        )
    
    @staticmethod
    def _automation_from_dict(data: Dict[str, Any]) -> 'AutomationLane':
        """Deserialize an automation lane"""
        lane = AutomationLane(
            id=data.get("id", str(uuid.uuid4())[:8]),
            target_track_id=data.get("target_track_id", ""),
            target_device_id=data.get("target_device_id", ""),
            target_param=data.get("target_param", ""),
        )
        
        for pt in data.get("points", []):
            lane.points.append(AutomationPoint(
                beat=pt.get("beat", 0.0),
                value=pt.get("value", 0.0),
                curve=pt.get("curve", "linear")
            ))
        
        return lane


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT FILE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

import json
from pathlib import Path
from datetime import datetime


class ProjectManager:
    """
    Handles project file operations:
    - Save/load .qsproj files
    - Recent projects list
    - Autosave/crash recovery
    """
    
    PROJECT_EXT = ".qsproj"
    AUTOSAVE_DIR = ".quantoniumos_autosave"
    RECENT_FILE = "recent_projects.json"
    MAX_RECENT = 10
    
    def __init__(self):
        self.config_dir = self._get_config_dir()
        self.autosave_dir = self.config_dir / self.AUTOSAVE_DIR
        self.autosave_dir.mkdir(parents=True, exist_ok=True)
        
        self.recent_projects: List[Dict[str, Any]] = []
        self._load_recent()
    
    def _get_config_dir(self) -> Path:
        """Get the config directory"""
        if os.name == 'nt':
            base = Path(os.environ.get('APPDATA', Path.home()))
        else:
            base = Path.home() / '.config'
        
        config_dir = base / 'quantoniumos'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    
    def save_project(self, session: Session, path: str) -> bool:
        """Save session to project file"""
        try:
            filepath = Path(path)
            if not filepath.suffix:
                filepath = filepath.with_suffix(self.PROJECT_EXT)
            
            data = session.to_dict()
            data["saved_at"] = datetime.now().isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Update recent projects
            self._add_to_recent(str(filepath), session.name)
            
            return True
        except Exception as e:
            print(f"Error saving project: {e}")
            return False
    
    def load_project(self, path: str) -> Optional[Session]:
        """Load session from project file"""
        try:
            filepath = Path(path)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = Session.from_dict(data)
            
            # Update recent projects
            self._add_to_recent(str(filepath), session.name)
            
            return session
        except Exception as e:
            print(f"Error loading project: {e}")
            return None
    
    def autosave(self, session: Session):
        """Save an autosave backup"""
        try:
            autosave_path = self.autosave_dir / f"autosave_{session.id}.qsproj"
            
            data = session.to_dict()
            data["autosaved_at"] = datetime.now().isoformat()
            
            with open(autosave_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Autosave failed: {e}")
    
    def recover_autosave(self) -> List[Dict[str, Any]]:
        """Get list of recoverable autosaves"""
        recoverable = []
        
        for autosave in self.autosave_dir.glob("autosave_*.qsproj"):
            try:
                with open(autosave, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                recoverable.append({
                    "path": str(autosave),
                    "name": data.get("name", "Unknown"),
                    "saved_at": data.get("autosaved_at", "Unknown"),
                })
            except Exception:
                pass
        
        return recoverable
    
    def clear_autosave(self, session_id: str):
        """Clear autosave for a session"""
        autosave_path = self.autosave_dir / f"autosave_{session_id}.qsproj"
        if autosave_path.exists():
            autosave_path.unlink()
    
    def _add_to_recent(self, path: str, name: str):
        """Add project to recent list"""
        # Remove if already exists
        self.recent_projects = [p for p in self.recent_projects if p["path"] != path]
        
        # Add to front
        self.recent_projects.insert(0, {
            "path": path,
            "name": name,
            "accessed_at": datetime.now().isoformat()
        })
        
        # Trim to max
        self.recent_projects = self.recent_projects[:self.MAX_RECENT]
        
        # Save
        self._save_recent()
    
    def _load_recent(self):
        """Load recent projects list"""
        try:
            recent_file = self.config_dir / self.RECENT_FILE
            if recent_file.exists():
                with open(recent_file, 'r', encoding='utf-8') as f:
                    self.recent_projects = json.load(f)
        except Exception:
            self.recent_projects = []
    
    def _save_recent(self):
        """Save recent projects list"""
        try:
            recent_file = self.config_dir / self.RECENT_FILE
            with open(recent_file, 'w', encoding='utf-8') as f:
                json.dump(self.recent_projects, f, indent=2)
        except Exception:
            pass
    
    def get_recent_projects(self) -> List[Dict[str, Any]]:
        """Get list of recent projects"""
        return self.recent_projects.copy()


# Global project manager instance
_project_manager: Optional[ProjectManager] = None


def get_project_manager() -> ProjectManager:
    """Get the global project manager"""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
# ═══════════════════════════════════════════════════════════════════════════════

class AudioEngine:
    """
    Real-time audio processing engine.
    Processes the session's audio graph each block.
    """
    
    def __init__(self, session: Session):
        self.session = session
        self.running = False
        
        # Pre-allocated buffers
        self._block_buffer = np.zeros(
            (2, session.block_size), dtype=np.float32
        )
        self._mix_buffer = np.zeros(
            (2, session.block_size), dtype=np.float32
        )
    
    def process_block(self) -> np.ndarray:
        """
        Process one audio block.
        Called by audio callback.
        
        Returns:
            Stereo output buffer (2, block_size)
        """
        session = self.session
        block_size = session.block_size
        
        # Clear mix buffer
        self._mix_buffer.fill(0)
        
        if not session.transport.playing:
            return self._mix_buffer
        
        current_beat = session.transport.position_beats
        
        # Check for solo
        any_solo = any(t.solo for t in session.tracks)
        
        # Process each track
        for track in session.tracks:
            # Skip muted tracks (unless soloed)
            if track.mute:
                continue
            if any_solo and not track.solo:
                continue
            
            # Get audio from clips at current position
            track_buffer = np.zeros((2, block_size), dtype=np.float32)
            
            for clip in track.get_clips_at(current_beat):
                clip_audio = clip.get_samples_at(
                    current_beat, 
                    session.beats_per_block,
                    session.sample_rate,
                    session.tempo_bpm
                )
                if clip_audio is not None:
                    # Add to track buffer (handle size mismatch)
                    samples = min(clip_audio.shape[1], block_size)
                    track_buffer[:, :samples] += clip_audio[:, :samples]
            
            # Process through device chain
            track_buffer = track.devices.process_block(
                track_buffer, session.sample_rate, block_size
            )
            
            # Apply track volume and pan
            track_buffer = track.apply_volume(track_buffer)
            track_buffer = track.apply_pan(track_buffer)
            
            # Sum to mix
            self._mix_buffer += track_buffer
        
        # Process through master chain
        output = session.master_track.devices.process_block(
            self._mix_buffer, session.sample_rate, block_size
        )
        output = session.master_track.apply_volume(output)
        
        # Advance transport
        session.transport.position_beats += session.beats_per_block
        session.transport.position_samples += block_size
        
        # Handle loop
        if session.transport.loop_enabled:
            if session.transport.position_beats >= session.transport.loop_end_beats:
                session.transport.position_beats = session.transport.loop_start_beats
        
        return output
    
    def start(self):
        """Start playback"""
        self.session.transport.playing = True
        self.running = True
    
    def stop(self):
        """Stop playback"""
        self.session.transport.playing = False
        self.running = False
    
    def seek(self, beat: float):
        """Seek to beat position"""
        self.session.transport.position_beats = beat
        self.session.transport.position_samples = int(beat * self.session.samples_per_beat)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'PHI', 'PHI_INV',
    'Domain', 'TrackKind', 'ClipKind', 'DeviceKind',
    'RecordingMode', 'AutomationMode',
    'WaveField', 'rft_forward', 'rft_inverse', 'rft_overlap_add',
    'Note', 'CCEvent', 'AudioPayload', 'MidiPayload', 'Clip',
    'AutomationPoint', 'AutomationLane',
    'DevicePort', 'DeviceConnection', 'DeviceNode', 'DeviceChain',
    'Track', 'TransportState', 'Session',
    'AudioEngine'
]
