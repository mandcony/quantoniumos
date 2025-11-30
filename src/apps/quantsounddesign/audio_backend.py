#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantSoundDesign Audio Backend - Real-time audio I/O with sounddevice.

EPIC 1: Rock-solid, low-latency audio engine for live recording and monitoring.
Integrates with Session/AudioEngine for proper audio graph processing.

Features:
- Configurable sample rate (44.1k/48k/96k)
- Configurable buffer size (64/128/256/512)
- Callback-based processing with minimal allocation
- XRun detection and performance monitoring
- Sample-accurate transport synchronization
"""

import numpy as np
import threading
import time
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Warning: sounddevice not installed. Audio playback disabled.")

# Golden ratio for RFT processing
PHI = (1 + np.sqrt(5)) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class LatencyMode(Enum):
    """Latency mode presets"""
    ULTRA_LOW = "ultra_low"    # 64 samples - minimum latency, high CPU
    LOW = "low"                # 128 samples - good for live playing
    BALANCED = "balanced"      # 256 samples - default, stable
    SAFE = "safe"              # 512 samples - for complex projects
    HIGH = "high"              # 1024 samples - maximum stability


class SampleRate(Enum):
    """Supported sample rates"""
    SR_44100 = 44100
    SR_48000 = 48000
    SR_88200 = 88200
    SR_96000 = 96000


# Buffer size presets by latency mode
BUFFER_SIZES = {
    LatencyMode.ULTRA_LOW: 64,
    LatencyMode.LOW: 128,
    LatencyMode.BALANCED: 256,
    LatencyMode.SAFE: 512,
    LatencyMode.HIGH: 1024,
}


@dataclass
class AudioDeviceInfo:
    """Information about an audio device"""
    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


@dataclass
class AudioSettings:
    """
    Complete audio configuration for QuantSoundDesign.
    
    This is the single source of truth for all audio settings.
    Can be saved per-session or globally.
    """
    # Core settings
    sample_rate: int = 48000
    buffer_size: int = 256
    channels: int = 2
    dtype: str = 'float32'
    
    # Device selection (None = system default)
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    
    # Latency mode (for presets)
    latency_mode: LatencyMode = LatencyMode.BALANCED
    
    # Performance tuning
    use_exclusive_mode: bool = False  # WASAPI exclusive on Windows
    use_low_latency: bool = True      # Request low latency from driver
    
    # Monitoring
    enable_performance_monitoring: bool = True
    xrun_callback: Optional[Callable[[str], None]] = None
    
    def get_latency_ms(self) -> float:
        """Calculate one-way latency in milliseconds"""
        return (self.buffer_size / self.sample_rate) * 1000
    
    def get_roundtrip_latency_ms(self) -> float:
        """Calculate roundtrip (input + output) latency in ms"""
        return self.get_latency_ms() * 2
    
    @classmethod
    def from_latency_mode(cls, mode: LatencyMode, 
                          sample_rate: int = 48000) -> "AudioSettings":
        """Create settings from a latency mode preset"""
        return cls(
            sample_rate=sample_rate,
            buffer_size=BUFFER_SIZES[mode],
            latency_mode=mode
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for saving"""
        return {
            "sample_rate": self.sample_rate,
            "buffer_size": self.buffer_size,
            "channels": self.channels,
            "input_device": self.input_device,
            "output_device": self.output_device,
            "latency_mode": self.latency_mode.value,
            "use_exclusive_mode": self.use_exclusive_mode,
            "use_low_latency": self.use_low_latency,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioSettings":
        """Deserialize from dict"""
        mode = LatencyMode(data.get("latency_mode", "balanced"))
        return cls(
            sample_rate=data.get("sample_rate", 48000),
            buffer_size=data.get("buffer_size", 256),
            channels=data.get("channels", 2),
            input_device=data.get("input_device"),
            output_device=data.get("output_device"),
            latency_mode=mode,
            use_exclusive_mode=data.get("use_exclusive_mode", False),
            use_low_latency=data.get("use_low_latency", True),
        )


@dataclass
class PerformanceStats:
    """Real-time performance statistics"""
    # XRun counts
    underruns: int = 0
    overruns: int = 0
    
    # CPU load estimation
    callback_time_us: float = 0.0  # Average callback processing time
    max_callback_time_us: float = 0.0
    buffer_time_us: float = 0.0  # Time available per buffer
    cpu_load_percent: float = 0.0
    
    # Timing
    last_callback_time: float = 0.0
    callbacks_total: int = 0
    
    # History for averaging
    _callback_times: List[float] = field(default_factory=list)
    _max_history: int = 100
    
    def record_callback(self, duration_us: float):
        """Record a callback duration"""
        self._callback_times.append(duration_us)
        if len(self._callback_times) > self._max_history:
            self._callback_times.pop(0)
        
        self.callback_time_us = np.mean(self._callback_times)
        self.max_callback_time_us = max(self.max_callback_time_us, duration_us)
        self.callbacks_total += 1
        
        if self.buffer_time_us > 0:
            self.cpu_load_percent = (self.callback_time_us / self.buffer_time_us) * 100
    
    def record_xrun(self, is_underrun: bool = True):
        """Record an XRun event"""
        if is_underrun:
            self.underruns += 1
        else:
            self.overruns += 1
    
    def reset(self):
        """Reset all stats"""
        self.underruns = 0
        self.overruns = 0
        self.callback_time_us = 0.0
        self.max_callback_time_us = 0.0
        self.cpu_load_percent = 0.0
        self.callbacks_total = 0
        self._callback_times.clear()


# Legacy compatibility
@dataclass
class AudioBackendConfig:
    """Audio backend configuration (legacy - use AudioSettings instead)"""
    sample_rate: int = 44100
    block_size: int = 512
    channels: int = 2
    dtype: str = 'float32'
    
    def to_audio_settings(self) -> AudioSettings:
        """Convert to new AudioSettings"""
        return AudioSettings(
            sample_rate=self.sample_rate,
            buffer_size=self.block_size,
            channels=self.channels,
            dtype=self.dtype
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE ENUMERATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_audio_devices() -> Tuple[List[AudioDeviceInfo], List[AudioDeviceInfo]]:
    """
    Get available audio input and output devices.
    
    Returns:
        Tuple of (input_devices, output_devices)
    """
    if sd is None:
        return [], []
    
    input_devices = []
    output_devices = []
    
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        
        for i, dev in enumerate(devices):
            info = AudioDeviceInfo(
                id=i,
                name=dev['name'],
                max_input_channels=dev['max_input_channels'],
                max_output_channels=dev['max_output_channels'],
                default_sample_rate=dev['default_samplerate'],
                is_default_input=(i == default_input),
                is_default_output=(i == default_output),
            )
            
            if dev['max_input_channels'] > 0:
                input_devices.append(info)
            if dev['max_output_channels'] > 0:
                output_devices.append(info)
    except Exception as e:
        print(f"Error querying audio devices: {e}")
    
    return input_devices, output_devices


def get_default_device_info() -> Dict[str, Any]:
    """Get information about default audio devices"""
    if sd is None:
        return {"available": False}
    
    try:
        return {
            "available": True,
            "default_input": sd.default.device[0],
            "default_output": sd.default.device[1],
            "default_samplerate": sd.default.samplerate,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class AudioBackend:
    """
    Real-time audio backend using sounddevice.
    
    Features:
    - Callback-based audio processing (minimal latency)
    - Configurable sample rate and buffer size
    - XRun detection and performance monitoring
    - Sample-accurate transport synchronization
    
    Integrates with Session/AudioEngine for:
    - Playing back DAW tracks through the audio graph
    - Synth input for real-time keyboard playing
    - Metronome and click track
    """
    
    def __init__(self, config: AudioBackendConfig = None, 
                 settings: AudioSettings = None,
                 session=None):
        # Handle both old and new config styles
        if settings is not None:
            self.settings = settings
        elif config is not None:
            self.settings = config.to_audio_settings()
        else:
            self.settings = AudioSettings()
        
        # Legacy config for backward compatibility
        self.config = AudioBackendConfig(
            sample_rate=self.settings.sample_rate,
            block_size=self.settings.buffer_size,
            channels=self.settings.channels,
            dtype=self.settings.dtype
        )
        
        self.stream: Optional[sd.OutputStream] = None
        self.is_running = False
        
        # Session and engine references
        self.session = session
        self.engine = session.engine if session else None
        
        # Playback state (fallback when no session)
        self.playing = False
        self.position_samples = 0
        self.tempo_bpm = 120.0
        self.metronome_enabled = True
        
        # Audio generators
        self.generators: List[Callable] = []
        
        # Clips to play (start_sample, audio_data) - legacy fallback
        self.clips: List[tuple] = []
        
        # Preview sounds (one-shots triggered by UI for instant playback)
        self.preview_sounds: List[tuple] = []  # (offset, audio_data)
        
        # Synth for real-time keyboard playback
        self.synth = None
        
        # Pattern player for drum/step sequencer playback
        self.pattern_player = None
        self.drum_synth = None
        self._last_step = -1  # Track which step was last triggered
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Metronome state
        self._metro_phase = 0.0
        self._last_beat = -1
        
        # Performance monitoring
        self.stats = PerformanceStats()
        self.stats.buffer_time_us = (self.settings.buffer_size / 
                                      self.settings.sample_rate) * 1_000_000
        
        # Callback timing
        self._callback_start_time = 0.0
    
    def attach_session(self, session):
        """Attach a Session after construction"""
        with self.lock:
            self.session = session
            self.engine = session.engine if session else None
            if session:
                self.tempo_bpm = session.tempo_bpm if hasattr(session, 'tempo_bpm') else session.tempo
    
    def set_synth(self, synth):
        """Set the synth for real-time playback"""
        with self.lock:
            self.synth = synth
    
    def set_pattern_player(self, pattern_player, drum_synth):
        """Set the pattern player and drum synth for step sequencer playback"""
        with self.lock:
            self.pattern_player = pattern_player
            self.drum_synth = drum_synth
    
    def apply_settings(self, settings: AudioSettings, restart: bool = True) -> bool:
        """
        Apply new audio settings.
        
        Args:
            settings: New audio settings
            restart: Whether to restart the stream
        
        Returns:
            True if successful
        """
        was_running = self.is_running
        
        if was_running and restart:
            self.stop()
        
        self.settings = settings
        self.config = AudioBackendConfig(
            sample_rate=settings.sample_rate,
            block_size=settings.buffer_size,
            channels=settings.channels,
            dtype=settings.dtype
        )
        
        # Update performance monitoring
        self.stats.buffer_time_us = (settings.buffer_size / 
                                      settings.sample_rate) * 1_000_000
        self.stats.reset()
        
        if was_running and restart:
            return self.start()
        
        return True
    
    def get_settings(self) -> AudioSettings:
        """Get current audio settings"""
        return self.settings
    
    def get_stats(self) -> PerformanceStats:
        """Get performance statistics"""
        return self.stats
        
    def start(self) -> bool:
        """Start the audio stream"""
        if sd is None:
            print("sounddevice not available")
            return False
            
        try:
            # Configure stream with settings
            latency = 'low' if self.settings.use_low_latency else 'high'
            
            self.stream = sd.OutputStream(
                samplerate=self.settings.sample_rate,
                blocksize=self.settings.buffer_size,
                channels=self.settings.channels,
                dtype=self.settings.dtype,
                device=self.settings.output_device,
                latency=latency,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_running = True
            
            latency_ms = self.settings.get_latency_ms()
            print(f"✓ Audio backend started: {self.settings.sample_rate}Hz, "
                  f"{self.settings.buffer_size} samples ({latency_ms:.1f}ms latency)")
            return True
        except Exception as e:
            print(f"✗ Failed to start audio: {e}")
            return False
    
    def stop(self):
        """Stop the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False
        print("Audio backend stopped")
    
    def play(self):
        """Start playback"""
        with self.lock:
            self.playing = True
            if self.session:
                self.session.transport.playing = True
    
    def pause(self):
        """Pause playback"""
        with self.lock:
            self.playing = False
            if self.session:
                self.session.transport.playing = False
    
    def stop_playback(self):
        """Stop and reset to beginning"""
        with self.lock:
            self.playing = False
            self.position_samples = 0
            self._last_beat = -1
            self._last_step = -1  # Reset step counter for pattern playback
            if self.session:
                self.session.transport.playing = False
                self.session.transport.position_beats = 0.0
                self.session.transport.position_samples = 0
    
    def set_tempo(self, bpm: float):
        """Set tempo in BPM"""
        bpm = max(20.0, min(300.0, bpm))
        with self.lock:
            self.tempo_bpm = bpm
            if self.session:
                self.session.tempo_bpm = bpm
    
    def set_position(self, beat: float):
        """Set position in beats"""
        samples_per_beat = (self.settings.sample_rate * 60) / self.tempo_bpm
        with self.lock:
            self.position_samples = int(beat * samples_per_beat)
            self._last_beat = int(beat) - 1
            if self.session:
                self.session.transport.position_beats = beat
                self.session.transport.position_samples = self.position_samples
    
    def get_position_beats(self) -> float:
        """Get current position in beats"""
        if self.session:
            return self.session.transport.position_beats
        samples_per_beat = (self.settings.sample_rate * 60) / self.tempo_bpm
        return self.position_samples / samples_per_beat
    
    def get_position_samples(self) -> int:
        """Get current position in samples (sample-accurate)"""
        if self.session:
            return self.session.transport.position_samples
        return self.position_samples
    
    def add_clip(self, start_beat: float, audio_data: np.ndarray):
        """Add an audio clip to play (legacy - prefer using Session tracks)"""
        samples_per_beat = (self.settings.sample_rate * 60) / self.tempo_bpm
        start_sample = int(start_beat * samples_per_beat)
        with self.lock:
            self.clips.append((start_sample, audio_data))
    
    def clear_clips(self):
        """Clear all clips"""
        with self.lock:
            self.clips.clear()

    def play_preview(self, audio_data: np.ndarray):
        """Queue a one-shot preview sound for immediate playback (e.g. drum hit when step toggled)"""
        with self.lock:
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            self.preview_sounds.append((0, audio_data))
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, 
                        time_info, status):
        """
        Audio callback - called by sounddevice for each block.
        This runs in a separate thread!
        
        IMPORTANT: Minimal work here - no disk I/O, no heavy allocations.
        """
        # Start timing for performance monitoring
        callback_start = time.perf_counter()
        
        # Check for XRuns
        if status:
            if status.output_underflow:
                self.stats.record_xrun(is_underrun=True)
                if self.settings.xrun_callback:
                    self.settings.xrun_callback("underrun")
            if status.output_overflow:
                self.stats.record_xrun(is_underrun=False)
                if self.settings.xrun_callback:
                    self.settings.xrun_callback("overrun")
        
        # Start with silence
        outdata.fill(0)
        
        with self.lock:
            # Always render synth (for keyboard playing even when stopped)
            if self.synth is not None:
                try:
                    synth_audio = self.synth.render(frames)
                    # Convert mono to stereo
                    outdata[:, 0] += synth_audio
                    outdata[:, 1] += synth_audio
                except Exception:
                    pass  # Ignore synth errors

            # Always mix preview sounds (drum hits, etc.) even when stopped
            self._mix_preview_sounds(outdata, frames)
            
            # Use engine if available, otherwise fallback
            if self.engine and self.session:
                self._process_with_engine(outdata, frames)
            else:
                self._process_fallback(outdata, frames)
        
        # Record callback performance (outside lock for accuracy)
        if self.settings.enable_performance_monitoring:
            callback_duration_us = (time.perf_counter() - callback_start) * 1_000_000
            self.stats.record_callback(callback_duration_us)
    
    def _process_with_engine(self, outdata: np.ndarray, frames: int):
        """Process audio through the Session's AudioEngine"""
        if not self.session.transport.playing:
            return
        
        try:
            # Get audio from the engine (processes all tracks and devices)
            engine_out = self.engine.process_block(frames)
            
            # Engine returns (channels, frames) or (frames,)
            if engine_out is not None:
                if engine_out.ndim == 1:
                    outdata[:, 0] += engine_out[:frames]
                    outdata[:, 1] += engine_out[:frames]
                else:
                    outdata[:, 0] += engine_out[0, :frames]
                    outdata[:, 1] += engine_out[1, :frames] if engine_out.shape[0] > 1 else engine_out[0, :frames]
            
            # Calculate timing for metronome
            samples_per_beat = (self.settings.sample_rate * 60) / self.session.tempo_bpm
            current_beat = self.session.transport.position_beats
            
            # Generate metronome clicks
            if self.metronome_enabled:
                self._generate_metronome(outdata, frames, current_beat, samples_per_beat)
            
            # Advance transport (sample-accurate)
            self.session.transport.position_samples += frames
            beat_increment = frames / samples_per_beat
            self.session.transport.position_beats += beat_increment
            
        except Exception as e:
            print(f"Engine process error: {e}")
    
    def _process_fallback(self, outdata: np.ndarray, frames: int):
        """Fallback processing without Session (metronome + legacy clips + patterns)"""
        if not self.playing:
            return
        
        # Calculate timing
        samples_per_beat = (self.settings.sample_rate * 60) / self.tempo_bpm
        current_beat = self.position_samples / samples_per_beat
        
        # Generate metronome clicks
        if self.metronome_enabled:
            self._generate_metronome(outdata, frames, current_beat, samples_per_beat)
        
        # Trigger pattern steps (16th notes = 4 steps per beat)
        self._process_pattern_steps(current_beat)
        
        # Mix in any legacy clips
        for start_sample, audio_data in self.clips:
            self._mix_clip(outdata, frames, start_sample, audio_data)
        
        # Advance position (sample-accurate)
        self.position_samples += frames
    
    def _process_pattern_steps(self, current_beat: float):
        """Trigger pattern steps at the correct time"""
        if not self.pattern_player or not self.drum_synth:
            return
        
        # Calculate current step (16th notes = 4 steps per beat)
        steps_per_beat = 4  # 16th notes
        current_step = int(current_beat * steps_per_beat)
        
        # Check if we crossed a step boundary
        if current_step > self._last_step:
            self._last_step = current_step
            
            # Trigger the pattern player at this step
            # PatternPlayer.trigger_step expects step index within pattern (0-15 for 16 steps)
            for track_id, pattern in self.pattern_player.patterns.items():
                pattern_step = current_step % pattern.steps
                
                for row, step_data in pattern.get_active_steps_at(pattern_step):
                    # Check probability
                    import numpy as np
                    if np.random.random() > step_data.probability:
                        continue
                    
                    velocity = step_data.velocity * row.volume
                    
                    if pattern.is_drum and row.drum_type:
                        # Synthesize and queue drum sound
                        audio = self.drum_synth.synthesize(
                            row.drum_type,
                            velocity=velocity,
                            duration=0.3
                        )
                        self.preview_sounds.append((0, audio))
    
    def _generate_metronome(self, outdata: np.ndarray, frames: int,
                            current_beat: float, samples_per_beat: float):
        """Generate metronome clicks"""
        current_beat_int = int(current_beat)
        
        # Check if we crossed a beat boundary
        if current_beat_int > self._last_beat:
            self._last_beat = current_beat_int
            
            # Determine click parameters
            is_downbeat = (current_beat_int % 4) == 0
            freq = 1000.0 if is_downbeat else 800.0
            amplitude = 0.5 if is_downbeat else 0.3
            click_samples = int(0.02 * self.settings.sample_rate)  # 20ms click
            
            # Generate click
            t = np.arange(click_samples) / self.settings.sample_rate
            
            # Sine with exponential decay
            envelope = np.exp(-t * 50)
            click = amplitude * np.sin(2 * np.pi * freq * t) * envelope
            
            # Add harmonics for richer sound
            click += amplitude * 0.3 * np.sin(2 * np.pi * freq * 2 * t) * envelope
            
            # Calculate where in the block the beat occurs
            beat_position_in_block = (current_beat - int(current_beat)) * samples_per_beat
            start_idx = max(0, int(frames - beat_position_in_block))
            
            # Mix click into output
            end_idx = min(frames, start_idx + len(click))
            click_len = end_idx - start_idx
            
            if click_len > 0 and start_idx < frames:
                outdata[start_idx:end_idx, 0] += click[:click_len]
                outdata[start_idx:end_idx, 1] += click[:click_len]
    
    def _mix_preview_sounds(self, outdata: np.ndarray, frames: int):
        """Mix one-shot preview sounds (drum hits, etc.) into output and advance their position"""
        new_previews = []
        for offset, audio in self.preview_sounds:
            if offset >= len(audio):
                continue  # Sound finished, discard
            end = min(offset + frames, len(audio))
            length = end - offset
            if length > 0:
                if audio.ndim == 1:
                    outdata[:length, 0] += audio[offset:end]
                    outdata[:length, 1] += audio[offset:end]
                else:
                    ch_count = min(audio.shape[0], 2)
                    for ch in range(ch_count):
                        outdata[:length, ch] += audio[ch, offset:end]
            # Keep if not finished
            if end < len(audio):
                new_previews.append((end, audio))
        self.preview_sounds = new_previews

    def _mix_clip(self, outdata: np.ndarray, frames: int,
                  start_sample: int, audio_data: np.ndarray):
        """Mix a clip into the output buffer (legacy)"""
        clip_start = start_sample - self.position_samples
        clip_end = clip_start + len(audio_data)
        
        # Check if clip overlaps with this block
        if clip_end < 0 or clip_start >= frames:
            return
        
        # Calculate overlap region
        out_start = max(0, clip_start)
        out_end = min(frames, clip_end)
        clip_offset = max(0, -clip_start)
        
        # Mix
        length = out_end - out_start
        if audio_data.ndim == 1:
            outdata[out_start:out_end, 0] += audio_data[clip_offset:clip_offset + length]
            outdata[out_start:out_end, 1] += audio_data[clip_offset:clip_offset + length]
        else:
            channels = min(audio_data.shape[0], 2)
            for ch in range(channels):
                outdata[out_start:out_end, ch] += audio_data[ch, clip_offset:clip_offset + length]


class TestToneGenerator:
    """Generate test tones for audio testing"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def sine(self, freq: float = 440.0, duration: float = 1.0, 
             amplitude: float = 0.5) -> np.ndarray:
        """Generate a sine wave"""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    
    def phi_chord(self, root: float = 220.0, duration: float = 2.0,
                  amplitude: float = 0.3) -> np.ndarray:
        """Generate a chord based on golden ratio intervals"""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        
        # Golden ratio frequency relationships
        freqs = [
            root,
            root * PHI,           # ~356 Hz
            root * PHI * PHI,     # ~576 Hz
            root * 2,             # Octave
        ]
        
        # Envelope
        attack = 0.1
        release = 0.5
        env = np.ones_like(t)
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        env[-release_samples:] = np.linspace(1, 0, release_samples)
        
        # Sum harmonics
        signal = np.zeros_like(t)
        for i, freq in enumerate(freqs):
            amp = amplitude / (i + 1)
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        return (signal * env).astype(np.float32)
    
    def drum_hit(self, duration: float = 0.3) -> np.ndarray:
        """Generate a simple drum-like sound"""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        
        # Pitch envelope (starts high, drops fast)
        pitch_env = 200 * np.exp(-t * 30) + 60
        
        # Amplitude envelope
        amp_env = np.exp(-t * 10)
        
        # Generate with frequency modulation
        phase = np.cumsum(2 * np.pi * pitch_env / self.sample_rate)
        signal = 0.6 * np.sin(phase) * amp_env
        
        # Add some noise for attack
        noise = np.random.randn(len(t)) * 0.2 * np.exp(-t * 50)
        signal += noise
        
        return signal.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL BACKEND MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Singleton backend instance
_backend: Optional[AudioBackend] = None
_settings: Optional[AudioSettings] = None


def get_audio_backend() -> AudioBackend:
    """Get or create the global audio backend"""
    global _backend, _settings
    if _backend is None:
        _backend = AudioBackend(settings=_settings or AudioSettings())
    return _backend


def get_audio_settings() -> AudioSettings:
    """Get current audio settings"""
    global _backend, _settings
    if _backend is not None:
        return _backend.get_settings()
    if _settings is None:
        _settings = AudioSettings()
    return _settings


def set_audio_settings(settings: AudioSettings) -> bool:
    """Set audio settings (will restart audio if running)"""
    global _backend, _settings
    _settings = settings
    if _backend is not None:
        return _backend.apply_settings(settings, restart=True)
    return True


def init_audio(settings: AudioSettings = None) -> bool:
    """Initialize the audio system with optional settings"""
    global _settings
    if settings is not None:
        _settings = settings
    backend = get_audio_backend()
    if settings is not None:
        backend.apply_settings(settings, restart=False)
    return backend.start()


def init_audio_with_session(session, settings: AudioSettings = None) -> AudioBackend:
    """Initialize audio with a Session for full DAW integration"""
    global _backend, _settings
    if settings is not None:
        _settings = settings
    
    if _backend is None:
        _backend = AudioBackend(settings=_settings or AudioSettings(), session=session)
    else:
        _backend.attach_session(session)
        if settings is not None:
            _backend.apply_settings(settings, restart=False)
    
    _backend.start()
    return _backend


def shutdown_audio():
    """Shutdown the audio system"""
    global _backend
    if _backend:
        _backend.stop()
        _backend = None


def get_performance_stats() -> Optional[PerformanceStats]:
    """Get current performance statistics"""
    global _backend
    if _backend:
        return _backend.get_stats()
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    'AudioSettings',
    'AudioDeviceInfo',
    'LatencyMode',
    'SampleRate',
    'PerformanceStats',
    'AudioBackendConfig',  # Legacy
    
    # Backend
    'AudioBackend',
    'TestToneGenerator',
    
    # Device enumeration
    'get_audio_devices',
    'get_default_device_info',
    
    # Global functions
    'get_audio_backend',
    'get_audio_settings',
    'set_audio_settings',
    'init_audio',
    'init_audio_with_session',
    'shutdown_audio',
    'get_performance_stats',
]