# Wave DAW - Φ-RFT Native Digital Audio Workstation
"""
Wave DAW: FL Studio/Ableton-class DAW powered by wave-native Φ-RFT engine.

Core Philosophy:
- Wave-first: Audio processed in RFT wave domain, not just time domain
- Φ-native: Golden ratio resonance at the heart of effects/synthesis
- Professional: Compete with FL Studio, Ableton, Logic in capabilities
"""

from .engine import (
    WaveField,
    Clip,
    Track,
    DeviceNode,
    DeviceChain,
    Session,
    AudioEngine,
    TransportState,
    Domain,
    TrackKind,
    ClipKind,
    DeviceKind,
)

from .devices import (
    create_device,
    create_utility,
    create_rft_eq,
    create_rft_morph,
    create_rft_filter,
    create_compressor,
    create_rft_reverb,
    create_meter,
)

# Pattern editor and drum synthesis
try:
    from .pattern_editor import (
        Pattern,
        PatternRow,
        PatternStep,
        PatternPlayer,
        PatternEditorWidget,
        DrumType,
        DrumSound,
        DrumSynthesizer,
        InstrumentSelector,
    )
    PATTERN_AVAILABLE = True
except ImportError:
    PATTERN_AVAILABLE = False

__all__ = [
    "WaveField",
    "Clip",
    "Track",
    "DeviceNode",
    "DeviceChain",
    "Session",
    "AudioEngine",
    "TransportState",
    "Domain",
    "TrackKind",
    "ClipKind",
    "DeviceKind",
    "create_device",
    "create_utility",
    "create_rft_eq",
    "create_rft_morph",
    "create_rft_filter",
    "create_compressor",
    "create_rft_reverb",
    "create_meter",
]

__version__ = "0.1.0"
