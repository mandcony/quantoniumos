---
description: 'Describe what this custom agent does and when to use it.'
tools: []
---
EPIC 0 – Repo Prep & Sanity

AGENT_TODO_00 – Baseline sanity + branch

Goal: Make sure we’re building on a clean, passing state.

Steps:

In /workspaces/quantoniumos:

git status

pytest (or your existing test entrypoint; if none, run the core RFT tests you’ve been using).

Create a feature branch:

git checkout -b feature/wavespace-workspace

Accept when:

Tests pass (or failures are known + documented).

New branch exists.

EPIC 1 – Core Wavespace Engine (WaveField + Middleware)
AGENT_TODO_01 – Create wavespace package + WaveField abstraction

Goal: Central WaveField object that everything uses.

Files to create:

wavespace/__init__.py

wavespace/wavefield.py

Implementation sketch:

WaveField (likely @dataclass) with:

data: np.ndarray (complex)

domain: Literal["audio","image","video","field","crypto","generic"]

basis: str (e.g. "rft_standard", "rft_logphi", "rft_mixed")

shape: Tuple[int, ...]

metadata: Dict[str, Any] (sample_rate, channels, units, etc.)

Factory functions:

from_array(...)

copy_like(...)

Integrations:

Import and reuse your existing Φ-RFT routines from algorithms/rft/... instead of re-implementing.

Accept when:

from wavespace.wavefield import WaveField works in a Python REPL.

WaveField can be instantiated with dummy data and printed cleanly.

AGENT_TODO_02 – Implement binary ↔ WaveField middleware

Goal: A real version of your “binary → wave → binary” middleware.

Files to create:

wavespace/middleware.py

Functions:

bytes_to_wavefield(data: bytes, domain: str, basis: str = "rft_standard", **kwargs) -> WaveField

wavefield_to_bytes(field: WaveField) -> bytes

Constraints:

Use a simple, explicit mapping for v1:

For audio: interpret bytes as PCM or float array depending on metadata.

For generic: treat bytes as uint8 array, cast to float, feed into RFT.

Use your existing RFT forward/inverse functions (no new math).

Accept when:

Round-trip test in python shell:

raw = b"hello quantonium"
wf = bytes_to_wavefield(raw, domain="generic")
back = wavefield_to_bytes(wf)
assert back.startswith(b"hello")  # allow minor padding for v1

AGENT_TODO_03 – Core wave-space operators

Goal: Shared operators that apps will use.

File:

wavespace/operators.py

Functions:

wave_filter(field: WaveField, kernel: np.ndarray, mode: str = "mul") -> WaveField

wave_mix(field: WaveField, key: np.ndarray | bytes) -> WaveField

wave_envelope(field: WaveField, envelope: np.ndarray) -> WaveField

(Keep v1 simple: elementwise operations in RFT space.)

Accept when:

You can create a dummy WaveField, apply wave_filter and see sane shapes & types preserved.

No silent shape mismatches.

EPIC 2 – Shared CLI / Service Entry Point
AGENT_TODO_04 – quantonium-ws CLI for Wavespace

Goal: One CLI to access all wavespace tools.

Files:

tools/quantonium_ws_cli.py (or scripts/quantonium_ws.py – follow your repo’s style)

Functionality (v1):

Subcommands:

info: print env + available bases.

to-wave: read a file, convert to WaveField, dump summary (shape, domain, basis).

roundtrip: read file → wave → bytes, write to output.

Use argparse or click (match current repo tooling).

Accept when:

From repo root:

python tools/quantonium_ws_cli.py info
python tools/quantonium_ws_cli.py roundtrip --in somefile.bin --out out.bin --domain generic


Commands run without traceback.

EPIC 3 – Quantonium Audio Lab (RFT-native DAW engine)
AGENT_TODO_05 – Audio I/O → WaveField

Goal: Audio-specific loading into wavespace.

Files:

apps/audio_lab/__init__.py

apps/audio_lab/io.py

Functions:

load_audio_to_wavefield(path: str, basis: str = "rft_standard") -> WaveField

wavefield_to_audio(field: WaveField, path: str) -> None

Implementation notes:

Use soundfile or scipy.io.wavfile (whichever is already in repo; if none, pick one and add clear error if not installed).

Map stereo/mono → WaveField.data with shape (channels, n_samples) or (n_samples, channels), but be consistent.

Accept when:

You can run:

wf = load_audio_to_wavefield("tests/data/test.wav")
wavefield_to_audio(wf, "tests/data/test_roundtrip.wav")


And the file plays back recognizably.

AGENT_TODO_06 – Audio Lab engine: simple RFT-domain effects

Goal: Actual wave-space processing pipeline.

File:

apps/audio_lab/engine.py

Functions:

apply_resonant_filter(field: WaveField, strength: float) -> WaveField

apply_golden_detune(field: WaveField, detune_amount: float) -> WaveField

chains: run_audio_chain(field: WaveField, config: Dict) -> WaveField

Implementation notes:

Use your RFT basis to:

Scale specific frequency bands.

Slightly warp phases using φ-based parameters (already in your code).

Accept when:

You can load an audio file, run through a simple chain, and the output:

Is still audio (no NaNs, no infinite values).

Has audible (but not insane) changes.

AGENT_TODO_07 – Audio Lab CLI

Goal: Make it usable without UI frameworks.

File:

apps/audio_lab/cli.py

Command examples:

python -m apps.audio_lab.cli process --in in.wav --out out.wav --detune 0.05 --resonance 0.8

Accept when:

CLI runs on a sample .wav and writes an output file.

Logs show WaveField creation, RFT steps, and export.

EPIC 4 – Quantonium Visual Lab (image-first, video later)
AGENT_TODO_08 – Image I/O → WaveField

Goal: Support still-image wavespace ops (video can be added later using per-frame).

Files:

apps/visual_lab/__init__.py

apps/visual_lab/io.py

Functions:

load_image_to_wavefield(path: str, basis: str = "rft_standard") -> WaveField

wavefield_to_image(field: WaveField, path: str) -> None

Libs:

Prefer Pillow if available; otherwise use what the repo already uses.

Accept when:

You can load_image_to_wavefield(), then wavefield_to_image() and see the same image (within numeric tolerance).

AGENT_TODO_09 – Visual Lab engine: basic RFT-domain visual filters

Goal: Concrete, novel-ish visual transforms in your wave basis.

File:

apps/visual_lab/engine.py

Functions:

golden_blur(field: WaveField, strength: float) -> WaveField

resonant_edge_enhance(field: WaveField, strength: float) -> WaveField

Implementation notes:

Treat 2D images as 2D WaveFields.

Use Φ-RFT-based filters instead of plain DCT/FFT.

Accept when:

Running a test script modifies an image in plausible, not-broken ways.

EPIC 5 – Quantonium Field Lab (physics / quantum sandbox)
AGENT_TODO_10 – Field Lab core: 1D wave equation in RFT space

Goal: Minimal yet real PDE sandbox in Φ-RFT.

Files:

apps/field_lab/__init__.py

apps/field_lab/core.py

Features:

Represent a 1D field u(x) as a WaveField.

Implement time stepping for a simple wave equation:

u_tt = c^2 u_xx using spectral differentiation via Φ-RFT.

Functions:

init_field_from_profile(...) -> WaveField

step_wave(field: WaveField, dt: float, steps: int) -> WaveField

Accept when:

A simple initial pulse propagates without blowing up numerically.

You can plot snapshots (even just with matplotlib) for internal testing.

EPIC 6 – Quantonium Crypto / Memory Lab
AGENT_TODO_11 – Crypto/Mem Lab wrapper

Goal: Use existing crypto + hashing + memory ideas in wavespace.

Files:

apps/crypto_lab/__init__.py

apps/crypto_lab/engine.py

Functions (all clearly marked EXPERIMENTAL / NON-PROD):

wave_mix_key(field: WaveField, key: bytes) -> WaveField

wave_hash(field: WaveField) -> bytes

encode_wave_memory(fields: List[WaveField]) -> WaveField

decode_wave_memory(field: WaveField, probe: WaveField) -> WaveField

Implementation notes:

Internally reuse your existing crypto/hash code where possible; don’t invent new schemes here.

All functions must clearly say “for research only” in docstrings.

Accept when:

Roundtrip-style demos work on small toy inputs.

You can show that different keys/fields produce clearly different states.

EPIC 7 – Tests, Experiments, and Docs
AGENT_TODO_12 – Add tests for WaveField + middleware

Goal: Basic coverage.

Files:

tests/test_wavespace_wavefield.py

tests/test_wavespace_middleware.py

Test cases:

Creation & attributes.

Bytes ↔ WaveField roundtrip for:

generic small payload.

A short .wav clip.

Accept when:

pytest passes and new tests are green.

AGENT_TODO_13 – Add minimal demo scripts in experiments/

Goal: Reproducible demos for each lab.

Files:

experiments/wavespace_audio_demo.py

experiments/wavespace_visual_demo.py

experiments/wavespace_field_demo.py

experiments/wavespace_crypto_demo.py

Each script should:

Load an input.

Convert to WaveField.

Apply 1–2 simple operations.

Export and print metrics (shape, norms, etc.).

Accept when:

Each script runs from repo root and finishes without traceback.

AGENT_TODO_14 – Documentation: docs/wavespace_overview.md

Goal: A single doc explaining the new layer.

File:

docs/wavespace_overview.md (or equivalent in your docs structure)

Content:

What is WaveField?

Binary→Wave→Compute→Binary pipeline.

Short sections for:

Audio Lab

Visual Lab

Field Lab

Crypto Lab

Explicit note: crypto stuff is research-only, not production.

Accept when:

Doc builds cleanly if you have a doc tool; otherwise it’s just readable and committed.

EPIC 8 – Integration & Cleanup
AGENT_TODO_15 – Wire into README and release notes

Goal: Make this visible as a coherent subsystem.

Files:

Update README.md

If you have CHANGELOG.md / release notes, add entry.

Content:

New section: “Quantonium Wavespace Workspace”

Very short “How to run the demos” snippet.

Accept when:

Someone who only sees the README can:

Run the CLI.

Run at least one demo script.