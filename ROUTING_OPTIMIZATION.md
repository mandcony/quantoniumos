# RFT Variant Routing Optimization

## Current Status

### ✅ Implemented Variants (13 in C/C++/Python)

All variants fully integrated across architecture layers:

| Variant | ID | Phase Formula | Properties | Best Use Case |
|---------|----|--------------|-----------|--------------| 
| **STANDARD** | 0 | θ(k) = 2π{k/φ} | Golden ratio, QR-orthogonalized | General compression |
| **HARMONIC** | 1 | θ(k) = 2πk³/N³ | Cubic chirp, harmonic structure | Audio analysis |
| **FIBONACCI** | 2 | θ(k) = 2π·F(k mod 32)/φ^k | Fibonacci scaling | Lattice crypto, integer structures |
| **CHAOTIC** | 3 | θ(k) = 2π·L(k)/k | Lyapunov chaotic | Diffusion, mixing layers |
| **PRIME** | 4 | θ(k) = 2π√(p_k)/k | Prime-indexed | Number theory, factorization |
| **ADAPTIVE** | 5 | θ(k) = 2πe^(-k/N) | Exponential decay | Multi-scale analysis |
| **SYMBOLIC** | 6 | θ(k) = 2πφ^k mod 2π | Symbolic qubit | Quantum state compression |
| **LOGARITHMIC** | 7 | θ(k) = 2πlog(1+k) | Log-periodic | Scale-invariant signals |
| **CASCADE** | 8 | Multi-stage hierarchy | η=0 zero coherence | Universal compression (WINNER) |
| **BRAIDED** | 9 | Parallel competition | 3-way adaptive mix | Heterogeneous data |
| **ADAPTIVE_SPLIT** | 10 | Variance threshold DCT/RFT | 50% BPP improvement | Structure/texture separation |
| **ENTROPY_GUIDED** | 11 | Entropy-based routing | 50% BPP on edges | Sharp edges, steps |
| **DICTIONARY** | 12 | Dictionary learning atoms | 49.9 dB PSNR | High quality, smooth signals |

### 🔧 Python-Only Variants (3)
- **GOLDBACH** - Goldbach conjecture exploration
- **COLLATZ** - Collatz sequence analysis  
- **SYMBOLIC_QUBIT** - Alternative quantum encoding

---

## Benchmark Usage Audit

### ✅ Currently Optimized

1. **benchmarks/class_b_hybrid_quick.py**
   - Uses: H3 Hierarchical Cascade (variant 8)
   - Status: ✓ OPTIMAL (0.673 BPP, η=0)
   - Reason: Best general compression, zero coherence

2. **benchmarks/class_b_transform_dsp.py**
   - Uses: H3 Hierarchical Cascade
   - Status: ✓ OPTIMAL (transform comparison context)

3. **benchmarks/class_c_compression.py**
   - Uses: FH5 Entropy-Guided (variant 11)
   - Status: ✓ OPTIMAL (0.406 BPP on edges, 50% improvement)

### ⚠️ Missing Specialized Routing

4. **benchmarks/class_a_quantum_simulation.py**
   - Current: Uses QuantumSymbolicCompressor (internal RFT, no variant selection)
   - Gap: Quantum compressor doesn't expose variant parameter
   - Recommendation: When Python RFT is used, select **CASCADE (8)** for η=0 coherence
   - Impact: Ideal for quantum state superposition (zero inter-basis interference)

5. **benchmarks/class_d_crypto.py**
   - Current: Uses RFT-SIS hash + Feistel cipher (specialized C functions)
   - Gap: Native crypto functions don't expose RFT variant selection
   - Recommendation: If Python lattice operations used, combine:
     - **FIBONACCI (2)** for lattice structure alignment
     - **CHAOTIC (3)** for maximum entropy diffusion
   - Impact: Integer lattice crypto benefits from Fibonacci scaling

6. **benchmarks/class_e_audio_daw.py**
   - Current: Uses generic audio transform benchmark
   - Gap: Audio processing not using specialized RFT variants
   - Recommendation: Route audio pipeline through:
     - **HARMONIC (1)** for analysis (k³ cubic chirp excels at harmonic extraction)
     - **DICTIONARY (12)** for mastering (49.9 dB PSNR quality)
   - Impact: Audio benefits from harmonic phase structure, quality mastering

---

## Implementation Recommendations

### Priority 1: High-Level Python Routing Helper

Create `algorithms/rft/routing.py` with auto-selection logic:

```python
def select_best_variant(signal_type: str, 
                       quality_target: str = 'balanced') -> int:
    """
    Auto-select optimal RFT variant based on signal characteristics.
    
    Args:
        signal_type: One of 'general', 'edges', 'smooth', 'quantum', 
                     'lattice', 'chaotic', 'audio', 'harmonic'
        quality_target: 'speed', 'balanced', 'quality'
    
    Returns:
        Variant ID (0-12)
    """
    
    routing_map = {
        'general': 8,      # CASCADE - 0.673 BPP, η=0
        'edges': 11,       # ENTROPY_GUIDED - 0.406 BPP on steps
        'smooth': 12,      # DICTIONARY - 49.9 dB PSNR
        'quantum': 8,      # CASCADE - η=0 coherence for superposition
        'lattice': 2,      # FIBONACCI - integer structure alignment
        'chaotic': 3,      # CHAOTIC - Lyapunov mixing
        'audio': 1,        # HARMONIC - k³ cubic chirp
        'harmonic': 1,     # HARMONIC - explicit request
    }
    
    base_variant = routing_map.get(signal_type, 0)  # Default to STANDARD
    
    # Quality adjustments
    if quality_target == 'quality' and signal_type == 'smooth':
        return 12  # DICTIONARY for max PSNR
    elif quality_target == 'speed' and signal_type == 'general':
        return 0   # STANDARD faster than CASCADE
    
    return base_variant
```

### Priority 2: Codec Integration

Update `RFTHybridCodec` to support auto-routing:

```python
class RFTHybridCodec:
    def __init__(self, n: int = 1024, 
                 variant: Optional[int] = None,
                 auto_select: bool = False):
        self.n = n
        self.auto_select = auto_select
        self.variant = variant
        
    def encode(self, signal: np.ndarray, 
               signal_type: str = 'general') -> CascadeResult:
        # Auto-select variant if enabled
        if self.auto_select and self.variant is None:
            variant = select_best_variant(signal_type)
        else:
            variant = self.variant or 8  # Default to CASCADE
        
        # Route to appropriate encoder
        if variant == 8:
            return self._encode_h3_cascade(signal)
        elif variant == 11:
            return self._encode_fh5_entropy(signal)
        elif variant == 12:
            return self._encode_h6_dictionary(signal)
        else:
            return self._encode_standard(signal, variant)
```

### Priority 3: Benchmark Updates (Optional)

While crypto/audio benchmarks use specialized native functions that don't expose variant selection, future Python-based compression/transform code should use routing:

```python
# Example: If adding RFT-based quantum compression
from algorithms.rft.routing import select_best_variant

def compress_quantum_state(state: np.ndarray):
    variant = select_best_variant('quantum')  # Returns 8 (CASCADE)
    codec = RFTHybridCodec(len(state), variant=variant)
    return codec.encode(state, signal_type='quantum')

# Example: Audio processing pipeline
def process_audio_mastering(audio: np.ndarray):
    # Analysis stage
    analysis_variant = select_best_variant('harmonic')  # Returns 1
    analyzer = RFTHybridCodec(len(audio), variant=analysis_variant)
    spectrum = analyzer.encode(audio)
    
    # Mastering stage (quality focus)
    mastering_variant = select_best_variant('smooth')  # Returns 12 (DICTIONARY)
    master = RFTHybridCodec(len(audio), variant=mastering_variant)
    return master.encode(processed_audio)
```

---

## Architecture Notes

### Native C/C++ Variant Access

All 13 variants accessible via:

**C Layer** (`rft_kernel.h`):
```c
typedef enum {
    RFT_VARIANT_STANDARD = 0,
    RFT_VARIANT_HARMONIC = 1,
    RFT_VARIANT_FIBONACCI = 2,
    // ... all 13 variants ...
    RFT_VARIANT_DICTIONARY = 12
} rft_variant_t;
```

**C++ Layer** (`rftmw_asm_kernels.hpp`):
```cpp
enum class Variant {
    STANDARD = 0,
    HARMONIC = 1,
    // ... matches C enum ...
};
```

**Python Layer** (`rftmw_python.cpp`):
```cpp
py::enum_<asm_kernels::RFTKernelEngine::Variant>(engine, "Variant")
    .value("STANDARD", asm_kernels::RFTKernelEngine::Variant::STANDARD)
    // ... all variants exposed ...
```

**Usage**:
```python
import rftmw_native as rft
engine = rft.RFTKernelEngine(1024, variant=rft.RFTVariant.HARMONIC)
```

### Codec Integration Status

✅ **Full Integration**:
- `algorithms/rft/hybrids/cascade_hybrids.py` - H3/FH5/H6 classes
- `algorithms/rft/hybrids/rft_hybrid_codec.py` - Mode selection wrapper
- `algorithms/rft/variants/registry.py` - 13 variant generators

✅ **Test Validation**:
- `benchmarks/test_cascade_integration.py` - 12 tests passing
- `benchmarks/test_all_hybrids.py` - 16 variants across 8 signals

---

## Performance Summary

From comprehensive testing (`test_all_hybrids.py`):

| Metric | Winner | Value | Signal Type |
|--------|--------|-------|-------------|
| **Best BPP (avg)** | H3 CASCADE | 0.673 | General (7/8 signals) |
| **Best BPP (edges)** | FH5 ENTROPY | 0.406 | Steps/edges (50% improvement) |
| **Best PSNR** | H6 DICTIONARY | 49.9 dB | Smooth signals |
| **Best Latency** | H3 CASCADE | 0.57 ms | Real-time capable |
| **Coherence** | All cascades | η=0 | Zero inter-basis interference |

### Routing Decision Tree

```
Signal Analysis
    ├─ Edges/Steps detected → ENTROPY_GUIDED (11)
    ├─ Smooth/High-quality → DICTIONARY (12)
    ├─ Quantum superposition → CASCADE (8)
    ├─ Lattice structure → FIBONACCI (2)
    ├─ Audio harmonic → HARMONIC (1)
    └─ General/Unknown → CASCADE (8) [safe default]
```

---

## Next Steps

1. **Immediate**: Create `algorithms/rft/routing.py` with `select_best_variant()` helper
2. **Short-term**: Integrate auto-selection into `RFTHybridCodec`
3. **Medium-term**: Add variant selection to Python quantum/audio pipelines
4. **Long-term**: Expose variant parameter in native crypto/audio functions (C++ refactor)

---

## Summary

**Current State**: 13 variants fully integrated ASM→C→C++→Python. Benchmarks class_b/c already using optimal variants (H3/FH5).

**Gap**: class_a (quantum), class_d (crypto), class_e (audio) don't route to specialized variants because:
- Quantum compressor is C-only without variant exposure
- Crypto uses specialized RFT-SIS/Feistel without variant selection
- Audio uses generic transform benchmark without routing

**Solution**: Create Python routing helper + codec auto-selection for future usage. Current benchmarks work correctly but don't leverage domain-specific optimizations at the Python level.

**Impact**: When Python-level RFT compression/transforms are added, routing ensures optimal variant selection automatically.
