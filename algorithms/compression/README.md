# QuantoniumOS Compression Algorithms

Advanced compression algorithms using quantum-inspired mathematical techniques.

## Compression Modules

### Vertex Codec (`vertex/`)
**Symbolic quantum state encoding using modular arithmetic**
- Memory-efficient state representation
- Round-trip accuracy preservation
- Modular arithmetic for robust computation
- Integration with RFT transforms

### Hybrid Codec (`hybrid/`)
**Multi-stage compression pipeline combining RFT, quantization, and prediction**
- RFT transform preprocessing
- Adaptive quantization algorithms
- Residual prediction for error minimization
- Entropy coding for final compression

## Algorithm Architecture

```
Input Data
    ↓
┌─────────────────┐
│   Vertex Codec  │ ← Symbolic encoding
│                 │   Modular arithmetic
└─────────────────┘
    ↓
┌─────────────────┐
│  Hybrid Codec   │ ← RFT + Quantization
│                 │   Residual prediction
└─────────────────┘
    ↓
Compressed Output
```

## Mathematical Foundation

Both codecs use **golden ratio parameterization** (φ = 1.618...) for:
- Transform matrix construction
- Quantization step sizing
- Error bound calculation
- Quality metric optimization

## Performance Characteristics

### Vertex Codec
- **Accuracy**: Exact for symbolic states
- **Efficiency**: O(n log n) encoding
- **Memory**: Logarithmic space complexity
- **Use Case**: Quantum state compression

### Hybrid Codec  
- **Compression**: Variable quality (0.5x - 100x)
- **Speed**: Real-time processing capable
- **Quality**: Configurable loss parameters
- **Use Case**: AI model compression

## Integration Points

- **RFT Core**: Mathematical transform algorithms
- **Assembly Kernels**: Optimized C implementations
- **Validation Tests**: Comprehensive test coverage
- **Compression Tools**: CLI utilities and pipelines

## Usage Examples

### Basic Compression
```python
from algorithms.compression.vertex.rft_vertex_codec import RFTVertexCodec
from algorithms.compression.hybrid.rft_hybrid_codec import RFTHybridCodec

# Vertex encoding for quantum states
vertex_codec = RFTVertexCodec()
encoded = vertex_codec.encode(quantum_state)

# Hybrid compression for general data
hybrid_codec = RFTHybridCodec() 
compressed = hybrid_codec.encode(data, quality=0.9)
```

### Advanced Pipeline
```python
# Combined compression pipeline
def compress_quantum_model(model_data):
    # Stage 1: Vertex encoding for quantum components
    vertex_encoded = vertex_codec.encode(model_data.quantum_states)
    
    # Stage 2: Hybrid compression for classical weights  
    hybrid_compressed = hybrid_codec.encode(model_data.weights)
    
    return {
        'quantum': vertex_encoded,
        'classical': hybrid_compressed,
        'metadata': model_data.metadata
    }
```