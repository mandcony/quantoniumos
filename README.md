# QuantoniumOS

**A signal processing and cryptographic research platform.**

**Research Platform**: This is an experimental implementation of signal processing techniques and cryptographic algorithms for educational and research purposes. Not production-ready cryptography.

**Mathematical Foundation**: Custom implementations of windowed DFT variants, geometric coordinate systems, and stream ciphers. See [MATHEMATICAL_JUSTIFICATION.md](MATHEMATICAL_JUSTIFICATION.md) for technical analysis.

## New to signal processing?

Check out our [Beginner's Guide](BEGINNERS_GUIDE.md) for explanations of key concepts.

## Quick Start

```bash
# Clone and run
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python setup_local_env.py
python app.py

# Test the API
curl http://localhost:5000/api/health
curl http://localhost:5000/docs  # Interactive API documentation
```

## What This Actually Is

**A Mathematical Research Implementation:**

**Windowed DFT Variants** - Modified Fourier transforms with custom weighting matrices  
**Geometric Coordinate Hashing** - Hash functions using golden ratio coordinate mappings  
**Stream Cipher Implementation** - XOR-based encryption with bit rotation and keystream generation  
**Entropy Generation System** - Adaptive randomness with feedback control mechanisms  
**Statistical Testing Framework** - Validation tools and empirical analysis    

## Research Implementation

**This project provides:**
- **Experimental signal processing algorithms** - Windowed DFT variants and geometric coordinate transformations
- **Educational cryptographic implementations** - Stream ciphers and hash functions for learning purposes
- **Statistical testing tools** - Entropy analysis and randomness evaluation
- **Cross-platform compatibility** - Python, C++, and Rust implementations
- **Reproducible experiments** - Deterministic validation and testing framework

## Architecture Explained

```
Web Interface (Flask) → Signal Processing Engine (C++/Python) → Mathematical Algorithms
     ↓                         ↓                                      ↓
  REST API               Windowed DFT/Stream Cipher         Experimental Math
```

**What Each Part Does:**
- `/api/` - REST endpoints for mathematical operations
- `/core/` - Mathematical implementations (Python with NumPy, C++ with Eigen)  
- `/encryption/` - Stream ciphers, geometric hashing, signal processing algorithms
- `/secure_core/` - High-performance C++ implementations
- `/tests/` - Comprehensive validation and statistical testing

## Usage Examples

### Python

```python
# Basic stream cipher encryption
from core.encryption.fixed_resonance_encrypt import fixed_resonance_encrypt, fixed_resonance_decrypt

# Encrypt message using stream cipher with keystream generation
message = "Hello, QuantoniumOS!"
key = "secure-key-2025"
encrypted = fixed_resonance_encrypt(message, key)
print(f"Encrypted: {len(encrypted)} bytes")

# Decrypt using same key
decrypted = fixed_resonance_decrypt(encrypted, key)
print(f"Decrypted: {decrypted}")

# Create a geometric coordinate hash
from core.encryption.geometric_waveform_hash import geometric_waveform_hash
waveform = [0.5 * math.sin(i * 0.1) for i in range(100)]
hash_result = geometric_waveform_hash(waveform)
print(f"Hash: {hash_result}")

# Perform windowed DFT
from core.encryption.resonance_fourier import perform_rft
signal = [1.0, 0.5, 0.2, 0.8, 0.3]
rft_result = perform_rft(signal, alpha=1.0)  # alpha controls windowing
print(f"DFT result: {rft_result}")
```

### C++

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

int main() {
    // Stream cipher encryption example
    std::string message = "Hello, QuantoniumOS!";
    std::string key = "secure-key-2025";
    
    // Note: C++ API would be similar to Python implementation
    // This is pseudocode showing the intended interface
    
    auto encrypted = encrypt_stream_cipher(message, key);
    std::cout << "Encrypted length: " << encrypted.size() << " bytes\n";
    
    auto decrypted = decrypt_stream_cipher(encrypted, key);
    std::cout << "Decrypted: " << decrypted << "\n";
    
    // Generate geometric hash
    std::vector<double> waveform;
    for (int i = 0; i < 100; ++i) {
        waveform.push_back(0.5 * std::sin(i * 0.1));
    }
    auto hash = compute_geometric_hash(waveform);
    std::cout << "Hash: " << hash << "\n";
    
    return 0;
}
```

### Rust

```rust
use resonance_core_rs::{ResonanceEncryption};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Stream cipher encryption
    let encryptor = ResonanceEncryption::new("secure-key-2025");
    let message = b"Hello, QuantoniumOS!";
    
    let encrypted = encryptor.encrypt(message)?;
    println!("Encrypted length: {} bytes", encrypted.len());
    
    let decrypted = encryptor.decrypt(&encrypted)?;
    println!("Decrypted: {}", String::from_utf8(decrypted)?);
    
    Ok(())
}
```

### Quickstart Test

```bash
# Clone and validate complete implementation
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos

# Run complete reproducible build and validation
.\make_repro.bat  # Windows
# OR
./make_repro.sh   # Linux/Mac

# This will:
# 1. Set up Python environment with pinned dependencies
# 2. Run comprehensive test suite (60+ tests)
# 3. Generate statistical validation reports
# 4. Create test vectors for validation
# 5. Verify mathematical implementations

# Quick functional test
python -c "
from core.encryption.fixed_resonance_encrypt import fixed_resonance_encrypt, fixed_resonance_decrypt
import math

# Test stream cipher
message = 'Testing QuantoniumOS stream cipher'
key = 'test-key-123'

print('Encrypting with stream cipher...')
encrypted = fixed_resonance_encrypt(message, key)
print(f'Encrypted: {len(encrypted)} bytes')

print('Decrypting...')
decrypted = fixed_resonance_decrypt(encrypted, key)
print(f'Decryption successful: {decrypted == message}')

# Test windowed DFT
from core.encryption.resonance_fourier import perform_rft
signal = [1.0, 0.5, 0.2, 0.8]
result = perform_rft(signal)
print(f'Windowed DFT result: {len(result)} components')
print('All implementations working correctly!')
"
```

## Documentation

- **[Windowed DFT Specification](WINDOWED_DFT_SPECIFICATION.md)** - Technical details of the windowed DFT implementation
- **[API Docs](http://localhost:5000/docs)** - Interactive Swagger interface  
- **[Developer Guide](QUANTONIUM_DEVELOPER_GUIDE.md)** - Setup and contribution guidelines
- **[Mathematical Analysis](MATHEMATICAL_JUSTIFICATION.md)** - Technical analysis of implementations
- **[Security Details](SECURITY.md)** - Security considerations and limitations
- **[License Details](LICENSE)** - Usage terms

### Implementation Status

**Mathematical Foundation**: Windowed DFT variants and stream cipher implementations

**Current Status**:
- Working stream cipher with XOR and bit rotation
- Geometric hash using golden ratio coordinate transformations
- Windowed DFT with custom weighting matrices
- Statistical testing and validation framework
- Research-grade implementations, not production cryptography

Run `python run_comprehensive_tests.py` to test current implementations.

## Important Notice

This project demonstrates **experimental signal processing and cryptographic concepts**:

- It runs on classical computers using standard mathematical operations
- It's designed for research, education, and experimentation
- Implementations are for learning purposes, not production security
- Mathematical techniques may have interesting properties but require further validation

## Implementation Status

**Research Implementation** - Signal processing algorithms and stream ciphers with educational value  
**Testing Framework** - Statistical analysis tools including entropy measurement and validation  
**Cross-Platform** - Python, C++, and Rust implementations  
**Reproducible** - Deterministic builds with comprehensive test coverage  
**Educational Purpose** - Not suitable for production cryptographic applications

## License

- **Academic Use:** [LICENSE](LICENSE) - Free for education and research
- **Other Uses:** [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) - Please contact us

To verify the complete implementation:

```powershell
# Windows - Complete validation suite
.\make_repro.bat

# Or specific components
python run_statistical_validation.py      # Statistical tests
python run_security_focused_tests.py      # Security analysis  
python run_comprehensive_tests.py         # Full test suite
```

```bash
# Linux/Mac - Complete validation suite  
./make_repro.sh

# Individual validation components
python3 run_statistical_validation.py
python3 run_security_focused_tests.py
python3 run_comprehensive_tests.py
```

## API Endpoints

The API exposes these mathematical operations:

- `/api/encrypt` - Stream cipher encryption with key-based keystream generation
- `/api/decrypt` - Corresponding decryption operation  
- `/api/hash` - Geometric waveform hashing using golden ratio coordinates
- `/api/entropy` - Entropy generation and statistical analysis
- `/api/rft` - Windowed DFT computation with custom weighting
- `/api/validate` - Statistical testing and validation tools
- `/quantum/*` - Interactive visualization and analysis endpoints

All endpoints include statistical validation and testing capabilities.

## Core Implementation

This repository contains implementations of:

- **Windowed DFT Variants** - Modified Fourier transforms with custom weighting matrices
- **Geometric Coordinate Hashing** - Hash functions using golden ratio coordinate transformations  
- **Stream Cipher Implementation** - XOR-based encryption with bit rotation and keystream generation
- **Entropy Generation System** - Adaptive randomness generation with feedback control
- **Statistical Testing Framework** - Validation tools and empirical analysis
- **Cross-Platform Bindings** - Python with NumPy, C++ with Eigen, and Rust implementations

All components have been tested through automated validation suites and statistical analysis.

## License

See [LICENSE](LICENSE) for terms of use.
