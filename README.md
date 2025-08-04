# QuantoniumOS
[![License: Patent Pending](https://img.shields.io/badge/License-Patent%20Pending-orange.svg)](LICENSE) [![Patent: USPTO #19/169,399](https://img.shields.io/badge/Patent-USPTO%20%2319%2F169%2C399-red.svg)](https://patents.uspto.gov/) [![Version](https://img.shields.io/badge/Version-1.0.0-blue)](https://github.com/mandcony/quantoniumos) [![API Status](https://img.shields.io/badge/API-Live-brightgreen)](http://localhost:5000/health)

**An educational cryptographic platform with quantum-inspired algorithms seeking academic peer review.**

## 🚀 Quick StartumOS
[![License: Patent Pending](https://img.shields.io/badge/License-Patent%20Pending-orange.svg)](LICENSE) [![Patent: USPTO #19/169,399](https://img.shields.io/badge/Patent-USPTO%20%2319%2F169%2C399-red.svg)](https://patents.uspto.gov/) [![Version](https://img.shields.io/badge/Version-1.0.0-blue)](https://github.com/mandcony/quantoniumos) [![API Status](https://img.shields.io/badge/API-Live-brightgreen)](http://localhost:5000/health)

**Patent-protected quantum cryptographic platform seeking academic peer review and validation.**

## � New to quantum cryptography?

Check out our [Beginner's Guide](BEGINNERS_GUIDE.md) for simple explanations of key concepts.

## �🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python setup_local_env.py
python main.py

# Test the API
curl http://localhost:5000/api/health
curl http://localhost:5000/docs  # Interactive API documentation
```

## ⚡ What This Is

**A Platform for Exploring New Cryptographic Concepts:**

✅ **Wave-based encryption** - Using mathematical wave patterns instead of traditional keys  
✅ **Alternative randomness generation** - Inspired by quantum physics concepts  
✅ **Educational framework** - Combining Python APIs with C++ performance  
✅ **Academic testing platform** - For research and validation  

## 🎯 Academic Research Focus

**This project is designed for:**
- **Computer science students** - Learning alternative cryptography concepts
- **Mathematics researchers** - Exploring wave-based mathematical patterns
- **Performance analysts** - Comparing implementation efficiency
- **Educational institutions** - Demonstrating cryptographic principles

## 🏗️ Architecture Explained

```
Web Interface (Flask) → Core Mathematics (C++) → Wave-Based Algorithms
     ↓                       ↓                        ↓
  Simple API           Eigen Math Library       Research Methods
```

**What Each Part Does:**
- `/api/` - Simple REST endpoints for accessing features
- `/core/` - Fast C++ mathematical engine  
- `/secure_core/` - Core algorithmic implementations
- `/auth/` - Basic security and access control
- `/tests/` - Validation test suite

## 💡 Usage Examples

### Python

```python
# Basic encryption
from quantoniumos import resonance_encrypt, resonance_decrypt

# Encrypt a message
message = "Hello, QuantoniumOS!"
key = "my-secure-key-2025"
encrypted = resonance_encrypt(message, key)
print(f"Encrypted: {encrypted[:20]}...{encrypted[-20:]}")

# Decrypt a message
decrypted = resonance_decrypt(encrypted, key)
print(f"Decrypted: {decrypted}")

# Create a secure hash
from quantoniumos import geometric_wave_hash
hash_result = geometric_wave_hash("Important data to hash")
print(f"Hash: {hash_result}")

# Run statistical tests
from quantoniumos.validation import run_statistical_suite
stats = run_statistical_suite(encrypted)
print(f"Passed tests: {stats['passed_tests']}/{stats['total_tests']}")
```

### C++

```cpp
#include <quantoniumos/resonance.hpp>
#include <iostream>
#include <string>

int main() {
    // Encrypt and decrypt
    std::string message = "Hello, QuantoniumOS!";
    std::string key = "my-secure-key-2025";
    
    auto encrypted = qos::resonance_encrypt(message, key);
    std::cout << "Encrypted length: " << encrypted.size() << " bytes\n";
    
    auto decrypted = qos::resonance_decrypt(encrypted, key);
    std::cout << "Decrypted: " << decrypted << "\n";
    
    // Create a secure hash
    auto hash = qos::geometric_wave_hash("Important data to hash");
    std::cout << "Hash: " << qos::bytes_to_hex(hash) << "\n";
    
    return 0;
}
```

### Rust

```rust
use resonance_core_rs::{resonance_encrypt, resonance_decrypt, geometric_wave_hash};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Encrypt and decrypt
    let message = "Hello, QuantoniumOS!";
    let key = "my-secure-key-2025";
    
    let encrypted = resonance_encrypt(message, key)?;
    println!("Encrypted length: {} bytes", encrypted.len());
    
    let decrypted = resonance_decrypt(&encrypted, key)?;
    println!("Decrypted: {}", decrypted);
    
    // Create a secure hash
    let hash = geometric_wave_hash("Important data to hash")?;
    println!("Hash: {}", hex::encode(hash));
    
    Ok(())
}
```

### Quickstart Roundtrip

```bash
# Clone and set up
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
python setup_local_env.py

# Run a full encrypt-decrypt-validate roundtrip
python -c "
from quantoniumos import resonance_encrypt, resonance_decrypt
from quantoniumos.validation import validate_encryption

# Create test message
message = 'Testing QuantoniumOS encryption'
key = 'secure-test-key-2025'

# Run roundtrip
print('Encrypting message...')
encrypted = resonance_encrypt(message, key)
print(f'Encrypted size: {len(encrypted)} bytes')

print('Decrypting message...')
decrypted = resonance_decrypt(encrypted, key)
print(f'Decryption successful: {decrypted == message}')

print('Validating encryption properties...')
validation = validate_encryption(message, key)
print(f'Passed validation: {validation}')
"
```

## 📖 Documentation

- **[API Docs](http://localhost:5000/docs)** - Interactive Swagger interface
- **[Developer Guide](QUANTONIUM_DEVELOPER_GUIDE.md)** - Setup and contribution
- **[License Details](LICENSE)** - Usage terms

## 🔒 Important Note

This project demonstrates **educational concepts** inspired by quantum principles, but:

- It runs on classical computers (not quantum hardware)
- It's designed for academic exploration, not production security
- It seeks peer review to validate mathematical approaches
- Patent pending status protects the specific mathematical implementations

## 🎯 Project Status

🟢 **Educational Tool** - Explores alternative cryptographic approaches  
🟢 **Research Platform** - Mathematical concepts open for review  
🟢 **Seeking Feedback** - Academic validation welcomed  
🟢 **Documentation** - Simplified explanations for accessibility

## License

- **Academic Use:** [LICENSE](LICENSE) - Free for education and research
- **Other Uses:** [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) - Please contact us

To verify the C++ implementation:

```powershell
.\run_simple_test.bat
```

For more comprehensive testing:

```powershell
.\run_robust_tests.bat
```

## API Endpoints

The simplified app exposes these key endpoints:

- `/api/status` - Check system status
- `/api/wave/compute` - Compute wave properties
- `/api/resonance/check` - Verify resonance calculations
- `/api/symbolic/eigenvector` - Calculate symbolic eigenvectors

## Implementation Notes

This repository contains:

- Complete wave-based mathematics
- Quantum-inspired geometric transformations
- Resonance-based process management
- High-performance C++ linear algebra operations using Eigen

All core scientific components have been verified through automated tests.

## Documentation

For more detailed information, see:
- [Local Setup Guide](LOCAL_SETUP.md)
- [Simplified Solution](SIMPLIFIED_SOLUTION.md)
- [Developer Guide](QUANTONIUM_DEVELOPER_GUIDE.md)

## License

See [LICENSE](LICENSE) for terms of use.
