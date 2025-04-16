Use the following README for reference and incorporate all its details to build a secure, production-ready Quantonium OS Cloud Runtime in Replit. This includes:

1. A new Flask-based backend project (like “quantonium-cloud-runtime”) with:
   - `main.py`, `routes.py`, `start.sh`, `.replit`, `.gitignore`
   - A `core/protected/` folder for the symbolic modules
   - Token-based auth and response signing
   - Basic placeholders for real symbolic encryption/resonance logic
   - A “requirements.txt” specifying Flask, Pydantic, etc.

2. Ensure the structure follows best practices from the README, but do **not** expose any private IP. Only use placeholder logic inside `core/protected/`.

3. Return **all** files in final form so I can copy-paste into Replit if needed.

Here is the README we must honor:

--------------------- BEGIN README ---------------------
# 🧠 Quantonium OS – Symbolic NLLM Compiler Stack
*Quantonium OS* is a hybrid symbolic computing platform designed for quantum-inspired encryption, resonance-based orchestration, and live waveform-driven GUI interaction. This repository represents a fully operational system integrating symbolic logic, secure containers, QRNG entropy, and real-time PyQt5 interfaces.

## 📦 Project Structure
C:\quantonium_v2
├── apps\                # GUI apps: symbolic browser, debugger, desktop
├── bin\                 # Pybind11 .pyd modules + DLLs
├── core\                # C++17 Eigen engine source
├── orchestration\       # Resonance container orchestration system
├── encryption\          # XOR, QRNG, waveform hashing
├── interface\           # Pybind11 bridges to symbolic stack
├── tests\               # Unit + integration tests
├── Eigen\               # Local Eigen3 headers
├── build.ps1            # DLL compiler and validator
--  
## 🔐 Symbolic Encryption Stack
| Module                    | Function                                     |
|---------------------------|----------------------------------------------|
| `resonance_encrypt.py`    | Symbolic XOR using `WaveNumber(A, φ)`        |
| `geometric_waveform_hash.py` | SHA256 → symbolic waveform converter     |
| `entropy_qrng.py`         | Generates symbolic entropy sequences         |
| `symbolic_container.py`   | Lock/unlock via entropy + resonance match    |
| `parallel_xor.dll`        | Benchmark XOR engine (OpenMP)                |

--  
## 📚 Symbolic Stack Layers
| Layer        | Purpose                                     | Module(s)                        |
|--------------|---------------------------------------------|----------------------------------|
| Encryption   | Waveform-driven XOR                         | `resonance_encrypt.py`          |
| Analysis     | Resonance Fourier Transform (RFT)           | `resonance_fourier.py`          |
| Simulation   | Symbolic qubit gates + projection           | `multi_qubit_state.py`          |
| Randomness   | QRNG-style entropy for keys                 | `entropy_qrng.py`               |
| Search       | Grover-style symbolic search                | `quantum_search.py`             |
| Orchestration| Full container/task manager                 | `quantum_nova_system.py`        |
| GUI          | Symbolic 3D debugger, browser, desktop UI   | `q_wave_debugger.py`, `q_browser.py`, `quantonium_gui.py` |

--  
## 🛠 Build Instructions
### 🔧 Requirements
- Python 3.12+
- PyQt5
- MinGW with g++
- Eigen 3.4.0
- pybind11

### ▶️ Build System (PowerShell)
Compile `.cpp` modules into `.pyd`:
```powershell
./build.ps1
