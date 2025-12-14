# Resonant Fourier Transform (RFT) - Canonical Definition

**USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"**

---

## 1. The RFT Definition

The **Resonant Fourier Transform (RFT)** is a multi-carrier transform that maps discrete data into a continuous waveform domain using **golden-ratio frequency and phase spacing**.

### 1.1 Basis Functions

$$
\Psi_k(t) = \exp\left(2\pi i \cdot f_k \cdot t + i \cdot \theta_k\right)
$$

Where:
- $f_k = (k+1) \times \varphi$ — **Resonant Frequency**
- $\theta_k = 2\pi k / \varphi$ — **Golden Phase**
- $\varphi = \frac{1+\sqrt{5}}{2} \approx 1.618$ — **Golden Ratio**

### 1.2 Forward Transform (Data → Wave)

$$
\text{RFT}(x)[t] = \sum_k x[k] \cdot \Psi_k(t)
$$

Transforms discrete data $x$ into a continuous waveform $W(t)$.

### 1.3 Inverse Transform (Wave → Data)

$$
x[k] = \langle W, \Psi_k \rangle = \frac{1}{T} \int_0^T W(t) \cdot \Psi_k^*(t) \, dt
$$

Recovers discrete data by correlating with basis functions.

---

## 2. Why "Resonant"?

The golden ratio creates **resonance** because:

1. **Self-similarity**: $\varphi^2 = \varphi + 1$ — the ONLY number with this property
2. **Fibonacci convergence**: $f_{k+1}/f_k \to \varphi$ for consecutive frequencies
3. **Quasi-periodicity**: Golden spacing creates beating patterns that never exactly repeat
4. **Optimal spreading**: The golden angle $2\pi/\varphi^2 \approx 137.5°$ is maximally irrational

The RFT basis functions **resonate** with signals having golden-ratio structure, enabling superior compression for natural signals (phyllotaxis, music, biological rhythms).

---

## 3. Key Innovation: Wave-Domain Computation

The RFT is designed for **computation IN the wave domain**:

### 3.1 Binary Encoding (BPSK)

| Bit | Symbol |
|-----|--------|
| 0   | -1     |
| 1   | +1     |

Binary data encodes as amplitude/phase modulation on resonant carriers:

```python
waveform = Σ_k symbol[k] × Ψ_k(t)
```

### 3.2 Logic Operations on Waveforms

Operations work **directly** on the waveform without decoding:

| Operation | Formula | Description |
|-----------|---------|-------------|
| **XOR**   | $-s_1 \times s_2$ | Negate product in BPSK |
| **AND**   | $+1$ if both $+1$ | Both bits set |
| **OR**    | $+1$ if either $+1$ | Either bit set |
| **NOT**   | $-w$ | Negate waveform |

### 3.3 Chained Operations

Complex expressions like `(A XOR B) AND (NOT C)` execute entirely in the wave domain, then decode once at the end.

---

## 4. Comparison to FFT

| Property | FFT | RFT |
|----------|-----|-----|
| **Frequencies** | $f_k = k$ (integers) | $f_k = k \times \varphi$ (irrational) |
| **Periodicity** | Exactly periodic | Quasi-periodic |
| **Aliasing** | At N boundaries | No exact aliasing |
| **Basis** | $e^{2\pi i k n/N}$ | $e^{2\pi i f_k t + i\theta_k}$ |
| **Computation** | O(N log N) | O(N²) naive, O(N) per coefficient |
| **Wave computation** | ❌ | ✅ |

---

## 5. Implementation

### 5.1 Core Module

```python
from algorithms.rft import rft, irft, BinaryRFT, PHI
```

### 5.2 Quick Usage

```python
import numpy as np
from algorithms.rft import rft_forward, rft_inverse, BinaryRFT

# Forward/Inverse RFT
x = np.array([1, 0, 1, 1, 0, 0, 1, 0])
wave = rft_forward(x)
recovered = rft_inverse(wave, len(x))

# Binary RFT (wave-domain computation)
brft = BinaryRFT(num_bits=8)

# Encode binary → wave
wave_a = brft.encode(0b10101010)
wave_b = brft.encode(0b11001100)

# Compute in wave domain
result_wave = brft.wave_xor(wave_a, wave_b)

# Decode wave → binary
result = brft.decode(result_wave)
print(f"XOR result: {result:08b}")  # 01100110
```

---

## 6. File Index

| Purpose | File |
|---------|------|
| **Canonical RFT** | `algorithms/rft/core/resonant_fourier_transform.py` |
| **Package exports** | `algorithms/rft/__init__.py` |
| **Wave-domain hash** | `algorithms/rft/core/symbolic_wave_computer.py` |

---

## 7. Patent Claims

This RFT definition implements:

- **Claim 1**: Binary → Wave encoding via BPSK on resonant carriers
- **Claim 2**: Wave-domain logic operations (XOR, AND, OR, NOT)
- **Claim 3**: Cryptographic hash using resonance structure
- **Claim 4**: Geometric feature preservation via golden-ratio basis

---

## 8. Citation

```bibtex
@misc{rft2025,
  title   = {Resonant Fourier Transform: Golden-Ratio Multi-Carrier Wave Encoding},
  author  = {Minier, Luis M.},
  year    = {2025},
  note    = {USPTO Patent 19/169,399}
}
```

---

*Canonical Definition - December 2025*
