# Reddit Reply Template

When someone claims QuantoniumOS is fake, use this calm, factual response:

---

**Repro steps (1 min):**
```bash
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
./make_repro.sh
```

**It prints:**
- Reconstruction error ~1e-15 (unitary RFT) 
- RFT vs DFT difference >0.1 (proves non-DFT)
- Entropy ≈ 7.9–8.0 bits/byte
- Avalanche ≈ 50% bits flipped from 1-bit change

**If any number doesn't match, open an issue with your log. Happy to iterate.**

**What it is:** Classical unitary transform R = Σᵢ wᵢ D_φᵢ C_σᵢ D_φᵢ† with working crypto demos. Novel math/packaging.

**What it's not:** A quantum computer. It's signal processing algebra, not qubits.

---

Keep it factual, reproducible, and avoid engaging with inflammatory comments.
