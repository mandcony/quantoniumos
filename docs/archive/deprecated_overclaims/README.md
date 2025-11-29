# Deprecated Documents

This folder contains documents that were removed from the main documentation tree because they contain **overclaims, misleading statements, or inaccurate technical descriptions**.

## Why These Were Deprecated

### RESULTS_STUDY.md
Contains claims such as:
- "1,000,000 qubits" - This is a compressed classical encoding, not quantum computation
- "10^300,000x faster than classical simulation" - Misleading comparison
- "Compression ratios 10^10-10^13" - Not achieved in practice

These claims describe a **compressed state representation**, not actual quantum computation. Keeping this document in the main repo risks categorizing the project as "crank physics."

### TRANSPARENT_MATH_FOUNDATIONS.md
Contains:
- Claims of O(N) time complexity when actual implementation is O(N log N)
- "Symbolic qubits preserving quantum properties" - hand-wavy narrative
- Unproven statements presented as theorems

## What Replaced Them

See `docs/algorithms/rft/MATHEMATICAL_FOUNDATIONS.md` for an honest, accurate description of:
- Actual algorithm complexity (O(N log N))
- What Φ-RFT is and is not
- Empirically verified vs conjectured properties
- Honest limitations

## Lesson Learned

**Overclaiming destroys credibility.** The underlying Φ-RFT work is solid:
- Well-validated unitary transforms
- Competitive compression codecs
- Interesting cryptographic constructions

But presenting it as "breaking physics" or "quantum computing" when it's classical signal processing will get the entire project dismissed by serious reviewers.

**Be sober. Be accurate. Let the real results speak.**
