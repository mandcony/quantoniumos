# Security Policy

## Reporting a Vulnerability

Please report security issues privately. Do not create public GitHub issues.

- Preferred: PGP-encrypted email to security@quantoniumos.example (provide your PGP key in the message to receive ours)
- Alternative: GitHub Security Advisories (privately)

We will acknowledge receipt within 72 hours and provide a remediation timeline.

## Scope

- Core RFT implementation
- Cryptographic primitives and AEAD
- Native kernels and bindings

## Non-Scope

- Experimental research prototypes marked as such

## Timing and Side-Channel Concerns

We aim to minimize timing/cache side-channel leakage. If you identify leaks, please report them with steps to reproduce. We will prioritize fixes and publish advisories as appropriate.
