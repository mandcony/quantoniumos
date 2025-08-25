# QuantoniumOS Public Test Vectors

Generated: 2025-08-02 19:14:35

## Encryption Test Vectors

- Algorithm: ResonanceEncryption
- Vectors Generated: 23
- Validation Status: PASSED
- Vector Files:
  - [JSON Format](vectors_ResonanceEncryption_encryption_latest.json)
  - [HTML Format](vectors_ResonanceEncryption_encryption_latest.html)

## Hash Test Vectors

- Algorithm: GeometricWaveformHash
- Vectors Generated: 19
- Validation Status: PASSED
- Vector Files:
  - [JSON Format](vectors_GeometricWaveformHash_hash_latest.json)
  - [HTML Format](vectors_GeometricWaveformHash_hash_latest.html)

## Usage Information

These test vectors are provided to validate implementations of QuantoniumOS cryptographic primitives across different platforms and languages. To use these vectors:

1. Implement the QuantoniumOS algorithms according to the specification
2. Process the inputs specified in the test vectors
3. Compare your output with the expected output in the test vectors
4. If all outputs match, your implementation is correct

## Vector Format

### Encryption Vectors

```json
{
  "id": 1,
  "type": "random",
  "plaintext_hex": "...",
  "key_hex": "...",
  "plaintext_size": 32,
  "key_size": 32,
  "ciphertext_hex": "..."
}
```

### Hash Vectors

```json
{
  "id": 1,
  "type": "random",
  "input_hex": "...",
  "input_size": 32,
  "hash_hex": "..."
}
```

## Contact

For questions or issues regarding these test vectors, please open an issue on the QuantoniumOS GitHub repository.
