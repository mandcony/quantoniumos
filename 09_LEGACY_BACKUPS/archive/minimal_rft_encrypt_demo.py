#!/usr/bin/env python3
"""""" Minimal Canonical RFT Encryption Demo === Reviewer-focused, dependency-light demonstration of applying the canonical True RFT to a simple XOR stream cipher style construction. Algorithm (demonstration only, NOT production crypto): 1. Derive 32-byte key from passphrase using SHA-256. 2. Chunk plaintext into blocks (length N). 3. For each block: a. Convert bytes->float vector. b. Apply canonical forward_true_rft(). c. Take real+imag parts interleaved, hash to produce keystream bytes. d. XOR with plaintext block. 4. Output hex ciphertext with embedded block size. Decrypt reverses using same deterministic keystream regeneration. Usage: python minimal_rft_encrypt_demo.py "secret pass" "Hello RFT World!" This file is intentionally tiny, auditable, and directly uses canonical_true_rft as single source of mathematical truth. """"""

import sys, hashlib, math from typing
import List from canonical_true_rft
import forward_true_rft, inverse_true_rft, PHI
def derive_key(passphrase: str) -> bytes:
        return hashlib.sha256(passphrase.encode('utf-8')).digest()
def rft_keystream(index: int, block_len: int, key: bytes) -> bytes:
"""
"""
        Deterministic keystream derivation independent of plaintext/ciphertext contents. Uses block index + key to form a seed vector passed through RFT then hashed.
"""
"""

        # Build deterministic seed vector from key + index counter seed = bytearray(key)
        for i in range(8): seed[i % len(seed)] ^= (index >> (8*i)) & 0xFF vec = [float(b)
        for b in seed[:block_len]]

        # Pad
        if needed
        if len(vec) < block_len: vec.extend([0.0]*(block_len-len(vec))) coeffs = forward_true_rft(vec[:block_len]) raw = []
        for c in coeffs: raw.append(c.real) raw.append(c.imag) packed = ('|'.join(f"{x:.17g}"
        for x in raw)).encode('utf-8') digest = hashlib.sha512(packed).digest()
        return digest[:block_len]
def encrypt(passphrase: str, plaintext: str, block_size: int = 32) -> str: key = derive_key(passphrase) data = plaintext.encode('utf-8') out = [] for bi, i in enumerate(range(0, len(data), block_size)): block = data[i:i+block_size] ks = rft_keystream(bi, len(block), key) out.append(bytes(a ^ b for a,b in zip(block, ks)))
        return f"RFT{block_size:02d}-" + b''.join(out).hex()
def decrypt(passphrase: str, ciphertext: str) -> str:
        if not ciphertext.startswith('RFT') or '-' not in ciphertext:
        raise ValueError('Bad ciphertext format') meta, hexdata = ciphertext.split('-',1) block_size = int(meta[3:5]) key = derive_key(passphrase) data = bytes.fromhex(hexdata) out_blocks = [] for bi, i in enumerate(range(0, len(data), block_size)): block = data[i:i+block_size] ks = rft_keystream(bi, len(block), key) out_blocks.append(bytes(a ^ b for a,b in zip(block, ks)))
        return b''.join(out_blocks).decode('utf-8', errors='strict')
if __name__ == '__main__':
if len(sys.argv) == 3: pw, pt = sys.argv[1], sys.argv[2] ct = encrypt(pw, pt) rt = decrypt(pw, ct)
print('Golden ratio phi =', PHI)
print('Ciphertext :', ct)
print('Round-trip :', rt)
print('Match :', pt == rt)
else:
print('Usage: python minimal_rft_encrypt_demo.py <passphrase> <plaintext>')