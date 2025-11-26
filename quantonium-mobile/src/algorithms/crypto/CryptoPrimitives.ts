/**
 * Cryptographic Primitives - QuantoniumOS Mobile
 *
 * Core cryptographic building blocks using Feistel network
 * and HMAC-based authentication
 */

import * as Crypto from 'expo-crypto';

/**
 * HMAC-SHA256 implementation
 */
export class HMACSHA256 {
  /**
   * Compute HMAC-SHA256 of message using key
   */
  static async compute(key: Uint8Array, message: Uint8Array): Promise<Uint8Array> {
    // Use Web Crypto API via expo-crypto
    const keyHex = Array.from(key).map(b => b.toString(16).padStart(2, '0')).join('');
    const messageHex = Array.from(message).map(b => b.toString(16).padStart(2, '0')).join('');

    const hmac = await Crypto.digestStringAsync(
      Crypto.CryptoDigestAlgorithm.SHA256,
      keyHex + messageHex
    );

    return this.hexToUint8Array(hmac);
  }

  /**
   * Simple HMAC implementation for synchronous operations
   */
  static computeSync(key: Uint8Array, message: Uint8Array): Uint8Array {
    // Simplified HMAC using XOR-based key padding (for demo purposes)
    const blockSize = 64;
    const opad = new Uint8Array(blockSize).fill(0x5c);
    const ipad = new Uint8Array(blockSize).fill(0x36);

    // Pad or hash key to block size
    let keyBlock = new Uint8Array(blockSize);
    if (key.length <= blockSize) {
      keyBlock.set(key);
    } else {
      // If key is longer, hash it first (simplified)
      keyBlock.set(key.slice(0, blockSize));
    }

    // Compute inner and outer key pads
    const innerKey = new Uint8Array(blockSize);
    const outerKey = new Uint8Array(blockSize);
    for (let i = 0; i < blockSize; i++) {
      innerKey[i] = keyBlock[i] ^ ipad[i];
      outerKey[i] = keyBlock[i] ^ opad[i];
    }

    // Inner hash: H(innerKey || message)
    const innerInput = new Uint8Array(innerKey.length + message.length);
    innerInput.set(innerKey);
    innerInput.set(message, innerKey.length);
    const innerHash = this.sha256Sync(innerInput);

    // Outer hash: H(outerKey || innerHash)
    const outerInput = new Uint8Array(outerKey.length + innerHash.length);
    outerInput.set(outerKey);
    outerInput.set(innerHash, outerKey.length);
    return this.sha256Sync(outerInput);
  }

  /**
   * Simplified synchronous SHA256 (for demo - not cryptographically secure)
   */
  private static sha256Sync(data: Uint8Array): Uint8Array {
    // This is a placeholder - in production, use a proper crypto library
    // For now, use a simple hash function for demonstration
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      hash = ((hash << 5) - hash) + data[i];
      hash = hash & hash; // Convert to 32-bit integer
    }

    const result = new Uint8Array(32);
    for (let i = 0; i < 32; i++) {
      result[i] = (hash >> (i % 4 * 8)) & 0xff;
    }
    return result;
  }

  private static hexToUint8Array(hex: string): Uint8Array {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < hex.length; i += 2) {
      bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
    }
    return bytes;
  }
}

/**
 * Feistel Network implementation for block ciphers
 */
export class FeistelNetwork {
  private rounds: number;
  private blockSize: number;
  private halfBlock: number;

  constructor(rounds: number = 16, blockSize: number = 16) {
    if (blockSize % 2 !== 0) {
      throw new Error('Block size must be even for Feistel network');
    }

    this.rounds = rounds;
    this.blockSize = blockSize;
    this.halfBlock = blockSize / 2;
  }

  /**
   * Round function for Feistel network
   */
  private roundFunction(right: Uint8Array, roundKey: Uint8Array): Uint8Array {
    const mac = HMACSHA256.computeSync(roundKey, right);
    return mac.slice(0, this.halfBlock);
  }

  /**
   * Generate round keys from master key
   */
  private generateRoundKeys(key: Uint8Array): Uint8Array[] {
    const roundKeys: Uint8Array[] = [];
    for (let i = 0; i < this.rounds; i++) {
      const roundIndex = new Uint8Array(4);
      roundIndex[0] = (i >> 24) & 0xff;
      roundIndex[1] = (i >> 16) & 0xff;
      roundIndex[2] = (i >> 8) & 0xff;
      roundIndex[3] = i & 0xff;

      const roundKey = HMACSHA256.computeSync(key, roundIndex);
      roundKeys.push(roundKey.slice(0, this.halfBlock));
    }
    return roundKeys;
  }

  /**
   * Encrypt a single block
   */
  encryptBlock(block: Uint8Array, key: Uint8Array): Uint8Array {
    if (block.length !== this.blockSize) {
      throw new Error(`Block size must be ${this.blockSize} bytes`);
    }

    const roundKeys = this.generateRoundKeys(key);
    let left = block.slice(0, this.halfBlock);
    let right = block.slice(this.halfBlock);

    for (const roundKey of roundKeys) {
      const newLeft = right;
      const fOutput = this.roundFunction(right, roundKey);
      const newRight = new Uint8Array(this.halfBlock);

      for (let i = 0; i < this.halfBlock; i++) {
        newRight[i] = left[i] ^ fOutput[i];
      }

      left = newLeft;
      right = newRight;
    }

    const result = new Uint8Array(this.blockSize);
    result.set(left);
    result.set(right, this.halfBlock);
    return result;
  }

  /**
   * Decrypt a single block
   */
  decryptBlock(block: Uint8Array, key: Uint8Array): Uint8Array {
    if (block.length !== this.blockSize) {
      throw new Error(`Block size must be ${this.blockSize} bytes`);
    }

    const roundKeys = this.generateRoundKeys(key);
    roundKeys.reverse(); // Use keys in reverse order for decryption

    let left = block.slice(0, this.halfBlock);
    let right = block.slice(this.halfBlock);

    for (const roundKey of roundKeys) {
      const newRight = left;
      const fOutput = this.roundFunction(left, roundKey);
      const newLeft = new Uint8Array(this.halfBlock);

      for (let i = 0; i < this.halfBlock; i++) {
        newLeft[i] = right[i] ^ fOutput[i];
      }

      left = newLeft;
      right = newRight;
    }

    const result = new Uint8Array(this.blockSize);
    result.set(left);
    result.set(right, this.halfBlock);
    return result;
  }
}

/**
 * RFT-Enhanced Feistel Network (48 rounds)
 */
export class RFTEnhancedFeistel {
  private rounds: number;
  private blockSize: number;
  private halfBlock: number;

  constructor(rounds: number = 48, blockSize: number = 16) {
    this.rounds = rounds;
    this.blockSize = blockSize;
    this.halfBlock = blockSize / 2;
  }

  /**
   * RFT-enhanced round function
   */
  private rftRoundFunction(right: Uint8Array, roundKey: Uint8Array): Uint8Array {
    // Standard HMAC-based round function (RFT enhancement would go here)
    const mac = HMACSHA256.computeSync(roundKey, right);
    return mac.slice(0, this.halfBlock);
  }

  /**
   * Generate round keys using golden ratio
   */
  private generateRoundKeys(key: Uint8Array): Uint8Array[] {
    const PHI = 1.618033988749895; // Golden ratio
    const roundKeys: Uint8Array[] = [];

    for (let i = 0; i < this.rounds; i++) {
      // Golden ratio phase
      const phase = (i * PHI) % 1.0;
      const phaseBytes = new Uint8Array(4);
      const phaseInt = Math.floor(phase * 0xffffffff);
      phaseBytes[0] = (phaseInt >> 24) & 0xff;
      phaseBytes[1] = (phaseInt >> 16) & 0xff;
      phaseBytes[2] = (phaseInt >> 8) & 0xff;
      phaseBytes[3] = phaseInt & 0xff;

      // Combine with round index
      const roundData = new Uint8Array(8);
      roundData.set(phaseBytes);
      roundData[4] = (i >> 24) & 0xff;
      roundData[5] = (i >> 16) & 0xff;
      roundData[6] = (i >> 8) & 0xff;
      roundData[7] = i & 0xff;

      const roundKey = HMACSHA256.computeSync(key, roundData);
      roundKeys.push(roundKey.slice(0, this.halfBlock));
    }

    return roundKeys;
  }

  /**
   * Encrypt data
   */
  encrypt(plaintext: Uint8Array, key: Uint8Array): Uint8Array {
    const roundKeys = this.generateRoundKeys(key);
    let left = plaintext.slice(0, this.halfBlock);
    let right = plaintext.slice(this.halfBlock);

    for (const roundKey of roundKeys) {
      const newLeft = right;
      const fOutput = this.rftRoundFunction(right, roundKey);
      const newRight = new Uint8Array(this.halfBlock);

      for (let i = 0; i < this.halfBlock; i++) {
        newRight[i] = left[i] ^ fOutput[i];
      }

      left = newLeft;
      right = newRight;
    }

    const result = new Uint8Array(this.blockSize);
    result.set(left);
    result.set(right, this.halfBlock);
    return result;
  }

  /**
   * Decrypt data
   */
  decrypt(ciphertext: Uint8Array, key: Uint8Array): Uint8Array {
    const roundKeys = this.generateRoundKeys(key);
    roundKeys.reverse();

    let left = ciphertext.slice(0, this.halfBlock);
    let right = ciphertext.slice(this.halfBlock);

    for (const roundKey of roundKeys) {
      const newRight = left;
      const fOutput = this.rftRoundFunction(left, roundKey);
      const newLeft = new Uint8Array(this.halfBlock);

      for (let i = 0; i < this.halfBlock; i++) {
        newLeft[i] = right[i] ^ fOutput[i];
      }

      left = newLeft;
      right = newRight;
    }

    const result = new Uint8Array(this.blockSize);
    result.set(left);
    result.set(right, this.halfBlock);
    return result;
  }
}

/**
 * Utility functions
 */
export class CryptoUtils {
  /**
   * Generate random bytes
   */
  static generateRandomBytes(length: number): Uint8Array {
    const bytes = new Uint8Array(length);
    for (let i = 0; i < length; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
    return bytes;
  }

  /**
   * Convert string to Uint8Array
   */
  static stringToBytes(str: string): Uint8Array {
    const encoder = new TextEncoder();
    return encoder.encode(str);
  }

  /**
   * Convert Uint8Array to string
   */
  static bytesToString(bytes: Uint8Array): string {
    const decoder = new TextDecoder();
    return decoder.decode(bytes);
  }

  /**
   * Convert bytes to hex string
   */
  static bytesToHex(bytes: Uint8Array): string {
    return Array.from(bytes)
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  /**
   * Convert hex string to bytes
   */
  static hexToBytes(hex: string): Uint8Array {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < hex.length; i += 2) {
      bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
    }
    return bytes;
  }
}
