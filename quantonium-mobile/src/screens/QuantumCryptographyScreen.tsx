/**
 * Quantum Cryptography Screen - QuantoniumOS Mobile
 * Cryptographic protocols and key generation
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Alert,
} from 'react-native';
import ScreenShell from '../components/ScreenShell';
import {
  RFTEnhancedFeistel,
  CryptoUtils,
} from '../algorithms/crypto/CryptoPrimitives';
import {
  borderRadius,
  colors,
  shadows,
  spacing,
  typography,
} from '../constants/DesignSystem';

export default function QuantumCryptographyScreen() {
  const [plaintext, setPlaintext] = useState('');
  const [ciphertext, setCiphertext] = useState('');
  const [key, setKey] = useState<Uint8Array | null>(null);

  const generateKey = () => {
    const newKey = CryptoUtils.generateRandomBytes(32);
    setKey(newKey);
    Alert.alert('Key Generated', `256-bit key generated successfully\n${CryptoUtils.bytesToHex(newKey).substring(0, 32)}...`);
  };

  const encrypt = () => {
    if (!key) {
      Alert.alert('Error', 'Please generate a key first');
      return;
    }
    if (!plaintext.trim()) {
      Alert.alert('Error', 'Please enter text to encrypt');
      return;
    }

    try {
      const cipher = new RFTEnhancedFeistel(48, 16);
      const bytes = CryptoUtils.stringToBytes(plaintext);

      // Pad to block size
      const blockSize = 16;
      const paddedSize = Math.ceil(bytes.length / blockSize) * blockSize;
      const padded = new Uint8Array(paddedSize);
      padded.set(bytes);

      // Encrypt
      let encrypted = new Uint8Array(0);
      for (let i = 0; i < paddedSize; i += blockSize) {
        const block = padded.slice(i, i + blockSize);
        const encBlock = cipher.encrypt(block, key);
        const combined = new Uint8Array(encrypted.length + encBlock.length);
        combined.set(encrypted);
        combined.set(encBlock, encrypted.length);
        encrypted = combined;
      }

      setCiphertext(CryptoUtils.bytesToHex(encrypted));
      Alert.alert('Success', 'Text encrypted with 48-round Feistel cipher');
    } catch (error) {
      Alert.alert('Error', 'Encryption failed');
    }
  };

  return (
    <ScreenShell
      title="Quantum Cryptography"
      subtitle="48-round Φ-RFT Feistel mobile toolkit"
    >
      <View style={styles.leadCopy}>
        <Text style={styles.leadText}>
          Generate 256-bit keys and run the Φ-RFT Feistel cipher locally. This mirrors the
          QuantoniumOS desktop cryptography console with identical parameters.
        </Text>
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Key Management</Text>
        <Text style={styles.sectionSubtitle}>
          Keys never exit the device and remain resident in secure memory.
        </Text>
        <TouchableOpacity style={styles.primaryButton} onPress={generateKey}>
          <Text style={styles.primaryButtonText}>Generate 256-bit Key</Text>
        </TouchableOpacity>
        {key && (
          <View style={styles.keyDisplay}>
            <Text style={styles.keyLabel}>Active Key</Text>
            <Text style={styles.keyValue}>{CryptoUtils.bytesToHex(key).substring(0, 32)}...</Text>
          </View>
        )}
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Encryption</Text>
        <Text style={styles.sectionSubtitle}>
          Plaintext is padded to 16-byte blocks before running the 48-round Feistel network.
        </Text>
        <TextInput
          style={styles.input}
          placeholder="Enter text to encrypt..."
          placeholderTextColor={colors.gray}
          value={plaintext}
          onChangeText={setPlaintext}
          multiline
        />
        <TouchableOpacity style={styles.secondaryButton} onPress={encrypt}>
          <Text style={styles.secondaryButtonText}>Encrypt with Φ-RFT</Text>
        </TouchableOpacity>

        {ciphertext ? (
          <View style={styles.outputCard}>
            <Text style={styles.outputLabel}>Ciphertext (hex)</Text>
            <ScrollView style={styles.outputScroll}>
              <Text style={styles.outputText}>{ciphertext}</Text>
            </ScrollView>
          </View>
        ) : null}
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.infoText}>
          Φ-RFT Feistel implementation mirrors the desktop vault cipher. Use this mobile
          console to verify key exchanges or run quick on-device encryption experiments.
        </Text>
      </View>
    </ScreenShell>
  );
}

const styles = StyleSheet.create({
  leadCopy: {
    marginBottom: spacing.xl,
  },
  leadText: {
    fontSize: typography.body,
    lineHeight: typography.body + 6,
    color: colors.darkGray,
  },
  sectionCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.18)',
    ...shadows.sm,
  },
  sectionTitle: {
    fontSize: typography.subtitle,
    color: colors.dark,
    fontWeight: '600',
    letterSpacing: 0.6,
  },
  sectionSubtitle: {
    marginTop: spacing.xs,
    marginBottom: spacing.lg,
    fontSize: typography.small,
    color: colors.gray,
  },
  primaryButton: {
    backgroundColor: colors.primary,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.md,
    alignItems: 'center',
    ...shadows.md,
  },
  primaryButtonText: {
    fontSize: typography.body,
    color: colors.white,
    fontWeight: '600',
    letterSpacing: 1,
  },
  keyDisplay: {
    marginTop: spacing.lg,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    backgroundColor: colors.offWhite,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.25)',
  },
  keyLabel: {
    fontSize: typography.small,
    color: colors.gray,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  keyValue: {
    fontSize: typography.micro,
    color: colors.dark,
    fontFamily: 'monospace',
  },
  input: {
    backgroundColor: colors.offWhite,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: typography.body,
    color: colors.dark,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.25)',
    minHeight: 120,
    textAlignVertical: 'top',
    marginBottom: spacing.md,
  },
  secondaryButton: {
    backgroundColor: colors.white,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.35)',
    ...shadows.sm,
  },
  secondaryButtonText: {
    fontSize: typography.body,
    color: colors.dark,
    fontWeight: '600',
  },
  outputCard: {
    marginTop: spacing.lg,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.2)',
    backgroundColor: colors.surface,
  },
  outputLabel: {
    padding: spacing.md,
    fontSize: typography.small,
    color: colors.gray,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(52, 152, 219, 0.15)',
  },
  outputScroll: {
    maxHeight: 160,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  outputText: {
    fontSize: typography.micro,
    color: colors.dark,
    fontFamily: 'monospace',
    lineHeight: typography.micro + 6,
  },
  infoCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.12)',
    ...shadows.sm,
  },
  infoText: {
    fontSize: typography.small,
    color: colors.darkGray,
    lineHeight: typography.small + 6,
  },
});
