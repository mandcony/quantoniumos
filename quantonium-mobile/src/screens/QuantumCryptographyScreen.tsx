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
import { LinearGradient } from 'expo-linear-gradient';
import { RFTEnhancedFeistel, CryptoUtils } from '../algorithms/crypto/CryptoPrimitives';
import { colors, spacing, typography } from '../constants/DesignSystem';

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
    <LinearGradient colors={colors.cryptoGradient} style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>🔐 Quantum Cryptography</Text>
          <Text style={styles.headerSubtitle}>RFT-Enhanced 48-Round Feistel Cipher</Text>
        </View>

        <View style={styles.controlsContainer}>
          <TouchableOpacity style={styles.keyButton} onPress={generateKey}>
            <Text style={styles.keyButtonText}>🔑 Generate Key</Text>
          </TouchableOpacity>
          {key && (
            <View style={styles.keyDisplay}>
              <Text style={styles.keyLabel}>Active Key:</Text>
              <Text style={styles.keyText}>{CryptoUtils.bytesToHex(key).substring(0, 32)}...</Text>
            </View>
          )}
        </View>

        <View style={styles.ioContainer}>
          <Text style={styles.sectionTitle}>Plaintext</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter text to encrypt..."
            placeholderTextColor="#aaa"
            value={plaintext}
            onChangeText={setPlaintext}
            multiline
          />

          <TouchableOpacity style={styles.encryptButton} onPress={encrypt}>
            <Text style={styles.encryptButtonText}>🔒 Encrypt</Text>
          </TouchableOpacity>

          {ciphertext && (
            <>
              <Text style={styles.sectionTitle}>Ciphertext (Hex)</Text>
              <View style={styles.output}>
                <ScrollView>
                  <Text style={styles.outputText}>{ciphertext}</Text>
                </ScrollView>
              </View>
            </>
          )}
        </View>

        <View style={styles.infoBox}>
          <Text style={styles.infoText}>
            Uses 48-round RFT-enhanced Feistel network with 256-bit keys.
            Provides quantum-resistant encryption for mobile data.
          </Text>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.lg,
    paddingBottom: spacing.xxl,
  },
  header: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  headerTitle: {
    fontSize: typography.title,
    fontWeight: 'bold',
    color: colors.white,
    marginBottom: spacing.xs,
  },
  headerSubtitle: {
    fontSize: typography.small,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  controlsContainer: {
    marginBottom: spacing.lg,
  },
  keyButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    padding: spacing.md,
    borderRadius: 12,
    alignItems: 'center',
  },
  keyButtonText: {
    fontSize: typography.body,
    fontWeight: 'bold',
    color: colors.white,
  },
  keyDisplay: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    padding: spacing.md,
    borderRadius: 12,
    marginTop: spacing.md,
  },
  keyLabel: {
    fontSize: typography.small,
    color: 'rgba(255, 255, 255, 0.8)',
    marginBottom: spacing.xs,
  },
  keyText: {
    fontSize: typography.micro,
    color: colors.white,
    fontFamily: 'monospace',
  },
  ioContainer: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    fontSize: typography.body,
    fontWeight: 'bold',
    color: colors.white,
    marginBottom: spacing.sm,
  },
  input: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: spacing.md,
    borderRadius: 12,
    fontSize: typography.body,
    color: '#333',
    minHeight: 100,
    textAlignVertical: 'top',
    marginBottom: spacing.md,
  },
  encryptButton: {
    backgroundColor: '#4caf50',
    padding: spacing.md,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  encryptButtonText: {
    fontSize: typography.body,
    fontWeight: 'bold',
    color: colors.white,
  },
  output: {
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    padding: spacing.md,
    borderRadius: 12,
    height: 150,
  },
  outputText: {
    fontSize: typography.micro,
    color: colors.white,
    fontFamily: 'monospace',
  },
  infoBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    padding: spacing.lg,
    borderRadius: 12,
  },
  infoText: {
    fontSize: typography.small,
    color: colors.white,
    lineHeight: 20,
  },
});
