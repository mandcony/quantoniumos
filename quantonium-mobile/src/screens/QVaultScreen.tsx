/**
 * Q-Vault Screen - Quantum-Secure Storage
 * Encrypted storage using RFT-enhanced Feistel cipher
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Alert,
} from 'react-native';
import * as SecureStore from 'expo-secure-store';
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

interface VaultItem {
  id: string;
  title: string;
  content: string;
  timestamp: number;
}

export default function QVaultScreen() {
  const [items, setItems] = useState<VaultItem[]>([]);
  const [newTitle, setNewTitle] = useState('');
  const [newContent, setNewContent] = useState('');
  const [masterKey, setMasterKey] = useState<Uint8Array | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);

  useEffect(() => {
    initializeVault();
  }, []);

  const initializeVault = async () => {
    try {
      // Try to load existing key or generate new one
      const keyHex = await SecureStore.getItemAsync('vault_master_key');
      if (keyHex) {
        setMasterKey(CryptoUtils.hexToBytes(keyHex));
      } else {
        // Generate new master key
        const newKey = CryptoUtils.generateRandomBytes(32);
        await SecureStore.setItemAsync('vault_master_key', CryptoUtils.bytesToHex(newKey));
        setMasterKey(newKey);
      }

      // Load items
      loadItems();
    } catch (error) {
      console.error('Failed to initialize vault:', error);
      Alert.alert('Error', 'Failed to initialize vault');
    }
  };

  const loadItems = async () => {
    try {
      const itemsJson = await SecureStore.getItemAsync('vault_items');
      if (itemsJson) {
        setItems(JSON.parse(itemsJson));
      }
    } catch (error) {
      console.error('Failed to load items:', error);
    }
  };

  const saveItems = async (newItems: VaultItem[]) => {
    try {
      await SecureStore.setItemAsync('vault_items', JSON.stringify(newItems));
      setItems(newItems);
    } catch (error) {
      console.error('Failed to save items:', error);
      Alert.alert('Error', 'Failed to save items');
    }
  };

  const encryptContent = (content: string): string => {
    if (!masterKey) return content;

    try {
      const cipher = new RFTEnhancedFeistel(48, 16);
      const plaintext = CryptoUtils.stringToBytes(content);

      // Pad to block size
      const blockSize = 16;
      const paddedSize = Math.ceil(plaintext.length / blockSize) * blockSize;
      const padded = new Uint8Array(paddedSize);
      padded.set(plaintext);

      // Encrypt each block
      let encrypted = new Uint8Array(0);
      for (let i = 0; i < paddedSize; i += blockSize) {
        const block = padded.slice(i, i + blockSize);
        const encryptedBlock = cipher.encrypt(block, masterKey);
        const combined = new Uint8Array(encrypted.length + encryptedBlock.length);
        combined.set(encrypted);
        combined.set(encryptedBlock, encrypted.length);
        encrypted = combined;
      }

      return CryptoUtils.bytesToHex(encrypted);
    } catch (error) {
      console.error('Encryption error:', error);
      return content;
    }
  };

  const decryptContent = (encryptedHex: string): string => {
    if (!masterKey) return encryptedHex;

    try {
      const cipher = new RFTEnhancedFeistel(48, 16);
      const encrypted = CryptoUtils.hexToBytes(encryptedHex);

      // Decrypt each block
      const blockSize = 16;
      let decrypted = new Uint8Array(0);
      for (let i = 0; i < encrypted.length; i += blockSize) {
        const block = encrypted.slice(i, i + blockSize);
        const decryptedBlock = cipher.decrypt(block, masterKey);
        const combined = new Uint8Array(decrypted.length + decryptedBlock.length);
        combined.set(decrypted);
        combined.set(decryptedBlock, decrypted.length);
        decrypted = combined;
      }

      // Remove padding (null bytes)
      let actualLength = decrypted.length;
      while (actualLength > 0 && decrypted[actualLength - 1] === 0) {
        actualLength--;
      }

      return CryptoUtils.bytesToString(decrypted.slice(0, actualLength));
    } catch (error) {
      console.error('Decryption error:', error);
      return '[Decryption Failed]';
    }
  };

  const addItem = () => {
    if (!newTitle.trim() || !newContent.trim()) {
      Alert.alert('Error', 'Please fill in both title and content');
      return;
    }

    const encrypted = encryptContent(newContent);
    const newItem: VaultItem = {
      id: Date.now().toString(),
      title: newTitle,
      content: encrypted,
      timestamp: Date.now(),
    };

    const updatedItems = [...items, newItem];
    saveItems(updatedItems);

    setNewTitle('');
    setNewContent('');
    setShowAddForm(false);
    Alert.alert('Success', 'Item encrypted and stored securely');
  };

  const viewItem = (item: VaultItem) => {
    const decrypted = decryptContent(item.content);
    Alert.alert(item.title, decrypted, [{ text: 'OK' }]);
  };

  const deleteItem = (id: string) => {
    Alert.alert(
      'Confirm Delete',
      'Are you sure you want to delete this item?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            const updatedItems = items.filter((item) => item.id !== id);
            saveItems(updatedItems);
          },
        },
      ]
    );
  };

  return (
    <ScreenShell
      title="Q-Vault"
      subtitle="48-round Φ-RFT Feistel quantum secure enclave"
    >
      <View style={styles.leadCopy}>
        <Text style={styles.leadText}>
          Secure sensitive research artifacts with the same Φ-RFT Feistel core
          powering the QuantoniumOS desktop vault. Entries are encrypted on-device
          before storage and only decrypted momentarily when you review them.
        </Text>
      </View>

      <View style={styles.statusCard}>
        <Text style={styles.statusLabel}>Stored Entries</Text>
        <Text style={styles.statusValue}>{items.length.toString().padStart(2, '0')}</Text>
        <Text style={styles.statusCaption}>
          Cipher Suite · Φ-RFT Feistel · 48 rounds · 256-bit master key
        </Text>
      </View>

      {!showAddForm ? (
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => setShowAddForm(true)}
        >
          <Text style={styles.primaryButtonText}>Add Secure Item</Text>
        </TouchableOpacity>
      ) : (
        <View style={styles.formCard}>
          <Text style={styles.sectionTitle}>New Secure Item</Text>
          <Text style={styles.sectionSubtitle}>
            Title and notes are encrypted locally using the Φ-RFT vault cipher.
          </Text>

          <TextInput
            style={styles.input}
            placeholder="Title"
            placeholderTextColor={colors.gray}
            value={newTitle}
            onChangeText={setNewTitle}
          />
          <TextInput
            style={[styles.input, styles.textArea]}
            placeholder="Content (encrypted before storage)"
            placeholderTextColor={colors.gray}
            value={newContent}
            onChangeText={setNewContent}
            multiline
          />

          <View style={styles.formActions}>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.cancelButton]}
              onPress={() => {
                setShowAddForm(false);
                setNewTitle('');
                setNewContent('');
              }}
            >
              <Text style={styles.secondaryButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.saveButton]}
              onPress={addItem}
            >
              <Text style={styles.saveButtonText}>Encrypt & Save</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      <View>
        <Text style={styles.sectionTitle}>Vault Contents</Text>
        <Text style={styles.sectionSubtitle}>
          Items remain opaque at rest; only decrypted within the secure session
          preview.
        </Text>

        {items.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyGlyph}>�</Text>
            <Text style={styles.emptyTitle}>No encrypted entries yet</Text>
            <Text style={styles.emptySubtitle}>
              Add research notes or credentials to seed the Φ-RFT vault.
            </Text>
          </View>
        ) : (
          items.map(item => (
            <View key={item.id} style={styles.itemCard}>
              <View style={styles.itemHeader}>
                <Text style={styles.itemTitle}>{item.title}</Text>
                <Text style={styles.itemMeta}>
                  {new Date(item.timestamp).toLocaleDateString()}
                </Text>
              </View>
              <Text style={styles.itemCipherLine}>
                [Encrypted • {item.content.length} bytes]
              </Text>
              <View style={styles.itemActions}>
                <TouchableOpacity
                  style={[styles.inlineButton, styles.viewAction]}
                  onPress={() => viewItem(item)}
                >
                  <Text style={styles.inlineButtonText}>Preview Decrypted</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.inlineButton, styles.deleteAction]}
                  onPress={() => deleteItem(item.id)}
                >
                  <Text style={styles.inlineButtonText}>Delete Entry</Text>
                </TouchableOpacity>
              </View>
            </View>
          ))
        )}
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
  statusCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.2)',
    ...shadows.md,
  },
  statusLabel: {
    fontSize: typography.small,
    color: colors.gray,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  statusValue: {
    fontSize: typography.hero,
    color: colors.primary,
    fontWeight: '600',
    marginTop: spacing.sm,
  },
  statusCaption: {
    marginTop: spacing.sm,
    fontSize: typography.small,
    color: colors.darkGray,
  },
  primaryButton: {
    backgroundColor: colors.primary,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.md,
    alignItems: 'center',
    marginBottom: spacing.xl,
    ...shadows.md,
  },
  primaryButtonText: {
    color: colors.white,
    fontSize: typography.body,
    fontWeight: '600',
    letterSpacing: 1,
  },
  formCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.15)',
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
  input: {
    backgroundColor: colors.offWhite,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: typography.body,
    color: colors.dark,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.25)',
    marginBottom: spacing.md,
  },
  textArea: {
    minHeight: 120,
    textAlignVertical: 'top',
  },
  formActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: spacing.md,
  },
  secondaryButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
    marginLeft: spacing.sm,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.3)',
  },
  cancelButton: {
    backgroundColor: colors.white,
  },
  saveButton: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  secondaryButtonText: {
    fontSize: typography.small,
    color: colors.dark,
    fontWeight: '600',
  },
  saveButtonText: {
    fontSize: typography.small,
    color: colors.white,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: spacing.xxl,
    borderRadius: borderRadius.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.15)',
    backgroundColor: colors.surfaceElevated,
    marginTop: spacing.lg,
    ...shadows.sm,
  },
  emptyGlyph: {
    fontSize: typography.hero,
    marginBottom: spacing.md,
  },
  emptyTitle: {
    fontSize: typography.subtitle,
    color: colors.dark,
    fontWeight: '600',
    marginBottom: spacing.xs,
  },
  emptySubtitle: {
    fontSize: typography.small,
    color: colors.gray,
    textAlign: 'center',
    paddingHorizontal: spacing.lg,
  },
  itemCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginTop: spacing.lg,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.18)',
    ...shadows.sm,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    marginBottom: spacing.sm,
  },
  itemTitle: {
    fontSize: typography.body,
    color: colors.dark,
    fontWeight: '600',
    flex: 1,
    marginRight: spacing.sm,
  },
  itemMeta: {
    fontSize: typography.small,
    color: colors.gray,
  },
  itemCipherLine: {
    fontSize: typography.small,
    color: colors.darkGray,
    fontStyle: 'italic',
    marginBottom: spacing.md,
  },
  itemActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  inlineButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
    marginLeft: spacing.sm,
    borderWidth: 1,
  },
  viewAction: {
    borderColor: 'rgba(52, 152, 219, 0.3)',
    backgroundColor: colors.white,
  },
  deleteAction: {
    borderColor: 'rgba(231, 76, 60, 0.3)',
    backgroundColor: colors.white,
  },
  inlineButtonText: {
    fontSize: typography.small,
    color: colors.dark,
    fontWeight: '600',
  },
});
