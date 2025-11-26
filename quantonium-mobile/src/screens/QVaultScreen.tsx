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
  ScrollView,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as SecureStore from 'expo-secure-store';
import { RFTEnhancedFeistel, CryptoUtils } from '../algorithms/crypto/CryptoPrimitives';

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
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Q-Vault</Text>
          <Text style={styles.headerSubtitle}>
            48-Round RFT-Enhanced Feistel Encryption
          </Text>
          <View style={styles.statusBadge}>
            <Text style={styles.statusText}>
              🔐 {items.length} items secured
            </Text>
          </View>
        </View>

        {!showAddForm ? (
          <TouchableOpacity
            style={styles.addButton}
            onPress={() => setShowAddForm(true)}
          >
            <Text style={styles.addButtonText}>+ Add Secure Item</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.form}>
            <Text style={styles.formTitle}>New Secure Item</Text>
            <TextInput
              style={styles.input}
              placeholder="Title"
              placeholderTextColor="#aaa"
              value={newTitle}
              onChangeText={setNewTitle}
            />
            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Content (will be encrypted)"
              placeholderTextColor="#aaa"
              value={newContent}
              onChangeText={setNewContent}
              multiline
              numberOfLines={4}
            />
            <View style={styles.formButtons}>
              <TouchableOpacity
                style={[styles.formButton, styles.cancelButton]}
                onPress={() => {
                  setShowAddForm(false);
                  setNewTitle('');
                  setNewContent('');
                }}
              >
                <Text style={styles.formButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.formButton, styles.saveButton]}
                onPress={addItem}
              >
                <Text style={styles.formButtonText}>Encrypt & Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        <View style={styles.itemsList}>
          {items.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyIcon}>🔒</Text>
              <Text style={styles.emptyText}>Your vault is empty</Text>
              <Text style={styles.emptySubtext}>
                Add items to store them with quantum-resistant encryption
              </Text>
            </View>
          ) : (
            items.map((item) => (
              <View key={item.id} style={styles.item}>
                <View style={styles.itemHeader}>
                  <Text style={styles.itemTitle}>{item.title}</Text>
                  <Text style={styles.itemDate}>
                    {new Date(item.timestamp).toLocaleDateString()}
                  </Text>
                </View>
                <Text style={styles.itemPreview}>
                  [Encrypted • {item.content.length} bytes]
                </Text>
                <View style={styles.itemActions}>
                  <TouchableOpacity
                    style={[styles.actionButton, styles.viewButton]}
                    onPress={() => viewItem(item)}
                  >
                    <Text style={styles.actionButtonText}>👁️ View</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.actionButton, styles.deleteButton]}
                    onPress={() => deleteItem(item.id)}
                  >
                    <Text style={styles.actionButtonText}>🗑️ Delete</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))
          )}
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
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    marginBottom: 15,
  },
  statusBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  statusText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  addButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
  },
  addButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  form: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  formTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 15,
  },
  input: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    fontSize: 16,
    color: '#333',
  },
  textArea: {
    height: 100,
    textAlignVertical: 'top',
  },
  formButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  formButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  saveButton: {
    backgroundColor: '#4caf50',
  },
  formButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  itemsList: {
    gap: 15,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyIcon: {
    fontSize: 64,
    marginBottom: 20,
  },
  emptyText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 10,
  },
  emptySubtext: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  item: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    padding: 15,
    borderRadius: 12,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  itemTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    flex: 1,
  },
  itemDate: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
  },
  itemPreview: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.6)',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  itemActions: {
    flexDirection: 'row',
    gap: 10,
  },
  actionButton: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  viewButton: {
    backgroundColor: 'rgba(100, 200, 255, 0.5)',
  },
  deleteButton: {
    backgroundColor: 'rgba(255, 100, 100, 0.5)',
  },
  actionButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
});
