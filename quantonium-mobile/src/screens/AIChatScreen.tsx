/**
 * AI Chat Screen - QuantoniumOS Mobile
 * Quantum-enhanced AI assistant
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing, typography } from '../constants/DesignSystem';
import {
  DOMAIN_OPTIONS,
  DomainKey,
  DEFAULT_TOP_K,
} from '../config/ai';
import { queryNeuralBrain } from '../utils/neuralBrainClient';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  meta?: {
    domain?: string;
    confidence?: number;
  };
}

export default function AIChatScreen() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m the QuantoniumOS AI assistant. I can help you understand quantum computing, RFT algorithms, and cryptographic protocols. How can I assist you?',
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<DomainKey>('auto');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    const trimmed = inputText.trim();
    if (!trimmed || isLoading) {
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: trimmed,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');

    try {
      setIsLoading(true);
      const response = await queryNeuralBrain(
        trimmed,
        selectedDomain,
        DEFAULT_TOP_K,
      );
      const aiMessage: Message = {
        id: `${Date.now()}-ai`,
        text: response.answer.trim(),
        sender: 'ai',
        timestamp: new Date(response.timestamp * 1000),
        meta: {
          domain: response.domain,
          confidence: response.confidence,
        },
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const aiMessage: Message = {
        id: `${Date.now()}-error`,
        text:
          (error as Error).message ??
          'Unable to reach the neural brain service. Ensure the Python backend is running.',
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, aiMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <LinearGradient colors={colors.aiGradient} style={styles.container}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={100}
      >
        <View style={styles.header}>
          <Text style={styles.headerTitle}>🤖 AI Chat</Text>
          <Text style={styles.headerSubtitle}>Quantum-Enhanced Assistant</Text>
        </View>

        <View style={styles.domainSelector}>
          {DOMAIN_OPTIONS.map(option => {
            const active = selectedDomain === option.value;
            return (
              <TouchableOpacity
                key={option.value}
                style={[styles.domainChip, active && styles.domainChipActive]}
                onPress={() => setSelectedDomain(option.value)}
              >
                <Text
                  style={[styles.domainChipText, active && styles.domainChipTextActive]}
                >
                  {option.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        <ScrollView
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
        >
          {messages.map(message => (
            <View
              key={message.id}
              style={[
                styles.messageBubble,
                message.sender === 'user' ? styles.userBubble : styles.aiBubble,
              ]}
            >
              <Text style={styles.messageText}>{message.text}</Text>
              {message.meta && message.meta.domain && (
                <Text style={styles.messageMeta}>
                  Domain: {message.meta.domain}
                  {typeof message.meta.confidence === 'number'
                    ? ` · Confidence ${(message.meta.confidence * 100).toFixed(0)}%`
                    : ''}
                </Text>
              )}
              <Text style={styles.messageTime}>
                {message.timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </Text>
            </View>
          ))}
        </ScrollView>

        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Ask about quantum computing, RFT, or crypto..."
            placeholderTextColor="#999"
            value={inputText}
            onChangeText={setInputText}
            onSubmitEditing={sendMessage}
            returnKeyType="send"
          />
          <TouchableOpacity
            style={[styles.sendButton, isLoading && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.sendButtonText}>📤</Text>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    alignItems: 'center',
    paddingTop: spacing.lg,
    paddingBottom: spacing.md,
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
  domainSelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.sm,
  },
  domainChip: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.4)',
  },
  domainChipActive: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderColor: colors.white,
  },
  domainChipText: {
    fontSize: typography.small,
    color: colors.white,
  },
  domainChipTextActive: {
    color: colors.white,
    fontWeight: '600',
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: spacing.md,
    borderRadius: 16,
    marginBottom: spacing.md,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  aiBubble: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  messageText: {
    fontSize: typography.body,
    color: colors.white,
    marginBottom: spacing.xs,
  },
  messageMeta: {
    fontSize: typography.micro,
    color: 'rgba(255, 255, 255, 0.7)',
    marginBottom: spacing.xs,
  },
  messageTime: {
    fontSize: typography.micro,
    color: 'rgba(255, 255, 255, 0.6)',
    alignSelf: 'flex-end',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: spacing.md,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    gap: spacing.sm,
  },
  input: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: spacing.md,
    borderRadius: 24,
    fontSize: typography.body,
    color: '#333',
  },
  sendButton: {
    backgroundColor: colors.primary,
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    opacity: 0.6,
  },
  sendButtonText: {
    fontSize: 24,
  },
});

