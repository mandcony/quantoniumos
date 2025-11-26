/**
 * AI Chat Screen - QuantoniumOS Mobile
 * Exact 1:1 match with desktop chat interface aesthetics
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
import ScreenShell from '../components/ScreenShell';
import { colors, spacing, typography, PHI, PHI_INV, BASE_UNIT } from '../constants/DesignSystem';
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
    <ScreenShell
      title="AI Chat"
      subtitle="Quantum-Enhanced Assistant"
    >
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={100}
      >
        {/* Domain selector with desktop minimal style */}
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

        {/* Messages area */}
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

        {/* Input area with desktop minimal aesthetics */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Ask about quantum computing, RFT, or crypto..."
            placeholderTextColor={colors.gray}
            value={inputText}
            onChangeText={setInputText}
            onSubmitEditing={sendMessage}
            returnKeyType="send"
            multiline
          />
          <TouchableOpacity
            style={[styles.sendButton, isLoading && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color={colors.white} />
            ) : (
              <Text style={styles.sendButtonText}>→</Text>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </ScreenShell>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  domainSelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(52, 152, 219, 0.15)',
  },
  domainChip: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.3)',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  domainChipActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  domainChipText: {
    fontSize: typography.small,
    color: colors.dark,
    fontWeight: '500',
  },
  domainChipTextActive: {
    color: colors.white,
    fontWeight: '600',
  },
  messagesContainer: {
    flex: 1,
    backgroundColor: colors.surface,
  },
  messagesContent: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: spacing.md,
    borderRadius: 12,
    marginBottom: spacing.md,
    borderWidth: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 2,
    elevation: 1,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: 'rgba(52, 152, 219, 0.1)', // Light blue tint
    borderColor: 'rgba(52, 152, 219, 0.3)',
  },
  aiBubble: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: 'rgba(52, 152, 219, 0.2)',
  },
  messageText: {
    fontSize: typography.body,
    color: colors.dark,
    lineHeight: typography.body + 6,
    marginBottom: spacing.xs,
  },
  messageMeta: {
    fontSize: typography.micro,
    color: colors.gray,
    marginBottom: spacing.xs,
    fontFamily: 'monospace',
  },
  messageTime: {
    fontSize: typography.micro,
    color: colors.gray,
    alignSelf: 'flex-end',
    fontFamily: 'monospace',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: spacing.md,
    backgroundColor: colors.white,
    borderTopWidth: 1,
    borderTopColor: 'rgba(52, 152, 219, 0.15)',
    gap: spacing.sm,
  },
  input: {
    flex: 1,
    backgroundColor: colors.surface,
    padding: spacing.md,
    borderRadius: 12,
    fontSize: typography.body,
    color: colors.dark,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.2)',
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: colors.primary,
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  sendButtonDisabled: {
    opacity: 0.5,
  },
  sendButtonText: {
    fontSize: 24,
    color: colors.white,
    fontWeight: 'bold',
  },
});
