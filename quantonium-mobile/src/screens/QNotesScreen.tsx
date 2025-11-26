/**
 * Q-Notes Screen - Encrypted Notes Application
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
  borderRadius,
  colors as dsColors,
  shadows,
  spacing,
  typography,
} from '../constants/DesignSystem';

interface Note {
  id: string;
  title: string;
  content: string;
  timestamp: number;
  color: string;
}

const NOTE_STORAGE_KEY = 'qnotes_data';

export default function QNotesScreen() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [showEditor, setShowEditor] = useState(false);
  const [currentNote, setCurrentNote] = useState<Note | null>(null);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');

  useEffect(() => {
    loadNotes();
  }, []);

  const loadNotes = async () => {
    try {
      const notesJson = await SecureStore.getItemAsync(NOTE_STORAGE_KEY);
      if (notesJson) {
        setNotes(JSON.parse(notesJson));
      }
    } catch (error) {
      console.error('Failed to load notes:', error);
    }
  };

  const saveNotes = async (newNotes: Note[]) => {
    try {
      await SecureStore.setItemAsync(NOTE_STORAGE_KEY, JSON.stringify(newNotes));
      setNotes(newNotes);
    } catch (error) {
      console.error('Failed to save notes:', error);
      Alert.alert('Error', 'Failed to save notes');
    }
  };

  const createNote = () => {
    setCurrentNote(null);
    setTitle('');
    setContent('');
    setShowEditor(true);
  };

  const editNote = (note: Note) => {
    setCurrentNote(note);
    setTitle(note.title);
    setContent(note.content);
    setShowEditor(true);
  };

  const saveNote = () => {
    if (!title.trim()) {
      Alert.alert('Error', 'Please enter a title');
      return;
    }

    if (currentNote) {
      // Update existing note
      const updatedNotes = notes.map(n =>
        n.id === currentNote.id
          ? { ...n, title, content, timestamp: Date.now() }
          : n
      );
      saveNotes(updatedNotes);
    } else {
      // Create new note
      const newNote: Note = {
        id: Date.now().toString(),
        title,
        content,
        timestamp: Date.now(),
        color: dsColors.primaryLight,
      };
      saveNotes([...notes, newNote]);
    }

    setShowEditor(false);
    setTitle('');
    setContent('');
    setCurrentNote(null);
  };

  const deleteNote = (id: string) => {
    Alert.alert('Confirm Delete', 'Are you sure you want to delete this note?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          const updatedNotes = notes.filter(n => n.id !== id);
          saveNotes(updatedNotes);
        },
      },
    ]);
  };

  return (
    <ScreenShell
      title="Q-Notes"
      subtitle="Encrypted research notebook aligned with Φ-RFT design"
    >
      <View style={styles.leadCopy}>
        <Text style={styles.leadText}>
          Draft experiments, patent notes, and system memos inside the QuantoniumOS
          notebook. Entries remain on-device and inherit the Φ-RFT vault
          encryption layer.
        </Text>
      </View>

      {showEditor ? (
        <View style={styles.editorCard}>
          <Text style={styles.sectionTitle}>
            {currentNote ? 'Update Note' : 'Create Note'}
          </Text>
          <Text style={styles.sectionSubtitle}>
            {currentNote
              ? 'Edit the secured entry. Changes save into the Φ-RFT notebook ledger.'
              : 'Compose a new encrypted entry. Upon save it is sealed with Φ-RFT crypto.'}
          </Text>
          <TextInput
            style={styles.input}
            placeholder="Note title"
            placeholderTextColor={dsColors.gray}
            value={title}
            onChangeText={setTitle}
          />
          <TextInput
            style={[styles.input, styles.textArea]}
            placeholder="Write your note..."
            placeholderTextColor={dsColors.gray}
            value={content}
            onChangeText={setContent}
            multiline
          />
          <View style={styles.editorActions}>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.cancelButton]}
              onPress={() => {
                setShowEditor(false);
                setTitle('');
                setContent('');
                setCurrentNote(null);
              }}
            >
              <Text style={styles.secondaryButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, styles.saveButton]}
              onPress={saveNote}
            >
              <Text style={styles.saveButtonText}>
                {currentNote ? 'Save Changes' : 'Save Note'}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <TouchableOpacity style={styles.primaryButton} onPress={createNote}>
          <Text style={styles.primaryButtonText}>Create New Note</Text>
        </TouchableOpacity>
      )}

      <View>
        <Text style={styles.sectionTitle}>Notebook Entries</Text>
        <Text style={styles.sectionSubtitle}>
          {notes.length === 0
            ? 'No encrypted notes yet. Start drafting to populate the ledger.'
            : 'Tap a card to edit; hold to remove. Decryption happens only in-session.'}
        </Text>

        {notes.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>Notebook is empty</Text>
            <Text style={styles.emptySubtitle}>
              Create a note to mirror the QuantoniumOS desktop research workspace.
            </Text>
          </View>
        ) : (
          notes.map(note => (
            <View key={note.id} style={styles.noteCard}>
              <View style={styles.noteHeader}>
                <View style={styles.noteAccent} />
                <View style={styles.noteHeadings}>
                  <Text style={styles.noteTitle} numberOfLines={2}>
                    {note.title}
                  </Text>
                  <Text style={styles.noteMeta}>
                    {new Date(note.timestamp).toLocaleDateString()}
                  </Text>
                </View>
              </View>
              <Text style={styles.notePreview} numberOfLines={5}>
                {note.content || '—'}
              </Text>
              <View style={styles.noteActions}>
                <TouchableOpacity
                  style={[styles.inlineButton, styles.editAction]}
                  onPress={() => editNote(note)}
                >
                  <Text style={styles.inlineButtonText}>Edit Entry</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.inlineButton, styles.deleteAction]}
                  onPress={() => deleteNote(note.id)}
                >
                  <Text style={styles.inlineButtonText}>Delete</Text>
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
    color: dsColors.darkGray,
  },
  primaryButton: {
    backgroundColor: dsColors.primary,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.md,
    alignItems: 'center',
    marginBottom: spacing.xl,
    ...shadows.md,
  },
  primaryButtonText: {
    color: dsColors.white,
    fontSize: typography.body,
    fontWeight: '600',
    letterSpacing: 1,
  },
  sectionTitle: {
    fontSize: typography.subtitle,
    color: dsColors.dark,
    fontWeight: '600',
    letterSpacing: 0.6,
  },
  sectionSubtitle: {
    marginTop: spacing.xs,
    marginBottom: spacing.lg,
    fontSize: typography.small,
    color: dsColors.gray,
    lineHeight: typography.small + 4,
  },
  editorCard: {
    backgroundColor: dsColors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.18)',
    ...shadows.sm,
  },
  input: {
    backgroundColor: dsColors.offWhite,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: typography.body,
    color: dsColors.dark,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.25)',
    marginBottom: spacing.md,
  },
  textArea: {
    minHeight: 140,
    textAlignVertical: 'top',
  },
  editorActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: spacing.md,
  },
  secondaryButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.28)',
    marginLeft: spacing.sm,
  },
  cancelButton: {
    backgroundColor: dsColors.white,
  },
  saveButton: {
    backgroundColor: dsColors.primary,
    borderColor: dsColors.primary,
  },
  secondaryButtonText: {
    fontSize: typography.small,
    color: dsColors.dark,
    fontWeight: '600',
  },
  saveButtonText: {
    fontSize: typography.small,
    color: dsColors.white,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: spacing.xxl,
    borderRadius: borderRadius.xl,
    backgroundColor: dsColors.surfaceElevated,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.15)',
    ...shadows.sm,
  },
  emptyTitle: {
    fontSize: typography.subtitle,
    color: dsColors.dark,
    fontWeight: '600',
    marginBottom: spacing.xs,
  },
  emptySubtitle: {
    fontSize: typography.small,
    color: dsColors.gray,
    textAlign: 'center',
    paddingHorizontal: spacing.lg,
    lineHeight: typography.small + 4,
  },
  noteCard: {
    backgroundColor: dsColors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.15)',
    ...shadows.sm,
  },
  noteHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  noteAccent: {
    width: 6,
    alignSelf: 'stretch',
    borderRadius: borderRadius.sm,
    backgroundColor: dsColors.primary,
    marginRight: spacing.md,
  },
  noteHeadings: {
    flex: 1,
  },
  noteTitle: {
    fontSize: typography.body,
    color: dsColors.dark,
    fontWeight: '600',
  },
  noteMeta: {
    marginTop: spacing.xs,
    fontSize: typography.small,
    color: dsColors.gray,
  },
  notePreview: {
    fontSize: typography.small,
    color: dsColors.darkGray,
    lineHeight: typography.small + 6,
    marginBottom: spacing.md,
  },
  noteActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  inlineButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    marginLeft: spacing.sm,
  },
  inlineButtonText: {
    fontSize: typography.small,
    fontWeight: '600',
    color: dsColors.dark,
  },
  editAction: {
    borderColor: 'rgba(52, 152, 219, 0.28)',
    backgroundColor: dsColors.white,
  },
  deleteAction: {
    borderColor: 'rgba(231, 76, 60, 0.28)',
    backgroundColor: dsColors.white,
  },
});
