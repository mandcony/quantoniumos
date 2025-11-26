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
  ScrollView,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as SecureStore from 'expo-secure-store';

interface Note {
  id: string;
  title: string;
  content: string;
  timestamp: number;
  color: string;
}

const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a29bfe'];

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
      const notesJson = await SecureStore.getItemAsync('qnotes_data');
      if (notesJson) {
        setNotes(JSON.parse(notesJson));
      }
    } catch (error) {
      console.error('Failed to load notes:', error);
    }
  };

  const saveNotes = async (newNotes: Note[]) => {
    try {
      await SecureStore.setItemAsync('qnotes_data', JSON.stringify(newNotes));
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
        color: colors[Math.floor(Math.random() * colors.length)],
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

  if (showEditor) {
    return (
      <LinearGradient colors={['#f093fb', '#f5576c']} style={styles.container}>
        <View style={styles.editor}>
          <TextInput
            style={styles.titleInput}
            placeholder="Note Title"
            placeholderTextColor="#aaa"
            value={title}
            onChangeText={setTitle}
          />
          <TextInput
            style={styles.contentInput}
            placeholder="Write your note here..."
            placeholderTextColor="#aaa"
            value={content}
            onChangeText={setContent}
            multiline
            textAlignVertical="top"
          />
          <View style={styles.editorButtons}>
            <TouchableOpacity
              style={[styles.editorButton, styles.cancelButton]}
              onPress={() => {
                setShowEditor(false);
                setTitle('');
                setContent('');
                setCurrentNote(null);
              }}
            >
              <Text style={styles.editorButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.editorButton, styles.saveButton]}
              onPress={saveNote}
            >
              <Text style={styles.editorButtonText}>Save Note</Text>
            </TouchableOpacity>
          </View>
        </View>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient colors={['#f093fb', '#f5576c']} style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Q-Notes</Text>
          <Text style={styles.headerSubtitle}>Secure Note Taking</Text>
        </View>

        <TouchableOpacity style={styles.createButton} onPress={createNote}>
          <Text style={styles.createButtonText}>+ Create New Note</Text>
        </TouchableOpacity>

        {notes.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyIcon}>📝</Text>
            <Text style={styles.emptyText}>No notes yet</Text>
            <Text style={styles.emptySubtext}>Create your first note to get started</Text>
          </View>
        ) : (
          <View style={styles.notesGrid}>
            {notes.map(note => (
              <TouchableOpacity
                key={note.id}
                style={[styles.noteCard, { backgroundColor: note.color }]}
                onPress={() => editNote(note)}
                onLongPress={() => deleteNote(note.id)}
              >
                <Text style={styles.noteTitle} numberOfLines={2}>
                  {note.title}
                </Text>
                <Text style={styles.noteContent} numberOfLines={5}>
                  {note.content}
                </Text>
                <Text style={styles.noteDate}>
                  {new Date(note.timestamp).toLocaleDateString()}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        )}
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
  },
  createButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
  },
  createButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
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
  },
  notesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 15,
  },
  noteCard: {
    width: '47%',
    minHeight: 150,
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
  },
  noteTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 10,
  },
  noteContent: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.9)',
    flex: 1,
  },
  noteDate: {
    fontSize: 10,
    color: 'rgba(255, 255, 255, 0.7)',
    marginTop: 10,
  },
  editor: {
    flex: 1,
    padding: 20,
  },
  titleInput: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: 15,
    borderRadius: 8,
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  contentInput: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: 15,
    borderRadius: 8,
    fontSize: 16,
    flex: 1,
    marginBottom: 15,
    color: '#333',
  },
  editorButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  editorButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  saveButton: {
    backgroundColor: '#4caf50',
  },
  editorButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
