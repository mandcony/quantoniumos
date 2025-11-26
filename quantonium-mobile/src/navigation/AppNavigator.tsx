/**
 * App Navigator - QuantoniumOS Mobile
 * Main navigation structure
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';

// Import screens
import LauncherScreen from '../screens/LauncherScreen';
import QVaultScreen from '../screens/QVaultScreen';
import QNotesScreen from '../screens/QNotesScreen';
import QuantumSimulatorScreen from '../screens/QuantumSimulatorScreen';
import RFTVisualizerScreen from '../screens/RFTVisualizerScreen';
import ValidationScreen from '../screens/ValidationScreen';
import SystemMonitorScreen from '../screens/SystemMonitorScreen';
import QuantumCryptographyScreen from '../screens/QuantumCryptographyScreen';
import AIChatScreen from '../screens/AIChatScreen';

export type RootStackParamList = {
  Launcher: undefined;
  QVault: undefined;
  QNotes: undefined;
  QuantumSimulator: undefined;
  RFTVisualizer: undefined;
  Validation: undefined;
  SystemMonitor: undefined;
  QuantumCryptography: undefined;
  AIChat: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <>
      <StatusBar style="light" />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Launcher"
          screenOptions={{
            headerStyle: {
              backgroundColor: '#1a1a2e',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
            contentStyle: {
              backgroundColor: '#16213e',
            },
          }}
        >
          <Stack.Screen
            name="Launcher"
            component={LauncherScreen}
            options={{
              title: 'QuantoniumOS',
              headerShown: false, // Fullscreen like desktop
            }}
          />
          <Stack.Screen
            name="QVault"
            component={QVaultScreen}
            options={{ title: 'Q-Vault' }}
          />
          <Stack.Screen
            name="QNotes"
            component={QNotesScreen}
            options={{ title: 'Q-Notes' }}
          />
          <Stack.Screen
            name="QuantumSimulator"
            component={QuantumSimulatorScreen}
            options={{ title: 'Quantum Simulator' }}
          />
          <Stack.Screen
            name="RFTVisualizer"
            component={RFTVisualizerScreen}
            options={{ title: 'RFT Visualizer' }}
          />
          <Stack.Screen
            name="Validation"
            component={ValidationScreen}
            options={{ title: 'RFT Validation Suite' }}
          />
          <Stack.Screen
            name="SystemMonitor"
            component={SystemMonitorScreen}
            options={{ title: 'System Monitor' }}
          />
          <Stack.Screen
            name="QuantumCryptography"
            component={QuantumCryptographyScreen}
            options={{ title: 'Quantum Cryptography' }}
          />
          <Stack.Screen
            name="AIChat"
            component={AIChatScreen}
            options={{ title: 'AI Chat' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}
