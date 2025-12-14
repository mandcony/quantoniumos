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
import AIChatScreen from '../screens/AIChatScreen';
import StructuralHealthScreen from '../screens/StructuralHealthScreen';

export type RootStackParamList = {
  Launcher: undefined;
  AIChat: undefined;
  StructuralHealth: undefined;
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
            name="AIChat"
            component={AIChatScreen}
            options={{ title: 'AI Chat' }}
          />
          <Stack.Screen
            name="StructuralHealth"
            component={StructuralHealthScreen}
            options={{ title: 'Structural Health Monitor' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}
