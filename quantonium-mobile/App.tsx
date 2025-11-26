/**
 * QuantoniumOS Mobile
 * Main Application Entry Point with brand-aligned splash
 */

import React, { useEffect, useState } from 'react';
import * as SplashScreen from 'expo-splash-screen';
import AppNavigator from './src/navigation/AppNavigator';
import QSplashScreen from './src/components/QSplashScreen';

SplashScreen.preventAutoHideAsync().catch(() => {
  /* no-op if already prevented */
});

export default function App() {
  const [isAppReady, setIsAppReady] = useState(false);
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    let isMounted = true;

    async function prepare() {
      try {
        // Lightweight pause keeps the Q splash visible long enough to register
        await new Promise(resolve => setTimeout(resolve, 900));
      } catch (error) {
        console.warn('Initialization sequence interrupted', error);
      } finally {
        if (!isMounted) {
          return;
        }

        setIsAppReady(true);

        try {
          await SplashScreen.hideAsync();
        } catch (error) {
          console.warn('Failed hiding Expo splash', error);
        }

        // Allow custom splash to linger briefly for smooth transition
        setTimeout(() => {
          if (isMounted) {
            setShowSplash(false);
          }
        }, 400);
      }
    }

    prepare();

    return () => {
      isMounted = false;
    };
  }, []);

  if (!isAppReady || showSplash) {
    return <QSplashScreen />;
  }

  return <AppNavigator />;
}
