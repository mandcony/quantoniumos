import React, { createContext, useContext, useMemo } from 'react';
import { useColorScheme } from 'react-native';

// Create Theme Context
const ThemeContext = createContext();

// Cyberpunk Colors
const cyberpunkColors = {
    dark: {
        primary: '#A786DF', // Soft lavender
        secondary: '#64B5F6', // Lighter Blue
        background: {
            start: '#131A26', // Very dark gray
            middle: '#1C212E', // Dark gray
            end: '#232936',  // Slightly lighter dark gray
        },
        text: '#000000',  // Always Black
        accent: '#FFC107', // Gold
        error: '#F44336', // Soft Red
    },
    light: {
        primary: '#5C6BC0', // Light purple
        secondary: '#26A69A', // Dark teal
        background: {
            start: '#F5F5F5',   // Very Light Gray
            middle: '#EEEEEE',  // Light Gray
            end: '#E0E0E0',    // Light Grey
        },
        text: '#000000',  // Always Black
       accent: '#FFA000', // Deep orange
        error: '#E57373', // Soft red
    },
};

// ThemeProvider Component
export function ThemeProvider({ children }) {
    const colorScheme = useColorScheme();
    const isDarkMode = colorScheme === 'dark';

    const colors = useMemo(() => (isDarkMode ? cyberpunkColors.dark : cyberpunkColors.light), [isDarkMode]);

    return (
        <ThemeContext.Provider value={{ colors }}>
            {children}
        </ThemeContext.Provider>
    );
}

// Custom hook for using the theme
export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
}