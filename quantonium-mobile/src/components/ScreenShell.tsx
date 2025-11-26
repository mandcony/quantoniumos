/**
 * ScreenShell - Shared Golden Ratio layout wrapper for QuantoniumOS Mobile
 * Mirrors desktop QuantoniumOS design language for every screen
 */

import React, { ReactNode, useEffect, useMemo, useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import QLogo from './QLogo';
import { colors, spacing, typography } from '../constants/DesignSystem';

interface ScreenShellProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  footerChildren?: ReactNode;
  scrollable?: boolean;
  contentPadding?: number;
}

function useSystemClock() {
  const [time, setTime] = useState(() => formatNow());
  const [date, setDate] = useState(() => formatDate(new Date()));

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setTime(formatNow(now));
      setDate(formatDate(now));
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return { time, date };
}

function formatNow(now: Date = new Date()) {
  const hours = now.getHours().toString().padStart(2, '0');
  const minutes = now.getMinutes().toString().padStart(2, '0');
  return `${hours}:${minutes}`;
}

function formatDate(now: Date) {
  return now.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
}

export default function ScreenShell({
  title,
  subtitle,
  children,
  footerChildren,
  scrollable = true,
  contentPadding = spacing.lg,
}: ScreenShellProps) {
  const { time, date } = useSystemClock();

  const ContentComponent = useMemo(() => (scrollable ? ScrollView : View), [scrollable]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={styles.topBar}>
          <QLogo size={64} color={colors.primary} />
          <View style={styles.timeBlock}>
            <Text style={styles.timeText}>{time}</Text>
            <Text style={styles.dateText}>{date}</Text>
          </View>
        </View>

        <View style={styles.header}>
          <Text style={styles.title}>{title}</Text>
          {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
          <View style={styles.headerLine} />
        </View>

        <ContentComponent
          style={scrollable ? styles.scrollView : styles.contentView}
          contentContainerStyle={scrollable ? [styles.scrollContent, { padding: contentPadding }] : undefined}
        >
          {!scrollable ? (
            <View style={[styles.contentInner, { padding: contentPadding }]}>{children}</View>
          ) : (
            children
          )}
        </ContentComponent>

        <View style={styles.footer}>
          {footerChildren}
          <Text style={styles.footerTitle}>QUANTONIUMOS</Text>
          <Text style={styles.footerSubtitle}>USPTO 19/169,399 · Symbolic Quantum-Inspired Computing</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.background, // White background matching desktop
  },
  container: {
    flex: 1,
    backgroundColor: colors.background, // White background matching desktop
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(52, 152, 219, 0.1)', // Subtle border like desktop
    paddingBottom: spacing.md,
  },
  timeBlock: {
    alignItems: 'flex-end',
  },
  timeText: {
    fontSize: typography.body,
    lineHeight: typography.body + 4,
    color: colors.darkGray, // #34495e matching desktop
    fontFamily: 'monospace',
    fontWeight: '400',
    letterSpacing: 1,
  },
  dateText: {
    fontSize: typography.small,
    color: colors.gray,
  },
  header: {
    paddingHorizontal: spacing.xl,
    paddingTop: spacing.lg,
    paddingBottom: spacing.sm,
  },
  title: {
    fontSize: typography.subtitle,
    color: colors.primary, // #3498db matching desktop
    fontWeight: '500',
    letterSpacing: 2,
    textTransform: 'uppercase',
  },
  subtitle: {
    marginTop: spacing.xs,
    fontSize: typography.body,
    color: colors.darkGray,
  },
  headerLine: {
    marginTop: spacing.md,
    height: 1,
    backgroundColor: 'rgba(52, 152, 219, 0.15)', // Matching desktop subtle line
    width: '100%',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: spacing.xxl,
  },
  contentView: {
    flex: 1,
  },
  contentInner: {
    flexGrow: 1,
  },
  footer: {
    alignItems: 'center',
    paddingBottom: spacing.lg,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: 'rgba(52, 152, 219, 0.1)', // Subtle border like desktop
  },
  footerTitle: {
    fontSize: typography.small,
    color: colors.primary, // #3498db
    fontWeight: '500',
    letterSpacing: 2,
  },
  footerSubtitle: {
    marginTop: spacing.xs,
    fontSize: typography.micro,
    color: colors.gray,
    textAlign: 'center',
  },
});
