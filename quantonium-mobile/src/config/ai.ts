export const AI_SERVER_URL =
  process.env.EXPO_PUBLIC_NEURAL_BRAIN_URL ?? 'http://127.0.0.1:8765';

export const DEFAULT_TOP_K = Number(
  process.env.EXPO_PUBLIC_NEURAL_BRAIN_TOP_K ?? 3,
);

export type DomainKey =
  | 'auto'
  | 'medicine'
  | 'physics'
  | 'computer_science'
  | 'psychology'
  | 'philosophy'
  | 'mathematics'
  | 'biology'
  | 'chemistry'
  | 'engineering'
  | 'art'
  | 'survival';

export const DOMAIN_OPTIONS: Array<{ label: string; value: DomainKey }> = [
  { label: 'Auto', value: 'auto' },
  { label: 'Medicine', value: 'medicine' },
  { label: 'Physics', value: 'physics' },
  { label: 'CompSci', value: 'computer_science' },
  { label: 'Psychology', value: 'psychology' },
  { label: 'Philosophy', value: 'philosophy' },
  { label: 'Mathematics', value: 'mathematics' },
  { label: 'Biology', value: 'biology' },
  { label: 'Chemistry', value: 'chemistry' },
  { label: 'Engineering', value: 'engineering' },
  { label: 'Art', value: 'art' },
  { label: 'Survival', value: 'survival' },
];
