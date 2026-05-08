import type { Settings } from '../types';

const KEY = 'drowsy-settings-v1';

export const DEFAULT_SETTINGS: Settings = {
  confidenceThreshold: 0.6,
  emergencyNumber: '112',
  alarmVolume: 1.0,
  keepScreenOn: true,
};

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: Settings) {
  localStorage.setItem(KEY, JSON.stringify(s));
}
