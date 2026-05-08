import { useEffect, useState } from 'react';
import { Trash2 } from 'lucide-react';
import { DEFAULT_SETTINGS, loadSettings, saveSettings } from '../services/settings';
import { clearTrips } from '../services/storage';
import type { Settings } from '../types';

const NUMBERS = ['112', '911', '999', '110'];

export function SettingsPage() {
  const [s, setS] = useState<Settings>(DEFAULT_SETTINGS);

  useEffect(() => {
    setS(loadSettings());
  }, []);

  const update = (patch: Partial<Settings>) => {
    const next = { ...s, ...patch };
    setS(next);
    saveSettings(next);
  };

  return (
    <div className="absolute inset-0 bg-bg overflow-y-auto">
      <header className="px-5 pt-[max(env(safe-area-inset-top),20px)] pb-4">
        <h1 className="text-2xl font-extrabold tracking-tight">Settings</h1>
        <p className="text-sm text-muted mt-1">Tune detection and alerts</p>
      </header>

      <div className="px-4 space-y-3 pb-6">
        <Section title="Detection">
          <Field label={`Confidence threshold — ${s.confidenceThreshold.toFixed(2)}`}>
            <input
              type="range"
              min={0.4}
              max={0.9}
              step={0.05}
              value={s.confidenceThreshold}
              onChange={(e) => update({ confidenceThreshold: parseFloat(e.target.value) })}
              className="w-full accent-primary"
            />
            <p className="text-xs text-muted mt-1">
              Predictions below this confidence are ignored. Higher = fewer false alarms.
            </p>
          </Field>
        </Section>

        <Section title="Emergency">
          <Field label="Emergency number">
            <div className="grid grid-cols-4 gap-2">
              {NUMBERS.map((n) => (
                <button
                  key={n}
                  onClick={() => update({ emergencyNumber: n })}
                  className={`py-2.5 rounded-xl font-bold tabular ${
                    s.emergencyNumber === n
                      ? 'bg-primary text-white'
                      : 'bg-surface2 text-muted'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
            <p className="text-xs text-muted mt-2">Visual demo only — no real call is placed.</p>
          </Field>
        </Section>

        <Section title="Audio">
          <Field label={`Alarm volume — ${Math.round(s.alarmVolume * 100)}%`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={s.alarmVolume}
              onChange={(e) => update({ alarmVolume: parseFloat(e.target.value) })}
              className="w-full accent-primary"
            />
          </Field>
        </Section>

        <Section title="Display">
          <ToggleRow
            label="Keep screen on while driving"
            value={s.keepScreenOn}
            onChange={(v) => update({ keepScreenOn: v })}
          />
        </Section>

        <Section title="Data">
          <button
            onClick={async () => {
              if (confirm('Delete all trip history?')) await clearTrips();
            }}
            className="w-full py-3 rounded-xl bg-danger/15 text-danger font-semibold inline-flex items-center justify-center gap-2"
          >
            <Trash2 size={16} /> Clear trip history
          </button>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl bg-surface border border-white/5 p-4">
      <div className="text-[11px] uppercase tracking-widest text-muted mb-3">{title}</div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-sm font-semibold mb-2">{label}</div>
      {children}
    </div>
  );
}

function ToggleRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="text-sm font-medium">{label}</div>
      <button
        onClick={() => onChange(!value)}
        className={`w-12 h-7 rounded-full p-0.5 transition ${value ? 'bg-primary' : 'bg-surface2'}`}
        aria-pressed={value}
      >
        <div
          className={`w-6 h-6 rounded-full bg-white transition ${value ? 'translate-x-5' : ''}`}
        />
      </button>
    </div>
  );
}
