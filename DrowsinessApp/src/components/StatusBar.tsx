import type { AlertLevel } from '../types';

const LABELS: Record<AlertLevel, { label: string; color: string }> = {
  none: { label: 'Alert', color: 'bg-ok' },
  'eyes-closing': { label: 'Eyes closing', color: 'bg-amber' },
  drowsy: { label: 'Drowsy', color: 'bg-amber' },
  warning: { label: 'Warning', color: 'bg-amber' },
  critical: { label: 'Critical', color: 'bg-danger' },
  emergency: { label: 'Emergency', color: 'bg-danger' },
};

export function StatusBar({
  level,
  closedMs,
  durationMs,
}: {
  level: AlertLevel;
  closedMs: number;
  durationMs: number;
}) {
  const meta = LABELS[level];
  const dur = formatDuration(durationMs);
  return (
    <div className="absolute top-0 left-0 right-0 z-10 px-4 pt-[max(env(safe-area-inset-top),12px)] pb-3 bg-gradient-to-b from-black/60 to-transparent">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${meta.color} ${level === 'critical' || level === 'emergency' ? 'animate-pulse' : ''}`} />
          <span className="text-sm font-semibold tracking-wide">{meta.label}</span>
        </div>
        <div className="text-xs text-muted tabular">{dur}</div>
      </div>
      {closedMs > 800 && (
        <div className="mt-2">
          <div className="text-[10px] uppercase tracking-widest text-muted">Eyes closed</div>
          <div className="h-1 mt-1 bg-white/10 rounded overflow-hidden">
            <div
              className="h-full bg-amber"
              style={{ width: `${Math.min(100, (closedMs / 15000) * 100)}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function formatDuration(ms: number) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, '0')}`;
}
