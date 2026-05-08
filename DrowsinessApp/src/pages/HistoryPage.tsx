import { useEffect, useState } from 'react';
import { Calendar, Clock, Eye, Siren } from 'lucide-react';
import { listTrips } from '../services/storage';
import type { Trip } from '../types';

export function HistoryPage() {
  const [trips, setTrips] = useState<Trip[]>([]);
  useEffect(() => {
    listTrips().then(setTrips);
  }, []);

  return (
    <div className="absolute inset-0 bg-bg overflow-y-auto">
      <header className="px-5 pt-[max(env(safe-area-inset-top),20px)] pb-4">
        <h1 className="text-2xl font-extrabold tracking-tight">History</h1>
        <p className="text-sm text-muted mt-1">Past trips and detected events</p>
      </header>
      <div className="px-4 pb-6 space-y-2">
        {trips.length === 0 && (
          <div className="text-muted text-center py-16">No trips yet.</div>
        )}
        {trips.map((t) => {
          const focus = t.events.filter((e) => e.type === 'yawn' || e.type === 'head-down').length;
          const drowsy = t.events.filter((e) => e.type === 'drowsy').length;
          const crit = t.events.filter((e) => e.type === 'critical' || e.type === 'emergency').length;
          const dur = t.endedAt - t.startedAt;
          return (
            <div key={t.id} className="rounded-2xl bg-surface border border-white/5 p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <Calendar size={16} className="text-muted" />
                  {new Date(t.startedAt).toLocaleString()}
                </div>
                <div className="flex items-center gap-1 text-xs text-muted tabular">
                  <Clock size={14} /> {formatDur(dur)}
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <Stat label="Focus events" value={focus} color="text-amber" />
                <Stat label="Drowsy" value={drowsy} color="text-amber" />
                <Stat label="Critical" value={crit} color="text-danger" />
              </div>
              <div className="mt-3 flex items-center gap-2 text-xs text-muted">
                <Eye size={14} />
                Longest closed-eye streak:{' '}
                <span className="text-text font-semibold tabular">
                  {(t.longestClosedMs / 1000).toFixed(1)}s
                </span>
                {crit > 0 && <Siren size={14} className="ml-2 text-danger" />}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded-xl bg-surface2 py-2.5">
      <div className={`text-xl font-extrabold tabular ${color}`}>{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-muted mt-0.5">{label}</div>
    </div>
  );
}

function formatDur(ms: number) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return `${m}m ${(s % 60)}s`;
}
