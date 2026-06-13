import { useEffect, useState } from "react";
import { Trip, TripEvent } from "@/lib/types";
import { loadTrips } from "@/lib/storage";

function fmtDate(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtDuration(ms: number): string {
  const s = Math.round(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

function countOf(events: TripEvent[], type: string): number {
  return events.filter((e) => e.type === type).length;
}

export function HistoryPage({ refreshKey }: { refreshKey: number }) {
  const [trips, setTrips] = useState<Trip[]>([]);

  useEffect(() => {
    loadTrips().then(setTrips).catch(() => setTrips([]));
  }, [refreshKey]);

  return (
    <div className="scroll">
      <h2>History</h2>
      {trips.length === 0 && (
        <p className="muted">
          No trips yet. Start a drive — finished sessions show up here.
        </p>
      )}
      {trips.map((t) => (
        <div key={t.id} className="card">
          <div className="row">
            <strong>{fmtDate(t.startedAt)}</strong>
            <span className="muted">{fmtDuration(t.endedAt - t.startedAt)}</span>
          </div>
          <div
            className="row"
            style={{ marginTop: 12, gap: 8, justifyContent: "flex-start" }}
          >
            <Stat label="Yawns" value={countOf(t.events, "yawn")} />
            <Stat label="Head drops" value={countOf(t.events, "headDown")} />
            <Stat
              label="Longest closed"
              value={`${(t.longestClosedMs / 1000).toFixed(1)}s`}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ flex: 1 }}>
      <div className="stat">{value}</div>
      <div className="muted" style={{ fontSize: 12 }}>
        {label}
      </div>
    </div>
  );
}
