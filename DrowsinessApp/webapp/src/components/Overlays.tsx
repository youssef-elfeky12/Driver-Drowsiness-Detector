// Full-screen alert overlays — ports lib/widgets/overlays.dart.
import { AlertLevel } from "@/lib/types";

export function PullOverOverlay({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="overlay amber">
      <div style={{ fontSize: 88 }}>⚠️</div>
      <div className="big">PULL OVER</div>
      <p style={{ maxWidth: 300, fontWeight: 500, opacity: 0.85 }}>
        Multiple drowsiness signals detected. Find a safe place to stop and rest.
      </p>
      <button
        className="btn-ghost"
        style={{ background: "var(--bg)", marginTop: 24 }}
        onClick={onDismiss}
      >
        I&apos;m OK
      </button>
    </div>
  );
}

export function WarningOverlay({ closedMs }: { closedMs: number }) {
  return (
    <div className="overlay amber">
      <div style={{ fontSize: 88 }}>👁️</div>
      <div className="big">EYES CLOSED</div>
      <div style={{ fontSize: 24, fontWeight: 700 }}>WAKE UP</div>
      <div style={{ marginTop: 18, fontSize: 18, fontVariantNumeric: "tabular-nums" }}>
        {(closedMs / 1000).toFixed(1)}s
      </div>
    </div>
  );
}

export function CriticalOverlay({
  countdown,
  number,
  onCancel,
}: {
  countdown: number;
  number: string;
  onCancel: () => void;
}) {
  return (
    <div className="overlay critical">
      <div style={{ fontSize: 80 }}>🚨</div>
      <div style={{ fontSize: 32, fontWeight: 800 }}>EMERGENCY</div>
      <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: 1.5 }}>
        CALLING {number} IN
      </div>
      <div className="countdown">{countdown}</div>
      <button
        className="btn-ghost"
        style={{ background: "rgba(11,15,20,0.85)", marginTop: 24 }}
        onClick={onCancel}
      >
        Cancel
      </button>
    </div>
  );
}

export function AlertOverlay({
  level,
  closedMs,
  countdown,
  number,
  onDismiss,
  onCancel,
}: {
  level: AlertLevel;
  closedMs: number;
  countdown: number;
  number: string;
  onDismiss: () => void;
  onCancel: () => void;
}) {
  // emergency uses the dialer + a red flash background but no blocking card
  if (level === AlertLevel.drowsy) return <PullOverOverlay onDismiss={onDismiss} />;
  if (level === AlertLevel.warning) return <WarningOverlay closedMs={closedMs} />;
  if (level === AlertLevel.critical || level === AlertLevel.emergency)
    return (
      <CriticalOverlay countdown={countdown} number={number} onCancel={onCancel} />
    );
  return null;
}
