import { AppSettings } from "@/lib/types";
import { clearTrips } from "@/lib/storage";

export function SettingsPage({
  settings,
  onChange,
  onHistoryCleared,
}: {
  settings: AppSettings;
  onChange: (s: AppSettings) => void;
  onHistoryCleared: () => void;
}) {
  const set = (patch: Partial<AppSettings>) => onChange({ ...settings, ...patch });

  return (
    <div className="scroll">
      <h2>Settings</h2>

      <div className="setting-row">
        <div className="head">
          <span>Confidence threshold</span>
          <span className="muted">
            {(settings.confidenceThreshold * 100).toFixed(0)}%
          </span>
        </div>
        <input
          type="range"
          min={0.3}
          max={0.95}
          step={0.05}
          value={settings.confidenceThreshold}
          onChange={(e) =>
            set({ confidenceThreshold: parseFloat(e.target.value) })
          }
        />
        <span className="muted" style={{ fontSize: 12 }}>
          Predictions below this are treated as uncertain.
        </span>
      </div>

      <div className="setting-row">
        <div className="head">
          <span>Emergency number</span>
          <input
            type="tel"
            value={settings.emergencyNumber}
            onChange={(e) =>
              set({ emergencyNumber: e.target.value.replace(/[^0-9*#]/g, "") })
            }
          />
        </div>
        <span className="muted" style={{ fontSize: 12 }}>
          Dialed (simulated) when eyes stay closed ≥ 15s.
        </span>
      </div>

      <div className="setting-row">
        <div className="head">
          <span>Alarm volume</span>
          <span className="muted">
            {(settings.alarmVolume * 100).toFixed(0)}%
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={settings.alarmVolume}
          onChange={(e) => set({ alarmVolume: parseFloat(e.target.value) })}
        />
      </div>

      <div className="setting-row">
        <div className="head">
          <span>Keep screen on while driving</span>
          <button
            className={`switch${settings.keepScreenOn ? " on" : ""}`}
            onClick={() => set({ keepScreenOn: !settings.keepScreenOn })}
          >
            <span className="knob" />
          </button>
        </div>
        <span className="muted" style={{ fontSize: 12 }}>
          Uses the Screen Wake Lock API (supported browsers only).
        </span>
      </div>

      <div className="setting-row" style={{ borderBottom: "none" }}>
        <button
          className="btn-ghost"
          style={{ color: "var(--danger)" }}
          onClick={async () => {
            await clearTrips();
            onHistoryCleared();
          }}
        >
          Clear trip history
        </button>
      </div>

      <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
        All data (history + settings) is stored only on this device. Nothing is
        sent to a server.
      </p>
    </div>
  );
}
