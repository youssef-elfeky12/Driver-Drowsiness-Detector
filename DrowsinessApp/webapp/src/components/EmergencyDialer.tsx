// GTA-style mini phone — ports lib/widgets/emergency_dialer.dart.
const KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"];

export function EmergencyDialer({
  digitsTyped,
  pressedDigit,
  callingActive,
  callConnected,
  onCancel,
}: {
  digitsTyped: string;
  pressedDigit: string | null;
  callingActive: boolean;
  callConnected: boolean;
  onCancel: () => void;
}) {
  const state = callConnected ? "Connected" : callingActive ? "Calling…" : "Dialing";
  return (
    <div className="dialer">
      <div className="dialer-header">
        <span className="dot" />
        <span>PHONE</span>
        <span style={{ marginLeft: "auto" }}>EMERGENCY</span>
      </div>
      <div className="dialer-screen">
        <div className="label">{state.toUpperCase()}</div>
        <div className="digits">{digitsTyped || " "}</div>
      </div>
      <div className="keypad">
        {KEYS.map((k) => (
          <div key={k} className={`key${pressedDigit === k ? " lit" : ""}`}>
            {k}
          </div>
        ))}
      </div>
      <div className="dialer-footer">
        <button style={{ color: "var(--ok)" }} disabled>
          📞 {callConnected ? "On call" : "Calling"}
        </button>
        <button style={{ color: "var(--danger)" }} onClick={onCancel}>
          ☎ End
        </button>
      </div>
    </div>
  );
}
