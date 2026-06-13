export type Tab = "drive" | "history" | "settings";

const TABS: { id: Tab; icon: string; label: string }[] = [
  { id: "drive", icon: "🚗", label: "Drive" },
  { id: "history", icon: "📊", label: "History" },
  { id: "settings", icon: "⚙️", label: "Settings" },
];

export function BottomTabs({
  active,
  onChange,
}: {
  active: Tab;
  onChange: (t: Tab) => void;
}) {
  return (
    <nav className="tabbar">
      {TABS.map((t) => (
        <button
          key={t.id}
          className={active === t.id ? "active" : ""}
          onClick={() => onChange(t.id)}
        >
          <span className="ico">{t.icon}</span>
          {t.label}
        </button>
      ))}
    </nav>
  );
}
