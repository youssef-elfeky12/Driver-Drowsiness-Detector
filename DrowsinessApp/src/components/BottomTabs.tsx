import { NavLink } from 'react-router-dom';
import { Car, History, Settings as SettingsIcon } from 'lucide-react';

const tabs = [
  { to: '/drive', label: 'Drive', icon: Car },
  { to: '/history', label: 'History', icon: History },
  { to: '/settings', label: 'Settings', icon: SettingsIcon },
];

export function BottomTabs() {
  return (
    <nav className="border-t border-white/5 bg-surface/95 backdrop-blur grid grid-cols-3 pb-[env(safe-area-inset-bottom)]">
      {tabs.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            `flex flex-col items-center justify-center gap-1 py-2.5 text-xs font-medium transition-colors ${
              isActive ? 'text-primary' : 'text-muted hover:text-text'
            }`
          }
        >
          <Icon size={22} strokeWidth={2} />
          <span>{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
