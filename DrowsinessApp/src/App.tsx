import { Navigate, Route, Routes } from 'react-router-dom';
import { BottomTabs } from './components/BottomTabs';
import { DrivePage } from './pages/DrivePage';
import { HistoryPage } from './pages/HistoryPage';
import { SettingsPage } from './pages/SettingsPage';

export default function App() {
  return (
    <div className="phone-shell flex flex-col">
      <div className="flex-1 min-h-0 relative">
        <Routes>
          <Route path="/" element={<Navigate to="/drive" replace />} />
          <Route path="/drive" element={<DrivePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </div>
      <BottomTabs />
    </div>
  );
}
