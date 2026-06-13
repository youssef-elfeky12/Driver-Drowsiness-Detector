"use client";

import { useEffect, useState } from "react";
import { BottomTabs, Tab } from "@/components/BottomTabs";
import { DrivePage } from "@/components/DrivePage";
import { HistoryPage } from "@/components/HistoryPage";
import { SettingsPage } from "@/components/SettingsPage";
import { AppSettings, defaultSettings } from "@/lib/types";
import { loadSettings, saveSettings } from "@/lib/storage";

export default function Home() {
  const [tab, setTab] = useState<Tab>("drive");
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [historyKey, setHistoryKey] = useState(0);

  // Load persisted settings after mount (localStorage is client-only).
  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  const updateSettings = (s: AppSettings) => {
    setSettings(s);
    saveSettings(s);
  };

  return (
    <main className="app-shell">
      {/* Drive stays mounted across tab switches (IndexedStack equivalent) so
          the camera + models don't re-initialize. */}
      <div className={`page${tab === "drive" ? "" : " hidden"}`}>
        <DrivePage
          settings={settings}
          onTripSaved={() => setHistoryKey((k) => k + 1)}
        />
      </div>
      <div className={`page${tab === "history" ? "" : " hidden"}`}>
        <HistoryPage refreshKey={historyKey} />
      </div>
      <div className={`page${tab === "settings" ? "" : " hidden"}`}>
        <SettingsPage
          settings={settings}
          onChange={updateSettings}
          onHistoryCleared={() => setHistoryKey((k) => k + 1)}
        />
      </div>

      <BottomTabs active={tab} onChange={setTab} />
    </main>
  );
}
