import { useEffect, useRef, useState } from 'react';
import { Pause, Play, Square } from 'lucide-react';
import { CameraView, type CameraHandle } from '../components/CameraView';
import { StatusBar } from '../components/StatusBar';
import { PullOverOverlay } from '../components/PullOverOverlay';
import { WarningOverlay } from '../components/WarningOverlay';
import { CriticalOverlay } from '../components/CriticalOverlay';
import { EmergencyDialer } from '../components/EmergencyDialer';
import { detectFrame, initDetector, isReady } from '../services/detector';
import { AlertEngine } from '../services/alertEngine';
import { loadSettings } from '../services/settings';
import { saveTrip } from '../services/storage';
import type { AlertLevel, TripEvent } from '../types';

export function DrivePage() {
  const camRef = useRef<CameraHandle>(null);
  const engineRef = useRef<AlertEngine | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameRef = useRef(0);
  const eventsRef = useRef<TripEvent[]>([]);
  const longestClosedRef = useRef(0);

  const [loadingMsg, setLoadingMsg] = useState('Loading…');
  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [tripStartedAt, setTripStartedAt] = useState(0);
  const [now, setNow] = useState(Date.now());

  const [level, setLevel] = useState<AlertLevel>('none');
  const [closedMs, setClosedMs] = useState(0);
  const [countdown, setCountdown] = useState(5);
  const [digitsTyped, setDigitsTyped] = useState('');
  const [callingActive, setCallingActive] = useState(false);
  const [callConnected, setCallConnected] = useState(false);

  // Init detector once
  useEffect(() => {
    initDetector(setLoadingMsg)
      .then(() => {
        setReady(true);
        setLoadingMsg('Ready');
      })
      .catch((e) => {
        console.error(e);
        setLoadingMsg('Failed to load: ' + (e?.message || e));
      });
  }, []);

  useEffect(() => {
    const i = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(i);
  }, []);

  const start = async () => {
    if (!ready) return;
    const settings = loadSettings();
    const engine = new AlertEngine({
      confidenceThreshold: settings.confidenceThreshold,
      onLevelChange: (l) => setLevel(l),
      onEvent: (ev) => {
        eventsRef.current.push(ev);
      },
      onClosedMs: (ms) => {
        setClosedMs(ms);
        if (ms > longestClosedRef.current) longestClosedRef.current = ms;
      },
      onDialerDigit: (d) => {
        setDigitsTyped((prev) => prev + d);
      },
      onCallingProgress: () => {
        setCallingActive(true);
        // mark connected when ringback finishes (3 plays). We approximate by listening to a timeout-free path:
        // since AudioEngine resets siren volume on calling end, we set callConnected after ~estimated 3× duration.
      },
      onCountdown: (s) => setCountdown(s),
    });
    engine.setMasterVolume(settings.alarmVolume);
    engineRef.current = engine;
    await engine.start();
    eventsRef.current = [];
    longestClosedRef.current = 0;
    setDigitsTyped('');
    setCallingActive(false);
    setCallConnected(false);
    setLevel('none');
    setRunning(true);
    setPaused(false);
    setTripStartedAt(performance.now());
    loop();

    if (settings.keepScreenOn && 'wakeLock' in navigator) {
      try {
        await (navigator as any).wakeLock.request('screen');
      } catch {}
    }
  };

  const loop = () => {
    rafRef.current = requestAnimationFrame(loop);
    const v = camRef.current?.video;
    if (!v || v.readyState < 2 || !engineRef.current) return;
    const t = performance.now();
    if (t - lastFrameRef.current < 100) return; // ~10 fps
    lastFrameRef.current = t;
    detectFrame(v, loadSettings().confidenceThreshold).then((res) => {
      camRef.current?.drawOverlay(res);
      if (!paused) engineRef.current?.ingest(res);
    });
  };

  const stop = async () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    engineRef.current?.stop();
    setRunning(false);
    setPaused(false);
    if (eventsRef.current.length > 0 || longestClosedRef.current > 0) {
      await saveTrip({
        id: crypto.randomUUID(),
        startedAt: Date.now() - (performance.now() - tripStartedAt),
        endedAt: Date.now(),
        events: eventsRef.current,
        longestClosedMs: longestClosedRef.current,
      });
    }
    setLevel('none');
    setClosedMs(0);
  };

  const dismiss = () => engineRef.current?.dismissAlert();

  // After call countdown reached 0 and ringback ends, we mark connected.
  // We poll periodically for a clean indicator while in emergency state.
  useEffect(() => {
    if (level !== 'emergency') {
      setCallingActive(false);
      setCallConnected(false);
      return;
    }
    const t1 = setTimeout(() => setCallingActive(true), 1500);
    const t2 = setTimeout(() => setCallConnected(true), 25_000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [level]);

  const tripDur = running ? performance.now() - tripStartedAt : 0;

  return (
    <div className="absolute inset-0 bg-bg overflow-hidden">
      <CameraView ref={camRef} />
      {running && <StatusBar level={level} closedMs={closedMs} durationMs={tripDur} />}

      {!running && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-end p-6 pb-12 bg-gradient-to-t from-bg via-bg/80 to-transparent">
          <div className="mb-6 text-center">
            <div className="text-2xl font-extrabold tracking-tight">Drowsiness Detector</div>
            <div className="text-sm text-muted mt-1">{loadingMsg}</div>
          </div>
          <button
            disabled={!ready}
            onClick={start}
            className="w-full max-w-xs py-4 rounded-2xl bg-primary text-white text-lg font-bold disabled:bg-surface2 disabled:text-muted shadow-lg active:scale-[0.98] transition"
          >
            <span className="inline-flex items-center gap-2">
              <Play size={20} /> Start Drive
            </span>
          </button>
        </div>
      )}

      {running && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 flex gap-2">
          <button
            onClick={() => setPaused((p) => !p)}
            className="px-5 py-3 rounded-xl bg-surface/95 backdrop-blur border border-white/10 text-text font-semibold inline-flex items-center gap-2"
          >
            {paused ? <><Play size={18}/> Resume</> : <><Pause size={18}/> Pause</>}
          </button>
          <button
            onClick={stop}
            className="px-5 py-3 rounded-xl bg-danger/90 text-white font-semibold inline-flex items-center gap-2"
          >
            <Square size={18}/> End
          </button>
        </div>
      )}

      {/* Overlays */}
      {level === 'drowsy' && <PullOverOverlay onDismiss={dismiss} />}
      {level === 'warning' && <WarningOverlay closedMs={closedMs} />}
      {level === 'critical' && <CriticalOverlay countdown={countdown} onCancel={dismiss} />}
      {level === 'emergency' && (
        <>
          {/* keep red flashing background under dialer */}
          <div className="absolute inset-0 z-30 animate-pulse-red pointer-events-none" />
          <EmergencyDialer
            number="112"
            digitsTyped={digitsTyped}
            callingActive={callingActive}
            callConnected={callConnected}
            onCancel={dismiss}
          />
        </>
      )}

      {/* tick to repaint timers */}
      <span className="hidden">{now}</span>
    </div>
  );
}
