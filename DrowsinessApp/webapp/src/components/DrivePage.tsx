"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Detector } from "@/lib/detector";
import { AudioEngine } from "@/lib/audioEngine";
import { AlertEngine } from "@/lib/alertEngine";
import { AlertLevel, AppSettings, DetectionResult, Trip, TripEvent } from "@/lib/types";
import { saveTrip } from "@/lib/storage";
import { getFix, toSpeech } from "@/lib/location";
import { warmUpVoices } from "@/lib/tts";
import { AlertOverlay } from "./Overlays";
import { EmergencyDialer } from "./EmergencyDialer";

const COLORS = {
  ok: "#22c55e",
  amber: "#f59e0b",
  danger: "#ef4444",
  primary: "#3b82f6",
  bg: "#0b0f14",
};

const LEVEL_LABEL: Record<AlertLevel, string> = {
  [AlertLevel.none]: "Monitoring",
  [AlertLevel.eyesClosing]: "Eyes closing",
  [AlertLevel.drowsy]: "Drowsy",
  [AlertLevel.warning]: "Wake up",
  [AlertLevel.critical]: "Critical",
  [AlertLevel.emergency]: "Emergency",
};

export function DrivePage({ settings, onTripSaved }: {
  settings: AppSettings;
  onTripSaved: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const detectorRef = useRef<Detector | null>(null);
  const audioRef = useRef<AudioEngine | null>(null);
  const engineRef = useRef<AlertEngine | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);
  const rafRef = useRef<number>(0);
  const busyRef = useRef(false);
  const runningRef = useRef(false);

  // Per-trip accumulators.
  const tripStartRef = useRef(0);
  const tripEventsRef = useRef<TripEvent[]>([]);
  const longestClosedRef = useRef(0);

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [level, setLevel] = useState<AlertLevel>(AlertLevel.none);
  const [closedMs, setClosedMs] = useState(0);
  const [countdown, setCountdown] = useState(5);

  // Dialer UI state.
  const [showDialer, setShowDialer] = useState(false);
  const [digitsTyped, setDigitsTyped] = useState("");
  const [pressedDigit, setPressedDigit] = useState<string | null>(null);
  const [callingActive, setCallingActive] = useState(false);

  // Keep engine config in sync with settings.
  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.confidenceThreshold = settings.confidenceThreshold;
      engineRef.current.emergencyNumber = settings.emergencyNumber;
    }
    audioRef.current?.setMasterVolume(settings.alarmVolume);
  }, [settings]);

  const drawOverlay = useCallback((result: DetectionResult) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (canvas.width !== result.frameWidth || canvas.height !== result.frameHeight) {
      canvas.width = result.frameWidth;
      canvas.height = result.frameHeight;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const W = result.frameWidth;

    for (const f of result.faces) {
      // Mirror x to match the selfie-mirrored <video>.
      const fx = W - f.box.x - f.box.w;
      ctx.lineWidth = 3;
      ctx.strokeStyle = COLORS.ok;
      ctx.strokeRect(fx, f.box.y, f.box.w, f.box.h);

      const yawnLabel = `${f.isYawn ? "yawn" : "no_yawn"} ${Math.round(f.yawnConf * 100)}%`;
      const headLabel = `${f.isHeadDown ? "down" : "front"} ${Math.round(f.headPoseConf * 100)}%`;
      tag(ctx, fx, f.box.y - 40, yawnLabel, f.isYawn ? COLORS.amber : COLORS.ok);
      tag(ctx, fx, f.box.y - 20, headLabel, f.isHeadDown ? COLORS.amber : COLORS.ok);

      for (const e of f.eyes) {
        const ex = W - e.box.x - e.box.w;
        const c = e.eyeClass === "Closed" ? COLORS.danger : COLORS.primary;
        ctx.lineWidth = 2;
        ctx.strokeStyle = c;
        ctx.strokeRect(ex, e.box.y, e.box.w, e.box.h);
        tag(ctx, ex, e.box.y - 16, `${e.eyeClass} ${Math.round(e.conf * 100)}%`, c, 11);
      }
    }
  }, []);

  const loop = useCallback(async () => {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const detector = detectorRef.current;
    if (video && detector?.isReady && video.readyState >= 2 && !busyRef.current) {
      busyRef.current = true;
      try {
        const result = await detector.detect(
          video,
          video.videoWidth,
          video.videoHeight,
        );
        drawOverlay(result);
        await engineRef.current?.ingest(result);
      } catch {
        /* skip frame on error */
      } finally {
        busyRef.current = false;
      }
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [drawOverlay]);

  const handleStart = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      warmUpVoices();
      // 1. Camera (front).
      setProgress("Requesting camera…");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      // 2. Audio + models.
      if (!audioRef.current) {
        audioRef.current = new AudioEngine();
        audioRef.current.init();
      }
      audioRef.current.setMasterVolume(settings.alarmVolume);
      await audioRef.current.unlock();

      if (!detectorRef.current) {
        detectorRef.current = new Detector();
        await detectorRef.current.init(setProgress);
      }

      // 3. Alert engine.
      engineRef.current = new AlertEngine(
        audioRef.current,
        settings.confidenceThreshold,
        settings.emergencyNumber,
        {
          onLevel: (l) => setLevel(l),
          onClosedMs: (ms) => {
            setClosedMs(ms);
            if (ms > longestClosedRef.current) longestClosedRef.current = ms;
          },
          onEvent: (ev) => tripEventsRef.current.push(ev),
          onDialerDigit: (digit) => {
            setShowDialer(true);
            setDigitsTyped((prev) => prev + digit);
            setPressedDigit(digit);
            setTimeout(() => setPressedDigit(null), 250);
          },
          onCountdown: (n) => setCountdown(n),
          onCallingStarted: () => setCallingActive(true),
        },
      );
      engineRef.current.start();

      // 4. Pre-resolve location for the dispatcher tail (best-effort).
      getFix().then((fix) => {
        audioRef.current?.setEmergencyLocationText(fix ? toSpeech(fix) : null);
      });

      // 5. Wake lock.
      if (settings.keepScreenOn && "wakeLock" in navigator) {
        try {
          wakeLockRef.current = await navigator.wakeLock.request("screen");
        } catch {
          /* not granted */
        }
      }

      // 6. Go.
      tripStartRef.current = Date.now();
      tripEventsRef.current = [];
      longestClosedRef.current = 0;
      setDigitsTyped("");
      setShowDialer(false);
      setCallingActive(false);
      setLevel(AlertLevel.none);
      runningRef.current = true;
      setRunning(true);
      setLoading(false);
      rafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      setLoading(false);
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Camera permission denied. Allow camera access and try again."
          : "Could not start the camera. " + (e as Error).message,
      );
    }
  }, [settings, loop]);

  const handleStop = useCallback(async () => {
    runningRef.current = false;
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    engineRef.current?.stop();
    setShowDialer(false);
    setCallingActive(false);
    setLevel(AlertLevel.none);
    setClosedMs(0);

    // Persist the trip.
    if (tripStartRef.current > 0) {
      const trip: Trip = {
        id: `${tripStartRef.current}-${Math.round(longestClosedRef.current)}`,
        startedAt: tripStartRef.current,
        endedAt: Date.now(),
        events: tripEventsRef.current,
        longestClosedMs: longestClosedRef.current,
      };
      // Only keep sessions that lasted more than a couple seconds.
      if (trip.endedAt - trip.startedAt > 2000) {
        try {
          await saveTrip(trip);
          onTripSaved();
        } catch {
          /* storage unavailable */
        }
      }
      tripStartRef.current = 0;
    }

    // Release camera + wake lock.
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    wakeLockRef.current?.release().catch(() => {});
    wakeLockRef.current = null;
  }, [onTripSaved]);

  const handleDismiss = useCallback(() => engineRef.current?.dismiss(), []);
  const handleCancel = useCallback(() => {
    engineRef.current?.dismiss();
    setShowDialer(false);
    setCallingActive(false);
  }, []);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      runningRef.current = false;
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      audioRef.current?.dispose();
      detectorRef.current?.dispose();
      wakeLockRef.current?.release().catch(() => {});
    };
  }, []);

  const levelColor =
    level === AlertLevel.critical || level === AlertLevel.emergency
      ? COLORS.danger
      : level === AlertLevel.warning || level === AlertLevel.drowsy
        ? COLORS.amber
        : COLORS.ok;

  return (
    <div className="drive-root">
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <video ref={videoRef} className="cam-video mirrored" playsInline muted />
      <canvas ref={canvasRef} className="cam-overlay" />

      {!running && !loading && (
        <div className="landing">
          <div className="logo">🚗💤</div>
          <h1>Drowsiness Detector</h1>
          <p>
            Mount your phone facing you and press Start. The camera watches for
            closed eyes, yawns and head drops — all processed on-device.
          </p>
          <span className="pill">
            <span className="dot" /> Ready
          </span>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <span>{progress || "Starting…"}</span>
        </div>
      )}

      {running && (
        <>
          <div className="top-status">
            <span className="status-chip" style={{ color: levelColor }}>
              <span className="dot" style={{ background: levelColor }} />
              {LEVEL_LABEL[level]}
            </span>
            {closedMs >= 800 && (
              <span className="status-chip" style={{ color: COLORS.amber }}>
                {(closedMs / 1000).toFixed(1)}s closed
              </span>
            )}
          </div>

          {level === AlertLevel.eyesClosing && (
            <div className="eyes-pill">Eyes closing…</div>
          )}

          <AlertOverlay
            level={level}
            closedMs={closedMs}
            countdown={countdown}
            number={settings.emergencyNumber}
            onDismiss={handleDismiss}
            onCancel={handleCancel}
          />

          {showDialer && (
            <EmergencyDialer
              digitsTyped={digitsTyped}
              pressedDigit={pressedDigit}
              callingActive={callingActive}
              callConnected={false}
              onCancel={handleCancel}
            />
          )}
        </>
      )}

      <div className="drive-controls">
        {!running ? (
          <button className="btn-primary" disabled={loading} onClick={handleStart}>
            {loading ? "Starting…" : "Start"}
          </button>
        ) : (
          <button className="btn-ghost" onClick={handleStop}>
            Stop
          </button>
        )}
      </div>

      {error && (
        <div className="loading" style={{ background: "rgba(11,15,20,0.95)" }}>
          <span style={{ color: COLORS.danger, padding: "0 32px", textAlign: "center" }}>
            {error}
          </span>
          <button className="btn-ghost" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}

function tag(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  color: string,
  fontSize = 12,
) {
  ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 8;
  const h = fontSize + 6;
  const ty = Math.max(0, y);
  ctx.fillStyle = color;
  ctx.fillRect(x, ty, w, h);
  ctx.fillStyle = COLORS.bg;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 4, ty + h / 2);
}
