// Browser port of lib/services/alert_engine.dart.
// Two independent state machines (focus reminders + critical eyes-closed)
// driven off the per-frame DetectionResult stream. Behavior matches DESIGN.md §4.

import { AudioEngine } from "./audioEngine";
import {
  AlertLevel,
  DetectionResult,
  FacePrediction,
  TripEvent,
} from "./types";

const SUSTAIN_MS = 500;
const COOLDOWN_MS = 3000;
const DROWSY_WINDOW_MS = 30000;
const DROWSY_THRESHOLD = 3;
const CALIBRATION_MS = 3000;
const FACE_LOST_TOLERANCE_MS = 3000;

const BLINK_IGNORE_MS = 800;
const WARNING_AT_MS = 5000;
const CRITICAL_AT_MS = 10000;
const EMERGENCY_AT_MS = 15000;
const EYE_OPEN_RESET_MS = 1000;

export interface AlertCallbacks {
  onLevel: (level: AlertLevel) => void;
  onEvent: (event: TripEvent) => void;
  onClosedMs: (ms: number) => void;
  onDialerDigit: (digit: string, index: number) => void;
  onCountdown: (value: number) => void;
  onCallingStarted: () => void;
}

function vibrate(ms: number) {
  if (typeof navigator !== "undefined" && navigator.vibrate) {
    try {
      navigator.vibrate(ms);
    } catch {
      /* not supported */
    }
  }
}

export class AlertEngine {
  confidenceThreshold: number;
  emergencyNumber: string;
  private audio: AudioEngine;
  private cb: AlertCallbacks;

  private _level = AlertLevel.none;
  get level() {
    return this._level;
  }

  private graceUntilMs = 0;

  // Track A
  private yawnSustainStart = 0;
  private headDownSustainStart = 0;
  private lastYawnEventAt = 0;
  private lastHeadDownEventAt = 0;
  private events: TripEvent[] = [];

  // Track B
  private closedSince = 0;
  private openSince = 0;
  private faceLostSince = 0;
  private inEmergencyFlow = false;

  private countdownTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    audio: AudioEngine,
    confidenceThreshold: number,
    emergencyNumber: string,
    cb: AlertCallbacks,
  ) {
    this.audio = audio;
    this.confidenceThreshold = confidenceThreshold;
    this.emergencyNumber = emergencyNumber;
    this.cb = cb;
  }

  start() {
    const now = Date.now();
    this.graceUntilMs = now + CALIBRATION_MS;
    this.events = [];
    this._level = AlertLevel.none;
    this.closedSince = 0;
    this.openSince = 0;
    this.inEmergencyFlow = false;
  }

  stop() {
    this.clearCountdown();
    this.audio.stopAll();
    this.setLevel(AlertLevel.none);
    this.inEmergencyFlow = false;
  }

  dismiss() {
    this.clearCountdown();
    this.audio.stopAll();
    this.events = [];
    this.closedSince = 0;
    this.openSince = 0;
    this.inEmergencyFlow = false;
    this.graceUntilMs = Date.now() + 10000;
    this.setLevel(AlertLevel.none);
  }

  async ingest(result: DetectionResult) {
    const now = result.tsMs;
    const inGrace = now < this.graceUntilMs;

    if (result.faceLost) {
      if (this.faceLostSince === 0) this.faceLostSince = now;
      if (now - this.faceLostSince > FACE_LOST_TOLERANCE_MS) this.cb.onClosedMs(0);
      return;
    }
    this.faceLostSince = 0;

    if (inGrace) {
      this.cb.onClosedMs(0);
      return;
    }

    const face = result.faces[0];

    // ---- Track A — yawn / head-down are INDEPENDENT signals ----
    const yawnPasses = face.isYawn && face.yawnConf >= this.confidenceThreshold;
    const headDownPasses =
      face.isHeadDown && face.headPoseConf >= this.confidenceThreshold;

    if (yawnPasses) {
      if (this.yawnSustainStart === 0) this.yawnSustainStart = now;
      if (
        now - this.yawnSustainStart >= SUSTAIN_MS &&
        now - this.lastYawnEventAt >= COOLDOWN_MS
      ) {
        this.lastYawnEventAt = now;
        await this.registerEvent({ ts: now, type: "yawn" });
      }
    } else {
      this.yawnSustainStart = 0;
    }

    if (headDownPasses) {
      if (this.headDownSustainStart === 0) this.headDownSustainStart = now;
      if (
        now - this.headDownSustainStart >= SUSTAIN_MS &&
        now - this.lastHeadDownEventAt >= COOLDOWN_MS
      ) {
        this.lastHeadDownEventAt = now;
        await this.registerEvent({ ts: now, type: "headDown" });
      }
    } else {
      this.headDownSustainStart = 0;
    }

    this.events = this.events.filter(
      (e) =>
        now - e.ts <= DROWSY_WINDOW_MS &&
        (e.type === "yawn" || e.type === "headDown"),
    );

    // ---- Track B ----
    const closed = this.classifyEyesClosed(face);
    if (closed) {
      if (this.closedSince === 0) this.closedSince = now;
      this.openSince = 0;
    } else {
      if (this.openSince === 0) this.openSince = now;
      if (now - this.openSince >= EYE_OPEN_RESET_MS) {
        if (this.closedSince !== 0) {
          const locked =
            this._level === AlertLevel.critical ||
            this._level === AlertLevel.emergency;
          if (!locked) {
            this.clearCountdown();
            this.audio.stopAll();
            this.inEmergencyFlow = false;
            this.closedSince = 0;
          }
        } else {
          this.closedSince = 0;
        }
      }
    }

    const closedMs = this.closedSince === 0 ? 0 : now - this.closedSince;
    this.cb.onClosedMs(closedMs);
    await this.updateLevel(closedMs);
  }

  private classifyEyesClosed(face: FacePrediction): boolean {
    const eyes = face.eyes.filter((e) => e.conf >= this.confidenceThreshold);
    if (eyes.length === 0) return true; // face but no confident eyes → likely closed
    return eyes.every((e) => e.eyeClass === "Closed");
  }

  private async registerEvent(ev: TripEvent) {
    this.events.push(ev);
    this.cb.onEvent(ev);
    await this.audio.playBuzz();
    vibrate(300);

    if (
      this.events.length >= DROWSY_THRESHOLD &&
      this._level !== AlertLevel.critical &&
      this._level !== AlertLevel.emergency
    ) {
      this.cb.onEvent({ ts: ev.ts, type: "drowsy" });
      this.setLevel(AlertLevel.drowsy);
      await this.audio.startPullover();
    }
  }

  private async updateLevel(closedMs: number) {
    if (closedMs >= EMERGENCY_AT_MS) {
      if (this._level !== AlertLevel.emergency) {
        this.setLevel(AlertLevel.emergency);
        this.cb.onEvent({ ts: Date.now(), type: "emergency" });
        await this.startEmergencyFlow();
      }
      return;
    }
    if (closedMs >= CRITICAL_AT_MS) {
      if (this._level !== AlertLevel.critical) {
        this.setLevel(AlertLevel.critical);
        this.cb.onEvent({ ts: Date.now(), type: "critical" });
        await this.audio.startSiren();
        this.startCountdown();
      }
      return;
    }
    if (closedMs >= WARNING_AT_MS) {
      if (this._level !== AlertLevel.warning) {
        this.setLevel(AlertLevel.warning);
        await this.audio.startPullover();
      }
      return;
    }
    if (closedMs >= BLINK_IGNORE_MS) {
      if (this._level === AlertLevel.none) this.setLevel(AlertLevel.eyesClosing);
      return;
    }
    if (
      closedMs === 0 &&
      (this._level === AlertLevel.warning ||
        this._level === AlertLevel.critical ||
        this._level === AlertLevel.emergency ||
        this._level === AlertLevel.eyesClosing)
    ) {
      this.clearCountdown();
      this.audio.stopAll();
      this.inEmergencyFlow = false;
      if (this.events.length >= DROWSY_THRESHOLD) {
        this.setLevel(AlertLevel.drowsy);
        await this.audio.startPullover();
      } else {
        this.setLevel(AlertLevel.none);
      }
    }
  }

  private setLevel(l: AlertLevel) {
    if (this._level === l) return;
    this._level = l;
    this.cb.onLevel(l);
  }

  private clearCountdown() {
    if (this.countdownTimer) {
      clearInterval(this.countdownTimer);
      this.countdownTimer = null;
    }
  }

  private startCountdown() {
    let n = 5;
    this.cb.onCountdown(n);
    this.clearCountdown();
    this.countdownTimer = setInterval(() => {
      n -= 1;
      this.cb.onCountdown(n < 0 ? 0 : n);
      if (n <= 0) this.clearCountdown();
    }, 1000);
  }

  private async startEmergencyFlow() {
    if (this.inEmergencyFlow) return;
    this.inEmergencyFlow = true;
    this.audio.stopPullover();
    this.audio.duckSiren();

    const offsets = await this.audio.playDialer();
    const number = this.emergencyNumber;
    for (let i = 0; i < offsets.length && i < number.length; i++) {
      setTimeout(() => this.cb.onDialerDigit(number[i], i), offsets[i]);
    }

    this.audio.onDialerEnd(async () => {
      this.cb.onCallingStarted();
      await this.audio.playCallingTimes(3);
    });
  }
}
