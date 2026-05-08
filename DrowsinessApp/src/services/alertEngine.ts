import type { AlertLevel, DetectionResult, TripEvent } from '../types';
import { AudioEngine } from './audio';

interface EngineOptions {
  confidenceThreshold: number;
  onLevelChange: (level: AlertLevel) => void;
  onEvent: (ev: TripEvent) => void;
  onClosedMs: (ms: number) => void;
  onDialerDigit: (digit: string, index: number) => void;
  onCallingProgress: (playsRemaining: number) => void;
  onCountdown: (seconds: number) => void;
}

const SUSTAIN_MS = 1500;
const COOLDOWN_MS = 3000;
const DROWSY_WINDOW_MS = 30_000;
const DROWSY_THRESHOLD = 3;
const CALIBRATION_MS = 3000;
const FACE_LOST_TOLERANCE_MS = 3000;

const BLINK_IGNORE_MS = 800;
const WARNING_AT_MS = 5000;
const CRITICAL_AT_MS = 10_000;
const EMERGENCY_AT_MS = 15_000;
const EYE_OPEN_RESET_MS = 500;

export class AlertEngine {
  private audio = new AudioEngine();
  private opts: EngineOptions;

  private startedAt = 0;
  private level: AlertLevel = 'none';

  // Track A — focus events
  private yawnSustainStart = 0;
  private headDownSustainStart = 0;
  private lastYawnEventAt = 0;
  private lastHeadDownEventAt = 0;
  private events: TripEvent[] = [];
  private grace = 0;

  // Track B — eyes
  private closedSince = 0;
  private openSince = 0;
  private faceLostSince = 0;
  private inEmergencyFlow = false;

  private countdownInterval: number | null = null;

  constructor(opts: EngineOptions) {
    this.opts = opts;
  }

  async start() {
    this.startedAt = performance.now();
    this.events = [];
    this.grace = this.startedAt + CALIBRATION_MS;
    await this.audio.unlock();
  }

  stop() {
    this.audio.stopAll();
    if (this.countdownInterval) clearInterval(this.countdownInterval);
    this.countdownInterval = null;
    this.setLevel('none');
    this.inEmergencyFlow = false;
  }

  setMasterVolume(v: number) {
    this.audio.setMasterVolume(v);
  }

  dismissAlert() {
    // user pressed "I'm OK" / cancel
    this.audio.stopAll();
    if (this.countdownInterval) clearInterval(this.countdownInterval);
    this.countdownInterval = null;
    this.events = [];
    this.closedSince = 0;
    this.openSince = 0;
    this.inEmergencyFlow = false;
    this.grace = performance.now() + 10_000;
    this.setLevel('none');
  }

  ingest(result: DetectionResult) {
    const now = result.ts;
    const inGrace = now < this.grace;

    // Face-lost handling
    if (result.faceLost) {
      if (this.faceLostSince === 0) this.faceLostSince = now;
      if (now - this.faceLostSince > FACE_LOST_TOLERANCE_MS) {
        // freeze counters
        this.opts.onClosedMs(0);
      }
      return;
    }
    this.faceLostSince = 0;

    if (inGrace) {
      this.opts.onClosedMs(0);
      return;
    }

    const face = result.faces[0];

    // ===== Track A: yawn / head-down =====
    const conf = face.faceConf;
    const passes = conf >= this.opts.confidenceThreshold;

    // yawn sustain
    if (passes && face.faceClass === 'yawn') {
      if (this.yawnSustainStart === 0) this.yawnSustainStart = now;
      if (
        now - this.yawnSustainStart >= SUSTAIN_MS &&
        now - this.lastYawnEventAt >= COOLDOWN_MS
      ) {
        this.lastYawnEventAt = now;
        this.registerEvent({ ts: now, type: 'yawn' });
      }
    } else {
      this.yawnSustainStart = 0;
    }

    // head-down sustain
    if (passes && face.faceClass === 'down') {
      if (this.headDownSustainStart === 0) this.headDownSustainStart = now;
      if (
        now - this.headDownSustainStart >= SUSTAIN_MS &&
        now - this.lastHeadDownEventAt >= COOLDOWN_MS
      ) {
        this.lastHeadDownEventAt = now;
        this.registerEvent({ ts: now, type: 'head-down' });
      }
    } else {
      this.headDownSustainStart = 0;
    }

    // prune sliding window
    this.events = this.events.filter(
      (e) => now - e.ts <= DROWSY_WINDOW_MS && (e.type === 'yawn' || e.type === 'head-down'),
    );

    // ===== Track B: eyes closed =====
    const eyesClosed = this.classifyEyesClosed(face);
    if (eyesClosed) {
      if (this.closedSince === 0) this.closedSince = now;
      this.openSince = 0;
    } else {
      if (this.openSince === 0) this.openSince = now;
      if (now - this.openSince >= EYE_OPEN_RESET_MS) {
        if (this.closedSince !== 0) {
          // reset full chain
          this.audio.stopAll();
          if (this.countdownInterval) clearInterval(this.countdownInterval);
          this.countdownInterval = null;
          this.inEmergencyFlow = false;
        }
        this.closedSince = 0;
      }
    }

    const closedMs = this.closedSince === 0 ? 0 : now - this.closedSince;
    this.opts.onClosedMs(closedMs);

    this.updateLevel(closedMs);
  }

  private classifyEyesClosed(face: import('../types').FacePrediction): boolean {
    const conf = this.opts.confidenceThreshold;
    const eyes = face.eyes.filter((e) => e.eyeConf >= conf);
    if (eyes.length === 0) {
      // no confident eyes detected within face → likely closed
      return true;
    }
    return eyes.every((e) => e.eyeClass === 'Closed');
  }

  private registerEvent(ev: TripEvent) {
    this.events.push(ev);
    this.opts.onEvent(ev);
    this.audio.playBuzz();

    const recent = this.events.length;
    if (recent >= DROWSY_THRESHOLD && this.level !== 'critical' && this.level !== 'emergency') {
      this.opts.onEvent({ ts: ev.ts, type: 'drowsy' });
      this.setLevel('drowsy');
      this.audio.startPullover();
    }
  }

  private updateLevel(closedMs: number) {
    // Critical track wins
    if (closedMs >= EMERGENCY_AT_MS) {
      if (this.level !== 'emergency') {
        this.setLevel('emergency');
        this.opts.onEvent({ ts: performance.now(), type: 'emergency' });
        this.startEmergencyFlow();
      }
      return;
    }
    if (closedMs >= CRITICAL_AT_MS) {
      if (this.level !== 'critical') {
        this.setLevel('critical');
        this.opts.onEvent({ ts: performance.now(), type: 'critical' });
        this.audio.startSiren();
        this.startCountdown();
      }
      return;
    }
    if (closedMs >= WARNING_AT_MS) {
      if (this.level !== 'warning') {
        this.setLevel('warning');
        this.audio.startPullover();
      }
      return;
    }
    if (closedMs >= BLINK_IGNORE_MS) {
      if (this.level === 'none' || this.level === 'drowsy') {
        // keep drowsy if it's already there; otherwise show eyes-closing
        if (this.level !== 'drowsy') this.setLevel('eyes-closing');
      }
      return;
    }
    // closed-eye timer reset path:
    if (closedMs === 0) {
      // if we were in warning/critical/emergency, fall back through dismissAlert path
      if (this.level === 'warning' || this.level === 'critical' || this.level === 'emergency') {
        this.audio.stopAll();
        if (this.countdownInterval) clearInterval(this.countdownInterval);
        this.countdownInterval = null;
        this.inEmergencyFlow = false;
        // Drowsy may still be active from track A — restore it.
        if (this.events.length >= DROWSY_THRESHOLD) {
          this.setLevel('drowsy');
          this.audio.startPullover();
        } else {
          this.setLevel('none');
        }
      } else if (this.level === 'eyes-closing') {
        this.setLevel(this.events.length >= DROWSY_THRESHOLD ? 'drowsy' : 'none');
      }
    }
  }

  private setLevel(l: AlertLevel) {
    if (this.level === l) return;
    this.level = l;
    this.opts.onLevelChange(l);
  }

  private startCountdown() {
    let n = 5;
    this.opts.onCountdown(n);
    if (this.countdownInterval) clearInterval(this.countdownInterval);
    this.countdownInterval = window.setInterval(() => {
      n -= 1;
      this.opts.onCountdown(Math.max(0, n));
      if (n <= 0) {
        if (this.countdownInterval) clearInterval(this.countdownInterval);
        this.countdownInterval = null;
      }
    }, 1000);
  }

  private startEmergencyFlow() {
    if (this.inEmergencyFlow) return;
    this.inEmergencyFlow = true;
    this.audio.stopPullover();
    this.audio.duckSiren();

    // 1) Dial 112
    const offsets = this.audio.playDialer();
    const number = '112';
    offsets.forEach((off, i) => {
      setTimeout(() => {
        if (i < number.length) this.opts.onDialerDigit(number[i], i);
      }, off * 1000);
    });

    // 2) After dialer ends → 3× ringback
    this.audio.onDialerEnd(() => {
      const ringbackPlays = 3;
      this.opts.onCallingProgress(ringbackPlays);
      this.audio.playCallingTimes(ringbackPlays);

      // Track remaining plays via polling (audio engine decrements internally)
      const start = performance.now();
      const poll = window.setInterval(() => {
        // best-effort: we just signal completion when calling track stops and counter is gone
        // real progress could be wired through, but visible end-state is enough.
        if (performance.now() - start > 60_000) clearInterval(poll);
      }, 500);
    });
  }
}
