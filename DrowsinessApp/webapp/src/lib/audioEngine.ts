// Browser port of lib/services/audio_engine.dart.
//
// Multi-track engine built on HTMLAudioElement. See DESIGN.md §4.
//   buzz     — one-shot focus reminder
//   pullover — looped pull-over voice
//   siren    — looped critical siren (volume ducking during dispatcher chain)
//   dialer   — one-shot 3-tone "112" dialing (digit offsets 71/437/701 ms)
//   calling  — ringback, played exactly 3 times
//   accept   — "911, what's your emergency"
//   intro    — pre-recorded dispatcher prefix
//   then live TTS speaks the location, and the siren ramps back up.

import { speak } from "./tts";

const SND = "/sounds";
export const DIAL_DIGIT_OFFSETS_MS = [71, 437, 701];

function mkAudio(src: string, loop = false): HTMLAudioElement {
  const a = new Audio(src);
  a.loop = loop;
  a.preload = "auto";
  return a;
}

export class AudioEngine {
  private buzz = mkAudio(`${SND}/buzz.mp3`);
  private pullover = mkAudio(`${SND}/PULLOVER.mp3`, true);
  private siren = mkAudio(`${SND}/sirenLoop.mp3`, true);
  private dialer = mkAudio(`${SND}/dialingButtons.m4a`);
  private calling = mkAudio(`${SND}/calling.mp3`);
  private accept = mkAudio(`${SND}/911accept.mp3`);
  private intro = mkAudio(`${SND}/dispatch_intro.mp3`);

  private master = 1.0;
  private callingPlaysLeft = 0;
  private emergencyLocationText: string | null = null;
  private dialerEndCb: (() => void) | null = null;
  private rampTimer: ReturnType<typeof setInterval> | null = null;

  init() {
    // Ringback chains: replay up to 3 times, then dispatcher pickup.
    this.calling.addEventListener("ended", () => {
      this.callingPlaysLeft -= 1;
      if (this.callingPlaysLeft > 0) {
        void this.play(this.calling, this.master);
      } else {
        void this.play(this.accept, this.master);
      }
    });
    // After dispatcher pickup → cached prefix.
    this.accept.addEventListener("ended", () => {
      void this.play(this.intro, this.master);
    });
    // After prefix → speak the location, then ramp the siren back up.
    this.intro.addEventListener("ended", async () => {
      await speak(this.emergencyLocationText ?? "an unknown location", this.master);
      this.rampVolume(this.siren, this.master, 200);
    });
    // Dialer one-shot end.
    this.dialer.addEventListener("ended", () => {
      const cb = this.dialerEndCb;
      this.dialerEndCb = null;
      cb?.();
    });
  }

  /** Browsers require a user gesture to unlock audio. Call from a click. */
  async unlock() {
    const all = [this.buzz, this.pullover, this.siren, this.dialer,
      this.calling, this.accept, this.intro];
    for (const a of all) {
      try {
        a.muted = true;
        await a.play();
        a.pause();
        a.currentTime = 0;
        a.muted = false;
      } catch {
        a.muted = false;
      }
    }
  }

  setMasterVolume(v: number) {
    this.master = Math.max(0, Math.min(1, v));
  }

  setEmergencyLocationText(text: string | null) {
    this.emergencyLocationText = text;
  }

  private async play(a: HTMLAudioElement, volume: number) {
    try {
      a.volume = Math.max(0, Math.min(1, volume));
      a.currentTime = 0;
      await a.play();
    } catch {
      /* autoplay blocked / interrupted — ignore */
    }
  }

  async playBuzz() {
    this.buzz.pause();
    await this.play(this.buzz, this.master);
  }

  async startPullover() {
    if (!this.pullover.paused) return;
    this.pullover.loop = true;
    await this.play(this.pullover, this.master);
  }
  stopPullover() {
    this.pullover.pause();
    this.pullover.currentTime = 0;
  }

  async startSiren() {
    if (!this.siren.paused) return;
    this.siren.loop = true;
    await this.play(this.siren, this.master);
  }
  stopSiren() {
    this.siren.pause();
    this.siren.currentTime = 0;
  }
  duckSiren() {
    this.rampVolume(this.siren, this.master * 0.1, 200);
  }
  unduckSiren() {
    this.rampVolume(this.siren, this.master, 200);
  }

  /** Plays the dialer one-shot. Returns the digit-offset timeline (ms). */
  async playDialer(): Promise<number[]> {
    await this.play(this.dialer, this.master);
    return DIAL_DIGIT_OFFSETS_MS;
  }

  onDialerEnd(cb: () => void) {
    this.dialerEndCb = cb;
  }

  async playCallingTimes(n: number) {
    this.callingPlaysLeft = n;
    await this.play(this.calling, this.master);
  }

  stopCalling() {
    this.callingPlaysLeft = 0;
    this.calling.pause();
    this.calling.currentTime = 0;
  }

  stopAll() {
    if (this.rampTimer) {
      clearInterval(this.rampTimer);
      this.rampTimer = null;
    }
    this.stopPullover();
    this.stopSiren();
    this.stopCalling();
    for (const a of [this.dialer, this.accept, this.intro]) {
      a.pause();
      a.currentTime = 0;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    this.dialerEndCb = null;
  }

  private rampVolume(a: HTMLAudioElement, target: number, ms: number) {
    if (this.rampTimer) clearInterval(this.rampTimer);
    const steps = 10;
    const stepMs = Math.round(ms / steps);
    const start = a.volume;
    let i = 0;
    this.rampTimer = setInterval(() => {
      i += 1;
      const v = start + (target - start) * (i / steps);
      a.volume = Math.max(0, Math.min(1, v));
      if (i >= steps && this.rampTimer) {
        clearInterval(this.rampTimer);
        this.rampTimer = null;
      }
    }, stepMs);
  }

  dispose() {
    this.stopAll();
  }
}
