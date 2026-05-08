/**
 * Multi-track audio engine.
 *
 * Tracks supported:
 *   buzz    — one-shot focus reminder (yawn / head-down event)
 *   pullover — looped pull-over voice (drowsy state OR eyes-closed 5–10 s warning)
 *   siren   — looped siren (eyes-closed 10 s+ critical)
 *   dialer  — one-shot 3-tone dialing for "112"  (timestamps: 0.071, 0.437, 0.701)
 *   calling — ringback tone, must play exactly 3 times then stop
 *
 * The dialer tone offsets are exposed so the UI can light up digits in sync.
 */

const FILES = {
  buzz: '/sounds/buzz.mp3',
  pullover: '/sounds/PULLOVER.mp3',
  siren: '/sounds/sirenLoop.mp3',
  dialer: '/sounds/dialingButtons.m4a',
  calling: '/sounds/calling.mp3',
} as const;

export const DIAL_DIGIT_OFFSETS_S = [0.071, 0.437, 0.701] as const;

class Track {
  el: HTMLAudioElement;
  baseVolume = 1;
  constructor(src: string, loop = false) {
    this.el = new Audio(src);
    this.el.loop = loop;
    this.el.preload = 'auto';
  }
  play() {
    this.el.currentTime = 0;
    this.el.volume = this.baseVolume;
    return this.el.play().catch(() => {});
  }
  stop() {
    this.el.pause();
    this.el.currentTime = 0;
  }
  setVolume(v: number) {
    this.baseVolume = v;
    this.el.volume = v;
  }
  duck(target: number, ms = 200) {
    rampVolume(this.el, target, ms);
  }
  isPlaying() {
    return !this.el.paused;
  }
}

function rampVolume(el: HTMLAudioElement, target: number, ms: number) {
  const start = el.volume;
  const t0 = performance.now();
  const tick = () => {
    const t = (performance.now() - t0) / ms;
    if (t >= 1) {
      el.volume = target;
      return;
    }
    el.volume = start + (target - start) * t;
    requestAnimationFrame(tick);
  };
  tick();
}

export class AudioEngine {
  private buzz: Track;
  private pullover: Track;
  private siren: Track;
  private dialer: Track;
  private calling: Track;
  private callingPlaysLeft = 0;
  private masterVolume = 1;

  constructor() {
    this.buzz = new Track(FILES.buzz, false);
    this.pullover = new Track(FILES.pullover, true);
    this.siren = new Track(FILES.siren, true);
    this.dialer = new Track(FILES.dialer, false);
    this.calling = new Track(FILES.calling, false);

    this.calling.el.addEventListener('ended', () => {
      this.callingPlaysLeft -= 1;
      if (this.callingPlaysLeft > 0) {
        this.calling.play();
      } else {
        // siren back to full volume after ringback finishes
        this.siren.duck(this.masterVolume * 1.0);
      }
    });
  }

  setMasterVolume(v: number) {
    this.masterVolume = v;
    this.buzz.setVolume(v);
    this.pullover.setVolume(v);
    if (this.siren.isPlaying()) this.siren.setVolume(v);
    this.dialer.setVolume(v);
    this.calling.setVolume(v);
  }

  /** Pre-warm by playing+pausing at zero volume — required on iOS Safari for autoplay later. */
  async unlock() {
    const tracks = [this.buzz, this.pullover, this.siren, this.dialer, this.calling];
    for (const t of tracks) {
      const oldVol = t.el.volume;
      t.el.volume = 0;
      try {
        await t.el.play();
      } catch {}
      t.el.pause();
      t.el.currentTime = 0;
      t.el.volume = oldVol;
    }
  }

  // ---- public API used by alertEngine ----
  playBuzz() {
    if (navigator.vibrate) navigator.vibrate(300);
    this.buzz.play();
  }

  startPullover() {
    if (!this.pullover.isPlaying()) this.pullover.play();
  }
  stopPullover() {
    this.pullover.stop();
  }

  startSiren() {
    if (!this.siren.isPlaying()) {
      this.siren.setVolume(this.masterVolume);
      this.siren.play();
    }
  }
  stopSiren() {
    this.siren.stop();
  }
  duckSiren() {
    this.siren.duck(this.masterVolume * 0.25);
  }
  unduckSiren() {
    this.siren.duck(this.masterVolume * 1.0);
  }

  /** Plays the 3-tone dialer once. Returns the digit-offset schedule. */
  playDialer(): readonly number[] {
    this.dialer.play();
    return DIAL_DIGIT_OFFSETS_S;
  }
  /** True when the dialer file has finished. */
  onDialerEnd(cb: () => void) {
    const handler = () => {
      this.dialer.el.removeEventListener('ended', handler);
      cb();
    };
    this.dialer.el.addEventListener('ended', handler);
  }

  playCallingTimes(n: number) {
    this.callingPlaysLeft = n;
    this.calling.play();
  }
  stopCalling() {
    this.callingPlaysLeft = 0;
    this.calling.stop();
  }

  stopAll() {
    this.stopPullover();
    this.stopSiren();
    this.stopCalling();
    this.dialer.stop();
  }
}
