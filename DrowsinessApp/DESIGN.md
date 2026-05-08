# Driver Drowsiness Detector — App Design

A phone-first drowsiness detection app. Designed to be mounted in front of the driver, watch them through the front-facing camera, and intervene when fatigue is detected.

This document is the locked specification for the application. The implementation must follow this spec; if reality forces a deviation, update this document first.

---

## 1. Platform & Stack

**Stack:** React + TypeScript + Vite + TailwindCSS, packaged as a **PWA**.

**Why a web-based PWA instead of native Flutter:**
- Genuinely **on-device** inference: TensorFlow.js runs the EfficientNet model in the browser; OpenCV.js runs the Haar cascades. No backend, no network — works offline once loaded.
- Runs on iPhone Safari **without a Mac, Apple Developer account, or App Store** — installed via Safari → "Add to Home Screen" → fullscreen, home-screen icon, indistinguishable from a native app for demo purposes.
- Same code path runs in Chrome on Windows for development.
- Zero install friction. Zero TFLite conversion friction.

**Inference:** `@tensorflow/tfjs` loading the EfficientNet-B0 model converted from `Models/drowsiness_efficientnet_b0.h5` via `tensorflowjs_converter`. Model is loaded once at app start.

**Detection (face / eyes):** OpenCV.js, using the same Haar cascade XMLs as the existing Python pipeline (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`).

**Storage:** Browser IndexedDB (via `idb` library) for trip history.

**Audio:** HTML5 `Audio` elements pre-loaded at session start. Multiple simultaneous tracks supported (siren + voice + dialer can play together).

---

## 2. Model Behavior (from notebook)

Single 6-class softmax model. Input **224×224 RGB, raw uint8 (no rescale — EfficientNetB0 has internal preprocessing)**.

| Index | Class | Trained on |
|---|---|---|
| 0 | yawn | Haar **face** crops |
| 1 | no_yawn | Haar **face** crops |
| 2 | Closed | **eye** region images |
| 3 | Open | **eye** region images |
| 4 | front | full **head** images |
| 5 | down | full **head** images |

The model is run **twice per frame** with two different crops:

| Crop | Output subset (renormalized) | Decision |
|---|---|---|
| Face crop, 224×224 | `{yawn, no_yawn, front, down}` | yawning? head-down? |
| Eye crop, 224×224 (per detected eye) | `{Closed, Open}` | eyes closed? |

Predictions below confidence **0.6** are treated as "uncertain" and do not extend or trigger any timer.

**Visual overlay on the live camera feed:**
- Green rectangle around the detected face, label `face: <class> <conf%>` next to it.
- Blue rectangle around each detected eye, label `eye: <class> <conf%>`.

---

## 3. Alert System — Two Independent Tracks

The alert engine runs two state machines in parallel. The Critical track always overrides the Focus track on screen.

### 3.1 Track A — Focus Reminders (yawn + head-down, combined)

Group A treats yawns and head-downs as the same kind of fatigue event.

**Event registration (debounced):**
- A **yawn event** is registered when class `yawn` is the top face-crop class for **≥ 1.5 s of continuous frames** above threshold, then locked out by a **3 s cooldown** before it can fire again.
- A **head-down event** is registered when class `down` is the top face-crop class for **≥ 1.5 s continuous**, with the same 3 s cooldown.
- Each registered event:
  - Plays `buzz.mp3` once (short focus reminder).
  - Triggers a 300 ms haptic vibration (`navigator.vibrate(300)`).
  - Is appended to a sliding event log with timestamp.

**Drowsy state trigger:**
- When the sliding log shows **≥ 3 events in the last 30 seconds** (yawn + head-down combined) → enter **Drowsy** state.

**Drowsy state UI/sound:**
- Full-screen **"PULL OVER"** card — large icon (steering wheel / warning), big text, calm but urgent palette (deep amber background).
- Loops `PULLOVER.mp3` until dismissed.
- Continuous haptic pulse pattern.
- One **"I'm OK"** button to dismiss.
- On dismiss: event log clears, **10 s grace period** before counting resumes.

### 3.2 Track B — Critical (eyes closed)

A single timer measures **continuous closed-eye time**. Closed = both eyes detected and classified `Closed`, OR (face detected but no eyes detected for >1 s — eyes likely closed and Haar can't find them).

**Reset rule:** the timer resets to zero only after eyes are detected `Open` for **≥ 0.5 s continuous**. This hysteresis prevents one stray frame from cancelling an active alarm.

**Threshold ladder:**

| Continuous closed time | State | UI | Audio |
|---|---|---|---|
| 0 – 0.8 s | normal blink | (nothing) | (nothing) |
| 0.8 – 5 s | "Eyes closing" warning | small amber pill at top of screen | (nothing) |
| 5 – 10 s | **WARNING** | screen dims, large amber overlay "EYES CLOSED — WAKE UP" | `PULLOVER.mp3` looping |
| 10 – 15 s | **CRITICAL** | screen flashes red, large overlay "EMERGENCY — CALLING SOON", countdown `5 → 0` displayed in center | `PULLOVER.mp3` + `sirenLoop.mp3` (both at full volume, layered) |
| ≥ 15 s | **EMERGENCY DIALER** | full-screen GTA-style phone in bottom-right; dial pad shows; digits **1, 1, 2** are typed in sequence | `sirenLoop.mp3` ducks to 25% volume; `dialingButtons.m4a` plays (tones at 0.071 s, 0.437 s, 0.701 s — UI digits appear synchronized to those tones); then `calling.mp3` plays exactly **3 times** in a row (ringback) |
| After ringback ends, eyes still closed | EMERGENCY ONGOING | dialer shows "Connected — call placed" | `sirenLoop.mp3` returns to full volume and continues |

**Cancel:** at any point in CRITICAL or EMERGENCY, the user can tap a "Cancel" button to dismiss the entire emergency stack. Eye-open hysteresis (≥0.5s) also fully resets.

**Emergency number** is configurable in Settings (default `112`; alternates `911`, `999`, `110`). The dialer animation always types the digits of the configured number in sequence; sound timing is mapped to the digit count of that number (default mapping is for 3-digit `112`; if a longer number is configured we tile/extend the dial tones).

---

## 4. Audio Spec — file mapping

All files live in `public/sounds/`, copied from `SoundEffects/` at the project root.

| File | Used for | Behavior |
|---|---|---|
| `buzz.mp3` | Single yawn / head-down event registered | One-shot play, ~300 ms |
| `PULLOVER.mp3` | (a) Drowsy state pull-over screen, (b) Eyes-closed 5–10 s warning | Looped while state active |
| `sirenLoop.mp3` | Eyes-closed 10 s+ critical alarm | Looped; ducks to 25% during dialer/ringback, returns to 100% after ringback |
| `dialingButtons.m4a` | Eyes-closed 15 s+ — typing 112 on the dialer | One-shot. Internal tones at `0.071`, `0.437`, `0.701` seconds. UI digits `1`, `1`, `2` are scheduled to appear at those exact offsets relative to playback start |
| `calling.mp3` | Eyes-closed 15 s+ — ringback tone after dialing completes | Played sequentially **exactly 3 times**, then stops |

**Audio engine rules:**
- Multiple tracks may play concurrently (siren + dialer + ringback all coexist via independent `Audio` objects).
- Volume ducking is implemented by ramping `audio.volume` linearly over 200 ms (avoids clicks).
- Looping is enabled via `audio.loop = true`.
- All audio is preloaded at app start, so playback is gapless.

---

## 5. Timeline — Worked Example for Eyes Closed

Driver's eyes close and stay closed continuously. T = 0 is the first closed frame:

```
T=0.0s  blink ignored
T=0.8s  small amber "Eyes closing" pill appears
T=5.0s  → WARNING state
        UI: amber dim overlay, "EYES CLOSED — WAKE UP"
        Audio: PULLOVER.mp3 starts looping
T=10.0s → CRITICAL state
        UI: red flash overlay, big "EMERGENCY — CALLING SOON", countdown 5
        Audio: sirenLoop.mp3 starts looping (full volume), PULLOVER.mp3 still looping
T=11.0s → countdown 4
T=12.0s → countdown 3
T=13.0s → countdown 2
T=14.0s → countdown 1
T=15.0s → EMERGENCY DIALER
        UI: phone slides in bottom-right (GTA-style), dial pad appears
        Audio: sirenLoop.mp3 ducks to 25%, PULLOVER.mp3 stops, dialingButtons.m4a starts
T=15.071s → digit "1" appears on dialer screen (synced to first tone)
T=15.437s → digit "1" appears (second 1)
T=15.701s → digit "2" appears
T=~16.5s  → dialingButtons.m4a finishes; calling.mp3 starts
            Loop count = 1
T=~22.5s  → calling.mp3 finishes; auto-replay (count = 2)
T=~28.5s  → finishes; auto-replay (count = 3)
T=~34.5s  → calling.mp3 done after exactly 3 plays
            UI: dialer shows "Connected — call placed"
            Audio: sirenLoop.mp3 ramps back to full volume
T=∞       → siren continues looping until eyes open ≥ 0.5 s OR user taps Cancel
```

(Times shown after T=15s are illustrative; actual `calling.mp3` duration depends on the file.)

---

## 6. Edge Cases

1. **No face detected for > 3 s** → show "Face lost — reposition camera" banner. Freeze all timers/counters. Do not count events.
2. **Calibration window**: first 3 s after pressing **Start Drive** — detect and draw boxes, but do not register any events.
3. **Confidence gate**: any prediction with top-class probability < 0.6 is "uncertain" and does not extend either timer.
4. **Eye detection robustness**: Haar eye cascade often returns 0–2 eyes per face, especially with head tilt or closed eyes (closed eyes have weak Haar response). Rules:
    - 2 eyes detected → both must classify `Closed` for "closed" state.
    - 1 eye detected → that eye's class is used.
    - 0 eyes detected for > 1 s **with** face still detected → treated as "likely closed" (extends the closed-eye timer).
5. **Window unfocused / tab hidden**: detection pauses (browser doesn't grant camera frames anyway). On return, all timers reset and a 3 s calibration runs.
6. **Both states at once**: Critical track always wins the screen. Focus events still register in the background log so the count is correct when Critical resolves.

---

## 7. Screens & Navigation

Three top-level screens, bottom tab bar:

### 7.1 Drive (default)

- Live full-screen camera feed (mirrored).
- Face/eye boxes overlaid.
- Top status bar: state badge (`Alert` / `Eyes closing` / `Drowsy` / `Critical`), confidence indicators, current trip duration.
- Bottom: **Start Drive** / **End Drive** button. While driving: **Pause**.
- Alert overlays (Pull-over, Warning, Critical, Emergency Dialer) render on top.

### 7.2 History

- List of past trips: date, duration, # focus events, longest closed-eye streak, # critical alarms.
- Tap a trip → detail screen with timeline of events.

### 7.3 Settings

- **Sensitivity**: confidence threshold slider (default 0.6).
- **Emergency number**: 112 / 911 / 999 / 110 / custom.
- **Alarm volume**.
- **Keep screen on while driving** (uses Wake Lock API).
- **Reset history**.

---

## 8. Visual Design

- **Tone**: clean, professional, automotive — like a modern car HUD. Not cartoonish.
- **Palette**:
  - Background: deep neutral `#0B0F14` (near-black with cool tint)
  - Surface: `#141B23`
  - Primary accent: `#3B82F6` (electric blue)
  - Alert amber: `#F59E0B`
  - Critical red: `#EF4444`
  - OK green: `#22C55E`
  - Text: `#E5E7EB` primary, `#9CA3AF` muted
- **Typography**: Inter or system sans-serif. Tabular numerals for timers.
- **Motion**: subtle. Pull-over screen slides up. Critical flash uses 1.5 Hz red opacity pulse. Dialer slides in from bottom-right with spring easing.
- **Iconography**: Lucide React icons (steering wheel, eye, phone, alert-triangle).
- **Frame**: app is locked to portrait, max-width 480 px on desktop centered in a phone bezel mock so the laptop preview *looks* like a phone for thesis screenshots.

---

## 9. Project Layout

```
DrowsinessApp/
  DESIGN.md                  ← this file
  README.md                  ← run instructions
  package.json
  vite.config.ts
  tailwind.config.js
  tsconfig.json
  index.html
  scripts/
    convert_model.py         ← runs tensorflowjs_converter on the .h5
  public/
    haarcascades/
      haarcascade_frontalface_default.xml
      haarcascade_eye.xml
    sounds/
      buzz.mp3
      PULLOVER.mp3
      sirenLoop.mp3
      dialingButtons.m4a
      calling.mp3
    models/
      efficientnet_b0/       ← TFJS graph model output
        model.json
        group1-shard*.bin
    manifest.webmanifest
  src/
    main.tsx
    App.tsx
    index.css
    types.ts
    pages/
      DrivePage.tsx
      HistoryPage.tsx
      SettingsPage.tsx
    components/
      CameraView.tsx
      StatusBar.tsx
      PullOverOverlay.tsx
      WarningOverlay.tsx
      CriticalOverlay.tsx
      EmergencyDialer.tsx
      BottomTabs.tsx
      PhoneFrame.tsx
    services/
      detector.ts             ← Haar + TFJS pipeline
      alertEngine.ts          ← state machine
      audio.ts                ← multi-track audio with ducking
      storage.ts              ← IndexedDB trip log
      settings.ts             ← persisted user prefs
```

---

## 10. Implementation Notes

- **Model conversion** is one-time. `scripts/convert_model.py` calls `tensorflowjs_converter` (Python package `tensorflowjs`) to produce `public/models/efficientnet_b0/`. Needs to be run once before the app can start.
- **Detection loop** runs on a `requestAnimationFrame` loop targeting ~10 fps (we throttle by tracking `lastFrameAt`). Higher fps wastes CPU; lower misses fast events.
- **Performance**: EfficientNet-B0 inference in TFJS WebGL backend on a typical laptop runs at ~50–100 ms per call; two calls per frame ≈ 100–200 ms ≈ 5–10 fps. Acceptable. On iPhone Safari the WebGL backend is also strong.
- **PWA**: a service worker caches the app shell + model + sounds + cascades on first load. Subsequent loads work fully offline.
