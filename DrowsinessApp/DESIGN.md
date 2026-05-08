# Driver Drowsiness Detector — App Design

A phone-first drowsiness detection app. Designed to be mounted in front of the driver, watch them through the front-facing camera, and intervene when fatigue is detected.

This document is the locked specification for the application. The implementation must follow this spec; if reality forces a deviation, update this document first.

---

## 1. Platform & Stack

**Framework:** **Flutter** (Dart). Single codebase compiled to **iOS** (target deployment), **Android**, and **Windows desktop** (used for laptop testing on this Windows machine).

**Inference:** `tflite_flutter` running an EfficientNet-B0 TFLite model converted from `Models/drowsiness_efficientnet_b0.h5` via the Python conversion script in `scripts/convert_model.py`. Model is loaded once at app start, runs entirely on-device.

**Detection (face / eyes):** `opencv_dart` — Dart bindings to native OpenCV with prebuilt binaries for Windows + iOS + Android. Uses the same Haar cascade XMLs as the existing Python pipeline (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`).

**Camera:** `camera` plugin (federated; covers iOS + Android + Windows desktop).

**Audio:** `audioplayers` — multiple `AudioPlayer` instances allow concurrent playback (siren + dialer + ringback can stack), looping is supported, volume is settable per track. Volume ramping (ducking) is implemented as a 10-step timer.

**Storage:** `sqflite` (with `sqflite_common_ffi` on Windows) for trip history. `shared_preferences` for app settings.

**System:**
- `vibration` — haptic pulses for buzz events.
- `wakelock_plus` — keep screen on during a drive.
- `permission_handler` — camera permission on iOS/Android.
- `go_router` — navigation between Drive / History / Settings.

---

## 2. Model Behavior (from notebook)

Single 6-class softmax model. Input **224×224 RGB, raw uint8 (no rescale — EfficientNetB0 has internal preprocessing)**, fed as `Float32List` of `[1,224,224,3]` containing the raw 0–255 pixel values.

| Index | Class | Trained on |
|---|---|---|
| 0 | yawn | Haar **face** crops |
| 1 | no_yawn | Haar **face** crops |
| 2 | Closed | **eye** region images |
| 3 | Open | **eye** region images |
| 4 | front | full **head** images |
| 5 | down | full **head** images |

The model is run **twice per detected face** with two different crops, and we read different output subsets for each:

| Crop | Output subset (renormalized) | Decision |
|---|---|---|
| Haar **face** crop, 224×224 | `{yawn, no_yawn, front, down}` | yawning? head-down? |
| Haar **eye** crop, 224×224 (per detected eye) | `{Closed, Open}` | eyes closed? |

Predictions below confidence **0.6** (configurable) are treated as "uncertain" and do not extend or trigger any timer.

**Visual overlay on the live camera feed:**
- Green rectangle around the detected face, label `face: <class> <conf%>`.
- Blue rectangle (Open) or red rectangle (Closed) around each detected eye, with `<class> <conf%>` label.
- Drawn via Flutter `CustomPainter` over the `CameraPreview`, mirrored for selfie view.

---

## 3. Alert System — Two Independent Tracks

The alert engine ([`lib/services/alert_engine.dart`](lib/services/alert_engine.dart)) runs two state machines in parallel. The Critical track always overrides the Focus track on screen.

### 3.1 Track A — Focus Reminders (yawn + head-down, combined)

Group A treats yawns and head-downs as the same kind of fatigue event.

**Event registration (debounced):**
- A **yawn event** is registered when class `yawn` is the top face-crop class for **≥ 1.5 s of continuous frames** above threshold, then locked out by a **3 s cooldown**.
- A **head-down event** is registered the same way for class `down`.
- Each registered event:
  - Plays `buzz.mp3` once.
  - Triggers a 300 ms haptic vibration.
  - Is appended to a sliding event log.

**Drowsy state trigger:**
- When the sliding log shows **≥ 3 events in the last 30 s** (yawn + head-down combined) → enter **Drowsy** state.

**Drowsy state UI/sound:**
- Full-screen **"PULL OVER"** card — large warning icon, big text, deep amber palette.
- Loops `PULLOVER.mp3` until dismissed.
- Single **"I'm OK"** button to dismiss.
- On dismiss: event log clears, **10 s grace period** before counting resumes.

### 3.2 Track B — Critical (eyes closed)

A single timer measures **continuous closed-eye time**. Closed = both eyes detected and classified `Closed`, OR (face detected but no eyes detected with high confidence — eyes likely closed and Haar can't find them).

**Reset rule:** the timer resets only after eyes are detected `Open` for **≥ 0.5 s continuous** (hysteresis).

**Threshold ladder:**

| Continuous closed time | State | UI | Audio |
|---|---|---|---|
| 0 – 0.8 s | normal blink | (nothing) | (nothing) |
| 0.8 – 5 s | "Eyes closing" pill | small amber pill at top | (nothing) |
| 5 – 10 s | **WARNING** | full-screen amber overlay "EYES CLOSED — WAKE UP" | `PULLOVER.mp3` looping |
| 10 – 15 s | **CRITICAL** | red flashing screen, large overlay "EMERGENCY — CALLING SOON", countdown `5 → 0` | `PULLOVER.mp3` + `sirenLoop.mp3` (both at full volume) |
| ≥ 15 s | **EMERGENCY DIALER** | GTA-style phone slides into bottom-right; dial pad shows; digits `1, 1, 2` light up in sequence | `sirenLoop.mp3` ducks to 25%; `dialingButtons.m4a` plays (digits appear at **0.071 s, 0.437 s, 0.701 s** offsets matching the audio); then `calling.mp3` plays exactly **3 times** in a row (ringback) |
| After ringback ends, eyes still closed | EMERGENCY ONGOING | dialer shows "Connected" | `sirenLoop.mp3` ramps back to full volume |

**Cancel:** at any point in CRITICAL or EMERGENCY, the user can tap **End** / **Cancel** to dismiss the entire emergency stack. Eye-open hysteresis (≥0.5s) also fully resets.

**Emergency number** is configurable in Settings (default `112`; alternates `911`, `999`, `110`).

---

## 4. Audio Spec — file mapping

All files live in `assets/sounds/`, copied from `SoundEffects/` at the project root.

| File | Used for | Behavior |
|---|---|---|
| `buzz.mp3` | Single yawn / head-down event registered | One-shot, low-latency mode |
| `PULLOVER.mp3` | (a) Drowsy state pull-over screen, (b) Eyes-closed 5–10 s warning | Looped while state active |
| `sirenLoop.mp3` | Eyes-closed 10 s+ critical alarm | Looped; ducks to 25% during dialer/ringback, returns to 100% after ringback |
| `dialingButtons.m4a` | Eyes-closed 15 s+ — typing 112 on the dialer | One-shot. Internal tones at `0.071`, `0.437`, `0.701` seconds. UI digits `1`, `1`, `2` appear at those exact offsets |
| `calling.mp3` | Eyes-closed 15 s+ — ringback after dialing completes | Played sequentially **exactly 3 times** via `onPlayerComplete` listener, then stops |

---

## 5. Timeline — Worked Example for Eyes Closed

Driver's eyes close and stay closed continuously. T = 0 is the first closed frame:

```
T=0.0s  blink ignored
T=0.8s  small amber "Eyes closing" pill appears
T=5.0s  → WARNING state
        UI: amber overlay "EYES CLOSED — WAKE UP"
        Audio: PULLOVER.mp3 starts looping
T=10.0s → CRITICAL state
        UI: red flashing screen, big "EMERGENCY — CALLING 112 IN 5"
        Audio: sirenLoop.mp3 starts looping (full volume), PULLOVER.mp3 still looping
T=11.0s → countdown 4
T=12.0s → countdown 3
T=13.0s → countdown 2
T=14.0s → countdown 1
T=15.0s → EMERGENCY DIALER
        UI: phone slides in bottom-right (GTA-style), dial pad
        Audio: sirenLoop.mp3 ducks to 25%, PULLOVER.mp3 stops, dialingButtons.m4a starts
T=15.071s → digit "1" appears on dialer screen (synced to first tone)
T=15.437s → digit "1" appears (second 1)
T=15.701s → digit "2" appears
T=~16.5s  → dialingButtons.m4a finishes; calling.mp3 starts
T=after 3 plays of calling.mp3:
            UI: dialer shows "Connected"
            Audio: sirenLoop.mp3 ramps back to full volume
T=∞       → siren continues looping until eyes open ≥ 0.5 s OR user taps Cancel
```

---

## 6. Edge Cases

1. **No face detected for > 3 s** → freeze counters. (Future: show banner "Face lost — reposition camera".)
2. **Calibration window**: first 3 s after pressing **Start Drive** — detect and draw boxes, but do not register any events.
3. **Confidence gate**: any prediction with top-class probability < 0.6 is ignored.
4. **Eye detection robustness**: Haar eye cascade often returns 0–2 eyes per face, especially with closed eyes (weak Haar response).
    - 2 eyes detected → both must classify `Closed` for "closed" state.
    - 1 eye detected → that eye's class is used.
    - 0 confident eyes within face → treated as "likely closed" (extends the closed-eye timer).
5. **App backgrounded / camera pauses**: the engine keeps state but no frames flow; on resume, the next-frame ingest will see continuous time and behave correctly.
6. **Both states at once**: Critical track always wins the screen. Focus events still register in the background log so the count is correct when Critical resolves.

---

## 7. Screens & Navigation

Three top-level screens, bottom tab bar (`go_router` ShellRoute):

### 7.1 Drive (default)

- Live full-screen `CameraPreview` (mirrored).
- `DetectionOverlay` `CustomPainter` draws face + eye boxes.
- Top status bar: state badge, eyes-closed progress bar, trip duration.
- Bottom: **Start Drive** initially. While driving: **Pause** + **End**.
- Alert overlays render on top.

### 7.2 History

- List of past trips: date, duration, focus events count, drowsy count, critical count, longest closed-eye streak.

### 7.3 Settings

- Sensitivity (confidence threshold slider, 0.4–0.9, default 0.6).
- Emergency number (112 / 911 / 999 / 110).
- Alarm volume.
- Keep screen on while driving (Wake Lock).
- Reset history.

---

## 8. Visual Design

- **Tone**: clean, professional, automotive — like a modern car HUD.
- **Palette** (`lib/theme.dart`):
  - Background: `#0B0F14`
  - Surface: `#141B23` / `#1C2530`
  - Primary: `#3B82F6` (electric blue)
  - Amber: `#F59E0B`
  - Danger: `#EF4444`
  - OK green: `#22C55E`
  - Text: `#E5E7EB` primary, `#9CA3AF` muted
- **Typography**: Inter via `google_fonts`. Tabular numerals for timers.
- **Motion**: Pull-over slides up; critical flashes red at 1.5 Hz; dialer slides in from right with a spring curve.
- **Iconography**: Material Icons.
- **Orientation**: locked to portrait (`SystemChrome.setPreferredOrientations`).

---

## 9. Project Layout

```
DrowsinessApp/
  DESIGN.md                  ← this file
  README.md                  ← run / install instructions
  pubspec.yaml
  analysis_options.yaml
  scripts/
    convert_model.py         ← .h5 → .tflite via TFLite converter
  assets/
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
      drowsiness_efficientnet_b0.tflite   ← produced by convert_model.py
  lib/
    main.dart                ← app + router
    theme.dart
    models/
      types.dart             ← DetectionResult, FaceBox, AlertLevel, AppSettings, Trip…
    pages/
      drive_page.dart
      history_page.dart
      settings_page.dart
    widgets/
      bottom_tabs.dart
      status_bar.dart
      detection_overlay.dart
      overlays.dart          ← PullOverOverlay, WarningOverlay, CriticalOverlay
      emergency_dialer.dart
    services/
      detector.dart          ← opencv_dart Haar + tflite_flutter inference
      alert_engine.dart      ← state machine
      audio_engine.dart      ← multi-track audio with ducking
      storage.dart           ← sqflite trip log
      settings.dart          ← shared_preferences
  android/  ios/  windows/   ← created by `flutter create .` on first setup
```

---

## 10. Implementation Notes

- **Model conversion** is one-time. `scripts/convert_model.py` produces `assets/models/drowsiness_efficientnet_b0.tflite` from `Models/drowsiness_efficientnet_b0.h5`. Run before first launch.
- **Frame loop** uses `CameraController.startImageStream()`. Frames arrive in BGRA8888 on iOS/Windows and YUV420 on Android — `Detector._matFromCameraImage` handles both. A `_busy` flag drops frames while inference is running, so effective fps is "as fast as inference allows."
- **Performance**: EfficientNet-B0 in `tflite_flutter` runs at ~50–100 ms per call on a typical laptop and most modern phones. Two calls per face plus Haar pre-detection ≈ 5–8 fps. Acceptable for the alert thresholds (1.5 s sustain, 0.5 s hysteresis, 5 s thresholds).
- **Audio ducking** is implemented in `AudioEngine._rampVolume` because `audioplayers` doesn't expose a native fade. 10 steps × 20 ms = 200 ms ramp.
- **Camera mirroring**: the preview is flipped via `Transform.scale(-1, 1)`, and the `DetectionOverlay` `CustomPainter` flips X coords to match.
