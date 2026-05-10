# Driver Drowsiness Detector — App Design

A phone-first drowsiness detection app. Designed to be mounted in front of the driver, watch them through the front-facing camera, and intervene when fatigue is detected.

This document is the high-level specification. For per-class implementation detail, read the source.

---

## 1. Platform & Stack

**Framework:** **Flutter** (Dart). Single codebase that compiles to:

- **iOS** — the target deployment for real driving use.
- **Android** — same codebase, untested but supported by all packages used.
- **Windows desktop** — used for daily development and laptop testing.

The app is portrait-locked and styled phone-first (mocked phone bezel on desktop) so the desktop build looks and behaves like the phone build.

**Inference:** **ResNet50V2** transfer-learned on the 6-class drowsiness task, converted from `Models/drowsiness_resnet50v2.h5` to a `.tflite` model via `scripts/convert_model.py`, loaded through `tflite_flutter`. Runs entirely on-device — no network calls during a drive. ResNet50V2 has no internal normalization layer, so the detector applies `preprocess_input` manually (BGR uint8 → `x/127.5 - 1.0`) before each forward pass.

**Detection (face / eyes):** `opencv_dart` runs **YuNet** (`face_detection_yunet_2023mar.onnx`, ~230 KB) for both face localization and eye landmarking. YuNet replaces the two Haar cascades the pipeline used to rely on. Two failure modes pushed the switch:

- `haarcascade_frontalface_default.xml` **drops the face when the head tilts down** — exactly the pose that signals drowsiness. Haar is trained on upright frontal faces and gives up once the visible feature layout drifts off-axis. YuNet is a small CNN trained on diverse poses and keeps producing a bounding box through the tilt, so the classifier still gets a crop and can make the front/down call itself instead of the detector silently swallowing it.
- `haarcascade_eye.xml` **drops the eye when it closes** — and "is the eye closed" is the central question the app is trying to answer. The cascade was trained on open eyes; the dark iris/sclera contrast it keys on disappears at lid closure. YuNet returns 5 facial landmarks (including right- and left-eye points) as part of the same detection pass, and those points are inferred from face geometry, so the eye crops keep flowing whether the eye is open or closed.

The face crop comes straight from YuNet's bbox. Each eye crop is a `face_w * 0.30` square centered on the corresponding eye landmark, clamped to the face rect. Both crops are fed to the model.

**Camera:**
- Mobile (iOS/Android): the official `camera` Flutter plugin.
- Windows desktop: OpenCV's `VideoCapture` directly (the Flutter `camera` plugin doesn't support frame streaming on Windows).

**Audio:** `audioplayers` runs five concurrent tracks (buzz, pull-over loop, siren loop, dial-tone, ringback, dispatcher pickup). Volume ducking is implemented as a brief ramp.

**Live TTS:** `flutter_tts` (Windows SAPI on desktop, native on iOS/Android) speaks the variable address tail of the dispatcher message, in a US female voice. The fixed prefix is pre-recorded as an asset to keep the alarm path snappy.

**Location:** `geolocator` resolves GPS coordinates; OpenStreetMap Nominatim (free HTTP API) reverse-geocodes them to a street address. Falls back gracefully to coordinates, then to "unknown location", if anything fails.

**Storage:** `sqflite` for trip history, `shared_preferences` for settings.

**System bits:** `vibration` for haptics, `wakelock_plus` to keep the screen on while driving, `permission_handler` for camera/location on mobile.

---

## 2. What the model does

A single 6-class softmax model trained on 224×224 BGR face/eye/head crops. The six classes are:

| Index | Class | What it indicates |
|---|---|---|
| 0 | yawn | mouth wide open / yawning |
| 1 | no_yawn | mouth closed |
| 2 | Closed | eye closed |
| 3 | Open | eye open |
| 4 | front | head facing forward |
| 5 | down | head dropped/tilted down |

The model is run twice per detected face:

1. On the **face crop** — we read it as **two independent binaries**: `yawn vs no_yawn` and `front vs down`. (Lumping all four into one argmax used to hide yawn signals because the front-pose probability dominated the softmax most of the time.)
2. On each **eye crop** — `Closed vs Open`.

Predictions below a configurable confidence threshold are treated as "uncertain" and don't trigger anything.

---

## 3. What's drawn on the live preview

- **Green rectangle** around the detected face.
- Above the face box, **two stacked tags**: yawn / no_yawn (top), front / down (bottom). Tags turn amber when their alarm side is firing.
- **Blue rectangle** (eye open) or **red rectangle** (eye closed) around each detected eye, with its class + confidence.
- The whole preview is mirrored horizontally (selfie view); labels are counter-flipped so the text reads correctly.

---

## 4. Alert system — what happens when drowsiness is detected

Two independent state machines run in parallel.

### 4.1 Focus reminders — yawning or head dropping

- A yawn or head-down event is registered after a brief sustained signal (debounced, with a cooldown so one yawn doesn't register five times).
- Each registered event plays a short buzz sound and a haptic pulse.
- If **3 events accumulate within 30 seconds**, the app enters **Drowsy** state:
  - Full-screen amber **PULL OVER** card with a single "I'm OK" button.
  - `PULLOVER.mp3` loops until dismissed.

### 4.2 Critical — eyes closed

A single timer measures continuous closed-eye time. The timer resets only after eyes have been continuously open for ~1 second (hysteresis prevents one stray frame from cancelling an active alarm).

| Continuous closed time | What happens |
|---|---|
| 0 – 0.8 s | normal blink, ignored |
| 0.8 – 5 s | small "Eyes closing" pill at the top of the screen |
| 5 – 10 s | full-screen amber **EYES CLOSED — WAKE UP** overlay; `PULLOVER.mp3` loops |
| 10 – 15 s | red flashing screen, big **EMERGENCY — CALLING SOON** with a 5→0 countdown; siren joins the pull-over loop |
| ≥ 15 s | **emergency dialer** flow (next section) |

After 10 s the alarm **locks**: opening your eyes no longer cancels it — only the manual Cancel/End button can.

### 4.3 The emergency dialer flow (≥ 15 s eyes closed)

A GTA-style mini phone slides into the bottom-right corner. The siren ducks to ~25% so the dispatcher chain is audible:

1. **Dial tones** play (the 3 keypress sounds for "112"). Each digit lights up briefly on the keypad in sync with its tone, then fades back.
2. **Ringback** (`calling.mp3`) plays exactly **3 times**.
3. **Dispatcher pickup** — a louder "911, what's your emergency" clip.
4. **Dispatcher prefix** — a pre-recorded fixed phrase ("A driver has become unresponsive. Please send help to,").
5. **Live TTS** speaks the address — resolved from GPS via reverse geocoding while the driver was still driving, so the address is ready when the alarm fires. Falls back to spoken coordinates, then to "unknown location".
6. **Siren ramps back** to full volume.

The emergency number, alarm volume, confidence threshold, and "keep screen on" are all configurable in the Settings screen.

---

## 5. Screens & navigation

Three screens behind a bottom tab bar (`IndexedStack` so they all stay alive — switching back to Drive doesn't re-bootstrap the camera/model):

- **Drive** — the camera feed with detection overlays + alert UI. Before pressing Start, shows a branded landing screen (animated logo, title, status pill) instead of the live camera.
- **History** — past trips with event counts, longest closed-eye streak, and a date/duration header. Stored in local SQLite.
- **Settings** — confidence threshold, emergency number, alarm volume, keep-screen-on toggle, clear history.

---

## 6. Cross-platform notes

The app is built mobile-first; Windows desktop is an explicit testing target with two adaptations:

- **Camera path** branches by platform (`opencv_dart` `VideoCapture` on desktop, `camera` plugin on mobile). The detector accepts a raw OpenCV Mat, so both paths feed it the same input.
- **Permissions UX** is mobile-only. On Windows, location and camera access are granted at the OS level (Settings → Privacy).

Once you have a Mac available, deploying to your iPhone is `flutter run -d <device>` — no code changes required. Same for Android if a device shows up.

---

## 7. Project layout (top-level)

```
DrowsinessApp/
  DESIGN.md              ← this file
  README.md              ← run instructions
  pubspec.yaml
  scripts/
    convert_model.py     ← .h5 → .tflite (one-time)
  assets/
    sounds/              ← buzz, pullover, siren, dial tones, ringback,
                           911 accept, dispatch prefix
    models/              ← YuNet ONNX + generated .tflite (.tflite gitignored)
  lib/
    main.dart, theme.dart
    models/types.dart
    pages/               ← drive, history, settings
    services/            ← detector, alert engine, audio, location,
                           desktop camera, settings, storage
    widgets/             ← bottom tabs, status bar, detection overlay,
                           alert overlays, emergency dialer
  windows/, ios/, android/, macos/, linux/   ← generated by `flutter create .`
```
