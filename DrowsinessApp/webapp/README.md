# Drowsiness Detector — Web App

A mobile-first **web** port of the Flutter drowsiness detector. Everything runs
**in the browser** — camera (getUserMedia), face/eye detection (YuNet via
onnxruntime-web), the 6-class drowsiness CNN (ONNX), alerts, the emergency
dialer simulation, spoken dispatcher (Web Speech), and GPS reverse-geocoding.
No backend, no database — trip history + settings live only on the device
(IndexedDB + localStorage). Deploys to **Vercel** as a static-ish Next.js app.

## How the ML pipeline maps to the browser

| Flutter (native)              | Web                                            |
| ----------------------------- | ---------------------------------------------- |
| `opencv_dart` YuNet ONNX      | `onnxruntime-web` + hand-written YuNet decoder |
| `tflite_flutter` ResNet50V2   | small CNN → ONNX, `onnxruntime-web`            |
| `camera` plugin               | `navigator.mediaDevices.getUserMedia`          |
| `audioplayers`                | `HTMLAudioElement`                             |
| `flutter_tts`                 | Web Speech API (`speechSynthesis`)             |
| `geolocator` + Nominatim      | Geolocation API + Nominatim                    |
| `sqflite` / `shared_prefs`    | IndexedDB / localStorage                       |

**Model:** `Models/drowsiness_eyeyawnnod.h5` (6-class, 145×145, BGR, `/255`)
converted to `public/models/classifier.onnx`. Class order:
`[yawn, no_yawn, Closed, Open, front, down]`.

## One-time model conversion

The two ONNX models are already committed under `public/models/`. To regenerate
them from the Keras source:

```bash
# from the repo root (parent of this folder)
pip install tensorflow==2.18.* tf2onnx onnx onnxruntime
python scripts/web_convert.py
```

This writes `webapp/public/models/classifier.onnx` (verified to match Keras to
< 1e-4) and copies the YuNet detector alongside it.

## Develop

```bash
cd webapp
npm install
npm run dev
```

Open the printed URL. **The camera only works over HTTPS or on `localhost`.**
On a phone, use `localhost` via a tunnel or deploy to Vercel (HTTPS).

## Deploy to Vercel

```bash
npm i -g vercel        # if needed
cd webapp
vercel                 # first run links/creates the project
vercel --prod          # production deploy
```

Or connect the repo in the Vercel dashboard and set **Root Directory =
`webapp`**. No environment variables or database are required. The
onnxruntime-web WebAssembly runtime is loaded from the jsDelivr CDN at runtime
(see `src/lib/detector.ts`), so nothing extra needs bundling.

## Notes & limits

- **HTTPS is mandatory** for the camera, geolocation, and wake lock — Vercel
  provides it automatically.
- First load downloads the YuNet (~230 KB) and classifier (~2 MB) models, then
  caches them.
- Inference runs on the main thread via WebAssembly (SIMD). On a typical phone
  this is ~8–15 fps, which is plenty for the alert timings.
- iOS Safari: audio/TTS unlock on the first Start tap (a user gesture is
  required). The Screen Wake Lock API is supported on iOS 16.4+.
- This is a **testing/demo** app — not a safety device.
