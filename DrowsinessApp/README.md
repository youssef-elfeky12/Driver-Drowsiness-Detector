# Drowsiness Detector — App

Phone-first PWA that watches the driver through the front camera and intervenes when fatigue is detected.

See **[DESIGN.md](DESIGN.md)** for the full spec.

## One-time setup

### 1. Install Node deps
```powershell
cd DrowsinessApp
npm install
```

### 2. Convert the Keras model to TF.js (one-time)
The app loads `public/models/efficientnet_b0/`. To produce it from the existing `.h5`:

```powershell
pip install tensorflow tensorflowjs
python scripts/convert_model.py
```

This calls `tensorflowjs_converter` on `../Models/drowsiness_efficientnet_b0.h5` and writes the converted graph model into `public/models/efficientnet_b0/`.

> If conversion errors out, the most common cause is a mismatched `tensorflow` / `tensorflowjs` version. The pair `tensorflow==2.15.*` + `tensorflowjs==4.20.*` is known to work.

## Run (laptop, dev)
```powershell
npm run dev
```
Open <http://localhost:5173>.

## Run (iPhone — same Wi-Fi as laptop)

1. Find your laptop's LAN IP: `ipconfig` → look for `IPv4 Address` (e.g. `192.168.1.42`).
2. On iPhone Safari: open `http://192.168.1.42:5173`.
3. Tap Share → **Add to Home Screen** → app installs as standalone (fullscreen, home-screen icon).

> ⚠️ iOS Safari requires HTTPS for camera access on non-localhost origins. For phone testing you'll need either:
> - Run `npm run dev -- --https` after generating a self-signed cert, **or**
> - Build (`npm run build`) and serve via a tool like `caddy` with a local cert, **or**
> - Use a tunnel like `ngrok http 5173` (gives you an HTTPS URL).
>
> The simplest option for thesis demos: `ngrok http 5173`.

## Run (production-style preview)
```powershell
npm run build
npm run preview
```

## Project layout
See `DESIGN.md §9`.

## Notes
- All inference is on-device (TF.js + OpenCV.js in the browser). No backend, no network calls during a drive.
- The first load downloads the model (~16 MB). After that the service worker caches everything and the app works offline.
