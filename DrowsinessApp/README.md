# Drowsiness Detector — Flutter App

Phone-first drowsiness detection app. Watches the driver via the front camera and intervenes (visual + audio + haptic alerts → simulated emergency call) when fatigue is detected.

See **[DESIGN.md](DESIGN.md)** for the full spec.

## Prerequisites

1. **Flutter SDK 3.24+** — install per the official guide:
   <https://docs.flutter.dev/get-started/install/windows>
   - For Windows desktop testing also enable: `flutter config --enable-windows-desktop`
   - Verify: `flutter doctor` (resolve any reported issues for Windows + iOS).
2. **Python 3.10 / 3.11** — for the one-time TFLite conversion.
3. (Optional, for iPhone deploy later) a Mac with Xcode, or use a CI service.

## One-time setup

### 1. Convert the model
The app loads `assets/models/drowsiness_resnet50v2.tflite`. Build it once from the existing `.h5`:

```powershell
cd DrowsinessApp
pip install "tensorflow==2.15.*"
python scripts/convert_model.py
```

This produces `assets/models/drowsiness_resnet50v2.tflite`.

### 2. Bootstrap platform folders
This repo ships only the Flutter source (`lib/`, `assets/`, `pubspec.yaml`). The platform-specific folders (`windows/`, `ios/`, `android/`) are created by Flutter:

```powershell
flutter create .
```

`flutter create .` adds platform scaffolding without overwriting existing files.

### 3. Install Dart dependencies
```powershell
flutter pub get
```

## Run

### Windows desktop (your laptop)
```powershell
flutter run -d windows
```
First-run note: the Windows build of `opencv_dart` and `tflite_flutter` ships native DLLs that take ~30 s to compile on first launch.

### iPhone (when you have Mac access)
```powershell
flutter run -d <your-iphone-id>
```
Requires Xcode + Apple Developer cert (free tier works for sideloading to your own device for 7 days).

### Android (if you ever get a device)
```powershell
flutter run -d <android-id>
```

## Project layout
See [DESIGN.md §9](DESIGN.md#9-project-layout).

## Notes
- **All inference is on-device**: TFLite + OpenCV native, no network calls during a drive.
- **First launch downloads nothing.** All assets ship in the bundle.
- **iOS permissions:** the camera permission usage description goes in `ios/Runner/Info.plist`. After `flutter create .` adds the file, edit it to include:
  ```xml
  <key>NSCameraUsageDescription</key>
  <string>Used to detect driver drowsiness.</string>
  ```
- **Android permissions:** `android/app/src/main/AndroidManifest.xml` should include:
  ```xml
  <uses-permission android:name="android.permission.CAMERA" />
  <uses-permission android:name="android.permission.VIBRATE" />
  <uses-permission android:name="android.permission.WAKE_LOCK" />
  ```
