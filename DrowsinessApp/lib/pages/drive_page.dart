import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:permission_handler/permission_handler.dart';
import 'package:uuid/uuid.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../models/types.dart';
import '../services/alert_engine.dart';
import '../services/audio_engine.dart';
import '../services/desktop_camera.dart';
import '../services/detector.dart';
import '../services/settings.dart';
import '../services/storage.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/emergency_dialer.dart';
import '../widgets/overlays.dart';
import '../widgets/status_bar.dart';

bool get _useDesktopCamera =>
    Platform.isWindows || Platform.isLinux || Platform.isMacOS;

class DrivePage extends StatefulWidget {
  const DrivePage({super.key});

  @override
  State<DrivePage> createState() => _DrivePageState();
}

class _DrivePageState extends State<DrivePage> {
  // Mobile camera (iOS / Android)
  CameraController? _camera;
  // Desktop camera (Windows / macOS / Linux) — uses opencv_dart VideoCapture
  final DesktopCamera _desktopCam = DesktopCamera();
  Timer? _desktopFrameTimer;
  ui.Image? _desktopFrame;

  final Detector _detector = Detector();
  final AudioEngine _audio = AudioEngine();
  AlertEngine? _engine;

  String _loadingMsg = 'Initializing…';
  bool _ready = false;
  bool _running = false;
  bool _paused = false;
  bool _busy = false;

  AlertLevel _level = AlertLevel.none;
  int _closedMs = 0;
  int _countdown = 5;
  String _digitsTyped = '';
  String? _pressedDigit;
  Timer? _pressedDigitTimer;
  bool _callingActive = false;
  bool _callConnected = false;

  DetectionResult? _lastResult;
  AppSettings _settings = const AppSettings();

  int _tripStartedAtMs = 0;
  Timer? _uiTicker;
  final List<TripEvent> _events = [];
  int _longestClosedMs = 0;

  Timer? _connectedTimer;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    setState(() => _loadingMsg = 'Loading audio…');
    await _audio.init();

    setState(() => _loadingMsg = 'Loading model…');
    await _detector.init(onProgress: (m) => setState(() => _loadingMsg = m));

    setState(() => _loadingMsg = 'Camera…');
    if (_useDesktopCamera) {
      // OpenCV VideoCapture path — same backend as the existing Python script.
      await _desktopCam.open();
    } else {
      await Permission.camera.request();
      final cams = await availableCameras();
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      _camera = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
      );
      await _camera!.initialize();
    }

    _settings = await SettingsService.load();
    _audio.setMasterVolume(_settings.alarmVolume);

    setState(() {
      _loadingMsg = 'Ready';
      _ready = true;
    });
  }

  @override
  void dispose() {
    _uiTicker?.cancel();
    _connectedTimer?.cancel();
    _pressedDigitTimer?.cancel();
    _desktopFrameTimer?.cancel();
    _desktopCam.close();
    _desktopFrame?.dispose();
    _camera?.dispose();
    _engine?.stop();
    _audio.dispose();
    _detector.dispose();
    WakelockPlus.disable();
    super.dispose();
  }

  Future<void> _start() async {
    if (!_ready) return;
    if (!_useDesktopCamera && _camera == null) return;
    _settings = await SettingsService.load();
    _audio.setMasterVolume(_settings.alarmVolume);

    final engine = AlertEngine(
      audio: _audio,
      confidenceThreshold: _settings.confidenceThreshold,
      emergencyNumber: _settings.emergencyNumber,
      onLevel: (l) => setState(() => _level = l),
      onEvent: (e) => _events.add(e),
      onClosedMs: (ms) {
        setState(() => _closedMs = ms);
        if (ms > _longestClosedMs) _longestClosedMs = ms;
      },
      onDialerDigit: (d, _) {
        setState(() {
          _digitsTyped = _digitsTyped + d;
          _pressedDigit = d;
        });
        // Flash for 250ms then clear so the key returns to neutral.
        _pressedDigitTimer?.cancel();
        _pressedDigitTimer = Timer(const Duration(milliseconds: 250), () {
          if (mounted) setState(() => _pressedDigit = null);
        });
      },
      onCountdown: (s) => setState(() => _countdown = s),
      onCallingStarted: () => setState(() => _callingActive = true),
    );
    engine.start();
    _engine = engine;
    _events.clear();
    _longestClosedMs = 0;

    setState(() {
      _running = true;
      _paused = false;
      _digitsTyped = '';
      _callingActive = false;
      _callConnected = false;
      _level = AlertLevel.none;
      _tripStartedAtMs = DateTime.now().millisecondsSinceEpoch;
    });

    if (_settings.keepScreenOn) await WakelockPlus.enable();

    _uiTicker = Timer.periodic(const Duration(milliseconds: 500), (_) {
      if (mounted) setState(() {});
    });

    if (_useDesktopCamera) {
      _desktopFrameTimer =
          Timer.periodic(const Duration(milliseconds: 100), (_) => _onDesktopTick());
    } else {
      await _camera!.startImageStream(_onFrame);
    }
  }

  Future<void> _stop() async {
    _desktopFrameTimer?.cancel();
    _desktopFrameTimer = null;
    if (_camera != null && _camera!.value.isStreamingImages) {
      await _camera!.stopImageStream();
    }
    _uiTicker?.cancel();
    _uiTicker = null;
    await _engine?.stop();
    _engine = null;
    await WakelockPlus.disable();

    if (_events.isNotEmpty || _longestClosedMs > 0) {
      await StorageService.saveTrip(Trip(
        id: const Uuid().v4(),
        startedAt: _tripStartedAtMs,
        endedAt: DateTime.now().millisecondsSinceEpoch,
        events: List.of(_events),
        longestClosedMs: _longestClosedMs,
      ));
    }

    setState(() {
      _running = false;
      _paused = false;
      _level = AlertLevel.none;
      _closedMs = 0;
      _lastResult = null;
    });
  }

  Future<void> _onFrame(CameraImage image) async {
    if (_busy || _paused || _engine == null) return;
    _busy = true;
    try {
      final result = _detector.detect(image, _settings.confidenceThreshold);
      _lastResult = result;
      await _engine!.ingest(result);
      if (mounted) setState(() {});
    } catch (e) {
      // swallow per-frame errors
    } finally {
      _busy = false;
    }
  }

  // Decouple display (~10 fps) from inference (~4 fps) — when a face is
  // detected, full Haar+TFLite work is ~150-300ms which would freeze the UI
  // every tick. Drowsiness thresholds are ≥1.5s of sustained signal so 4 fps
  // inference is plenty.
  static const _inferenceIntervalMs = 250;
  int _lastInferenceMs = 0;

  Future<void> _onDesktopTick() async {
    if (_paused) return;
    cv.Mat? frame;
    try {
      frame = _desktopCam.readMat();
      if (frame == null) return;

      // 1) Always update the preview (fast: cvtColor + decodeImageFromPixels).
      final img = await DesktopCamera.matToUiImage(frame);
      final old = _desktopFrame;
      _desktopFrame = img;
      old?.dispose();

      // 2) Run detection only when the inference interval has elapsed AND a
      //    previous inference isn't still running (`_busy`).
      final now = DateTime.now().millisecondsSinceEpoch;
      if (!_busy &&
          _engine != null &&
          now - _lastInferenceMs >= _inferenceIntervalMs) {
        _busy = true;
        try {
          final result =
              _detector.detectMat(frame, _settings.confidenceThreshold);
          _lastResult = result;
          await _engine!.ingest(result);
          _lastInferenceMs = DateTime.now().millisecondsSinceEpoch;
        } finally {
          _busy = false;
        }
      }

      if (mounted) setState(() {});
    } catch (_) {
      // swallow per-frame errors
    } finally {
      frame?.dispose();
    }
  }

  Future<void> _dismiss() async {
    await _engine?.dismiss();
    setState(() {
      _digitsTyped = '';
      _callingActive = false;
      _callConnected = false;
    });
  }

  void _watchCallConnected() {
    _connectedTimer?.cancel();
    if (_level == AlertLevel.emergency) {
      _connectedTimer = Timer(const Duration(seconds: 25), () {
        if (mounted && _level == AlertLevel.emergency) {
          setState(() => _callConnected = true);
        }
      });
    } else {
      _callConnected = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    _watchCallConnected();
    final tripDur = _running
        ? DateTime.now().millisecondsSinceEpoch - _tripStartedAtMs
        : 0;

    return Container(
      color: AppColors.bg,
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview + detection boxes (only while running).
          // Wrap BOTH in the same Transform + FittedBox so the overlay shares
          // the cover-fit transform and the mirror flip — keeps boxes aligned
          // with the visible faces regardless of the window aspect ratio.
          if (_running && _useDesktopCamera && _desktopFrame != null)
            Positioned.fill(
              child: ClipRect(
                child: Transform(
                  alignment: Alignment.center,
                  transform: Matrix4.identity()..scale(-1.0, 1.0, 1.0),
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: _desktopFrame!.width.toDouble(),
                      height: _desktopFrame!.height.toDouble(),
                      child: Stack(
                        children: [
                          RawImage(
                            image: _desktopFrame,
                            width: _desktopFrame!.width.toDouble(),
                            height: _desktopFrame!.height.toDouble(),
                          ),
                          if (_lastResult != null)
                            DetectionOverlay(
                              result: _lastResult,
                              previewSize: Size(
                                _desktopFrame!.width.toDouble(),
                                _desktopFrame!.height.toDouble(),
                              ),
                              mirrored: false, // parent Transform already mirrors
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            )
          else if (_running &&
              !_useDesktopCamera &&
              _camera != null &&
              _camera!.value.isInitialized)
            Transform(
              alignment: Alignment.center,
              transform: Matrix4.identity()..scale(-1.0, 1.0, 1.0),
              child: Stack(
                children: [
                  CameraPreview(_camera!),
                  if (_lastResult != null)
                    LayoutBuilder(builder: (ctx, c) {
                      return DetectionOverlay(
                        result: _lastResult,
                        previewSize: Size(c.maxWidth, c.maxHeight),
                        mirrored: false,
                      );
                    }),
                ],
              ),
            ),

          // Status bar
          if (_running)
            StatusBar(
                level: _level, closedMs: _closedMs, durationMs: tripDur),

          // Landing screen — branded hero shown when not driving.
          if (!_running)
            Positioned.fill(
              child: _LandingHero(
                ready: _ready,
                loadingMsg: _loadingMsg,
                onStart: _start,
              ),
            ),

          // Bottom controls when running
          if (_running)
            Positioned(
              bottom: 16,
              left: 0,
              right: 0,
              child: Center(
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _ctrlButton(
                      label: _paused ? 'Resume' : 'Pause',
                      icon: _paused ? Icons.play_arrow : Icons.pause,
                      bg: AppColors.surface.withOpacity(0.95),
                      onTap: () => setState(() => _paused = !_paused),
                    ),
                    const SizedBox(width: 8),
                    _ctrlButton(
                      label: 'End',
                      icon: Icons.stop,
                      bg: AppColors.danger.withOpacity(0.9),
                      fg: Colors.white,
                      onTap: _stop,
                    ),
                  ],
                ),
              ),
            ),

          // Alert overlays
          if (_level == AlertLevel.drowsy)
            PullOverOverlay(onDismiss: _dismiss),
          if (_level == AlertLevel.warning)
            WarningOverlay(closedMs: _closedMs),
          if (_level == AlertLevel.critical)
            CriticalOverlay(
              countdown: _countdown,
              number: _settings.emergencyNumber,
              onCancel: _dismiss,
            ),
          if (_level == AlertLevel.emergency) ...[
            // Faint red flash under the dialer
            Positioned.fill(
              child: IgnorePointer(
                child: TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0, end: 1),
                  duration: const Duration(milliseconds: 660),
                  builder: (_, t, __) => Container(
                    color: AppColors.danger.withOpacity(0.18),
                  ),
                ),
              ),
            ),
            EmergencyDialer(
              digitsTyped: _digitsTyped,
              number: _settings.emergencyNumber,
              pressedDigit: _pressedDigit,
              callingActive: _callingActive,
              callConnected: _callConnected,
              onCancel: _dismiss,
            ),
          ],
        ],
      ),
    );
  }

  Widget _ctrlButton({
    required String label,
    required IconData icon,
    required Color bg,
    Color fg = AppColors.text,
    required VoidCallback onTap,
  }) {
    return Material(
      color: bg,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(14),
        child: Padding(
          padding:
              const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 18, color: fg),
              const SizedBox(width: 6),
              Text(label,
                  style: TextStyle(
                      color: fg,
                      fontWeight: FontWeight.w700,
                      fontSize: 14)),
            ],
          ),
        ),
      ),
    );
  }
}

/// Landing hero — shown on the Drive page when no trip is running.
/// Big animated logo + title + Start button, no live camera in the background.
class _LandingHero extends StatefulWidget {
  final bool ready;
  final String loadingMsg;
  final VoidCallback onStart;
  const _LandingHero({
    required this.ready,
    required this.loadingMsg,
    required this.onStart,
  });

  @override
  State<_LandingHero> createState() => _LandingHeroState();
}

class _LandingHeroState extends State<_LandingHero>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: RadialGradient(
          center: Alignment(0, -0.3),
          radius: 1.1,
          colors: [
            Color(0xFF1A2330),
            Color(0xFF0B0F14),
          ],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(28, 64, 28, 36),
      child: Column(
        children: [
          const Spacer(flex: 2),

          // Animated logo: stacked steering wheel + eye glyph
          AnimatedBuilder(
            animation: _ctrl,
            builder: (_, __) {
              final t = _ctrl.value;
              return Container(
                width: 156,
                height: 156,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      AppColors.primary,
                      Color.lerp(AppColors.primary, AppColors.amber, t)!,
                    ],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.primary.withValues(alpha: 0.35 + 0.15 * t),
                      blurRadius: 40,
                      spreadRadius: 4,
                    ),
                  ],
                ),
                child: const Center(
                  child: Icon(
                    Icons.remove_red_eye_outlined,
                    size: 80,
                    color: Colors.white,
                  ),
                ),
              );
            },
          ),

          const SizedBox(height: 28),

          // Title + tagline
          ShaderMask(
            shaderCallback: (rect) => const LinearGradient(
              colors: [Colors.white, Color(0xFFB6D2FF)],
            ).createShader(rect),
            child: const Text(
              'DROWSY',
              style: TextStyle(
                fontSize: 44,
                fontWeight: FontWeight.w900,
                letterSpacing: 6,
                color: Colors.white,
              ),
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Eyes on the road. Always.',
            style: TextStyle(
              color: AppColors.muted,
              fontSize: 14,
              fontWeight: FontWeight.w500,
              letterSpacing: 0.4,
            ),
          ),

          const Spacer(flex: 3),

          // Status pill
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: Row(
              key: ValueKey(widget.ready),
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: widget.ready ? AppColors.ok : AppColors.amber,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  widget.loadingMsg,
                  style: const TextStyle(
                    color: AppColors.muted,
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 18),

          // Start button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: widget.ready ? widget.onStart : null,
              icon: const Icon(Icons.play_arrow_rounded, size: 26),
              label: const Text('Start Drive'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
                disabledBackgroundColor: AppColors.surface2,
                disabledForegroundColor: AppColors.muted,
                padding: const EdgeInsets.symmetric(vertical: 18),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20)),
                textStyle: const TextStyle(
                    fontSize: 18, fontWeight: FontWeight.w800, letterSpacing: 0.5),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
