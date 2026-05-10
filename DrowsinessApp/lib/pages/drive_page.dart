import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
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
import '../services/inference_worker.dart';
import '../services/location_service.dart';
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

  final InferenceWorker _worker = InferenceWorker();
  final AudioEngine _audio = AudioEngine();
  AlertEngine? _engine;

  String _loadingMsg = 'Initializing…';
  bool _ready = false;
  bool _running = false;
  bool _paused = false;
  // Two independent in-flight flags — the cheap detect path and the heavy
  // classify path each have their own worker isolate, so neither blocks
  // the other and they each drop their own frames on busy.
  bool _detectBusy = false;
  bool _classifyBusy = false;

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
    await _worker.init(onProgress: (m) => setState(() => _loadingMsg = m));

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
    _locationTimer?.cancel();
    _desktopFrameTimer?.cancel();
    _desktopCam.close();
    _desktopFrame?.dispose();
    _camera?.dispose();
    _engine?.stop();
    _audio.dispose();
    _worker.dispose();
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

    // Resolve location in the background so the dispatcher TTS has an address
    // ready by the time the emergency flow fires (~15 s in). We don't block
    // the start — emergency falls back to coords/"unknown" if this isn't done
    // by then. Re-resolves silently every 30 s.
    _refreshLocationLoop();

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
    if (_paused || _engine == null) return;
    // Convert on main isolate (CameraImage isn't sendable), then ship raw
    // BGR bytes to the worker isolates.
    final mat = Detector.matFromCameraImage(image);
    if (mat == null) return;
    final Uint8List bytes;
    final int w, h;
    try {
      bytes = Uint8List.fromList(mat.data);
      w = mat.cols;
      h = mat.rows;
    } finally {
      mat.dispose();
    }
    _dispatch(bytes, w, h);
  }

  /// Common dispatch path shared by the mobile [CameraImage] callback and
  /// the desktop frame timer. Fires a detection request every tick (cheap,
  /// drives the live box) and a classification request at the throttled
  /// rate (heavy, drives labels + alert engine).
  void _dispatch(Uint8List bytes, int w, int h) {
    if (!_detectBusy) {
      _detectBusy = true;
      // ignore: unawaited_futures
      _runDetect(bytes, w, h);
    }
    final now = DateTime.now().millisecondsSinceEpoch;
    if (!_classifyBusy && now - _lastInferenceMs >= _inferenceIntervalMs) {
      _classifyBusy = true;
      // ignore: unawaited_futures
      _runClassify(bytes, w, h);
    }
  }

  // Classification cadence. The detect path runs every tick regardless.
  static const _inferenceIntervalMs = 250;
  int _lastInferenceMs = 0;

  Timer? _locationTimer;
  Future<void> _refreshLocationLoop() async {
    Future<void> doFetch() async {
      try {
        final fix = await LocationService.getFix();
        if (fix != null) {
          _audio.setEmergencyLocationText(fix.toSpeech());
        }
      } catch (_) {/* ignore */}
    }
    await doFetch();
    _locationTimer?.cancel();
    _locationTimer = Timer.periodic(
        const Duration(seconds: 30), (_) => doFetch());
  }

  Future<void> _runDetect(Uint8List bytes, int w, int h) async {
    try {
      final detResult = await _worker.detect(
        bgrBytes: bytes,
        width: w,
        height: h,
      );
      if (!mounted || detResult == null) return;
      _lastResult = _mergeDetectIntoLast(detResult);
      if (mounted) setState(() {});
    } catch (_) {
      // swallow per-frame errors
    } finally {
      _detectBusy = false;
    }
  }

  Future<void> _runClassify(Uint8List bytes, int w, int h) async {
    try {
      final result = await _worker.classify(
        bgrBytes: bytes,
        width: w,
        height: h,
        threshold: _settings.confidenceThreshold,
      );
      if (!mounted || _engine == null || result == null) return;
      _lastResult = result;
      await _engine!.ingest(result);
      _lastInferenceMs = DateTime.now().millisecondsSinceEpoch;
      if (mounted) setState(() {});
    } catch (_) {
      // swallow per-frame errors
    } finally {
      _classifyBusy = false;
    }
  }

  /// Merge fresh box positions from a detect-only reply onto the
  /// last-known classification labels. If the face count changed (face
  /// entered/left frame), fall back to the detect result as-is — labels
  /// will be neutral until the next classify reply lands ~250 ms later.
  DetectionResult _mergeDetectIntoLast(DetectionResult det) {
    final prev = _lastResult;
    if (prev == null || prev.faces.length != det.faces.length) {
      return det;
    }
    final merged = <FacePrediction>[];
    for (var i = 0; i < det.faces.length; i++) {
      final freshBox = det.faces[i].box;
      final freshEyes = det.faces[i].eyes;
      final prevFace = prev.faces[i];
      // Update eye box positions from detect, keep eye class+conf from
      // the last classify reply when count matches.
      final mergedEyes = <EyePrediction>[];
      for (var j = 0; j < freshEyes.length; j++) {
        if (j < prevFace.eyes.length) {
          mergedEyes.add(prevFace.eyes[j].copyWith(box: freshEyes[j].box));
        } else {
          mergedEyes.add(freshEyes[j]);
        }
      }
      merged.add(prevFace.copyWith(box: freshBox, eyes: mergedEyes));
    }
    return DetectionResult(
      faces: merged,
      frameWidth: det.frameWidth,
      frameHeight: det.frameHeight,
      tsMs: det.tsMs,
      aligned: prev.aligned,
    );
  }

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

      // 2) Snapshot BGR bytes once and hand them to _dispatch, which
      //    decides per-tick whether to fire a detect (every tick) and/or
      //    a classify (throttled). The Mat itself can't cross an isolate
      //    boundary and is disposed in `finally`.
      if (_engine != null) {
        final bytes = Uint8List.fromList(frame.data);
        final fw = frame.cols;
        final fh = frame.rows;
        _dispatch(bytes, fw, fh);
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
