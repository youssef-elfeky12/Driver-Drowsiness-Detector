import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:uuid/uuid.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../models/types.dart';
import '../services/alert_engine.dart';
import '../services/audio_engine.dart';
import '../services/detector.dart';
import '../services/settings.dart';
import '../services/storage.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/emergency_dialer.dart';
import '../widgets/overlays.dart';
import '../widgets/status_bar.dart';

class DrivePage extends StatefulWidget {
  const DrivePage({super.key});

  @override
  State<DrivePage> createState() => _DrivePageState();
}

class _DrivePageState extends State<DrivePage> {
  CameraController? _camera;
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
    setState(() => _loadingMsg = 'Permissions…');
    await Permission.camera.request();

    setState(() => _loadingMsg = 'Loading audio…');
    await _audio.init();

    setState(() => _loadingMsg = 'Loading model…');
    await _detector.init(onProgress: (m) => setState(() => _loadingMsg = m));

    setState(() => _loadingMsg = 'Camera…');
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
    _camera?.dispose();
    _engine?.stop();
    _audio.dispose();
    _detector.dispose();
    WakelockPlus.disable();
    super.dispose();
  }

  Future<void> _start() async {
    if (!_ready || _camera == null) return;
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
      onDialerDigit: (d, _) =>
          setState(() => _digitsTyped = _digitsTyped + d),
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

    await _camera!.startImageStream(_onFrame);
  }

  Future<void> _stop() async {
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
          // Camera preview (mirrored for selfie view)
          if (_camera != null && _camera!.value.isInitialized)
            Transform(
              alignment: Alignment.center,
              transform: Matrix4.identity()..scale(-1.0, 1.0, 1.0),
              child: CameraPreview(_camera!),
            )
          else
            const Center(child: CircularProgressIndicator()),

          // Detection boxes
          if (_running && _lastResult != null)
            LayoutBuilder(builder: (ctx, c) {
              return DetectionOverlay(
                result: _lastResult,
                previewSize: Size(c.maxWidth, c.maxHeight),
              );
            }),

          // Status bar
          if (_running)
            StatusBar(
                level: _level, closedMs: _closedMs, durationMs: tripDur),

          // Start screen
          if (!_running)
            Positioned.fill(
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      Colors.transparent,
                      AppColors.bg.withOpacity(0.85),
                      AppColors.bg,
                    ],
                  ),
                ),
                padding: const EdgeInsets.fromLTRB(24, 24, 24, 48),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    const Text(
                      'Drowsiness Detector',
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -0.5,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(_loadingMsg,
                        style: const TextStyle(
                            color: AppColors.muted, fontSize: 14)),
                    const SizedBox(height: 24),
                    SizedBox(
                      width: 280,
                      child: ElevatedButton.icon(
                        onPressed: _ready ? _start : null,
                        icon: const Icon(Icons.play_arrow_rounded),
                        label: const Text('Start Drive'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primary,
                          foregroundColor: Colors.white,
                          disabledBackgroundColor: AppColors.surface2,
                          disabledForegroundColor: AppColors.muted,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(18)),
                          textStyle: const TextStyle(
                              fontSize: 18, fontWeight: FontWeight.w800),
                        ),
                      ),
                    ),
                  ],
                ),
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
