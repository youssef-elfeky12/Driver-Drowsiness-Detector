import 'dart:async';
import 'package:vibration/vibration.dart';

import '../models/types.dart';
import 'audio_engine.dart';

typedef LevelCallback = void Function(AlertLevel level);
typedef EventCallback = void Function(TripEvent event);
typedef MsCallback = void Function(int ms);
typedef DigitCallback = void Function(String digit, int index);
typedef IntCallback = void Function(int value);

class AlertEngine {
  static const _sustainMs = 500;
  static const _cooldownMs = 3000;
  static const _drowsyWindowMs = 30000;
  static const _drowsyThreshold = 3;
  static const _calibrationMs = 3000;
  static const _faceLostToleranceMs = 3000;

  static const _blinkIgnoreMs = 800;
  static const _warningAtMs = 5000;
  static const _criticalAtMs = 10000;
  static const _emergencyAtMs = 15000;
  static const _eyeOpenResetMs = 1000;

  final AudioEngine audio;
  double confidenceThreshold;
  String emergencyNumber;

  final LevelCallback onLevel;
  final EventCallback onEvent;
  final MsCallback onClosedMs;
  final DigitCallback onDialerDigit;
  final IntCallback onCountdown;
  final void Function() onCallingStarted;

  AlertLevel _level = AlertLevel.none;
  AlertLevel get level => _level;

  int _graceUntilMs = 0;

  // Track A
  int _yawnSustainStart = 0;
  int _headDownSustainStart = 0;
  int _lastYawnEventAt = 0;
  int _lastHeadDownEventAt = 0;
  final List<TripEvent> _events = [];

  // Track B
  int _closedSince = 0;
  int _openSince = 0;
  int _faceLostSince = 0;
  bool _inEmergencyFlow = false;

  Timer? _countdownTimer;

  AlertEngine({
    required this.audio,
    required this.confidenceThreshold,
    required this.emergencyNumber,
    required this.onLevel,
    required this.onEvent,
    required this.onClosedMs,
    required this.onDialerDigit,
    required this.onCountdown,
    required this.onCallingStarted,
  });

  void start() {
    final now = DateTime.now().millisecondsSinceEpoch;
    _graceUntilMs = now + _calibrationMs;
    _events.clear();
    _level = AlertLevel.none;
    _closedSince = 0;
    _openSince = 0;
    _inEmergencyFlow = false;
  }

  Future<void> stop() async {
    _countdownTimer?.cancel();
    _countdownTimer = null;
    await audio.stopAll();
    _setLevel(AlertLevel.none);
    _inEmergencyFlow = false;
  }

  Future<void> dismiss() async {
    _countdownTimer?.cancel();
    _countdownTimer = null;
    await audio.stopAll();
    _events.clear();
    _closedSince = 0;
    _openSince = 0;
    _inEmergencyFlow = false;
    _graceUntilMs = DateTime.now().millisecondsSinceEpoch + 10000;
    _setLevel(AlertLevel.none);
  }

  Future<void> ingest(DetectionResult result) async {
    final now = result.tsMs;
    final inGrace = now < _graceUntilMs;

    if (result.faceLost) {
      if (_faceLostSince == 0) _faceLostSince = now;
      if (now - _faceLostSince > _faceLostToleranceMs) onClosedMs(0);
      return;
    }
    _faceLostSince = 0;

    if (inGrace) {
      onClosedMs(0);
      return;
    }

    final face = result.faces.first;

    // ---- Track A — yawn and head-down are INDEPENDENT signals.
    // Use the per-signal binary confidence, not the combined `face.conf`,
    // so a yawning forward-facing user still registers as yawn.
    final yawnPasses = face.isYawn && face.yawnConf >= confidenceThreshold;
    final headDownPasses =
        face.isHeadDown && face.headPoseConf >= confidenceThreshold;

    if (yawnPasses) {
      if (_yawnSustainStart == 0) _yawnSustainStart = now;
      if (now - _yawnSustainStart >= _sustainMs &&
          now - _lastYawnEventAt >= _cooldownMs) {
        _lastYawnEventAt = now;
        await _registerEvent(TripEvent(now, EventType.yawn));
      }
    } else {
      _yawnSustainStart = 0;
    }

    if (headDownPasses) {
      if (_headDownSustainStart == 0) _headDownSustainStart = now;
      if (now - _headDownSustainStart >= _sustainMs &&
          now - _lastHeadDownEventAt >= _cooldownMs) {
        _lastHeadDownEventAt = now;
        await _registerEvent(TripEvent(now, EventType.headDown));
      }
    } else {
      _headDownSustainStart = 0;
    }

    _events.removeWhere((e) =>
        now - e.ts > _drowsyWindowMs ||
        (e.type != EventType.yawn && e.type != EventType.headDown));

    // ---- Track B ----
    final closed = _classifyEyesClosed(face);
    if (closed) {
      if (_closedSince == 0) _closedSince = now;
      _openSince = 0;
    } else {
      if (_openSince == 0) _openSince = now;
      if (now - _openSince >= _eyeOpenResetMs) {
        if (_closedSince != 0) {
          // Lock past 10s: once we've escalated to CRITICAL or EMERGENCY, only
          // a manual Cancel/dismiss can stop the alarm. Eye-open won't reset.
          final locked = _level == AlertLevel.critical ||
              _level == AlertLevel.emergency;
          if (!locked) {
            _countdownTimer?.cancel();
            _countdownTimer = null;
            await audio.stopAll();
            _inEmergencyFlow = false;
            _closedSince = 0;
          }
        } else {
          _closedSince = 0;
        }
      }
    }

    final closedMs = _closedSince == 0 ? 0 : now - _closedSince;
    onClosedMs(closedMs);
    await _updateLevel(closedMs);
  }

  bool _classifyEyesClosed(FacePrediction face) {
    final eyes = face.eyes.where((e) => e.conf >= confidenceThreshold).toList();
    if (eyes.isEmpty) return true; // face but no confident eyes → likely closed
    return eyes.every((e) => e.eyeClass == EyeClass.closed);
  }

  Future<void> _registerEvent(TripEvent ev) async {
    _events.add(ev);
    onEvent(ev);
    await audio.playBuzz();
    if (await Vibration.hasVibrator() ?? false) {
      Vibration.vibrate(duration: 300);
    }

    if (_events.length >= _drowsyThreshold &&
        _level != AlertLevel.critical &&
        _level != AlertLevel.emergency) {
      onEvent(TripEvent(ev.ts, EventType.drowsy));
      _setLevel(AlertLevel.drowsy);
      await audio.startPullover();
    }
  }

  Future<void> _updateLevel(int closedMs) async {
    if (closedMs >= _emergencyAtMs) {
      if (_level != AlertLevel.emergency) {
        _setLevel(AlertLevel.emergency);
        onEvent(TripEvent(
            DateTime.now().millisecondsSinceEpoch, EventType.emergency));
        await _startEmergencyFlow();
      }
      return;
    }
    if (closedMs >= _criticalAtMs) {
      if (_level != AlertLevel.critical) {
        _setLevel(AlertLevel.critical);
        onEvent(TripEvent(
            DateTime.now().millisecondsSinceEpoch, EventType.critical));
        await audio.startSiren();
        _startCountdown();
      }
      return;
    }
    if (closedMs >= _warningAtMs) {
      if (_level != AlertLevel.warning) {
        _setLevel(AlertLevel.warning);
        await audio.startPullover();
      }
      return;
    }
    if (closedMs >= _blinkIgnoreMs) {
      if (_level == AlertLevel.none) _setLevel(AlertLevel.eyesClosing);
      return;
    }
    if (closedMs == 0 &&
        (_level == AlertLevel.warning ||
            _level == AlertLevel.critical ||
            _level == AlertLevel.emergency ||
            _level == AlertLevel.eyesClosing)) {
      _countdownTimer?.cancel();
      _countdownTimer = null;
      await audio.stopAll();
      _inEmergencyFlow = false;
      if (_events.length >= _drowsyThreshold) {
        _setLevel(AlertLevel.drowsy);
        await audio.startPullover();
      } else {
        _setLevel(AlertLevel.none);
      }
    }
  }

  void _setLevel(AlertLevel l) {
    if (_level == l) return;
    _level = l;
    onLevel(l);
  }

  void _startCountdown() {
    var n = 5;
    onCountdown(n);
    _countdownTimer?.cancel();
    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      n -= 1;
      onCountdown(n < 0 ? 0 : n);
      if (n <= 0) {
        t.cancel();
        _countdownTimer = null;
      }
    });
  }

  Future<void> _startEmergencyFlow() async {
    if (_inEmergencyFlow) return;
    _inEmergencyFlow = true;
    await audio.stopPullover();
    await audio.duckSiren();

    final offsets = await audio.playDialer();
    final number = emergencyNumber;
    for (var i = 0; i < offsets.length && i < number.length; i++) {
      Future.delayed(Duration(milliseconds: offsets[i]), () {
        onDialerDigit(number[i], i);
      });
    }

    audio.onDialerEnd(() async {
      onCallingStarted();
      await audio.playCallingTimes(3);
    });
  }
}
