import 'dart:async';
import 'dart:io';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_tts/flutter_tts.dart';

import 'windows_tts.dart';

/// Multi-track engine. See DESIGN.md §4 for behavior.
///
/// Tracks:
///   buzz     — one-shot focus reminder
///   pullover — looped pull-over voice
///   siren    — looped critical siren (with volume ducking)
///   dialer   — one-shot 3-tone dialing for "112" (offsets at 0.071, 0.437, 0.701 s)
///   calling  — ringback played exactly 3 times then stops
class AudioEngine {
  final AudioPlayer _buzz = AudioPlayer();
  final AudioPlayer _pullover = AudioPlayer();
  final AudioPlayer _siren = AudioPlayer();
  final AudioPlayer _dialer = AudioPlayer();
  final AudioPlayer _calling = AudioPlayer();
  // Dispatcher pickup ("911, what's your emergency?") — plays once after the
  // ringback ends. Siren stays ducked until the whole dispatcher chain finishes.
  final AudioPlayer _accept = AudioPlayer();
  StreamSubscription<void>? _acceptSub;

  // Pre-recorded fixed prefix: "A driver has become unresponsive. Please send
  // help to ...". Generated externally (TTSforge) and shipped as an asset so we
  // don't pay live-TTS latency for this every emergency.
  final AudioPlayer _intro = AudioPlayer();
  StreamSubscription<void>? _introSub;

  // Live TTS for the variable location tail (Windows uses native SAPI).
  final FlutterTts _tts = FlutterTts();
  String? _emergencyLocationText;
  bool _ttsConfigured = false;

  static const dialDigitOffsetsMs = <int>[71, 437, 701];

  double _master = 1.0;
  int _callingPlaysLeft = 0;
  StreamSubscription<void>? _callingSub;

  Future<void> init() async {
    await _pullover.setReleaseMode(ReleaseMode.loop);
    await _siren.setReleaseMode(ReleaseMode.loop);
    await _buzz.setPlayerMode(PlayerMode.lowLatency);

    _callingSub = _calling.onPlayerComplete.listen((_) async {
      _callingPlaysLeft -= 1;
      if (_callingPlaysLeft > 0) {
        await _calling.play(AssetSource('sounds/calling.mp3'),
            volume: _master);
      } else {
        // Ringback finished → play the dispatcher pickup (911 accept).
        // Siren stays ducked across the full dispatcher chain.
        await _accept.play(AssetSource('sounds/911accept.mp3'),
            volume: _master);
      }
    });

    // After the dispatcher's "911 what's your emergency" → play the cached
    // prefix ("A driver has become unresponsive. Please send help to").
    _acceptSub = _accept.onPlayerComplete.listen((_) async {
      await _intro.play(AssetSource('sounds/dispatch_intro.mp3'),
          volume: _master);
    });

    // After the cached prefix → speak the location tail live, then ramp the
    // siren back up once TTS is done.
    _introSub = _intro.onPlayerComplete.listen((_) async {
      await _speakLocation();
      await _rampVolume(_siren, _master, 200);
    });

    await _configureTts();
  }

  Future<void> _configureTts() async {
    if (_ttsConfigured) return;
    try {
      await _tts.setLanguage('en-US');
      await _tts.setSpeechRate(0.45); // slower & clearer
      await _tts.setPitch(1.0);
      await _tts.setVolume(1.0);
      // Try to pick a US female voice. Voice catalogue varies per platform —
      // we look for one that contains "female" or the typical Windows SAPI
      // names ("Zira" / "Aria"). If none match, we accept the default.
      final voices = await _tts.getVoices;
      if (voices is List) {
        for (final v in voices) {
          if (v is Map) {
            final name = (v['name'] ?? '').toString().toLowerCase();
            final locale = (v['locale'] ?? '').toString().toLowerCase();
            final isUs = locale.startsWith('en-us') || locale.startsWith('en_us');
            final isFemale = name.contains('female') ||
                name.contains('zira') ||
                name.contains('aria') ||
                name.contains('jenny');
            if (isUs && isFemale) {
              await _tts.setVoice({
                'name': v['name'].toString(),
                'locale': v['locale'].toString(),
              });
              break;
            }
          }
        }
      }
      _ttsConfigured = true;
    } catch (_) {
      // best-effort; default voice is fine if any setter fails
    }
  }

  /// Set by the alert engine (or DrivePage) before the emergency flow starts.
  /// If null when the chain reaches TTS, we speak a generic fallback.
  void setEmergencyLocationText(String? text) {
    _emergencyLocationText = text;
  }

  // Dedicated player for the boosted-gain TTS WAV on Windows.
  final AudioPlayer _ttsPlayer = AudioPlayer();

  Future<void> _speakLocation() async {
    final text = _emergencyLocationText ?? 'an unknown location';

    // On Windows, render TTS to a WAV via SAPI then amplify so the dispatcher
    // location matches the volume of the pre-recorded prefix. flutter_tts
    // can't boost above 1.0 on its own.
    if (Platform.isWindows) {
      final path = await WindowsTts.synthesize(text: text, gainDb: 8.0);
      if (path != null) {
        final completer = Completer<void>();
        late StreamSubscription<void> sub;
        sub = _ttsPlayer.onPlayerComplete.listen((_) {
          sub.cancel();
          if (!completer.isCompleted) completer.complete();
        });
        try {
          await _ttsPlayer.play(DeviceFileSource(path), volume: _master);
        } catch (_) {
          if (!completer.isCompleted) completer.complete();
        }
        return completer.future.timeout(const Duration(seconds: 15),
            onTimeout: () {});
      }
      // PowerShell synth failed → fall through to flutter_tts.
    }

    final completer = Completer<void>();
    _tts.setCompletionHandler(() {
      if (!completer.isCompleted) completer.complete();
    });
    _tts.setErrorHandler((_) {
      if (!completer.isCompleted) completer.complete();
    });
    try {
      await _tts.awaitSpeakCompletion(true);
      await _tts.speak(text);
      if (!completer.isCompleted) completer.complete();
    } catch (_) {
      if (!completer.isCompleted) completer.complete();
    }
    return completer.future.timeout(const Duration(seconds: 12),
        onTimeout: () {});
  }

  Future<void> dispose() async {
    await _callingSub?.cancel();
    await _acceptSub?.cancel();
    await _introSub?.cancel();
    await _tts.stop();
    await _buzz.dispose();
    await _pullover.dispose();
    await _siren.dispose();
    await _dialer.dispose();
    await _calling.dispose();
    await _accept.dispose();
    await _intro.dispose();
    await _ttsPlayer.dispose();
  }

  void setMasterVolume(double v) {
    _master = v.clamp(0, 1);
  }

  // ----- public ops -----
  Future<void> playBuzz() async {
    await _buzz.stop();
    await _buzz.play(AssetSource('sounds/buzz.mp3'), volume: _master);
  }

  Future<void> startPullover() async {
    if (_pullover.state == PlayerState.playing) return;
    // Re-assert loop each start: audioplayers can drop the release mode after
    // a stop()/play() cycle on Windows.
    await _pullover.setReleaseMode(ReleaseMode.loop);
    await _pullover.play(AssetSource('sounds/PULLOVER.mp3'), volume: _master);
  }

  Future<void> stopPullover() => _pullover.stop();

  Future<void> startSiren() async {
    if (_siren.state == PlayerState.playing) return;
    await _siren.setReleaseMode(ReleaseMode.loop);
    await _siren.setVolume(_master);
    await _siren.play(AssetSource('sounds/sirenLoop.mp3'), volume: _master);
  }

  Future<void> stopSiren() => _siren.stop();
  // Aggressive duck during the dispatcher chain so the operator + TTS are
  // clearly audible over the alarm.
  Future<void> duckSiren() => _rampVolume(_siren, _master * 0.10, 200);
  Future<void> unduckSiren() => _rampVolume(_siren, _master, 200);

  /// Plays the dialer one-shot. Returns the digit-offset timeline (ms) so the UI
  /// can light up digits in sync.
  Future<List<int>> playDialer() async {
    await _dialer.stop();
    await _dialer.play(AssetSource('sounds/dialingButtons.m4a'),
        volume: _master);
    return dialDigitOffsetsMs;
  }

  Future<void> onDialerEnd(void Function() cb) async {
    late StreamSubscription<void> sub;
    sub = _dialer.onPlayerComplete.listen((_) {
      sub.cancel();
      cb();
    });
  }

  Future<void> playCallingTimes(int n) async {
    _callingPlaysLeft = n;
    await _calling.play(AssetSource('sounds/calling.mp3'), volume: _master);
  }

  Future<void> stopCalling() async {
    _callingPlaysLeft = 0;
    await _calling.stop();
  }

  Future<void> stopAll() async {
    await Future.wait([
      stopPullover(),
      stopSiren(),
      stopCalling(),
      _dialer.stop(),
      _accept.stop(),
      _intro.stop(),
      _tts.stop(),
    ]);
  }

  Future<void> _rampVolume(AudioPlayer p, double target, int ms) async {
    // audioplayers has no built-in fade; do it in N steps.
    const steps = 10;
    final stepMs = (ms / steps).round();
    final start = await _readVolume(p);
    for (var i = 1; i <= steps; i++) {
      final v = start + (target - start) * (i / steps);
      await p.setVolume(v.clamp(0, 1));
      await Future.delayed(Duration(milliseconds: stepMs));
    }
  }

  // audioplayers doesn't expose current volume; store last set externally.
  // We approximate by ramping from `_master` for the unduck case and
  // from `_master * 0.25` for duck.
  Future<double> _readVolume(AudioPlayer p) async {
    if (identical(p, _siren)) return _master; // best-effort
    return _master;
  }
}
