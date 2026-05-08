import 'dart:async';
import 'package:audioplayers/audioplayers.dart';

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
        // After ringback ends, restore siren to full volume.
        await _rampVolume(_siren, _master, 200);
      }
    });
  }

  Future<void> dispose() async {
    await _callingSub?.cancel();
    await _buzz.dispose();
    await _pullover.dispose();
    await _siren.dispose();
    await _dialer.dispose();
    await _calling.dispose();
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
    await _pullover.play(AssetSource('sounds/PULLOVER.mp3'), volume: _master);
  }

  Future<void> stopPullover() => _pullover.stop();

  Future<void> startSiren() async {
    if (_siren.state == PlayerState.playing) return;
    await _siren.setVolume(_master);
    await _siren.play(AssetSource('sounds/sirenLoop.mp3'), volume: _master);
  }

  Future<void> stopSiren() => _siren.stop();
  Future<void> duckSiren() => _rampVolume(_siren, _master * 0.25, 200);
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
