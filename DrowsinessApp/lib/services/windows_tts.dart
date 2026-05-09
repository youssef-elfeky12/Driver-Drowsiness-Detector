import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

/// Render text → boosted WAV using Windows' built-in SAPI (no extra deps).
/// flutter_tts on Windows caps volume at 1.0; we work around that by writing
/// our own WAV through PowerShell's System.Speech and then amplifying the
/// 16-bit PCM samples directly.
class WindowsTts {
  /// Synthesize [text] to a WAV file with [gainDb] amplification applied,
  /// using a US female SAPI voice when available. Returns the file path on
  /// success, null on any failure (caller should fall back to flutter_tts).
  static Future<String?> synthesize({
    required String text,
    double gainDb = 8.0,
    int rate = -1, // SAPI rate range -10..10 (0 = default; negative = slower)
  }) async {
    if (!Platform.isWindows) return null;
    try {
      final tmp = await Directory.systemTemp.createTemp('drowsy_tts_');
      final rawPath = '${tmp.path}\\raw.wav';
      final outPath = '${tmp.path}\\loud.wav';

      // Single-quoted strings in PS — escape any embedded apostrophes.
      final safeText = text.replaceAll("'", "''");
      final ps = """
\$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
\$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
\$synth.Volume = 100
\$synth.Rate = $rate
foreach (\$v in \$synth.GetInstalledVoices()) {
    \$info = \$v.VoiceInfo
    if (\$info.Gender -eq 'Female' -and \$info.Culture.Name -like 'en*') {
        \$synth.SelectVoice(\$info.Name); break
    }
}
\$synth.SetOutputToWaveFile('$rawPath')
\$synth.Speak('$safeText')
\$synth.Dispose()
""";

      final r = await Process.run(
        'powershell',
        ['-NoProfile', '-NonInteractive', '-Command', ps],
      );
      if (r.exitCode != 0) {
        return null;
      }

      final raw = await File(rawPath).readAsBytes();
      final boosted = _amplifyWav16BitPcm(raw, gainDb: gainDb);
      await File(outPath).writeAsBytes(boosted);
      return outPath;
    } catch (_) {
      return null;
    }
  }

  /// Multiply every 16-bit PCM sample by 10^(gainDb/20). Clamps to int16 range.
  /// Assumes a standard PCM WAV — finds the 'data' chunk and processes its bytes.
  static Uint8List _amplifyWav16BitPcm(Uint8List wav, {required double gainDb}) {
    if (wav.length < 44) return wav;
    final factor = pow(10.0, gainDb / 20.0).toDouble();

    // Locate the 'data' subchunk header. Standard PCM WAVs put it at offset 36,
    // but some include extra fmt subchunk bytes — scan to be safe.
    int dataStart = -1;
    int dataLen = 0;
    for (var i = 12; i < wav.length - 8; i++) {
      if (wav[i] == 0x64 && // 'd'
          wav[i + 1] == 0x61 && // 'a'
          wav[i + 2] == 0x74 && // 't'
          wav[i + 3] == 0x61) {
        // little-endian uint32 chunk length
        dataLen = wav[i + 4] |
            (wav[i + 5] << 8) |
            (wav[i + 6] << 16) |
            (wav[i + 7] << 24);
        dataStart = i + 8;
        break;
      }
    }
    if (dataStart < 0) return wav;

    final out = Uint8List.fromList(wav);
    final end = (dataStart + dataLen).clamp(0, out.length);
    for (var i = dataStart; i + 1 < end; i += 2) {
      // little-endian signed 16-bit
      var s = out[i] | (out[i + 1] << 8);
      if (s & 0x8000 != 0) s -= 0x10000; // sign-extend
      var v = (s * factor).round();
      if (v > 32767) v = 32767;
      if (v < -32768) v = -32768;
      if (v < 0) v += 0x10000;
      out[i] = v & 0xFF;
      out[i + 1] = (v >> 8) & 0xFF;
    }
    return out;
  }
}
