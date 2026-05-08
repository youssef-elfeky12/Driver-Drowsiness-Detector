import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:opencv_dart/opencv_dart.dart' as cv;

/// Desktop camera capture using OpenCV's VideoCapture (DirectShow on Windows,
/// AVFoundation on macOS, V4L2 on Linux). Used because the Flutter `camera`
/// plugin's Windows implementation doesn't support frame streaming.
class DesktopCamera {
  cv.VideoCapture? _cap;
  int width = 0;
  int height = 0;

  bool get isOpen => _cap != null && _cap!.isOpened;

  Future<void> open({int index = 0, int width = 640, int height = 480}) async {
    final cap = cv.VideoCapture.fromDevice(index);
    if (!cap.isOpened) {
      throw StateError('Could not open webcam (index $index).');
    }
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width.toDouble());
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height.toDouble());
    _cap = cap;
    this.width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
    this.height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
  }

  /// Read the next frame as a BGR cv.Mat. Caller MUST dispose() it.
  cv.Mat? readMat() {
    if (_cap == null) return null;
    final (ok, frame) = _cap!.read();
    if (!ok || frame.isEmpty) {
      frame.dispose();
      return null;
    }
    return frame;
  }

  /// Convert a BGR Mat to a Flutter ui.Image for rendering.
  static Future<ui.Image> matToUiImage(cv.Mat bgr) async {
    final rgba = cv.cvtColor(bgr, cv.COLOR_BGR2RGBA);
    final bytes = Uint8List.fromList(rgba.data);
    final w = rgba.cols;
    final h = rgba.rows;
    rgba.dispose();
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      bytes,
      w,
      h,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );
    return completer.future;
  }

  void close() {
    _cap?.release();
    _cap?.dispose();
    _cap = null;
  }
}
