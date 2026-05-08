import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/types.dart';

/// Loads the TFLite model + Haar cascades, and runs detection on a frame.
///
/// Class index map (from notebook):
/// 0=yawn, 1=no_yawn, 2=Closed, 3=Open, 4=front, 5=down
class Detector {
  static const int imgSize = 224;
  static const _faceIndices = [0, 1, 4, 5];
  static const _eyeIndices = [2, 3];
  static const _faceLabels = [
    FaceClass.yawn,
    FaceClass.noYawn,
    FaceClass.front,
    FaceClass.down,
  ];

  Interpreter? _interp;
  cv.CascadeClassifier? _faceCascade;
  cv.CascadeClassifier? _eyeCascade;

  bool get isReady =>
      _interp != null && _faceCascade != null && _eyeCascade != null;

  Future<void> init({void Function(String)? onProgress}) async {
    onProgress?.call('Loading model…');
    _interp = await Interpreter.fromAsset(
      'assets/models/drowsiness_efficientnet_b0.tflite',
    );
    _interp!.allocateTensors();

    onProgress?.call('Loading cascades…');
    final faceXml = await _writeAssetToTemp(
        'assets/haarcascades/haarcascade_frontalface_default.xml',
        'face.xml');
    final eyeXml = await _writeAssetToTemp(
        'assets/haarcascades/haarcascade_eye.xml', 'eye.xml');
    _faceCascade = cv.CascadeClassifier.fromFile(faceXml);
    _eyeCascade = cv.CascadeClassifier.fromFile(eyeXml);

    onProgress?.call('Ready');
  }

  Future<String> _writeAssetToTemp(String asset, String name) async {
    final dir = await getTemporaryDirectory();
    final path = p.join(dir.path, name);
    final f = File(path);
    if (!await f.exists()) {
      final bytes = await rootBundle.load(asset);
      await f.writeAsBytes(bytes.buffer.asUint8List());
    }
    return path;
  }

  /// Run detection on a CameraImage frame (mobile path).
  DetectionResult detect(CameraImage image, double confThreshold) {
    if (!isReady) {
      return DetectionResult(
        faces: const [],
        frameWidth: image.width,
        frameHeight: image.height,
        tsMs: DateTime.now().millisecondsSinceEpoch,
      );
    }
    final mat = _matFromCameraImage(image);
    if (mat == null) {
      return DetectionResult(
        faces: const [],
        frameWidth: image.width,
        frameHeight: image.height,
        tsMs: DateTime.now().millisecondsSinceEpoch,
      );
    }
    try {
      return detectMat(mat, confThreshold);
    } finally {
      mat.dispose();
    }
  }

  /// Run detection on an already-decoded BGR cv.Mat (desktop path).
  /// Caller owns the Mat lifecycle.
  DetectionResult detectMat(cv.Mat mat, double confThreshold) {
    if (!isReady) {
      return DetectionResult(
        faces: const [],
        frameWidth: mat.cols,
        frameHeight: mat.rows,
        tsMs: DateTime.now().millisecondsSinceEpoch,
      );
    }
    try {
      final gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY);
      final faces =
          _faceCascade!.detectMultiScale(gray, scaleFactor: 1.3, minNeighbors: 5);
      gray.dispose();

      final out = <FacePrediction>[];
      for (final r in faces) {
        final fb = FaceBox(r.x, r.y, r.width, r.height);
        final probs = _classify(mat, fb);
        final faceProbs = _renormalized(probs, _faceIndices);
        var bestI = 0;
        for (var i = 1; i < faceProbs.length; i++) {
          if (faceProbs[i] > faceProbs[bestI]) bestI = i;
        }
        final faceClass = _faceLabels[bestI];
        final faceConf = faceProbs[bestI];

        // Eyes inside face ROI (upper 60%)
        final eyeRoi = mat.region(cv.Rect(r.x, r.y, r.width, r.height));
        final grayE = cv.cvtColor(eyeRoi, cv.COLOR_BGR2GRAY);
        final eyes =
            _eyeCascade!.detectMultiScale(grayE, scaleFactor: 1.1, minNeighbors: 5);
        grayE.dispose();
        eyeRoi.dispose();

        final eyePreds = <EyePrediction>[];
        // Sort eyes by area, take 2 largest, filter to upper 60% of face
        final filtered = eyes
            .where((e) => e.y + e.height / 2 < r.height * 0.6)
            .toList()
          ..sort((a, b) => (b.width * b.height) - (a.width * a.height));
        for (final e in filtered.take(2)) {
          final eb = FaceBox(r.x + e.x, r.y + e.y, e.width, e.height);
          final ep = _classify(mat, eb);
          final ev = _renormalized(ep, _eyeIndices);
          final ei = ev[0] > ev[1] ? 0 : 1;
          eyePreds.add(EyePrediction(
            eb,
            ei == 0 ? EyeClass.closed : EyeClass.open,
            ev[ei],
          ));
        }

        out.add(FacePrediction(fb, faceClass, faceConf, eyePreds));
      }

      return DetectionResult(
        faces: out,
        frameWidth: mat.cols,
        frameHeight: mat.rows,
        tsMs: DateTime.now().millisecondsSinceEpoch,
      );
    } catch (_) {
      return DetectionResult(
        faces: const [],
        frameWidth: mat.cols,
        frameHeight: mat.rows,
        tsMs: DateTime.now().millisecondsSinceEpoch,
      );
    }
  }

  cv.Mat? _matFromCameraImage(CameraImage img) {
    try {
      // BGRA8888 (Windows / iOS when configured): single plane, 4 channels.
      if (img.format.group == ImageFormatGroup.bgra8888) {
        final bytes = img.planes[0].bytes;
        final mat = cv.Mat.fromList(
          img.height,
          img.width,
          cv.MatType.CV_8UC4,
          bytes,
        );
        return cv.cvtColor(mat, cv.COLOR_BGRA2BGR);
      }
      // YUV420 (Android default): convert via opencv_dart helper.
      if (img.format.group == ImageFormatGroup.yuv420) {
        // opencv_dart can take an interleaved YUV NV21 buffer; we build it.
        final y = img.planes[0].bytes;
        final u = img.planes[1].bytes;
        final v = img.planes[2].bytes;
        final nv21 = Uint8List(y.length + u.length + v.length);
        nv21.setAll(0, y);
        for (var i = 0, j = y.length; i < u.length; i++) {
          nv21[j++] = v[i];
          nv21[j++] = u[i];
        }
        final yuvMat = cv.Mat.fromList(
          (img.height * 3 ~/ 2),
          img.width,
          cv.MatType.CV_8UC1,
          nv21,
        );
        return cv.cvtColor(yuvMat, cv.COLOR_YUV2BGR_NV21);
      }
    } catch (_) {}
    return null;
  }

  /// Run model on a 224x224 BGR crop, return raw 6-class softmax outputs.
  Float32List _classify(cv.Mat src, FaceBox box) {
    final roi = src.region(cv.Rect(box.x, box.y, box.w, box.h));
    final resized = cv.resize(roi, (imgSize, imgSize));
    final rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB);
    roi.dispose();
    resized.dispose();

    // Build a [1,224,224,3] float32 tensor with raw uint8 values
    // (EfficientNetB0 has internal preprocessing).
    final input = Float32List(1 * imgSize * imgSize * 3);
    final raw = rgb.data; // Uint8List of HxWx3
    for (var i = 0; i < raw.length; i++) {
      input[i] = raw[i].toDouble();
    }
    rgb.dispose();

    final output = List.filled(6, 0.0).reshape([1, 6]);
    _interp!.run(input.reshape([1, imgSize, imgSize, 3]), output);
    final probs = Float32List(6);
    for (var i = 0; i < 6; i++) {
      probs[i] = (output[0][i] as num).toDouble();
    }
    return probs;
  }

  List<double> _renormalized(Float32List probs, List<int> indices) {
    final subset = indices.map((i) => probs[i]).toList();
    final sum = subset.fold<double>(0, (a, b) => a + b);
    if (sum == 0) return subset.map((_) => 0.0).toList();
    return subset.map((v) => v / sum).toList();
  }

  void dispose() {
    _interp?.close();
    _faceCascade?.dispose();
    _eyeCascade?.dispose();
  }
}
