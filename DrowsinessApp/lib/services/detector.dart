import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/types.dart';

/// Loads the TFLite model + YuNet face detector, and runs detection on a
/// frame.
///
/// Both face *and* eye localization come from YuNet:
/// - face bbox (cols 0..3 of the [N,15] output)
/// - 5 landmarks (cols 4..13) — we use the right/left eye points to derive
///   eye crops directly. This replaces `haarcascade_eye.xml`, which lost
///   the eye whenever it closed (the cascade was trained on open eyes).
///   YuNet infers landmarks from face geometry, so closed eyes still yield
///   valid eye points.
///
/// Class index map (from notebook):
/// 0=yawn, 1=no_yawn, 2=Closed, 3=Open, 4=front, 5=down
class Detector {
  static const int imgSize = 224;
  static const _yawnIndices = [0, 1]; // yawn vs no_yawn (binary)
  static const _headIndices = [4, 5]; // front vs down (binary)
  static const _eyeIndices = [2, 3];

  // Eye crop side length as a fraction of face width. ~0.30 covers the eye
  // plus enough surrounding skin for the classifier to see lid context.
  static const double _eyeSideFrac = 0.30;

  Interpreter? _interp;
  cv.FaceDetectorYN? _faceDetector;
  (int, int)? _yunetInputSize;

  bool get isReady => _faceDetector != null;
  bool get canClassify => _interp != null;

  /// Asset-bundle init. Only safe on the root isolate (rootBundle requires
  /// the platform-channel binary messenger). Extracts the two model files
  /// to temp and forwards to [initFromPaths].
  Future<void> init({void Function(String)? onProgress}) async {
    onProgress?.call('Extracting models…');
    final tflitePath = await _writeAssetToTemp(
        'assets/models/drowsiness_resnet50v2.tflite',
        'drowsiness_resnet50v2.tflite');
    final yunetPath = await _writeAssetToTemp(
        'assets/models/face_detection_yunet_2023mar.onnx',
        'face_detection_yunet_2023mar.onnx');
    initFromPaths(
      tflitePath: tflitePath,
      yunetPath: yunetPath,
      onProgress: onProgress,
    );
  }

  /// Isolate-safe init. Both files must already exist on disk (the caller
  /// is responsible for extracting them from assets on the root isolate).
  ///
  /// Pass [enableClassification] = false on a detection-only worker — it
  /// avoids loading the ~92 MB TFLite model on isolates that will only
  /// ever call [detectFacesOnly].
  void initFromPaths({
    required String yunetPath,
    String? tflitePath,
    bool enableClassification = true,
    void Function(String)? onProgress,
  }) {
    if (enableClassification) {
      assert(tflitePath != null,
          'tflitePath is required when enableClassification is true');
      onProgress?.call('Loading model…');
      _interp = Interpreter.fromFile(File(tflitePath!));
      _interp!.allocateTensors();
    }

    onProgress?.call('Loading face detector…');
    // Input size is reset per-frame via setInputSize; the (320,320) here is
    // just a starting point.
    _faceDetector = cv.FaceDetectorYN.fromFile(
      yunetPath,
      '',
      (320, 320),
      scoreThreshold: 0.6,
      nmsThreshold: 0.3,
      topK: 50,
    );
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
    final mat = matFromCameraImage(image);
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

  /// YuNet-only detection. Returns boxes + neutral class labels; runs no
  /// classifier passes. Cheap (~10–15 ms on a typical CPU) and intended
  /// to be called every frame so the on-screen bounding box can track the
  /// face fluidly while the slower [detectMat] catches up at a fixed rate.
  DetectionResult detectFacesOnly(cv.Mat mat) {
    if (!isReady) {
      return DetectionResult(
        faces: const [],
        frameWidth: mat.cols,
        frameHeight: mat.rows,
        tsMs: DateTime.now().millisecondsSinceEpoch,
      );
    }
    try {
      final faces = _detectFacesYuNet(mat);
      final out = <FacePrediction>[];
      for (final f in faces) {
        final r = f.rect;
        final fb = FaceBox(r.x, r.y, r.width, r.height);
        final eyeBoxes = <EyePrediction>[];
        for (final pt in [f.rightEye, f.leftEye]) {
          final eb = _eyeBoxFromLandmark(pt, r, mat.cols, mat.rows);
          if (eb != null) {
            // Neutral eye state — overlay will keep showing the last
            // classified state via the merge in drive_page.
            eyeBoxes.add(EyePrediction(eb, EyeClass.open, 0));
          }
        }
        out.add(FacePrediction(
          fb,
          FaceClass.front,
          0,
          eyeBoxes,
          isYawn: false,
          yawnConf: 0,
          isHeadDown: false,
          headPoseConf: 0,
        ));
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

  /// Run detection on an already-decoded BGR cv.Mat (desktop path).
  /// Caller owns the Mat lifecycle.
  DetectionResult detectMat(cv.Mat mat, double confThreshold) {
    if (!isReady || !canClassify) {
      return DetectionResult(
        faces: const [],
        frameWidth: mat.cols,
        frameHeight: mat.rows,
        tsMs: DateTime.now().millisecondsSinceEpoch,
      );
    }
    try {
      final faces = _detectFacesYuNet(mat);

      final out = <FacePrediction>[];
      for (final f in faces) {
        final r = f.rect;
        final fb = FaceBox(r.x, r.y, r.width, r.height);
        final probs = _classify(mat, fb);

        // Two independent binaries instead of a single 4-way argmax.
        // The 4-way argmax was hiding yawn signals because front-pose
        // probability dominates the softmax most of the time.
        final yawnP = _renormalized(probs, _yawnIndices);   // [yawn, no_yawn]
        final headP = _renormalized(probs, _headIndices);   // [front, down]
        final isYawn = yawnP[0] > yawnP[1];
        final yawnConf = isYawn ? yawnP[0] : yawnP[1];
        final isHeadDown = headP[1] > headP[0];
        final headConf = isHeadDown ? headP[1] : headP[0];

        // Pick a single display label: surface the most "alarming" signal.
        FaceClass faceClass;
        double faceConf;
        if (isYawn) {
          faceClass = FaceClass.yawn;
          faceConf = yawnConf;
        } else if (isHeadDown) {
          faceClass = FaceClass.down;
          faceConf = headConf;
        } else {
          // both neutral — surface whichever binary is more confident
          faceClass = yawnConf >= headConf ? FaceClass.noYawn : FaceClass.front;
          faceConf = (yawnConf + headConf) / 2;
        }

        // Eye crops come straight from YuNet landmarks — no eye detector
        // pass, so closed eyes still produce valid crops.
        final eyePreds = <EyePrediction>[];
        for (final pt in [f.rightEye, f.leftEye]) {
          final eb = _eyeBoxFromLandmark(pt, r, mat.cols, mat.rows);
          if (eb == null) continue;
          final ep = _classify(mat, eb);
          final ev = _renormalized(ep, _eyeIndices);
          final ei = ev[0] > ev[1] ? 0 : 1;
          eyePreds.add(EyePrediction(
            eb,
            ei == 0 ? EyeClass.closed : EyeClass.open,
            ev[ei],
          ));
        }

        out.add(FacePrediction(
          fb,
          faceClass,
          faceConf,
          eyePreds,
          isYawn: isYawn,
          yawnConf: yawnConf,
          isHeadDown: isHeadDown,
          headPoseConf: headConf,
        ));
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

  /// Convert a [CameraImage] to a BGR cv.Mat. Returns null on unsupported
  /// formats. Caller owns the Mat. Exposed so the inference-worker layer
  /// can do the conversion on the root isolate (where the camera frame
  /// lives) before shipping raw bytes to a worker isolate.
  static cv.Mat? matFromCameraImage(CameraImage img) {
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
  ///
  /// IMPORTANT: training (notebook) used `cv2.imread` which returns BGR and
  /// never converted to RGB before fitting. The model therefore expects BGR
  /// inputs — do NOT cvtColor here.
  ///
  /// ResNet50V2 has no internal normalization layer, so we apply
  /// `preprocess_input` manually: scale [0, 255] uint8 to [-1, 1] float.
  Float32List _classify(cv.Mat src, FaceBox box) {
    final roi = src.region(cv.Rect(box.x, box.y, box.w, box.h));
    final resized = cv.resize(roi, (imgSize, imgSize));
    roi.dispose();

    // Build a [1,224,224,3] float32 tensor, applying ResNet50V2's
    // preprocess_input (tf mode: x/127.5 - 1.0).
    final input = Float32List(1 * imgSize * imgSize * 3);
    final raw = resized.data; // Uint8List of HxWx3 in BGR order
    for (var i = 0; i < raw.length; i++) {
      input[i] = raw[i] / 127.5 - 1.0;
    }
    resized.dispose();

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

  /// Run YuNet on a BGR Mat and return faces (bbox + eye landmarks) in image
  /// coordinates.
  ///
  /// YuNet's `detect` returns a [N,15] float Mat: 0..3 = x,y,w,h;
  /// 4..5 = right eye, 6..7 = left eye, 8..9 = nose, 10..13 = mouth corners,
  /// 14 = score. We keep bbox + eye points; nose/mouth are unused.
  List<_YuNetFace> _detectFacesYuNet(cv.Mat mat) {
    final size = (mat.cols, mat.rows);
    if (_yunetInputSize == null ||
        _yunetInputSize!.$1 != size.$1 ||
        _yunetInputSize!.$2 != size.$2) {
      _faceDetector!.setInputSize(size);
      _yunetInputSize = size;
    }
    final result = _faceDetector!.detect(mat);
    try {
      final faces = <_YuNetFace>[];
      for (var i = 0; i < result.rows; i++) {
        final x = result.atNum(i, 0).toInt();
        final y = result.atNum(i, 1).toInt();
        final w = result.atNum(i, 2).toInt();
        final h = result.atNum(i, 3).toInt();
        // Clamp to frame bounds — YuNet can return slightly negative
        // coords or boxes that spill past the edge on extreme poses.
        final x0 = x.clamp(0, mat.cols - 1);
        final y0 = y.clamp(0, mat.rows - 1);
        final x1 = (x + w).clamp(0, mat.cols);
        final y1 = (y + h).clamp(0, mat.rows);
        final cw = x1 - x0;
        final ch = y1 - y0;
        if (cw <= 0 || ch <= 0) continue;
        faces.add(_YuNetFace(
          rect: cv.Rect(x0, y0, cw, ch),
          rightEye: (
            result.atNum(i, 4).toDouble(),
            result.atNum(i, 5).toDouble(),
          ),
          leftEye: (
            result.atNum(i, 6).toDouble(),
            result.atNum(i, 7).toDouble(),
          ),
        ));
      }
      return faces;
    } finally {
      result.dispose();
    }
  }

  /// Build a square eye crop around a YuNet eye landmark.
  ///
  /// Side length is a fraction of face width so the crop scales with the
  /// subject's distance from the camera. We clamp the box to the frame *and*
  /// the face rect — if a landmark drifts outside the face on an extreme
  /// pose, this keeps the crop sensible instead of running the classifier
  /// on background.
  FaceBox? _eyeBoxFromLandmark(
    (double, double) pt,
    cv.Rect face,
    int frameW,
    int frameH,
  ) {
    final side = (face.width * _eyeSideFrac).round();
    if (side <= 1) return null;
    final half = side ~/ 2;
    final cx = pt.$1.round();
    final cy = pt.$2.round();
    final faceX1 = face.x + face.width;
    final faceY1 = face.y + face.height;
    int clampI(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);
    final x0 = clampI(clampI(cx - half, face.x, faceX1 - 1), 0, frameW - 1);
    final y0 = clampI(clampI(cy - half, face.y, faceY1 - 1), 0, frameH - 1);
    final x1 = clampI(clampI(cx + half, face.x, faceX1), 0, frameW);
    final y1 = clampI(clampI(cy + half, face.y, faceY1), 0, frameH);
    final w = x1 - x0;
    final h = y1 - y0;
    if (w <= 1 || h <= 1) return null;
    return FaceBox(x0, y0, w, h);
  }

  void dispose() {
    _interp?.close();
    _faceDetector?.dispose();
  }
}

class _YuNetFace {
  final cv.Rect rect;
  final (double, double) rightEye;
  final (double, double) leftEye;
  _YuNetFace({
    required this.rect,
    required this.rightEye,
    required this.leftEye,
  });
}
