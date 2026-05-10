import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/types.dart';
import 'detector.dart';

/// Two-isolate worker so the cheap face detector (YuNet, ~10–15 ms) can
/// run on every camera tick while the heavy classifier (YuNet + 3×
/// ResNet50V2, ~120–200 ms) runs at a throttled rate, without either
/// blocking the other.
///
/// - [detect] → goes to the **detection isolate**. Cheap. Use every tick.
/// - [classify] → goes to the **classification isolate**. Use at a fixed
///   rate (e.g. every 250 ms). Caller is expected to gate on the previous
///   call having returned (drop-on-busy).
class InferenceWorker {
  Isolate? _detectIsolate;
  Isolate? _classifyIsolate;
  SendPort? _detectSendPort;
  SendPort? _classifySendPort;
  ReceivePort? _detectReceivePort;
  ReceivePort? _classifyReceivePort;

  final Map<int, Completer<DetectionResult?>> _detectPending = {};
  final Map<int, Completer<DetectionResult?>> _classifyPending = {};
  final Completer<SendPort> _detectReady = Completer<SendPort>();
  final Completer<SendPort> _classifyReady = Completer<SendPort>();

  int _nextDetectId = 0;
  int _nextClassifyId = 0;

  bool get isReady => _detectSendPort != null && _classifySendPort != null;

  Future<void> init({void Function(String)? onProgress}) async {
    onProgress?.call('Extracting models…');
    final tflitePath = await _writeAssetToTemp(
        'assets/models/drowsiness_resnet50v2.tflite',
        'drowsiness_resnet50v2.tflite');
    final yunetPath = await _writeAssetToTemp(
        'assets/models/face_detection_yunet_2023mar.onnx',
        'face_detection_yunet_2023mar.onnx');

    onProgress?.call('Starting detection worker…');
    final dRp = ReceivePort();
    _detectReceivePort = dRp;
    dRp.listen(_onDetectMessage);
    _detectIsolate = await Isolate.spawn<_WorkerInit>(
      _detectionEntrypoint,
      _WorkerInit(dRp.sendPort, null, yunetPath),
    );
    _detectSendPort = await _detectReady.future;

    onProgress?.call('Starting classification worker…');
    final cRp = ReceivePort();
    _classifyReceivePort = cRp;
    cRp.listen(_onClassifyMessage);
    _classifyIsolate = await Isolate.spawn<_WorkerInit>(
      _classificationEntrypoint,
      _WorkerInit(cRp.sendPort, tflitePath, yunetPath),
    );
    _classifySendPort = await _classifyReady.future;

    onProgress?.call('Ready');
  }

  void _onDetectMessage(dynamic msg) {
    if (msg is SendPort) {
      if (!_detectReady.isCompleted) _detectReady.complete(msg);
    } else if (msg is _Reply) {
      _detectPending.remove(msg.id)?.complete(msg.result);
    }
  }

  void _onClassifyMessage(dynamic msg) {
    if (msg is SendPort) {
      if (!_classifyReady.isCompleted) _classifyReady.complete(msg);
    } else if (msg is _Reply) {
      _classifyPending.remove(msg.id)?.complete(msg.result);
    }
  }

  /// YuNet-only pass — returns boxes + landmark-derived eye crops, no
  /// class labels. Use for live box tracking.
  Future<DetectionResult?> detect({
    required Uint8List bgrBytes,
    required int width,
    required int height,
  }) {
    final sp = _detectSendPort;
    if (sp == null) return Future.value(null);
    final id = _nextDetectId++;
    final c = Completer<DetectionResult?>();
    _detectPending[id] = c;
    final transferable = TransferableTypedData.fromList([bgrBytes]);
    sp.send(_Request(id, transferable, width, height, 0));
    return c.future;
  }

  /// Full pipeline — YuNet + ResNet on the face and each eye crop.
  /// Returns the full [DetectionResult] with class labels filled in.
  Future<DetectionResult?> classify({
    required Uint8List bgrBytes,
    required int width,
    required int height,
    required double threshold,
  }) {
    final sp = _classifySendPort;
    if (sp == null) return Future.value(null);
    final id = _nextClassifyId++;
    final c = Completer<DetectionResult?>();
    _classifyPending[id] = c;
    final transferable = TransferableTypedData.fromList([bgrBytes]);
    sp.send(_Request(id, transferable, width, height, threshold));
    return c.future;
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

  void dispose() {
    _detectIsolate?.kill(priority: Isolate.immediate);
    _classifyIsolate?.kill(priority: Isolate.immediate);
    _detectReceivePort?.close();
    _classifyReceivePort?.close();
    _detectSendPort = null;
    _classifySendPort = null;
    _detectIsolate = null;
    _classifyIsolate = null;
    for (final c in _detectPending.values) {
      if (!c.isCompleted) c.complete(null);
    }
    for (final c in _classifyPending.values) {
      if (!c.isCompleted) c.complete(null);
    }
    _detectPending.clear();
    _classifyPending.clear();
  }
}

// ---------------------------------------------------------------------------
// Worker-side messages
// ---------------------------------------------------------------------------

class _WorkerInit {
  final SendPort reply;
  final String? tflitePath;
  final String yunetPath;
  _WorkerInit(this.reply, this.tflitePath, this.yunetPath);
}

class _Request {
  final int id;
  final TransferableTypedData data;
  final int width, height;
  final double threshold;
  _Request(this.id, this.data, this.width, this.height, this.threshold);
}

class _Reply {
  final int id;
  final DetectionResult? result;
  _Reply(this.id, this.result);
}

// ---------------------------------------------------------------------------
// Worker entrypoints
// ---------------------------------------------------------------------------

void _detectionEntrypoint(_WorkerInit init) {
  final rp = ReceivePort();
  init.reply.send(rp.sendPort);

  final detector = Detector();
  detector.initFromPaths(
    yunetPath: init.yunetPath,
    enableClassification: false,
  );

  rp.listen((msg) {
    if (msg is! _Request) return;
    final bytes = msg.data.materialize().asUint8List();
    DetectionResult? result;
    cv.Mat? mat;
    try {
      mat = cv.Mat.fromList(
        msg.height,
        msg.width,
        cv.MatType.CV_8UC3,
        bytes,
      );
      result = detector.detectFacesOnly(mat);
    } catch (_) {
      result = null;
    } finally {
      mat?.dispose();
    }
    init.reply.send(_Reply(msg.id, result));
  });
}

void _classificationEntrypoint(_WorkerInit init) {
  final rp = ReceivePort();
  init.reply.send(rp.sendPort);

  final detector = Detector();
  detector.initFromPaths(
    yunetPath: init.yunetPath,
    tflitePath: init.tflitePath,
  );

  rp.listen((msg) {
    if (msg is! _Request) return;
    final bytes = msg.data.materialize().asUint8List();
    DetectionResult? result;
    cv.Mat? mat;
    try {
      mat = cv.Mat.fromList(
        msg.height,
        msg.width,
        cv.MatType.CV_8UC3,
        bytes,
      );
      result = detector.detectMat(mat, msg.threshold);
    } catch (_) {
      result = null;
    } finally {
      mat?.dispose();
    }
    init.reply.send(_Reply(msg.id, result));
  });
}
