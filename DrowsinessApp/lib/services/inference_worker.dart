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

/// Runs the [Detector] on a background isolate so heavy work
/// (YuNet face detection + ResNet50V2 inference × 3 per frame) never
/// blocks the root isolate's event loop. The UI / preview keep running
/// at full camera fps; results trail behind by whatever inference takes
/// (~50–200 ms).
///
/// Use [infer] for one-off requests. If you call [infer] again before
/// the previous one resolves you'll get two queued requests on the
/// worker — prefer to guard with an in-flight flag on the caller side.
class InferenceWorker {
  Isolate? _isolate;
  SendPort? _sendPort;
  ReceivePort? _receivePort;
  final Map<int, Completer<DetectionResult?>> _pending = {};
  final Completer<SendPort> _ready = Completer<SendPort>();
  int _nextId = 0;

  bool get isReady => _sendPort != null;

  Future<void> init({void Function(String)? onProgress}) async {
    onProgress?.call('Extracting models…');
    final tflitePath = await _writeAssetToTemp(
        'assets/models/drowsiness_resnet50v2.tflite',
        'drowsiness_resnet50v2.tflite');
    final yunetPath = await _writeAssetToTemp(
        'assets/models/face_detection_yunet_2023mar.onnx',
        'face_detection_yunet_2023mar.onnx');

    onProgress?.call('Starting inference worker…');
    final rp = ReceivePort();
    _receivePort = rp;
    rp.listen(_onMessage);
    _isolate = await Isolate.spawn<_WorkerInit>(
      _workerEntrypoint,
      _WorkerInit(rp.sendPort, tflitePath, yunetPath),
    );
    _sendPort = await _ready.future;
    onProgress?.call('Ready');
  }

  void _onMessage(dynamic msg) {
    if (msg is SendPort) {
      if (!_ready.isCompleted) _ready.complete(msg);
    } else if (msg is _Reply) {
      final c = _pending.remove(msg.id);
      c?.complete(msg.result);
    }
  }

  /// Run detection on a BGR pixel buffer. Returns null if the worker
  /// isn't ready or the request fails inside the isolate.
  Future<DetectionResult?> infer({
    required Uint8List bgrBytes,
    required int width,
    required int height,
    required double threshold,
  }) {
    final sp = _sendPort;
    if (sp == null) return Future.value(null);
    final id = _nextId++;
    final c = Completer<DetectionResult?>();
    _pending[id] = c;
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
    _isolate?.kill(priority: Isolate.immediate);
    _receivePort?.close();
    _sendPort = null;
    _isolate = null;
    for (final c in _pending.values) {
      if (!c.isCompleted) c.complete(null);
    }
    _pending.clear();
  }
}

// ---------------------------------------------------------------------------
// Worker-side
// ---------------------------------------------------------------------------

class _WorkerInit {
  final SendPort reply;
  final String tflitePath;
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

void _workerEntrypoint(_WorkerInit init) {
  final rp = ReceivePort();
  init.reply.send(rp.sendPort);

  final detector = Detector();
  detector.initFromPaths(
    tflitePath: init.tflitePath,
    yunetPath: init.yunetPath,
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
