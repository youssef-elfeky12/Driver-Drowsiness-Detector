import 'package:flutter/material.dart';
import '../models/types.dart';
import '../theme.dart';

class DetectionOverlay extends StatelessWidget {
  final DetectionResult? result;
  final Size previewSize;
  final bool mirrored;
  const DetectionOverlay({
    super.key,
    required this.result,
    required this.previewSize,
    this.mirrored = true,
  });

  @override
  Widget build(BuildContext context) {
    if (result == null || result!.faces.isEmpty) {
      return const SizedBox.expand();
    }
    return CustomPaint(
      size: previewSize,
      painter: _Painter(result!, mirrored),
    );
  }
}

class _Painter extends CustomPainter {
  final DetectionResult result;
  final bool mirrored;
  _Painter(this.result, this.mirrored);

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / result.frameWidth;
    final sy = size.height / result.frameHeight;

    for (final f in result.faces) {
      _drawBox(
        canvas,
        f.box,
        sx,
        sy,
        size,
        AppColors.ok,
        'face: ${f.faceClass.label}  ${(f.conf * 100).toInt()}%',
        thick: 3,
      );
      for (final e in f.eyes) {
        final c = e.eyeClass == EyeClass.closed
            ? AppColors.danger
            : AppColors.primary;
        _drawBox(
          canvas,
          e.box,
          sx,
          sy,
          size,
          c,
          '${e.eyeClass.label} ${(e.conf * 100).toInt()}%',
          thick: 2,
        );
      }
    }
  }

  void _drawBox(
    Canvas canvas,
    FaceBox b,
    double sx,
    double sy,
    Size size,
    Color color,
    String label, {
    double thick = 2,
  }) {
    var left = b.x * sx;
    final top = b.y * sy;
    final w = b.w * sx;
    final h = b.h * sy;
    if (mirrored) left = size.width - left - w;
    final rect = Rect.fromLTWH(left, top, w, h);

    final stroke = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = thick;
    canvas.drawRect(rect, stroke);

    final tp = TextPainter(
      text: TextSpan(
        text: ' $label ',
        style: TextStyle(
          color: AppColors.bg,
          fontWeight: FontWeight.w700,
          fontSize: thick == 3 ? 13 : 10,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final tagRect = Rect.fromLTWH(
      left,
      (top - tp.height - 2).clamp(0, size.height),
      tp.width + 6,
      tp.height + 2,
    );
    canvas.drawRect(tagRect, Paint()..color = color);
    tp.paint(canvas, tagRect.topLeft + const Offset(3, 1));
  }

  @override
  bool shouldRepaint(covariant _Painter old) => old.result != result;
}

extension on FaceClass {
  String get label => switch (this) {
        FaceClass.yawn => 'yawn',
        FaceClass.noYawn => 'no_yawn',
        FaceClass.front => 'front',
        FaceClass.down => 'down',
      };
}

extension on EyeClass {
  String get label => switch (this) {
        EyeClass.closed => 'Closed',
        EyeClass.open => 'Open',
      };
}
