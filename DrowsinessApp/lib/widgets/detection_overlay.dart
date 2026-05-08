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
      // Two stacked labels above the face box: yawn binary, then head pose.
      // Each tag colored amber when its alarm-side is firing, green otherwise.
      final yawnLabel = f.isYawn ? 'yawn' : 'no_yawn';
      final yawnColor = f.isYawn ? AppColors.amber : AppColors.ok;
      final headLabel = f.isHeadDown ? 'down' : 'front';
      final headColor = f.isHeadDown ? AppColors.amber : AppColors.ok;

      _drawFaceBoxWithStackedTags(
        canvas,
        f.box,
        sx,
        sy,
        size,
        AppColors.ok,
        topLabel: '$yawnLabel ${(f.yawnConf * 100).toInt()}%',
        topColor: yawnColor,
        bottomLabel: '$headLabel ${(f.headPoseConf * 100).toInt()}%',
        bottomColor: headColor,
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

  void _drawFaceBoxWithStackedTags(
    Canvas canvas,
    FaceBox b,
    double sx,
    double sy,
    Size size,
    Color boxColor, {
    required String topLabel,
    required Color topColor,
    required String bottomLabel,
    required Color bottomColor,
  }) {
    var left = b.x * sx;
    final top = b.y * sy;
    final w = b.w * sx;
    final h = b.h * sy;
    if (mirrored) left = size.width - left - w;

    // Box outline.
    canvas.drawRect(
      Rect.fromLTWH(left, top, w, h),
      Paint()
        ..color = boxColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3,
    );

    // Tags above the box, stacked: top tag = yawn binary, second = head pose.
    final tags = [
      (topLabel, topColor),
      (bottomLabel, bottomColor),
    ];
    final painters = tags
        .map((t) => TextPainter(
              text: TextSpan(
                text: ' ${t.$1} ',
                style: const TextStyle(
                  color: AppColors.bg,
                  fontWeight: FontWeight.w700,
                  fontSize: 12,
                ),
              ),
              textDirection: TextDirection.ltr,
            )..layout())
        .toList();

    final lineHeight = painters[0].height + 2;
    final stackHeight = lineHeight * tags.length + 2;
    var tagTop = (top - stackHeight).clamp(0.0, size.height - stackHeight);

    for (var i = 0; i < tags.length; i++) {
      final tp = painters[i];
      final color = tags[i].$2;
      final tagRect = Rect.fromLTWH(
        left,
        tagTop,
        tp.width + 6,
        tp.height + 2,
      );
      canvas.drawRect(tagRect, Paint()..color = color);

      // Counter-flip text against the parent's mirror Transform.
      final cx = tagRect.center.dx;
      canvas.save();
      canvas.translate(cx, 0);
      canvas.scale(-1.0, 1.0);
      canvas.translate(-cx, 0);
      tp.paint(canvas, tagRect.topLeft + const Offset(3, 1));
      canvas.restore();

      tagTop += lineHeight;
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

    // The whole detection layer is wrapped in a Transform.scale(-1, 1) for the
    // mirrored selfie view. Rectangles are symmetric so they look fine, but
    // text gets visually flipped. Counter-flip locally around the label's
    // horizontal centre so glyphs read correctly on screen.
    final cx = tagRect.center.dx;
    canvas.save();
    canvas.translate(cx, 0);
    canvas.scale(-1.0, 1.0);
    canvas.translate(-cx, 0);
    tp.paint(canvas, tagRect.topLeft + const Offset(3, 1));
    canvas.restore();
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
