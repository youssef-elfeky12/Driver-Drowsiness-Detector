import 'package:flutter/material.dart';
import '../models/types.dart';
import '../theme.dart';

class StatusBar extends StatelessWidget {
  final AlertLevel level;
  final int closedMs;
  final int durationMs;
  const StatusBar({
    super.key,
    required this.level,
    required this.closedMs,
    required this.durationMs,
  });

  static const _meta = {
    AlertLevel.none: ('Alert', AppColors.ok),
    AlertLevel.eyesClosing: ('Eyes closing', AppColors.amber),
    AlertLevel.drowsy: ('Drowsy', AppColors.amber),
    AlertLevel.warning: ('Warning', AppColors.amber),
    AlertLevel.critical: ('Critical', AppColors.danger),
    AlertLevel.emergency: ('Emergency', AppColors.danger),
  };

  String _formatDur(int ms) {
    final s = ms ~/ 1000;
    final m = s ~/ 60;
    final r = s % 60;
    return '$m:${r.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final (label, color) = _meta[level]!;
    return Positioned(
      top: 0,
      left: 0,
      right: 0,
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.black.withOpacity(0.6), Colors.transparent],
          ),
        ),
        padding: EdgeInsets.fromLTRB(
            16, MediaQuery.of(context).padding.top + 12, 16, 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    color: color,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  label,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                    letterSpacing: 0.3,
                  ),
                ),
                const Spacer(),
                Text(
                  _formatDur(durationMs),
                  style: const TextStyle(
                    color: AppColors.muted,
                    fontFeatures: [FontFeature.tabularFigures()],
                    fontSize: 12,
                  ),
                ),
              ],
            ),
            if (closedMs > 800) ...[
              const SizedBox(height: 10),
              const Text(
                'EYES CLOSED',
                style: TextStyle(
                  color: AppColors.muted,
                  fontSize: 10,
                  letterSpacing: 1.5,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 4),
              ClipRRect(
                borderRadius: BorderRadius.circular(2),
                child: LinearProgressIndicator(
                  value: (closedMs / 15000).clamp(0, 1),
                  backgroundColor: Colors.white.withOpacity(0.1),
                  valueColor:
                      const AlwaysStoppedAnimation<Color>(AppColors.amber),
                  minHeight: 4,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
