import 'package:flutter/material.dart';
import '../theme.dart';

class PullOverOverlay extends StatelessWidget {
  final VoidCallback onDismiss;
  const PullOverOverlay({super.key, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: TweenAnimationBuilder<double>(
        tween: Tween(begin: 0, end: 1),
        duration: const Duration(milliseconds: 350),
        builder: (_, t, child) => Opacity(
          opacity: t,
          child: Transform.translate(
            offset: Offset(0, (1 - t) * 40),
            child: child,
          ),
        ),
        child: Container(
          color: AppColors.amber.withOpacity(0.95),
          padding: const EdgeInsets.all(32),
          alignment: Alignment.center,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.warning_amber_rounded,
                  size: 96, color: AppColors.bg),
              const SizedBox(height: 16),
              const Text(
                'PULL OVER',
                style: TextStyle(
                  color: AppColors.bg,
                  fontSize: 48,
                  fontWeight: FontWeight.w800,
                  letterSpacing: -1,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Multiple drowsiness signals detected.\nFind a safe place to stop and rest.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: AppColors.bg.withOpacity(0.8),
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 40),
              ElevatedButton(
                onPressed: onDismiss,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.bg,
                  foregroundColor: AppColors.text,
                  padding: const EdgeInsets.symmetric(
                      horizontal: 32, vertical: 16),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16)),
                ),
                child: const Text(
                  "I'm OK",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class WarningOverlay extends StatelessWidget {
  final int closedMs;
  const WarningOverlay({super.key, required this.closedMs});

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: Container(
        color: AppColors.amber.withOpacity(0.8),
        alignment: Alignment.center,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.remove_red_eye, size: 96, color: AppColors.bg),
            const SizedBox(height: 16),
            const Text('EYES CLOSED',
                style: TextStyle(
                  color: AppColors.bg,
                  fontSize: 40,
                  fontWeight: FontWeight.w800,
                )),
            const SizedBox(height: 4),
            const Text('WAKE UP',
                style: TextStyle(
                  color: AppColors.bg,
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                )),
            const SizedBox(height: 24),
            Text(
              '${(closedMs / 1000).toStringAsFixed(1)}s',
              style: TextStyle(
                color: AppColors.bg.withOpacity(0.85),
                fontSize: 18,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class CriticalOverlay extends StatefulWidget {
  final int countdown;
  final String number;
  final VoidCallback onCancel;
  const CriticalOverlay({
    super.key,
    required this.countdown,
    required this.number,
    required this.onCancel,
  });

  @override
  State<CriticalOverlay> createState() => _CriticalOverlayState();
}

class _CriticalOverlayState extends State<CriticalOverlay>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctl;

  @override
  void initState() {
    super.initState();
    _ctl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 660),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: AnimatedBuilder(
        animation: _ctl,
        builder: (_, __) => Container(
          color: AppColors.danger.withOpacity(0.0 + 0.55 * _ctl.value),
          alignment: Alignment.center,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.emergency, size: 96, color: AppColors.text),
              const SizedBox(height: 8),
              const Text('EMERGENCY',
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.w800,
                  )),
              const SizedBox(height: 4),
              Text('CALLING ${widget.number} IN',
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.5,
                  )),
              const SizedBox(height: 12),
              Text(
                '${widget.countdown}',
                style: const TextStyle(
                  fontSize: 120,
                  fontWeight: FontWeight.w800,
                  height: 1,
                  fontFeatures: [FontFeature.tabularFigures()],
                ),
              ),
              const SizedBox(height: 32),
              ElevatedButton(
                onPressed: widget.onCancel,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.bg.withOpacity(0.85),
                  foregroundColor: AppColors.text,
                  padding: const EdgeInsets.symmetric(
                      horizontal: 40, vertical: 16),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(18)),
                ),
                child: const Text('Cancel',
                    style: TextStyle(
                        fontSize: 18, fontWeight: FontWeight.w700)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
