import 'package:flutter/material.dart';
import '../theme.dart';

class EmergencyDialer extends StatelessWidget {
  final String digitsTyped;
  final String number;
  // The single digit currently being "pressed" — null when no key is held.
  // Drives the brief flash on the keypad. Cleared 250 ms after each press by
  // the parent (DrivePage).
  final String? pressedDigit;
  final bool callingActive;
  final bool callConnected;
  final VoidCallback onCancel;
  const EmergencyDialer({
    super.key,
    required this.digitsTyped,
    required this.number,
    required this.pressedDigit,
    required this.callingActive,
    required this.callConnected,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      right: 12,
      bottom: 12,
      child: TweenAnimationBuilder<double>(
        tween: Tween(begin: 0, end: 1),
        duration: const Duration(milliseconds: 500),
        curve: Curves.easeOutBack,
        builder: (_, t, child) => Transform.translate(
          offset: Offset((1 - t) * 100, 0),
          child: Opacity(opacity: t, child: child),
        ),
        child: Container(
          width: 260,
          decoration: BoxDecoration(
            color: const Color(0xFF0E0E0E),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.white.withOpacity(0.1)),
            boxShadow: const [
              BoxShadow(
                color: Color(0x80000000),
                blurRadius: 20,
                offset: Offset(0, 8),
              ),
            ],
          ),
          clipBehavior: Clip.antiAlias,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _header(),
              _screen(),
              _keypad(),
              _footer(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header() => Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF1A1A1A), Color(0xFF0E0E0E)],
          ),
        ),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        child: Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: const BoxDecoration(
                  color: AppColors.ok, shape: BoxShape.circle),
            ),
            const SizedBox(width: 8),
            const Text('PHONE',
                style: TextStyle(
                    color: AppColors.muted,
                    fontFamily: 'monospace',
                    fontSize: 11,
                    letterSpacing: 1)),
            const Spacer(),
            const Text('EMERGENCY',
                style: TextStyle(
                    color: AppColors.muted,
                    fontFamily: 'monospace',
                    fontSize: 10,
                    letterSpacing: 1)),
          ],
        ),
      );

  Widget _screen() => Container(
        color: const Color(0xFF0A0A0A),
        padding: const EdgeInsets.symmetric(vertical: 22),
        alignment: Alignment.center,
        child: Column(
          children: [
            Text(
              callConnected
                  ? 'Connected'
                  : callingActive
                      ? 'Calling…'
                      : 'Dialing',
              style: const TextStyle(
                  color: AppColors.muted,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 2),
            ),
            const SizedBox(height: 6),
            SizedBox(
              height: 38,
              child: Text(
                digitsTyped.isEmpty ? ' ' : digitsTyped,
                style: const TextStyle(
                  fontSize: 30,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 4,
                  fontFeatures: [FontFeature.tabularFigures()],
                ),
              ),
            ),
          ],
        ),
      );

  Widget _keypad() {
    final keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '0', '#'];
    return Container(
      padding: const EdgeInsets.all(10),
      color: const Color(0xFF0E0E0E),
      child: GridView.count(
        shrinkWrap: true,
        crossAxisCount: 3,
        mainAxisSpacing: 6,
        crossAxisSpacing: 6,
        childAspectRatio: 2.1,
        physics: const NeverScrollableScrollPhysics(),
        children: keys.map((k) {
          // Only the digit currently being pressed lights up — like a real
          // phone keypad. The blue fades back to neutral 250 ms later when the
          // parent clears `pressedDigit`.
          final lit = pressedDigit == k;
          return AnimatedContainer(
            duration: const Duration(milliseconds: 120),
            decoration: BoxDecoration(
              color: lit
                  ? AppColors.primary.withOpacity(0.32)
                  : const Color(0xFF1A1A1A),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                color: lit
                    ? AppColors.primary.withOpacity(0.65)
                    : Colors.transparent,
              ),
            ),
            alignment: Alignment.center,
            child: AnimatedDefaultTextStyle(
              duration: const Duration(milliseconds: 120),
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: lit ? AppColors.primary : AppColors.muted,
              ),
              child: Text(k),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _footer() => Container(
        decoration: BoxDecoration(
          color: const Color(0xFF0E0E0E),
          border: Border(
              top: BorderSide(color: Colors.white.withOpacity(0.05))),
        ),
        child: Row(
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 12),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.phone, size: 16, color: AppColors.ok),
                    const SizedBox(width: 6),
                    Text(callConnected ? 'On call' : 'Calling',
                        style: const TextStyle(
                            color: AppColors.ok,
                            fontWeight: FontWeight.w700,
                            fontSize: 12)),
                  ],
                ),
              ),
            ),
            Expanded(
              child: InkWell(
                onTap: onCancel,
                child: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.call_end, size: 16, color: AppColors.danger),
                      SizedBox(width: 6),
                      Text('End',
                          style: TextStyle(
                              color: AppColors.danger,
                              fontWeight: FontWeight.w700,
                              fontSize: 12)),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      );
}
