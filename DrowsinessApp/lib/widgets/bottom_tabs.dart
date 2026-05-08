import 'package:flutter/material.dart';
import '../theme.dart';

class BottomTabs extends StatelessWidget {
  final int activeIndex;
  final ValueChanged<int> onTap;
  const BottomTabs({
    super.key,
    required this.activeIndex,
    required this.onTap,
  });

  static const _items = [
    (label: 'Drive', icon: Icons.directions_car_filled),
    (label: 'History', icon: Icons.history),
    (label: 'Settings', icon: Icons.settings),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface.withValues(alpha: 0.95),
        border: Border(
          top: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
        ),
      ),
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).padding.bottom,
      ),
      child: Row(
        children: List.generate(_items.length, (i) {
          final it = _items[i];
          final active = i == activeIndex;
          return Expanded(
            child: InkWell(
              onTap: () => onTap(i),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      it.icon,
                      size: 22,
                      color: active ? AppColors.primary : AppColors.muted,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      it.label,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: active ? AppColors.primary : AppColors.muted,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }),
      ),
    );
  }
}
