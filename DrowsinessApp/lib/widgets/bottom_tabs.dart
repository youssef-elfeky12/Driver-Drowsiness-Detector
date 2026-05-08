import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../theme.dart';

class BottomTabs extends StatelessWidget {
  const BottomTabs({super.key});

  static const _items = [
    (label: 'Drive', path: '/drive', icon: Icons.directions_car_filled),
    (label: 'History', path: '/history', icon: Icons.history),
    (label: 'Settings', path: '/settings', icon: Icons.settings),
  ];

  @override
  Widget build(BuildContext context) {
    final loc = GoRouterState.of(context).uri.path;
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface.withOpacity(0.95),
        border: Border(
          top: BorderSide(color: Colors.white.withOpacity(0.05)),
        ),
      ),
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).padding.bottom,
      ),
      child: Row(
        children: _items.map((it) {
          final active = loc == it.path;
          return Expanded(
            child: InkWell(
              onTap: () => context.go(it.path),
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
        }).toList(),
      ),
    );
  }
}
