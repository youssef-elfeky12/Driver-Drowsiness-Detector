import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppColors {
  static const bg = Color(0xFF0B0F14);
  static const surface = Color(0xFF141B23);
  static const surface2 = Color(0xFF1C2530);
  static const primary = Color(0xFF3B82F6);
  static const amber = Color(0xFFF59E0B);
  static const danger = Color(0xFFEF4444);
  static const ok = Color(0xFF22C55E);
  static const muted = Color(0xFF9CA3AF);
  static const text = Color(0xFFE5E7EB);
}

ThemeData buildTheme() {
  final base = ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.bg,
    colorScheme: const ColorScheme.dark(
      surface: AppColors.bg,
      primary: AppColors.primary,
      secondary: AppColors.amber,
      error: AppColors.danger,
    ),
    useMaterial3: true,
  );
  return base.copyWith(
    textTheme: GoogleFonts.interTextTheme(base.textTheme).apply(
      bodyColor: AppColors.text,
      displayColor: AppColors.text,
    ),
  );
}
