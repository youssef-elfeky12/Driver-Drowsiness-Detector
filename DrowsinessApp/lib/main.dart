import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';

import 'pages/drive_page.dart';
import 'pages/history_page.dart';
import 'pages/settings_page.dart';
import 'theme.dart';
import 'widgets/bottom_tabs.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  runApp(const DrowsinessApp());
}

class DrowsinessApp extends StatelessWidget {
  const DrowsinessApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'Drowsiness Detector',
      theme: buildTheme(),
      debugShowCheckedModeBanner: false,
      routerConfig: _router,
    );
  }
}

final _router = GoRouter(
  initialLocation: '/drive',
  routes: [
    ShellRoute(
      builder: (ctx, state, child) => Scaffold(
        backgroundColor: AppColors.bg,
        body: SafeArea(
          top: false,
          bottom: false,
          child: Column(
            children: [
              Expanded(child: child),
              const BottomTabs(),
            ],
          ),
        ),
      ),
      routes: [
        GoRoute(path: '/drive', builder: (_, __) => const DrivePage()),
        GoRoute(path: '/history', builder: (_, __) => const HistoryPage()),
        GoRoute(path: '/settings', builder: (_, __) => const SettingsPage()),
      ],
    ),
  ],
);
