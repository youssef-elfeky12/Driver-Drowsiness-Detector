import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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
    return MaterialApp(
      title: 'Drowsiness Detector',
      theme: buildTheme(),
      debugShowCheckedModeBanner: false,
      home: const HomeShell(),
    );
  }
}

/// Tab shell using IndexedStack — keeps all three pages alive across taps so
/// switching back to Drive doesn't re-bootstrap (camera, model, audio).
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});
  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  late final List<Widget> _pages = const [
    DrivePage(),
    HistoryPage(),
    SettingsPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        top: false,
        bottom: false,
        child: Column(
          children: [
            Expanded(
              child: IndexedStack(
                index: _index,
                children: _pages,
              ),
            ),
            BottomTabs(
              activeIndex: _index,
              onTap: (i) => setState(() => _index = i),
            ),
          ],
        ),
      ),
    );
  }
}
