import 'package:flutter/material.dart';

import '../models/types.dart';
import '../services/settings.dart';
import '../services/storage.dart';
import '../theme.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  AppSettings _s = const AppSettings();
  static const _numbers = ['112', '911', '999', '110'];

  @override
  void initState() {
    super.initState();
    SettingsService.load().then((s) => setState(() => _s = s));
  }

  void _update(AppSettings next) {
    setState(() => _s = next);
    SettingsService.save(next);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.bg,
      child: SafeArea(
        bottom: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 20, 16, 24),
          children: [
            const Padding(
              padding: EdgeInsets.fromLTRB(4, 0, 4, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Settings',
                      style: TextStyle(
                          fontSize: 26, fontWeight: FontWeight.w800)),
                  SizedBox(height: 4),
                  Text('Tune detection and alerts',
                      style: TextStyle(
                          color: AppColors.muted, fontSize: 14)),
                ],
              ),
            ),
            _section(
              title: 'Detection',
              children: [
                _label(
                    'Confidence threshold — ${_s.confidenceThreshold.toStringAsFixed(2)}'),
                Slider(
                  min: 0.4,
                  max: 0.9,
                  divisions: 10,
                  value: _s.confidenceThreshold,
                  onChanged: (v) =>
                      _update(_s.copyWith(confidenceThreshold: v)),
                  activeColor: AppColors.primary,
                ),
                const Text(
                  'Predictions below this confidence are ignored. Higher = fewer false alarms.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _section(
              title: 'Emergency',
              children: [
                _label('Emergency number'),
                const SizedBox(height: 8),
                Row(
                  children: _numbers.map((n) {
                    final active = _s.emergencyNumber == n;
                    return Expanded(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                        child: ElevatedButton(
                          onPressed: () =>
                              _update(_s.copyWith(emergencyNumber: n)),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: active
                                ? AppColors.primary
                                : AppColors.surface2,
                            foregroundColor: active
                                ? Colors.white
                                : AppColors.muted,
                            padding:
                                const EdgeInsets.symmetric(vertical: 12),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                          ),
                          child: Text(n,
                              style: const TextStyle(
                                fontWeight: FontWeight.w800,
                                fontSize: 16,
                                fontFeatures: [
                                  FontFeature.tabularFigures()
                                ],
                              )),
                        ),
                      ),
                    );
                  }).toList(),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Visual demo only — no real call is placed.',
                  style: TextStyle(color: AppColors.muted, fontSize: 12),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _section(
              title: 'Audio',
              children: [
                _label(
                    'Alarm volume — ${(_s.alarmVolume * 100).round()}%'),
                Slider(
                  value: _s.alarmVolume,
                  onChanged: (v) => _update(_s.copyWith(alarmVolume: v)),
                  activeColor: AppColors.primary,
                ),
              ],
            ),
            const SizedBox(height: 12),
            _section(
              title: 'Display',
              children: [
                Row(
                  children: [
                    const Expanded(
                      child: Text(
                        'Keep screen on while driving',
                        style: TextStyle(
                            fontSize: 14, fontWeight: FontWeight.w600),
                      ),
                    ),
                    Switch(
                      value: _s.keepScreenOn,
                      onChanged: (v) =>
                          _update(_s.copyWith(keepScreenOn: v)),
                      activeColor: AppColors.primary,
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 12),
            _section(
              title: 'Data',
              children: [
                ElevatedButton.icon(
                  onPressed: () async {
                    final ok = await showDialog<bool>(
                      context: context,
                      builder: (_) => AlertDialog(
                        backgroundColor: AppColors.surface,
                        title: const Text('Delete history?'),
                        content: const Text('All trips will be removed.'),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(context, false),
                            child: const Text('Cancel'),
                          ),
                          TextButton(
                            onPressed: () => Navigator.pop(context, true),
                            child: const Text('Delete',
                                style: TextStyle(color: AppColors.danger)),
                          ),
                        ],
                      ),
                    );
                    if (ok == true) await StorageService.clearTrips();
                  },
                  icon: const Icon(Icons.delete_outline, size: 18),
                  label: const Text('Clear trip history'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.danger.withOpacity(0.15),
                    foregroundColor: AppColors.danger,
                    minimumSize: const Size.fromHeight(44),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                    textStyle: const TextStyle(fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _section({required String title, required List<Widget> children}) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withOpacity(0.05)),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title.toUpperCase(),
              style: const TextStyle(
                color: AppColors.muted,
                fontSize: 11,
                letterSpacing: 1.5,
                fontWeight: FontWeight.w700,
              )),
          const SizedBox(height: 12),
          ...children,
        ],
      ),
    );
  }

  Widget _label(String s) => Text(s,
      style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600));
}
