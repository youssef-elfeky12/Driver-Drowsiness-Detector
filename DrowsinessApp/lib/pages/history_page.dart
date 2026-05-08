import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/types.dart';
import '../services/storage.dart';
import '../theme.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  late Future<List<Trip>> _future;

  @override
  void initState() {
    super.initState();
    _future = StorageService.listTrips();
  }

  String _formatDur(int ms) {
    final s = ms ~/ 1000;
    return '${s ~/ 60}m ${s % 60}s';
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.bg,
      child: SafeArea(
        bottom: false,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text('History',
                      style: TextStyle(
                          fontSize: 26, fontWeight: FontWeight.w800)),
                  SizedBox(height: 4),
                  Text('Past trips and detected events',
                      style: TextStyle(
                          color: AppColors.muted, fontSize: 14)),
                ],
              ),
            ),
            Expanded(
              child: FutureBuilder<List<Trip>>(
                future: _future,
                builder: (ctx, snap) {
                  if (!snap.hasData) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  final trips = snap.data!;
                  if (trips.isEmpty) {
                    return const Center(
                      child: Text('No trips yet.',
                          style: TextStyle(color: AppColors.muted)),
                    );
                  }
                  return ListView.separated(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
                    itemCount: trips.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 8),
                    itemBuilder: (_, i) => _tripCard(trips[i]),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _tripCard(Trip t) {
    final focus = t.events
        .where((e) =>
            e.type == EventType.yawn || e.type == EventType.headDown)
        .length;
    final drowsy =
        t.events.where((e) => e.type == EventType.drowsy).length;
    final crit = t.events
        .where((e) =>
            e.type == EventType.critical || e.type == EventType.emergency)
        .length;
    final dur = t.endedAt - t.startedAt;
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
          Row(
            children: [
              const Icon(Icons.calendar_today,
                  size: 14, color: AppColors.muted),
              const SizedBox(width: 6),
              Text(
                DateFormat.yMMMd().add_jm().format(
                    DateTime.fromMillisecondsSinceEpoch(t.startedAt)),
                style: const TextStyle(
                    fontSize: 13, fontWeight: FontWeight.w600),
              ),
              const Spacer(),
              const Icon(Icons.access_time,
                  size: 14, color: AppColors.muted),
              const SizedBox(width: 4),
              Text(_formatDur(dur),
                  style: const TextStyle(
                      color: AppColors.muted,
                      fontSize: 12,
                      fontFeatures: [FontFeature.tabularFigures()])),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                  child: _stat('Focus', focus, AppColors.amber)),
              const SizedBox(width: 8),
              Expanded(
                  child: _stat('Drowsy', drowsy, AppColors.amber)),
              const SizedBox(width: 8),
              Expanded(
                  child: _stat('Critical', crit, AppColors.danger)),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              const Icon(Icons.remove_red_eye,
                  size: 14, color: AppColors.muted),
              const SizedBox(width: 6),
              const Text('Longest closed: ',
                  style: TextStyle(
                      fontSize: 12, color: AppColors.muted)),
              Text(
                '${(t.longestClosedMs / 1000).toStringAsFixed(1)}s',
                style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    fontFeatures: [FontFeature.tabularFigures()]),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _stat(String label, int value, Color color) => Container(
        decoration: BoxDecoration(
          color: AppColors.surface2,
          borderRadius: BorderRadius.circular(12),
        ),
        padding: const EdgeInsets.symmetric(vertical: 10),
        child: Column(
          children: [
            Text('$value',
                style: TextStyle(
                  color: color,
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  fontFeatures: const [FontFeature.tabularFigures()],
                )),
            const SizedBox(height: 2),
            Text(label.toUpperCase(),
                style: const TextStyle(
                  color: AppColors.muted,
                  fontSize: 10,
                  letterSpacing: 1.2,
                  fontWeight: FontWeight.w700,
                )),
          ],
        ),
      );
}
