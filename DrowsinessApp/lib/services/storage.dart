import 'dart:convert';
import 'dart:io' show Platform;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import '../models/types.dart';

class StorageService {
  static Database? _db;

  static Future<Database> _open() async {
    if (_db != null) return _db!;

    if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
      sqfliteFfiInit();
      databaseFactory = databaseFactoryFfi;
    }

    final dir = await getApplicationDocumentsDirectory();
    final path = p.join(dir.path, 'drowsy.db');
    _db = await openDatabase(
      path,
      version: 1,
      onCreate: (db, _) => db.execute('''
        CREATE TABLE trips(
          id TEXT PRIMARY KEY,
          started_at INTEGER NOT NULL,
          ended_at INTEGER NOT NULL,
          events_json TEXT NOT NULL,
          longest_closed_ms INTEGER NOT NULL
        )'''),
    );
    return _db!;
  }

  static Future<void> saveTrip(Trip t) async {
    final db = await _open();
    await db.insert('trips', {
      'id': t.id,
      'started_at': t.startedAt,
      'ended_at': t.endedAt,
      'events_json': jsonEncode(t.events.map((e) => e.toJson()).toList()),
      'longest_closed_ms': t.longestClosedMs,
    });
  }

  static Future<List<Trip>> listTrips() async {
    final db = await _open();
    final rows = await db.query('trips', orderBy: 'started_at DESC');
    return rows.map((r) {
      final raw = jsonDecode(r['events_json'] as String) as List;
      return Trip(
        id: r['id'] as String,
        startedAt: r['started_at'] as int,
        endedAt: r['ended_at'] as int,
        events: raw
            .map((e) => TripEvent.fromJson(e as Map<String, dynamic>))
            .toList(),
        longestClosedMs: r['longest_closed_ms'] as int,
      );
    }).toList();
  }

  static Future<void> clearTrips() async {
    final db = await _open();
    await db.delete('trips');
  }
}
