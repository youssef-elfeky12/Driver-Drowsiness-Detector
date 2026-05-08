import 'package:shared_preferences/shared_preferences.dart';
import '../models/types.dart';

class SettingsService {
  static const _kConf = 'conf';
  static const _kNumber = 'number';
  static const _kVolume = 'volume';
  static const _kKeepOn = 'keepOn';

  static Future<AppSettings> load() async {
    final p = await SharedPreferences.getInstance();
    return AppSettings(
      confidenceThreshold: p.getDouble(_kConf) ?? 0.6,
      emergencyNumber: p.getString(_kNumber) ?? '112',
      alarmVolume: p.getDouble(_kVolume) ?? 1.0,
      keepScreenOn: p.getBool(_kKeepOn) ?? true,
    );
  }

  static Future<void> save(AppSettings s) async {
    final p = await SharedPreferences.getInstance();
    await p.setDouble(_kConf, s.confidenceThreshold);
    await p.setString(_kNumber, s.emergencyNumber);
    await p.setDouble(_kVolume, s.alarmVolume);
    await p.setBool(_kKeepOn, s.keepScreenOn);
  }
}
