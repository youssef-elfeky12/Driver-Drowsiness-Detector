import 'dart:convert';

import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;

/// Resolved location for the emergency dispatcher message.
class LocationFix {
  final double latitude;
  final double longitude;
  final String? address; // human-readable, may be null if reverse geocoding failed
  const LocationFix({
    required this.latitude,
    required this.longitude,
    this.address,
  });

  /// What we hand to the TTS — prefer the human address, fall back to coords.
  String toSpeech() {
    if (address != null && address!.isNotEmpty) return address!;
    final lat = latitude.toStringAsFixed(4);
    final lon = longitude.toStringAsFixed(4);
    return 'coordinates $lat ${latitude >= 0 ? 'north' : 'south'}, '
        '$lon ${longitude >= 0 ? 'east' : 'west'}';
  }
}

class LocationService {
  /// Best-effort fix. Returns null if location services are off or denied.
  /// Does not throw — emergencies should never crash on a missing GPS.
  static Future<LocationFix?> getFix({
    Duration timeout = const Duration(seconds: 6),
  }) async {
    try {
      final enabled = await Geolocator.isLocationServiceEnabled();
      if (!enabled) return null;

      var perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.denied ||
          perm == LocationPermission.deniedForever) {
        return null;
      }

      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.medium,
        ),
      ).timeout(timeout);

      final addr = await _reverseGeocode(pos.latitude, pos.longitude);
      return LocationFix(
        latitude: pos.latitude,
        longitude: pos.longitude,
        address: addr,
      );
    } catch (_) {
      return null;
    }
  }

  /// Free reverse geocoding via OpenStreetMap Nominatim. The official
  /// `geocoding` plugin has no Windows backend, so we hit the HTTP API.
  /// Nominatim's TOS requires a real User-Agent and ≤1 req/s. Both are fine
  /// for a thesis demo.
  static Future<String?> _reverseGeocode(double lat, double lon) async {
    try {
      final url = Uri.parse(
        'https://nominatim.openstreetmap.org/reverse'
        '?format=jsonv2&lat=$lat&lon=$lon&zoom=18&addressdetails=1',
      );
      final r = await http.get(
        url,
        headers: {
          'User-Agent': 'DrowsinessDetector/0.1 (thesis project)',
          'Accept-Language': 'en',
        },
      ).timeout(const Duration(seconds: 4));
      if (r.statusCode != 200) return null;
      final body = jsonDecode(r.body) as Map<String, dynamic>;
      final addr = body['address'] as Map<String, dynamic>?;
      if (addr == null) {
        return body['display_name']?.toString();
      }
      // Build a maximally informative speakable address. Nominatim's data is
      // sparser outside dense western cities, so we layer multiple fallbacks
      // from the most specific to the most general.
      final parts = <String>[];
      final road = addr['road'] ??
          addr['pedestrian'] ??
          addr['footway'] ??
          addr['path'];
      final house = addr['house_number'];
      if (road != null) {
        parts.add(house != null ? '$house $road' : road.toString());
      }
      final neighbourhood = addr['neighbourhood'] ??
          addr['quarter'] ??
          addr['hamlet'] ??
          addr['suburb'];
      if (neighbourhood != null) parts.add(neighbourhood.toString());
      final district = addr['city_district'] ??
          addr['district'] ??
          addr['borough'] ??
          addr['county'];
      if (district != null) parts.add(district.toString());
      final city = addr['city'] ?? addr['town'] ?? addr['village'];
      if (city != null) parts.add(city.toString());
      final state = addr['state'] ?? addr['region'];
      if (state != null) parts.add(state.toString());

      if (parts.isEmpty) {
        return body['display_name']?.toString();
      }
      // Keep it under ~6 components so the TTS doesn't read forever.
      return parts.take(5).join(', ');
    } catch (_) {
      return null;
    }
  }
}
