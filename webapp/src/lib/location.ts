// Browser port of lib/services/location_service.dart.
// Geolocation API + OpenStreetMap Nominatim reverse geocoding. Best-effort;
// never throws — an emergency must not crash on a missing GPS fix.

export interface LocationFix {
  latitude: number;
  longitude: number;
  address: string | null;
}

export function toSpeech(fix: LocationFix): string {
  if (fix.address && fix.address.length > 0) return fix.address;
  const lat = fix.latitude.toFixed(4);
  const lon = fix.longitude.toFixed(4);
  return (
    `coordinates ${lat} ${fix.latitude >= 0 ? "north" : "south"}, ` +
    `${lon} ${fix.longitude >= 0 ? "east" : "west"}`
  );
}

export async function getFix(timeoutMs = 6000): Promise<LocationFix | null> {
  if (typeof navigator === "undefined" || !navigator.geolocation) return null;
  try {
    const pos = await new Promise<GeolocationPosition>((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        enableHighAccuracy: false,
        timeout: timeoutMs,
        maximumAge: 30000,
      });
    });
    const addr = await reverseGeocode(pos.coords.latitude, pos.coords.longitude);
    return {
      latitude: pos.coords.latitude,
      longitude: pos.coords.longitude,
      address: addr,
    };
  } catch {
    return null;
  }
}

async function reverseGeocode(
  lat: number,
  lon: number,
): Promise<string | null> {
  try {
    const url =
      `https://nominatim.openstreetmap.org/reverse` +
      `?format=jsonv2&lat=${lat}&lon=${lon}&zoom=18&addressdetails=1`;
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 4000);
    const r = await fetch(url, {
      headers: { "Accept-Language": "en" },
      signal: ctrl.signal,
    });
    clearTimeout(t);
    if (!r.ok) return null;
    const body = await r.json();
    const addr = body.address as Record<string, string> | undefined;
    if (!addr) return body.display_name ?? null;

    const parts: string[] = [];
    const road = addr.road ?? addr.pedestrian ?? addr.footway ?? addr.path;
    const house = addr.house_number;
    if (road) parts.push(house ? `${house} ${road}` : road);
    const neighbourhood =
      addr.neighbourhood ?? addr.quarter ?? addr.hamlet ?? addr.suburb;
    if (neighbourhood) parts.push(neighbourhood);
    const district =
      addr.city_district ?? addr.district ?? addr.borough ?? addr.county;
    if (district) parts.push(district);
    const city = addr.city ?? addr.town ?? addr.village;
    if (city) parts.push(city);
    const state = addr.state ?? addr.region;
    if (state) parts.push(state);

    if (parts.length === 0) return body.display_name ?? null;
    return parts.slice(0, 5).join(", ");
  } catch {
    return null;
  }
}
