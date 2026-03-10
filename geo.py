# geo.py — Geographic context: reverse geocoding, directional offset math,
#           and nearby feature lookup via OpenStreetMap / Overpass API.
#
# Run standalone to test:  python geo.py

from __future__ import annotations
import math
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

import config

# Nominatim rate-limits to 1 req/sec — thread-safe lock + timestamp
_nominatim_lock = threading.Lock()
_last_nominatim_call = 0.0
_geocoder = Nominatim(user_agent=config.NOMINATIM_USER_AGENT)


# ── Direction → bearing offset ─────────────────────────────────────────────────

DIRECTION_OFFSETS: dict[str, float] = {
    # Absolute headings
    "north": 0.0,   "n": 0.0,
    "south": 180.0, "s": 180.0,
    "east":  90.0,  "e": 90.0,
    "west":  270.0, "w": 270.0,
    "northeast": 45.0,  "ne": 45.0,
    "northwest": 315.0, "nw": 315.0,
    "southeast": 135.0, "se": 135.0,
    "southwest": 225.0, "sw": 225.0,
    # Relative to aircraft heading (resolved at call time)
    "ahead":  "AHEAD",
    "front":  "AHEAD",
    "forward":"AHEAD",
    "behind": "BEHIND",
    "back":   "BEHIND",
    "left":   "LEFT",
    "right":  "RIGHT",
    # Clock positions (relative)
    "12 o'clock": "AHEAD",
    "3 o'clock":  "RIGHT",
    "6 o'clock":  "BEHIND",
    "9 o'clock":  "LEFT",
    "1 o'clock":  30.0,   # relative — resolved below
    "2 o'clock":  60.0,
    "4 o'clock":  120.0,
    "5 o'clock":  150.0,
    "7 o'clock":  210.0,
    "8 o'clock":  240.0,
    "10 o'clock": 300.0,
    "11 o'clock": 330.0,
}

RELATIVE_LABELS = {"AHEAD": 0.0, "RIGHT": 90.0, "BEHIND": 180.0, "LEFT": 270.0}


def resolve_direction(direction_key: str, heading: float) -> float | None:
    """Return absolute bearing (0–360) for a direction word, given aircraft heading.
    Returns None if direction_key is unknown."""
    key = direction_key.strip().lower()
    val = DIRECTION_OFFSETS.get(key)
    if val is None:
        return None
    if isinstance(val, float):
        return (heading + val) % 360.0
    if val in RELATIVE_LABELS:
        return (heading + RELATIVE_LABELS[val]) % 360.0
    return None


def look_distance_nm(altitude_ft: float) -> float:
    """Dynamic look-ahead distance based on altitude."""
    dist = altitude_ft / 1000.0 * config.LOOK_DISTANCE_MULTIPLIER
    return max(config.MIN_LOOK_DISTANCE_NM, min(config.MAX_LOOK_DISTANCE_NM, dist))


# ── Coordinate projection ──────────────────────────────────────────────────────

_NM_TO_KM = 1.852

def project_point(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> tuple[float, float]:
    """Return (lat, lon) of a point at distance_nm along bearing_deg from (lat, lon).
    Uses spherical Earth approximation — accurate enough for our purposes."""
    d_km = distance_nm * _NM_TO_KM
    R = 6371.0  # Earth radius km
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    brng = math.radians(bearing_deg)
    d_r = d_km / R

    lat2 = math.asin(
        math.sin(lat_r) * math.cos(d_r) +
        math.cos(lat_r) * math.sin(d_r) * math.cos(brng)
    )
    lon2 = lon_r + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(lat_r),
        math.cos(d_r) - math.sin(lat_r) * math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)


# ── Reverse geocoding (Nominatim) ──────────────────────────────────────────────

def _nominatim_rate_limit():
    global _last_nominatim_call
    with _nominatim_lock:
        elapsed = time.time() - _last_nominatim_call
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        _last_nominatim_call = time.time()


def reverse_geocode(lat: float, lon: float) -> dict:
    """Return place information for (lat, lon).
    Returns a dict with keys: display_name, country, state, county, city, etc."""
    _nominatim_rate_limit()
    try:
        location = _geocoder.reverse((lat, lon), exactly_one=True, timeout=10)
        if location is None:
            return {"display_name": f"{lat:.4f}, {lon:.4f}", "raw": {}}
        addr = location.raw.get("address", {})
        return {
            "display_name": location.address,
            "country":  addr.get("country", ""),
            "state":    addr.get("state", ""),
            "county":   addr.get("county", ""),
            "city":     addr.get("city") or addr.get("town") or addr.get("village", ""),
            "suburb":   addr.get("suburb", ""),
            "raw":      addr,
        }
    except GeocoderTimedOut:
        return {"display_name": "Geocoder timed out", "raw": {}}
    except Exception as e:
        return {"display_name": f"Geocoding error: {e}", "raw": {}}


# ── Overpass feature lookup ────────────────────────────────────────────────────

# Feature types we care about, mapped from OSM tags
_FEATURE_TAGS = [
    ("natural", ["peak", "water", "lake", "river", "valley", "cliff", "volcano",
                 "glacier", "bay", "strait", "cape", "island"]),
    ("waterway", ["river", "stream", "canal"]),
    ("highway",  ["motorway", "trunk", "primary", "secondary"]),
    ("place",    ["city", "town", "village", "island"]),
    ("landuse",  ["reservoir"]),
    ("leisure",  ["park", "nature_reserve"]),
    ("boundary", ["national_park", "protected_area"]),
]

def _build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    """Build Overpass QL query to find named features near a point."""
    parts = ['[out:json][timeout:{}];('.format(config.OVERPASS_TIMEOUT)]
    for tag_key, tag_values in _FEATURE_TAGS:
        for val in tag_values:
            parts.append(
                f'  node["{tag_key}"="{val}"]["name"](around:{radius_m},{lat},{lon});'
            )
            parts.append(
                f'  way["{tag_key}"="{val}"]["name"](around:{radius_m},{lat},{lon});'
            )
            parts.append(
                f'  relation["{tag_key}"="{val}"]["name"](around:{radius_m},{lat},{lon});'
            )
    parts.append(');')
    parts.append('out center tags;')
    return '\n'.join(parts)


def nearby_features(lat: float, lon: float, radius_m: int | None = None) -> list[dict]:
    """Query Overpass for named geographic features near (lat, lon).
    Returns list of {name, type, subtype, lat, lon}."""
    if radius_m is None:
        radius_m = config.OVERPASS_RADIUS_M
    query = _build_overpass_query(lat, lon, radius_m)
    for attempt in range(config.OVERPASS_RETRIES + 1):
        try:
            resp = requests.post(
                config.OVERPASS_URL,
                data={"data": query},
                timeout=config.OVERPASS_TIMEOUT + 5,
            )
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break
        except Exception as e:
            if attempt == 2:
                return [{"name": f"Feature lookup error: {e}", "type": "error"}]
            time.sleep(3)
    else:
        return []

    features = []
    seen = set()
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name or name in seen:
            continue
        seen.add(name)

        # Determine feature type label
        ftype = "unknown"
        for key in ["natural", "waterway", "highway", "place", "landuse", "leisure", "boundary"]:
            if key in tags:
                ftype = f"{key}:{tags[key]}"
                break

        # Coordinates
        if el.get("type") == "node":
            feat_lat, feat_lon = el.get("lat", lat), el.get("lon", lon)
        else:  # way / relation have 'center'
            center = el.get("center", {})
            feat_lat = center.get("lat", lat)
            feat_lon = center.get("lon", lon)

        features.append({"name": name, "type": ftype, "lat": feat_lat, "lon": feat_lon})

    # Sort by proximity
    features.sort(key=lambda f: _dist_km(lat, lon, f["lat"], f["lon"]))
    return features[:15]  # cap at 15


def _dist_km(lat1, lon1, lat2, lon2) -> float:
    """Quick approximate distance in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── High-level context builder ─────────────────────────────────────────────────

def get_geographic_context(
    lat: float,
    lon: float,
    altitude_ft: float,
    heading: float,
    direction: str | None = None,
    need_features: bool = True,
) -> dict:
    """Build a geographic context dict for a given position + optional direction.

    need_features: if False, skips the slow Overpass feature lookup entirely.
                   Set False for LOCATION questions (just need place name).

    Runs reverse geocode + feature lookups in parallel when a direction is given.
    """
    ctx: dict = {}

    # Calculate look-point coordinates if a direction was given
    look_lat = look_lon = None
    bearing = dist_nm = None
    if direction:
        bearing = resolve_direction(direction, heading)
        if bearing is not None:
            dist_nm = look_distance_nm(altitude_ft)
            look_lat, look_lon = project_point(lat, lon, bearing, dist_nm)

    # Run all slow network calls in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_geocode_here  = pool.submit(reverse_geocode, lat, lon)
        f_features_here = pool.submit(nearby_features, lat, lon) if need_features else None
        f_geocode_look  = pool.submit(reverse_geocode, look_lat, look_lon) if look_lat is not None else None
        f_features_look = pool.submit(nearby_features, look_lat, look_lon) if (look_lat is not None and need_features) else None

        ctx["position"] = {
            "lat": lat,
            "lon": lon,
            "geocode": f_geocode_here.result(),
        }
        ctx["features_here"] = f_features_here.result() if f_features_here else []

        if look_lat is not None:
            ctx["look_point"] = {
                "lat": look_lat,
                "lon": look_lon,
                "distance_nm": round(dist_nm, 1),
                "bearing": round(bearing, 1),
                "geocode": f_geocode_look.result() if f_geocode_look else {},
                "features": f_features_look.result() if f_features_look else [],
            }

    return ctx


def context_to_text(ctx: dict) -> str:
    """Convert a geographic context dict to a readable string for LLM prompts."""
    lines = []

    pos = ctx.get("position", {})
    geo = pos.get("geocode", {})
    lines.append(f"Current position: {geo.get('display_name', 'unknown')}")

    here = ctx.get("features_here", [])
    if here:
        names = ", ".join(f["name"] for f in here[:5])
        lines.append(f"Nearby features: {names}")

    lp = ctx.get("look_point")
    if lp:
        lgeo = lp.get("geocode", {})
        lines.append(
            f"Looking {lp['bearing']:.0f}° at ~{lp['distance_nm']} nm: "
            f"{lgeo.get('display_name', 'unknown')}"
        )
        feats = lp.get("features", [])
        if feats:
            fnames = ", ".join(f["name"] for f in feats[:5])
            lines.append(f"Features in that direction: {fnames}")

    return "\n".join(lines)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test near Grand Canyon South Rim, looking west
    test_lat, test_lon = 36.0544, -112.1401
    test_alt = 7000
    test_heading = 270.0

    print("=== Reverse geocode current position ===")
    geo = reverse_geocode(test_lat, test_lon)
    print(geo["display_name"])

    print("\n=== Nearby features ===")
    feats = nearby_features(test_lat, test_lon, radius_m=10000)
    for f in feats[:8]:
        print(f"  {f['name']}  ({f['type']})")

    print("\n=== Full context (looking ahead) ===")
    ctx = get_geographic_context(test_lat, test_lon, test_alt, test_heading, direction="ahead")
    print(context_to_text(ctx))
