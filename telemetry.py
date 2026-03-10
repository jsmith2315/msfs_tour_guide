# telemetry.py — Flight data from SimConnect (or mock data for testing).
#
# Usage:
#   from telemetry import get_telemetry_source
#   telem = get_telemetry_source()   # returns Mock or SimConnect based on config
#   data  = telem.snapshot()         # dict of current flight data
#
# Run standalone to test:  python telemetry.py

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Optional

import config
from mock_locations import MOCK_LOCATIONS_BY_NAME


@dataclass
class FlightData:
    """All flight data used by the tour guide."""
    lat:           float = 0.0   # decimal degrees
    lon:           float = 0.0
    altitude:      float = 0.0   # feet MSL
    altitude_agl:  float = 0.0   # feet above ground level
    heading:       float = 0.0   # degrees true (0 = N, 90 = E, ...)
    airspeed:      float = 0.0   # knots indicated
    groundspeed:   float = 0.0   # knots
    vertical_speed: float = 0.0  # feet per minute (negative = descending)
    aircraft:      str   = ""    # e.g. "Cessna 172 Skyhawk"
    on_ground:     bool  = False
    # Optional enrichment fields (filled when available)
    wind_speed:    float = 0.0   # knots
    wind_dir:      float = 0.0   # degrees
    temperature:   float = 15.0  # Celsius
    visibility:    float = 9999.0  # metres
    sim_time:      str   = ""    # "HH:MM" local sim time

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Short human-readable summary for use in prompts."""
        direction = _heading_to_cardinal(self.heading)
        state = "on the ground" if self.on_ground else f"at {self.altitude:,.0f} ft MSL ({self.altitude_agl:,.0f} ft AGL)"
        vs = ""
        if not self.on_ground and abs(self.vertical_speed) > 50:
            vs = f", {'climbing' if self.vertical_speed > 0 else 'descending'} at {abs(self.vertical_speed):,.0f} fpm"
        return (
            f"Aircraft: {self.aircraft}. "
            f"Position: {self.lat:.4f}°N, {self.lon:.4f}°E. "
            f"{state}, heading {self.heading:.0f}° ({direction}){vs}. "
            f"Airspeed: {self.airspeed:.0f} kts. "
            f"Visibility: {self.visibility/1000:.1f} km."
        )


def _heading_to_cardinal(heading: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(heading / 22.5) % 16
    return dirs[idx]


# ── Mock telemetry ─────────────────────────────────────────────────────────────

class MockTelemetry:
    """Returns static flight data from mock_locations.py.
    Perfect for testing the full pipeline without MSFS running."""

    def __init__(self, location_name: str):
        if location_name not in MOCK_LOCATIONS_BY_NAME:
            available = ", ".join(MOCK_LOCATIONS_BY_NAME.keys())
            raise ValueError(
                f"Unknown mock location '{location_name}'. "
                f"Available: {available}"
            )
        self._loc = MOCK_LOCATIONS_BY_NAME[location_name]
        print(f"[MockTelemetry] Using location: {location_name}")

    def snapshot(self) -> FlightData:
        loc = self._loc
        return FlightData(
            lat=loc["lat"],
            lon=loc["lon"],
            altitude=loc["altitude"],
            altitude_agl=loc["altitude_agl"],
            heading=loc["heading"],
            airspeed=loc["airspeed"],
            groundspeed=loc["groundspeed"],
            vertical_speed=loc["vertical_speed"],
            aircraft=loc["aircraft"],
            on_ground=loc["on_ground"],
        )

    def close(self):
        pass


# ── SimConnect telemetry ───────────────────────────────────────────────────────

class SimConnectTelemetry:
    """Reads live flight data from MSFS 2024 via SimConnect."""

    # Map FlightData field → SimConnect variable name (underscores, as used by the Python lib)
    _VARS = {
        "lat":            "PLANE_LATITUDE",
        "lon":            "PLANE_LONGITUDE",
        "altitude":       "PLANE_ALTITUDE",
        "altitude_agl":   "PLANE_ALT_ABOVE_GROUND",
        "heading":        "PLANE_HEADING_DEGREES_TRUE",
        "airspeed":       "AIRSPEED_INDICATED",
        "groundspeed":    "GPS_GROUND_SPEED",
        "vertical_speed": "VERTICAL_SPEED",
        "wind_speed":     "AMBIENT_WIND_VELOCITY",
        "wind_dir":       "AMBIENT_WIND_DIRECTION",
        "temperature":    "AMBIENT_TEMPERATURE",
        "visibility":     "AMBIENT_VISIBILITY",
    }

    def __init__(self):
        try:
            from SimConnect import SimConnect, AircraftRequests
            self._sm = SimConnect()
            self._aq = AircraftRequests(self._sm, _time=2000)
            print("[SimConnectTelemetry] Connected to MSFS.")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to MSFS SimConnect: {e}\n"
                "Make sure MSFS 2024 is running and SimConnect is enabled."
            ) from e

    def snapshot(self) -> FlightData:
        data = FlightData()
        for field, var_name in self._VARS.items():
            try:
                val = self._aq.get(var_name)
                if val is not None:
                    setattr(data, field, float(val))
            except Exception:
                pass  # leave default value if a variable fails

        # Aircraft title is a string, handle separately
        try:
            title = self._aq.get("TITLE")
            data.aircraft = str(title) if title else "Unknown"
        except Exception:
            data.aircraft = "Unknown"

        # on_ground is a bool
        try:
            og = self._aq.get("SIM_ON_GROUND")
            data.on_ground = bool(og)
        except Exception:
            pass

        return data

    def close(self):
        try:
            self._sm.exit()
        except Exception:
            pass


# ── Factory function ───────────────────────────────────────────────────────────

def get_telemetry_source() -> MockTelemetry | SimConnectTelemetry:
    """Return the appropriate telemetry source based on config."""
    if config.USE_MOCK_TELEMETRY:
        return MockTelemetry(config.MOCK_LOCATION_NAME)
    return SimConnectTelemetry()


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing telemetry — press Ctrl+C to stop.\n")
    telem = get_telemetry_source()
    try:
        while True:
            data = telem.snapshot()
            print(data.summary())
            time.sleep(config.TELEMETRY_POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        telem.close()
