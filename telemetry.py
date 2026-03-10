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
import threading
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

_SIMCONNECT_RETRY_SEC = 20  # seconds between reconnect attempts

class SimConnectTelemetry:
    """Reads live flight data from MSFS 2024 via SimConnect.

    Starts immediately regardless of whether MSFS is running.  A background
    thread keeps trying to (re)connect every 20 seconds.  snapshot() always
    returns the latest data; check .connected to know if it's live.
    """

    # Map FlightData field → SimConnect variable name
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
        self._lock      = threading.Lock()
        self._data      = FlightData()   # last known good snapshot
        self._connected = False
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True, name="SimConnectPoller")
        self._thread.start()

    @property
    def connected(self) -> bool:
        return self._connected

    # ── background thread ──────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.is_set():
            sm = aq = None
            try:
                from SimConnect import SimConnect, AircraftRequests
                sm = SimConnect()
                aq = AircraftRequests(sm, _time=2000)
                self._connected = True
                print("[SimConnectTelemetry] Connected to MSFS.", flush=True)

                # Poll loop — runs until an error or stop request
                while not self._stop.is_set():
                    try:
                        snap = self._read(aq)
                        with self._lock:
                            self._data = snap
                    except Exception as e:
                        print(f"[SimConnectTelemetry] Poll error: {e}", flush=True)
                        break
                    self._stop.wait(config.TELEMETRY_POLL_INTERVAL_SEC)

            except Exception as e:
                print(f"[SimConnectTelemetry] Cannot connect: {e}", flush=True)
            finally:
                self._connected = False
                if sm is not None:
                    try:
                        sm.exit()
                    except Exception:
                        pass

            if not self._stop.is_set():
                print(f"[SimConnectTelemetry] Retrying in {_SIMCONNECT_RETRY_SEC}s...", flush=True)
                self._stop.wait(_SIMCONNECT_RETRY_SEC)

    def _read(self, aq) -> FlightData:
        """Read one snapshot from SimConnect. Raises on critical failure."""
        data = FlightData()
        for field, var_name in self._VARS.items():
            try:
                val = aq.get(var_name)
                if val is not None:
                    setattr(data, field, float(val))
            except Exception:
                pass

        try:
            title = aq.get("TITLE")
            data.aircraft = str(title) if title else "Unknown"
        except Exception:
            data.aircraft = "Unknown"

        try:
            og = aq.get("SIM_ON_GROUND")
            data.on_ground = bool(og)
        except Exception:
            pass

        # When MSFS closes, SimConnect stops raising and just returns zeros.
        # Lat=0 and lon=0 simultaneously is a reliable disconnection signal
        # (no real airport or flight is at 0°N, 0°E).
        if data.lat == 0.0 and data.lon == 0.0:
            raise ConnectionError("Telemetry all-zero — simulator likely disconnected")

        return data

    # ── public API ─────────────────────────────────────────────────────────────

    def snapshot(self) -> FlightData:
        with self._lock:
            return self._data

    def close(self):
        self._stop.set()


# ── Factory function ───────────────────────────────────────────────────────────

def get_telemetry_source() -> MockTelemetry | SimConnectTelemetry:
    """Return the appropriate telemetry source based on config.
    SimConnectTelemetry starts a background thread and returns immediately —
    it will connect to MSFS whenever it becomes available."""
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
