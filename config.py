# config.py — All tweakable settings in one place.
# Change values here to tune behaviour without touching any logic files.

# ── Ollama models ─────────────────────────────────────────────────────────────
CLASSIFIER_MODEL = "llama3.2:latest"  # Same model for both — avoids Ollama reload penalty
ANSWER_MODEL     = "llama3.2:latest"  # Quality answer model
# Other available models: llama3.2:latest (2GB, faster), dolphin-mixtral:8x7b (26GB, best quality)
OLLAMA_BASE_URL  = "http://localhost:11434"

# ── Telemetry ─────────────────────────────────────────────────────────────────
TELEMETRY_POLL_INTERVAL_SEC = 2.0  # How often to read SimConnect data

# Set to True to use mock data instead of connecting to MSFS
USE_MOCK_TELEMETRY = False
# Which location from mock_locations.py to use when mocking
MOCK_LOCATION_NAME = "Grand Canyon South Rim"

# ── Geographic look-ahead distance ────────────────────────────────────────────
# When user asks about something "ahead" / "left" / etc. we project a point
# at this distance from the aircraft and look up what's there.
# Formula: look_nm = clamp(altitude_ft / 1000 * LOOK_DISTANCE_MULTIPLIER,
#                          MIN_LOOK_DISTANCE_NM, MAX_LOOK_DISTANCE_NM)
LOOK_DISTANCE_MULTIPLIER = 3   # nautical miles per 1000 ft of altitude
MIN_LOOK_DISTANCE_NM = 1
MAX_LOOK_DISTANCE_NM = 50

# ── OpenStreetMap / Overpass ──────────────────────────────────────────────────
NOMINATIM_USER_AGENT = "msfs-tour-guide/1.0"
OVERPASS_URL      = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT  = 6     # seconds per attempt — fail fast rather than hang
OVERPASS_RETRIES  = 1     # max retry attempts (1 = try twice total)
# Radius (metres) around a projected point to search for OSM features
OVERPASS_RADIUS_M = 5000

# ── Web search ────────────────────────────────────────────────────────────────
SEARCH_MAX_RESULTS = 5         # Number of DuckDuckGo snippets to fetch

# ── Conversation history ──────────────────────────────────────────────────────
HISTORY_MAX_TURNS = 10         # How many Q&A pairs to keep in context

# ── Web server ────────────────────────────────────────────────────────────────
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
