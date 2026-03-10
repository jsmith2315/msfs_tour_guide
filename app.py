# app.py — FastAPI web server.
#
# Endpoints:
#   GET  /              → serves static/index.html
#   GET  /telemetry     → current flight data as JSON
#   POST /ask           → ask the tour guide a question, returns answer
#   POST /clear         → clear conversation history
#   GET  /config        → server config flags (e.g. test_mode)
#   POST /save-log      → append a Q&A review entry to review_log.jsonl (TEST_MODE only)
#
# Start:  python app.py
# Then open:  http://localhost:8000

from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

import config
import llm
from orchestrator import TourGuide

# ── App state ──────────────────────────────────────────────────────────────────

_guide: TourGuide | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _guide
    llm.warmup()       # load model into Ollama VRAM before first question
    _guide = TourGuide()
    yield
    if _guide:
        _guide.close()
    llm.unload()  # evict models from VRAM on clean shutdown


app = FastAPI(title="MSFS Tour Guide", lifespan=lifespan)

# Serve static files (index.html, etc.)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Models ─────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    question: str
    diagnostics: list[dict]       # [{step, detail, elapsed_ms}, ...]
    # Structured fields for test/review logging (always present)
    timestamp: str
    question_type: str
    direction: str | None
    location: str
    flight_snapshot: dict
    search_ran: bool
    search_query: str | None
    overpass_features: list[str]
    mock_mode: bool


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/telemetry")
def get_telemetry():
    if _guide is None:
        raise HTTPException(503, "Tour guide not initialised")
    result = _guide._telem.snapshot().to_dict()
    # Add connection state so the UI can show "waiting for simulator"
    result["sim_connected"] = getattr(_guide._telem, "connected", True)
    return result


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if _guide is None:
        raise HTTPException(503, "Tour guide not initialised")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")
    result = _guide.ask(question)
    return AskResponse(
        answer=result["answer"],
        question=question,
        diagnostics=result["diagnostics"],
        timestamp=result["timestamp"],
        question_type=result["question_type"],
        direction=result["direction"],
        location=result["location"],
        flight_snapshot=result["flight_snapshot"],
        search_ran=result["search_ran"],
        search_query=result["search_query"],
        overpass_features=result["overpass_features"],
        mock_mode=result["mock_mode"],
    )


@app.post("/clear")
def clear_history():
    if _guide is None:
        raise HTTPException(503, "Tour guide not initialised")
    _guide.clear_history()
    return {"status": "history cleared"}


@app.get("/config")
def get_config():
    return {"test_mode": config.TEST_MODE}


@app.post("/save-log")
def save_log(entry: dict):
    if not config.TEST_MODE:
        raise HTTPException(403, "TEST_MODE is not enabled")
    log_path = Path(__file__).parent / config.LOG_FILE
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "saved"}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting MSFS Tour Guide at http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run(
        "app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
    )
