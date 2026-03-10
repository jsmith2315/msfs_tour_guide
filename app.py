# app.py — FastAPI web server.
#
# Endpoints:
#   GET  /              → serves static/index.html
#   GET  /telemetry     → current flight data as JSON
#   POST /ask           → ask the tour guide a question, returns answer
#   POST /clear         → clear conversation history
#
# Start:  python app.py
# Then open:  http://localhost:8000

from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

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
    diagnostics: list[dict]  # [{step, detail, elapsed_ms}, ...]


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
    return AskResponse(answer=result["answer"], question=question, diagnostics=result["diagnostics"])


@app.post("/clear")
def clear_history():
    if _guide is None:
        raise HTTPException(503, "Tour guide not initialised")
    _guide.clear_history()
    return {"status": "history cleared"}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting MSFS Tour Guide at http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run(
        "app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
    )
