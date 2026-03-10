# orchestrator.py — The main Q&A pipeline.
#
# Ties all components together:
#   classify question -> gather context -> generate answer -> store in history
#
# Usage:
#   from orchestrator import TourGuide
#   guide = TourGuide()
#   result = guide.ask("What mountain is that ahead?")
#   print(result["answer"])
#   print(result["diagnostics"])  # list of {step, detail, elapsed_ms}
#
# Run standalone (terminal chat with mock telemetry):  python orchestrator.py

from __future__ import annotations
import time

import config
import llm
import geo
import search as search_mod
from telemetry import get_telemetry_source, FlightData
from history import ConversationHistory
from classifier import classify_question

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an enthusiastic and knowledgeable aviation tour guide flying alongside the user
in Microsoft Flight Simulator 2024. Your job is to answer questions about what the
user sees out the window.

Guidelines:
- Respond like a real tour guide or knowledgeable co-pilot — natural, conversational, friendly.
- NEVER recite raw flight data (coordinates, altitude, airspeed, heading numbers, etc.) unless
  the user explicitly asks for that information. That data is given to you as background context
  only, so you know where the plane is — not for you to read back to the user.
- Focus your answer on the interesting geography, history, or features — not the flight instruments.
- Keep answers concise. Simple questions like "where are we?" deserve a short answer
  (1-2 sentences). Only elaborate if the user asks for more detail or history.
- Don't open with "Welcome aboard!" or similar every time — just answer the question.
- If directional context is given, focus on what's in that direction.
- ONLY describe features, landmarks, and activities that are explicitly mentioned in the
  geographic context or search results provided. Do NOT invent, extrapolate, or add details
  not present in the data. If the data doesn't confirm something, don't say it.
- If you are not sure about something, say so rather than inventing details.
""".strip()


class TourGuide:
    def __init__(self):
        self._telem  = get_telemetry_source()
        self._hist   = ConversationHistory()

    def ask(self, question: str) -> dict:
        """Process a user question. Returns:
          {
            "answer": str,
            "diagnostics": [{"step": str, "detail": str, "elapsed_ms": int}, ...]
          }
        """
        diag = []
        t_total = time.perf_counter()

        # 1. Get current flight data
        t = time.perf_counter()
        flight = self._telem.snapshot()
        diag.append(_d("Telemetry", f"lat={flight.lat:.4f} lon={flight.lon:.4f} alt={flight.altitude:.0f}ft hdg={flight.heading:.0f}°", t))

        # 2. Classify the question
        t = time.perf_counter()
        classification = classify_question(question)
        qtype     = classification["question_type"]
        direction = classification.get("direction")
        diag.append(_d("Classify", f"type={qtype}  direction={direction}  (model: {config.CLASSIFIER_MODEL})", t))

        # 3. Build geographic context
        # Skip Overpass feature lookup for LOCATION questions — we only need the place name.
        need_features = qtype not in ("LOCATION", "WEATHER", "AIRCRAFT")
        t = time.perf_counter()
        geo_ctx = geo.get_geographic_context(
            lat=flight.lat,
            lon=flight.lon,
            altitude_ft=flight.altitude,
            heading=flight.heading,
            direction=direction,
            need_features=need_features,
        )
        geo_text = geo.context_to_text(geo_ctx)
        diag.append(_d("Geo lookup", geo_text.split("\n")[0], t))  # first line as summary

        # 4. Web search (for HISTORY, FEATURE, and GENERAL — cast a wide net)
        search_text = ""
        if qtype in ("HISTORY", "FEATURE", "GENERAL"):
            t = time.perf_counter()
            place_name = (
                geo_ctx.get("look_point", {}).get("geocode", {}).get("display_name")
                or geo_ctx.get("position", {}).get("geocode", {}).get("display_name")
                or f"{flight.lat:.4f}, {flight.lon:.4f}"
            )
            search_query = _build_search_query(question, place_name, qtype)
            results = search_mod.web_search(search_query)
            search_text = search_mod.results_to_text(results)
            diag.append(_d("Web search", f'"{search_query}" → {len(results)} results', t))
        else:
            diag.append({"step": "Web search", "detail": "skipped (not needed for this question type)", "elapsed_ms": 0})

        # 5. Build prompt context
        flight_note = f"Aircraft: {flight.aircraft}."
        if flight.on_ground:
            flight_note += " Currently on the ground."
        context_parts = [
            "## Flight Context",
            flight_note,
            "",
            "## Geographic Context",
            geo_text,
        ]
        if search_text:
            context_parts += ["", "## Web Search Results", search_text]
        if len(self._hist) > 0:
            context_parts += ["", self._hist.to_prompt_text()]

        context_block = "\n".join(context_parts)
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"User question: {question}"
        )

        # 6. Generate answer
        t = time.perf_counter()
        response = llm.answer(_SYSTEM_PROMPT, user_prompt)
        diag.append(_d("LLM answer", f"model: {config.ANSWER_MODEL}  ({len(response)} chars)", t))

        # 7. Store in history
        self._hist.add(question, response)

        total_ms = int((time.perf_counter() - t_total) * 1000)
        diag.append({"step": "TOTAL", "detail": "", "elapsed_ms": total_ms})

        return {"answer": response, "diagnostics": diag}

    def clear_history(self):
        self._hist.clear()

    def close(self):
        self._telem.close()


def _d(step: str, detail: str, t_start: float) -> dict:
    """Build a diagnostic entry with elapsed ms since t_start."""
    return {
        "step":       step,
        "detail":     detail,
        "elapsed_ms": int((time.perf_counter() - t_start) * 1000),
    }


def _build_search_query(question: str, place_name: str, qtype: str) -> str:
    clean_q = question.strip().rstrip("?")
    if qtype == "HISTORY":
        return f"history of {place_name}"
    if qtype == "FEATURE":
        return f"{clean_q} near {place_name}"
    return f"{clean_q} {place_name}"


# ── Standalone terminal chat ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MSFS Tour Guide — terminal mode")
    print(f"Mock telemetry: {config.USE_MOCK_TELEMETRY} ({config.MOCK_LOCATION_NAME})")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    guide = TourGuide()
    try:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            result = guide.ask(question)
            print(f"\nGuide: {result['answer']}\n")
            for d in result["diagnostics"]:
                ms = f"{d['elapsed_ms']}ms" if d["elapsed_ms"] else ""
                print(f"  [{d['step']}] {ms}  {d['detail']}")
            print()
    finally:
        guide.close()
