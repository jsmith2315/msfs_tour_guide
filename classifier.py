# classifier.py — Classify user questions using the small LLM.
#
# Determines:
#   - question_type: what kind of information the user wants
#   - direction: spatial direction the user is asking about (if any)
#
# Run standalone to test:  python classifier.py

from __future__ import annotations
import json
import re
import llm


# ── Question types ─────────────────────────────────────────────────────────────

QUESTION_TYPES = {
    "LOCATION":  "User wants to know where they are (city, country, region, etc.)",
    "FEATURE":   "User is asking about a specific geographic or man-made feature (mountain, lake, road, bridge, city, etc.)",
    "HISTORY":   "User wants historical information about a place, feature, or area",
    "AIRCRAFT":  "User is asking about the aircraft they are flying",
    "WEATHER":   "User is asking about current weather or conditions",
    "GENERAL":   "Any other question or general curiosity",
}

# ── Direction vocabulary (plain English → canonical key) ──────────────────────

DIRECTION_VOCAB = [
    "ahead", "front", "forward",
    "behind", "back",
    "left", "right",
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "ne", "nw", "se", "sw",
    "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock",
    "5 o'clock", "6 o'clock", "7 o'clock", "8 o'clock",
    "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock",
]

# ── Classifier prompt ─────────────────────────────────────────────────────────

_CLASSIFY_TEMPLATE = """
Classify the following user question asked during a flight in Microsoft Flight Simulator.

Question: "{question}"

Rules:
- LOCATION: user asks where they are (region, city, country, airspace)
- FEATURE: user asks what a specific visible thing IS (mountain, lake, road, bridge, building, etc.)
- HISTORY: user asks for history, background, or facts about a place or thing
- AIRCRAFT: user asks about the aircraft they are flying
- WEATHER: user asks about weather, visibility, wind, or conditions
- GENERAL: anything else (speed, altitude, calculations, general chat)

Direction rules:
- Only set direction if the question EXPLICITLY mentions a direction word ("ahead", "left", "north", "3 o'clock", etc.)
- If no direction is mentioned, set direction to null

Examples:
  "Where are we?"                  -> {{"question_type": "LOCATION",  "direction": null}}
  "What is that mountain to my left?" -> {{"question_type": "FEATURE",   "direction": "left"}}
  "Tell me the history of this area"  -> {{"question_type": "HISTORY",   "direction": null}}
  "What plane are we flying?"      -> {{"question_type": "AIRCRAFT",  "direction": null}}
  "What lake is ahead?"            -> {{"question_type": "FEATURE",   "direction": "ahead"}}
  "What's the weather like?"       -> {{"question_type": "WEATHER",   "direction": null}}
  "Tell me the history of this area" -> {{"question_type": "HISTORY",  "direction": null}}
  "What's the history of this place" -> {{"question_type": "HISTORY",  "direction": null}}
  "What river is to the north?"    -> {{"question_type": "FEATURE",   "direction": "north"}}

Output a JSON object with exactly these two fields:
  "question_type": one of {types}
  "direction": the spatial direction mentioned in the question, or null if none

Direction must be one of: {directions}

Respond with ONLY the raw JSON object, no markdown, no extra text.
""".strip()


def classify_question(question: str) -> dict:
    """Return dict with 'question_type' and 'direction' (or None)."""
    types_list = ", ".join(QUESTION_TYPES.keys())
    directions_list = ", ".join(f'"{d}"' for d in DIRECTION_VOCAB) + ", null"

    prompt = _CLASSIFY_TEMPLATE.format(
        question=question,
        types=types_list,
        directions=directions_list,
    )

    raw = llm.classify(prompt)

    # Try to parse the JSON — be lenient in case model adds backticks etc.
    json_str = _extract_json(raw)
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: safe defaults
        result = {"question_type": "GENERAL", "direction": None}

    # Normalise
    qt = result.get("question_type", "GENERAL").upper()
    if qt not in QUESTION_TYPES:
        qt = "GENERAL"
    direction = result.get("direction")
    if direction:
        direction = direction.strip().lower()
        if direction not in DIRECTION_VOCAB:
            direction = None

    return {"question_type": qt, "direction": direction}


def _extract_json(text: str) -> str:
    """Pull out the first {...} block from model output."""
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    return match.group(0) if match else text


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "Where are we right now?",
        "What is that mountain to my left?",
        "Tell me the history of this area.",
        "What plane are we flying?",
        "What's that lake ahead of us?",
        "What's the weather like?",
        "What river is that to the north?",
        "What's at 3 o'clock?",
        "Tell me about the Grand Canyon.",
        "How fast are we going?",
    ]

    print("Classifying test questions...\n")
    for q in test_questions:
        result = classify_question(q)
        print(f"Q: {q!r}")
        print(f"   -> type={result['question_type']}, direction={result['direction']}\n")
