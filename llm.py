# llm.py — Ollama interface.
#
# Two entry points:
#   classify(prompt)  → uses the small/fast CLASSIFIER_MODEL
#   answer(prompt)    → uses the larger ANSWER_MODEL
#
# Both return plain strings.
#
# Run standalone to test:  python llm.py

from __future__ import annotations
import threading
import ollama
import config

# Model stay-alive strategy:
#   - Each chat call sets keep_alive=_KEEP_ALIVE_SEC (1 hour).
#   - A background thread pings every 30 min to reset the timer while the
#     app is running, so the model never unloads mid-session.
#   - On clean shutdown, unload() sends keep_alive=0 to evict immediately.
_KEEP_ALIVE_SEC    = 3600  # 1 hour — how long Ollama holds the model after last use
_PING_INTERVAL_SEC = 1800  # 30 min — how often the keepalive thread resets the timer

_keepalive_stop   = threading.Event()
_keepalive_thread: threading.Thread | None = None


def _keepalive_loop(models: set[str]) -> None:
    """Runs in a daemon thread. Pings each model every 30 min to prevent eviction."""
    while not _keepalive_stop.wait(_PING_INTERVAL_SEC):
        for model in models:
            try:
                ollama.generate(model=model, prompt="", keep_alive=_KEEP_ALIVE_SEC)
            except Exception:
                pass  # Ollama may be briefly busy — next ping will succeed


def warmup() -> None:
    """Load models into Ollama VRAM at startup and start the keepalive thread.
    Models stay loaded for 1 hour after last use; the keepalive ping resets
    that timer every 30 min while the app is running."""
    global _keepalive_thread
    models_needed = {config.CLASSIFIER_MODEL, config.ANSWER_MODEL}  # set deduplicates
    for model in models_needed:
        print(f"[LLM] Warming up {model}...", flush=True)
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.0},
            keep_alive=_KEEP_ALIVE_SEC,
        )
    print("[LLM] Models ready.", flush=True)

    _keepalive_stop.clear()
    _keepalive_thread = threading.Thread(
        target=_keepalive_loop, args=(models_needed,),
        daemon=True, name="OllamaKeepalive",
    )
    _keepalive_thread.start()


def unload() -> None:
    """Evict models from Ollama VRAM on shutdown (keep_alive=0 = unload now)."""
    _keepalive_stop.set()  # stop the ping thread
    models_needed = {config.CLASSIFIER_MODEL, config.ANSWER_MODEL}
    for model in models_needed:
        try:
            print(f"[LLM] Unloading {model}...", flush=True)
            ollama.generate(model=model, prompt="", keep_alive=0)
        except Exception:
            pass
    print("[LLM] Models unloaded.", flush=True)


def _chat(model: str, system: str, user: str) -> str:
    """Send a chat request to Ollama and return the assistant's reply as a string."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options={"temperature": 0.3},  # lower = more factual, less creative
        keep_alive=_KEEP_ALIVE_SEC,    # reset the 1-hour eviction timer on each call
    )
    return response["message"]["content"].strip()


def classify(prompt: str) -> str:
    """Run a classification or parsing task with the small fast model."""
    return _chat(
        model=config.CLASSIFIER_MODEL,
        system=(
            "You are a precise classification assistant. "
            "Follow the instructions exactly and output only what is requested. "
            "Do not add explanations or extra text."
        ),
        user=prompt,
    )


def answer(system_prompt: str, user_question: str) -> str:
    """Generate a tour guide answer with the larger model."""
    return _chat(
        model=config.ANSWER_MODEL,
        system=system_prompt,
        user=user_question,
    )


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing LLM connection...\n")

    print("=== Classifier model ===")
    resp = classify("What is the capital of France? Answer with just the city name.")
    print(f"Response: {resp}\n")

    print("=== Answer model ===")
    resp = answer(
        system_prompt="You are a friendly aviation tour guide.",
        user_question="What is the Grand Canyon and why is it famous?",
    )
    print(f"Response: {resp}")
