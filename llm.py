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
import ollama
import config


def warmup():
    """Load the model into Ollama's memory at startup so the first real request
    doesn't pay the cold-load penalty.  keep_alive=-1 keeps it resident until
    the Ollama server is restarted or explicitly unloaded."""
    models_needed = {config.CLASSIFIER_MODEL, config.ANSWER_MODEL}
    for model in models_needed:
        print(f"[LLM] Warming up {model}...", flush=True)
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.0},
            keep_alive=-1,
        )
    print("[LLM] Models ready.", flush=True)


def _chat(model: str, system: str, user: str) -> str:
    """Send a chat request to Ollama and return the assistant's reply as a string."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options={"temperature": 0.3},  # lower = more factual, less creative
        keep_alive=-1,                 # keep model in VRAM between calls
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
