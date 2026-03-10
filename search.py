# search.py — DuckDuckGo web search wrapper.
#
# Usage:
#   from search import web_search
#   results = web_search("history of Grand Canyon Arizona")
#   # returns list of {title, body, href}
#
# Run standalone to test:  python search.py

from __future__ import annotations
from ddgs import DDGS
import config


def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search DuckDuckGo and return text snippets.

    Returns list of dicts with keys: title, body, href
    Returns empty list on any error.
    """
    if max_results is None:
        max_results = config.SEARCH_MAX_RESULTS
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"[search] Search failed: {e}")
        return []


def results_to_text(results: list[dict]) -> str:
    """Convert search results to a readable string for LLM prompts."""
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body  = r.get("body", "")
        lines.append(f"{i}. {title}: {body}")
    return "\n".join(lines)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "Grand Canyon Arizona history geology"
    print(f"Searching: {query!r}\n")
    results = web_search(query)
    print(results_to_text(results))
