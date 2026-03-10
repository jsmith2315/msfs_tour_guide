# history.py — Sliding-window conversation history.
#
# Keeps the last N question/answer pairs in memory so the LLM can
# answer follow-up questions with context.
#
# Usage:
#   from history import ConversationHistory
#   hist = ConversationHistory()
#   hist.add("Where are we?", "You are flying over the Grand Canyon.")
#   print(hist.to_prompt_text())

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import config


@dataclass
class Turn:
    question: str
    answer:   str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class ConversationHistory:
    def __init__(self, max_turns: int | None = None):
        self._max = max_turns if max_turns is not None else config.HISTORY_MAX_TURNS
        self._turns: list[Turn] = []

    def add(self, question: str, answer: str) -> None:
        """Add a Q&A pair. Drops the oldest turn if at capacity."""
        self._turns.append(Turn(question=question, answer=answer))
        if len(self._turns) > self._max:
            self._turns.pop(0)

    def to_prompt_text(self) -> str:
        """Format history as a string suitable for injecting into an LLM prompt."""
        if not self._turns:
            return ""
        lines = ["--- Conversation history ---"]
        for t in self._turns:
            lines.append(f"User: {t.question}")
            lines.append(f"Guide: {t.answer}")
        lines.append("--- End of history ---")
        return "\n".join(lines)

    def to_messages(self) -> list[dict]:
        """Format history as a list of chat messages (for multi-turn chat APIs)."""
        msgs = []
        for t in self._turns:
            msgs.append({"role": "user",      "content": t.question})
            msgs.append({"role": "assistant", "content": t.answer})
        return msgs

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)
