"""Run state. `RunState.snapshot()` is exactly what the decision layer sees at each step."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    id: int
    title: str
    url: str
    summary: str = ""
    relevant: bool = True
    fetched: bool = False   # True if the page was opened; False if only the search snippet was used


@dataclass
class Decision:
    action: str                       # search | clarify | answer
    rationale: str                    # human-readable "why", shown in the UI
    confidence: float                 # confidence in the chosen action, 0-1
    evidence: float | None = None     # how well the sources support an answer, 0-1
    details: dict = field(default_factory=dict)
    decider: str = ""
    latency_s: float = 0.0


@dataclass
class Review:
    accept: bool
    rationale: str
    details: dict = field(default_factory=dict)
    latency_s: float = 0.0


@dataclass
class RunState:
    question: str
    clarifications: list[dict] = field(default_factory=list)   # {question, answer}
    searches: list[dict] = field(default_factory=list)         # {query, new_sources}
    sources: list[Source] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)          # {step, action, confidence}
    gaps: list[str] = field(default_factory=list)
    confidence: float | None = None

    def snapshot(self, cfg) -> dict:
        return {
            "question": self.question,
            "clarifications": self.clarifications,
            "search_history": self.searches,
            "sources": [
                {"id": s.id, "title": s.title, "url": s.url, "relevant": s.relevant, "summary": s.summary}
                for s in self.sources
            ],
            "previous_actions": self.actions,
            "information_gaps": self.gaps,
            "confidence": self.confidence,
            "budget": {
                "searches_used": len(self.searches),
                "max_searches": cfg.max_searches,
                "clarifications_used": len(self.clarifications),
                "max_clarifications": cfg.max_clarifications,
            },
        }
