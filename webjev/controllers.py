"""The decision layer. This is the ONLY thing that differs between the two modes.

  BaselineController - WebGPT-style: the LLM decides the next action and reviews its own draft.
  JevOrchestrator    - Jev decides (Choice + Score + Noul); the LLM only writes text.

Both are used by DecisionLoopAgent (agent.py), which runs the same loop for each. "LLM as a tool for Jev"
(Option 2) needs a different action set, so it is a new Agent subclass in agent.py, not a controller here.
"""
from __future__ import annotations

import json
import time

from . import jev as J
from .llm import LLM
from .state import Decision, Review

ACTIONS = ("search", "clarify", "answer")


class Controller:
    name = "controller"

    def decide(self, snapshot: dict) -> Decision:
        raise NotImplementedError

    def review(self, snapshot: dict, draft: str) -> Review:
        raise NotImplementedError


# --- baseline: the LLM decides everything ---------------------------------------
DECIDE_SYSTEM = (
    "You control a web research agent. Given the agent's state, choose its next action:\n"
    "- search: gather more information with another web search\n"
    "- clarify: ask the user a clarifying question (only when the question is genuinely ambiguous "
    "in a way only the user can resolve)\n"
    "- answer: stop searching and write the final cited answer\n"
    "Give a one- or two-sentence rationale the user will read, your confidence (0-1) that this is the "
    "right action, and how sufficient the current evidence is for a truthful answer (0-1)."
)
DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "rationale": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_sufficiency": {"type": "number"},
    },
    "required": ["action", "rationale", "confidence", "evidence_sufficiency"],
    "additionalProperties": False,
}
REVIEW_SYSTEM = (
    "You review a research agent's draft answer before it is shown to the user. Accept it only if every "
    "factual claim is supported by the listed sources and it actually answers the user's question. "
    "Give a short rationale the user will read."
)
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"accept": {"type": "boolean"}, "rationale": {"type": "string"}},
    "required": ["accept", "rationale"],
    "additionalProperties": False,
}


def _clamp(x) -> float:
    return min(1.0, max(0.0, float(x)))


class BaselineController(Controller):
    def __init__(self, llm: LLM):
        self.llm = llm
        self.name = f"baseline-llm({llm.name})"

    def decide(self, snapshot):
        t0 = time.perf_counter()
        out = self.llm.json("decide", DECIDE_SYSTEM, json.dumps(snapshot, ensure_ascii=False, indent=1),
                            DECIDE_SCHEMA, effort="medium", meta=snapshot)
        return Decision(
            action=out["action"], rationale=out["rationale"], confidence=_clamp(out["confidence"]),
            evidence=_clamp(out["evidence_sufficiency"]), decider=self.name,
            latency_s=time.perf_counter() - t0,
        )

    def review(self, snapshot, draft):
        t0 = time.perf_counter()
        prompt = json.dumps({**snapshot, "draft_answer": draft}, ensure_ascii=False, indent=1)
        out = self.llm.json("review", REVIEW_SYSTEM, prompt, REVIEW_SCHEMA, effort="medium", meta=snapshot)
        return Review(accept=out["accept"], rationale=out["rationale"], latency_s=time.perf_counter() - t0)


# --- Jev as orchestrator -------------------------------------------------------
DECIDE_QUESTIONS = {
    "next_action": J.choice(
        "The best next action for a web research agent answering `question`, given `sources`, "
        "`information_gaps`, `clarifications`, `previous_actions` and `budget`.",
        {
            "search": "`sources` do not yet contain enough reliable evidence to answer `question`, and "
                      "another web search is likely to close an entry in `information_gaps`.",
            "clarify": "`question` is ambiguous or underspecified in a way only the user can resolve, "
                       "and `clarifications` does not already resolve it.",
            "answer": "`sources` already support a truthful, complete answer to `question`, or further "
                      "searching is unlikely to add anything.",
        },
    ),
    "evidence": J.score(
        "How well `sources` support a truthful, complete answer to `question`.",
        [
            "No relevant evidence has been retrieved.",
            "Related material only; nothing addresses the question directly.",
            "Partial evidence: part of the question is answered, or a single weak source supports it.",
            "Good evidence: a reliable source answers the question directly, with minor gaps.",
            "Strong evidence: multiple reliable sources agree and fully answer the question.",
        ],
    ),
    "ambiguous": J.noul(
        "`question` has more than one plausible interpretation that would change the answer, and "
        "`clarifications` does not settle which one the user means."
    ),
}
REVIEW_QUESTIONS = {
    "supported": J.noul("Every factual claim in `draft_answer` is directly supported by the summaries in `sources`."),
    "responsive": J.noul("`draft_answer` directly and completely answers `question` as refined by `clarifications`."),
}


class JevOrchestrator(Controller):
    def __init__(self, backend: J.JevBackend, min_answer_confidence: float = 0.5, review_threshold: float = 0.5):
        self.jev = backend
        self.min_answer_confidence = min_answer_confidence
        self.review_threshold = review_threshold

    @property
    def name(self):
        return f"jev-orchestrator({self.jev.name})"

    def decide(self, snapshot):
        t0 = time.perf_counter()
        answers = self.jev.ask(snapshot, DECIDE_QUESTIONS)
        latency = time.perf_counter() - t0

        pick, evidence, ambiguous = answers["next_action"], answers["evidence"], answers["ambiguous"]["noul"]
        action, confidence = pick["choice"], pick["confidence"]
        probs = pick["probabilities"]
        level = (evidence.get("legend") or {}).get(str(round(evidence["score"])), "")
        dist = ", ".join(f"{k} {probs[k]:.0%}" for k in sorted(probs, key=probs.get, reverse=True))
        rationale = (
            f"Jev chose {action.upper()} with confidence {confidence:.2f} ({dist}). "
            f"Evidence level {evidence['score']:.1f}: {level} "
            f"Probability the question is ambiguous: {ambiguous:.0%}."
        )
        if action == "answer" and confidence < self.min_answer_confidence:
            rationale += (f" Confidence is below the {self.min_answer_confidence:.2f} threshold for answering, "
                          "so the agent searches again instead.")
            action = "search"
        return Decision(
            action=action, rationale=rationale, confidence=confidence,
            evidence=J.normalized_score(evidence), decider=self.name, latency_s=latency,
            details={"probabilities": probs, "evidence_score": evidence["score"],
                     "evidence_level": level, "ambiguity": ambiguous, "jev_choice": pick["choice"]},
        )

    def review(self, snapshot, draft):
        t0 = time.perf_counter()
        answers = self.jev.ask({**snapshot, "draft_answer": draft}, REVIEW_QUESTIONS)
        supported, responsive = answers["supported"]["noul"], answers["responsive"]["noul"]
        accept = min(supported, responsive) >= self.review_threshold
        rationale = (
            f"Jev review: claims supported by sources {supported:.0%}, answers the question {responsive:.0%} "
            f"(both must reach {self.review_threshold:.0%}). " + ("Draft accepted." if accept else "Draft rejected.")
        )
        return Review(accept=accept, rationale=rationale, latency_s=time.perf_counter() - t0,
                      details={"supported": supported, "responsive": responsive})
