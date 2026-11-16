"""Jev: TypeSafe's structured decision model.

Wire format (docs.typesafe.ai): POST {state, model, questions} -> {model, answers, usage}.
  choice -> {choice, probabilities, confidence}
  score  -> {score, legend, probabilities, confidence}   (score = probability-weighted mean level)
  noul   -> {noul}                                       (0-1 probability of "yes", no confidence)

Three interchangeable backends return answers in that same format:
  JevHTTP  - the real API (needs TYPESAFE_API_KEY)
  LLMJev   - an LLM emulating Jev's I/O. NOT calibrated; a stand-in until the key arrives.
  MockJev  - deterministic heuristics for offline tests.
"""
from __future__ import annotations

import json
import os
import time

import requests

from .llm import LLM, mock_policy


class JevError(RuntimeError):
    pass


# --- question builders -------------------------------------------------------
def choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


# --- helpers -----------------------------------------------------------------
def concentration(probabilities: dict) -> float:
    """Confidence as concentration of the distribution: 1 when one option holds all the mass.
    Matches the formula the Jev docs give for three options, (3*max - 1) / 2."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (n * max(probabilities.values()) - 1) / (n - 1))


def noul_confidence(p: float) -> float:
    """Nouls carry no confidence field; use distance from 0.5."""
    return abs(2 * p - 1)


def normalized_score(answer: dict) -> float:
    """Map a score answer onto 0-1 regardless of whether levels are 0- or 1-indexed."""
    keys = sorted(int(k) for k in (answer.get("legend") or answer.get("probabilities") or {}))
    if len(keys) < 2:
        return 0.0
    return min(1.0, max(0.0, (answer["score"] - keys[0]) / (keys[-1] - keys[0])))


def _wire_answers(questions: dict, raw: dict) -> dict:
    """Turn raw per-option weights into Jev's answer format (used by the stand-in backends)."""
    answers = {}
    for qid, q in questions.items():
        value = raw[qid]
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": min(1.0, max(0.0, float(value)))}
            continue
        total = sum(max(0.0, float(v)) for v in value.values()) or 1.0
        probs = {k: max(0.0, float(v)) / total for k, v in value.items()}
        if q["type"] == "choice":
            answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get),
                            "probabilities": probs, "confidence": concentration(probs)}
        else:
            answers[qid] = {"type": "score", "score": sum(int(k) * p for k, p in probs.items()),
                            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                            "probabilities": probs, "confidence": concentration(probs)}
    return answers


# --- backends ----------------------------------------------------------------
class JevBackend:
    name = "jev"

    def __init__(self):
        self.calls = 0

    def ask(self, state: dict, questions: dict) -> dict:
        raise NotImplementedError


class JevHTTP(JevBackend):
    def __init__(self, url: str, model: str, api_key: str | None = None):
        super().__init__()
        self.url, self.model = url, model
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY is not set (add it to .env).")
        self.name = model
        self.input_tokens = 0
        self.output_tokens = 0

    def ask(self, state, questions):
        payload = {"state": state, "model": self.model, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(4):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
            except requests.RequestException as exc:
                if attempt == 3:
                    raise JevError(f"Could not reach Jev: {exc}") from exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == 3:
                    raise JevError(f"Jev returned {resp.status_code}: {resp.text[:300]}")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                raise JevError(f"Jev returned {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            self.calls += 1
            self.name = body.get("model", self.model)
            usage = body.get("usage") or {}
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            missing = [qid for qid in questions if qid not in body.get("answers", {})]
            if missing:
                raise JevError(f"Jev response is missing answers for {missing}: {str(body)[:300]}")
            return body["answers"]
        raise JevError("unreachable")


class LLMJev(JevBackend):
    """Stand-in: an LLM constrained to Jev's question types. Its probabilities are NOT calibrated."""

    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm
        self.name = f"llm-standin({llm.name})"

    def ask(self, state, questions):
        props = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                props[qid] = {"type": "number"}
                continue
            keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            props[qid] = {"type": "object", "properties": {k: {"type": "number"} for k in keys},
                          "required": keys, "additionalProperties": False}
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        system = (
            "You emulate a structured decision model. Evaluate each question independently against the "
            "state. For a choice question give a probability (0-1) for every option; for a score question "
            "give a probability for every level index (levels are listed in order, 0 = first); for a noul "
            "question give the probability (0-1) that the statement is true. Be calibrated."
        )
        prompt = f"STATE:\n{json.dumps(state, ensure_ascii=False, indent=1)}\n\nQUESTIONS:\n{json.dumps(questions, indent=1)}"
        raw = self.llm.json("jev", system, prompt, schema, effort="low")
        self.calls += 1
        return _wire_answers(questions, raw)


class MockJev(JevBackend):
    name = "mock-jev"

    def ask(self, state, questions):
        self.calls += 1
        action = mock_policy(state)
        raw = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                raw[qid] = {k: (0.85 if k == action else 0.15 / max(1, len(q["criteria"]) - 1)) for k in q["criteria"]}
            elif q["type"] == "score":
                top = len(q["criteria"]) - 2 if action == "answer" else 0
                raw[qid] = {str(i): (0.8 if i == top else 0.2 / (len(q["criteria"]) - 1))
                            for i in range(len(q["criteria"]))}
            else:
                raw[qid] = 0.9 if (qid != "ambiguous" or action == "clarify") else 0.1
        return _wire_answers(questions, raw)
