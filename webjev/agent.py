"""The research agents.

Two independent axes:
  which system runs  - an `Agent` subclass, registered in AGENTS and selected by mode name.
  what it runs on    - `Parts`: one LLM, one Search and (if the agent needs it) one Jev backend, each real
                       or mock, built from the config by `build_parts`.

BaselineAgent and JevOrchestratorAgent share DecisionLoopAgent's loop, so the controller is the only thing
that differs between them. The LLM writes all text (prompts.py); in Jev mode it makes no control-flow
decisions. A new system (e.g. Option 2, "LLM as a tool for Jev") is a new Agent subclass with its own
`_loop`, plus one entry in AGENTS.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .controllers import BaselineController, Controller, JevOrchestrator
from .jev import Jev, JevBackend, MockJev
from .llm import LLM, AnthropicLLM, MockLLM, OpenAILLM
from . import tools
from .prompts import write_answer
from .search import Search, make_search
from .state import RunState


# --- what an agent runs on ---------------------------------------------------------
@dataclass
class Parts:
    llm: LLM
    search: Search
    jev: JevBackend | None = None


def make_llm(cfg: Config) -> LLM:
    if cfg.offline:
        return MockLLM()
    provider, model = cfg.resolved_llm()
    return OpenAILLM(model) if provider == "openai" else AnthropicLLM(model)


def make_jev(cfg: Config) -> JevBackend:
    """The real Jev API, or the mock when offline. Raises JevError if TYPESAFE_API_KEY is not set."""
    return MockJev() if cfg.offline else Jev(cfg.jev_url, cfg.jev_model)


def build_parts(cfg: Config, needs_jev: bool) -> Parts:
    llm, search = make_llm(cfg), make_search(cfg)
    return Parts(llm, search, make_jev(cfg) if needs_jev else None)


def describe_backends(cfg: Config) -> dict:
    """What would be used right now (for the UI banner and eval header)."""
    info = {"llm": "mock-llm" if cfg.offline else cfg.resolved_llm()[1]}
    try:
        info["search"] = make_search(cfg).name
    except Exception as exc:
        info["search"] = f"unavailable ({exc})"
    if cfg.offline:
        info["jev"] = "mock-jev"
    else:
        info["jev"] = f"Jev API ({cfg.jev_model})"
        if not os.environ.get("TYPESAFE_API_KEY"):
            info["jev"] += ", TYPESAFE_API_KEY not set: Jev mode will fail"
    return info


# --- which system runs -------------------------------------------------------------
class Agent(ABC):
    """One research system. `run` is shared (setup, metrics, error handling, logging); `_loop` differs."""
    mode: str
    uses_jev: bool = False

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.from_env()

    def run(self, question: str, ask_user: Callable[[str], str] | None = None,
            on_event: Callable[[dict], None] | None = None, save: bool = True) -> dict:
        """Run one question. `ask_user(question) -> answer` enables clarification requests."""
        cfg = self.cfg
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        st = RunState(question=question)
        events: list[dict] = []
        t0 = time.perf_counter()
        m = dict(searches=0, pages_visited=0, search_answer_cycles=0, clarification_requests=0,
                 answer_attempts=0, steps=0, decision_time_s=0.0, user_wait_s=0.0)

        def emit(type_: str, **data):
            event = {"type": type_, "t": round(time.perf_counter() - t0, 2), **data}
            events.append(event)
            if on_event:
                on_event(event)

        parts, controller, final, error = None, None, None, None
        try:
            parts = build_parts(cfg, self.uses_jev)
            controller = self.make_controller(parts)
            final = self._loop(parts, controller, st, m, emit, ask_user)
        except Exception as exc:  # surface any backend failure in the trace instead of losing the run
            error = f"{type(exc).__name__}: {exc}"
            emit("error", message=error)

        wall = time.perf_counter() - t0
        m["time_to_answer_s"] = round(wall, 2)                     # wall clock, includes waiting on the user
        m["agent_time_s"] = round(wall - m["user_wait_s"], 2)      # excludes waiting on the user
        m["decision_time_s"] = round(m["decision_time_s"], 3)
        m["user_wait_s"] = round(m["user_wait_s"], 2)
        if parts:
            m.update(llm_calls=parts.llm.calls, llm_input_tokens=parts.llm.input_tokens,
                     llm_output_tokens=parts.llm.output_tokens, jev_calls=parts.jev.calls if parts.jev else 0)
        record = {
            "id": run_id, "question": question, "mode": self.mode,
            "backends": {"controller": controller.name if controller else None, "llm": parts.llm.name,
                         "search": parts.search.name} if parts else {},
            "answer": final, "error": error, "metrics": m,
            "sources": [vars(s) for s in st.sources], "clarifications": st.clarifications,
            "events": events, "config": cfg.public(),
        }
        if final is not None:
            emit("final", answer=final, sources=record["sources"], metrics=m, backends=record["backends"])
        if save:
            save_run(record, cfg)
        return record

    @abstractmethod
    def make_controller(self, parts: Parts) -> Controller:
        """The decision layer this agent uses. Its name goes in the run record."""

    @abstractmethod
    def _loop(self, parts: Parts, controller: Controller, st: RunState, m: dict,
              emit: Callable[..., None], ask_user: Callable[[str], str] | None) -> str | None:
        """Research `st.question`, updating `st` and the metrics `m`, and return the final answer."""


class DecisionLoopAgent(Agent):
    """The search / clarify / answer loop. The controller is the only variable between subclasses."""

    def _loop(self, parts, controller, st, m, emit, ask_user):
        cfg = self.cfg
        final, draft = None, None
        searches_since_draft, rejected = 0, False
        for step in range(1, cfg.max_steps + 1):
            m["steps"] = step
            snap = st.snapshot(cfg)
            decision = controller.decide(snap)
            m["decision_time_s"] += decision.latency_s
            action, override = _guard(decision.action, st, cfg, step, ask_user is not None,
                                      need_new_evidence=rejected and searches_since_draft == 0)
            st.confidence = decision.evidence
            st.actions.append({"step": step, "action": action, "confidence": round(decision.confidence, 3)})
            emit("decision", step=step, action=action, chosen=decision.action, override=override,
                 why=decision.rationale, confidence=round(decision.confidence, 3),
                 evidence=None if decision.evidence is None else round(decision.evidence, 3),
                 details=decision.details, decider=decision.decider, latency_s=round(decision.latency_s, 3))

            if action == "search":
                found = tools.search(parts.llm, parts.search, st, cfg, snap)
                m["searches"] += 1
                m["pages_visited"] += found.pages_read
                searches_since_draft += 1
                emit("search", step=step, query=found.query, results=found.n_results, gaps=st.gaps,
                     sources=found.new_sources)

            elif action == "clarify":
                def asked(question: str, step=step):
                    m["clarification_requests"] += 1
                    emit("clarify_request", step=step, question=question)

                got = tools.clarify(parts.llm, snap, ask_user, on_question=asked)
                m["user_wait_s"] += got.waited_s
                st.clarifications.append({"question": got.question, "answer": got.reply})
                emit("clarify_answer", step=step, answer=got.reply)

            else:
                draft = write_answer(parts.llm, snap, cfg.answer_style, rejected=draft if rejected else None)
                m["answer_attempts"] += 1
                if searches_since_draft:
                    m["search_answer_cycles"] += 1
                searches_since_draft = 0
                emit("draft", step=step, attempt=m["answer_attempts"], answer=draft)
                can_retry = (m["answer_attempts"] < cfg.max_answer_attempts and step < cfg.max_steps
                             and len(st.searches) < cfg.max_searches)
                if can_retry:
                    review = controller.review(st.snapshot(cfg), draft)
                    m["decision_time_s"] += review.latency_s
                    emit("review", step=step, accept=review.accept, why=review.rationale,
                         details=review.details, latency_s=round(review.latency_s, 3))
                    if not review.accept:
                        rejected = True
                        st.gaps.append("A draft answer was rejected as not fully supported by the sources "
                                       "or not responsive to the question.")
                        continue
                final = draft
                break
        if final is None:
            final = draft or write_answer(parts.llm, st.snapshot(cfg), cfg.answer_style)
        return final


class BaselineAgent(DecisionLoopAgent):
    """WebGPT-style: the LLM decides every action and reviews its own draft."""
    mode = "baseline"

    def make_controller(self, parts: Parts) -> Controller:
        return BaselineController(parts.llm)


class JevOrchestratorAgent(DecisionLoopAgent):
    """Option 1: Jev decides each action and reviews the draft; the LLM only writes text."""
    mode = "jev"
    uses_jev = True

    def make_controller(self, parts: Parts) -> Controller:
        return JevOrchestrator(parts.jev, self.cfg.min_answer_confidence, self.cfg.review_threshold)


AGENTS: dict[str, type[Agent]] = {a.mode: a for a in (JevOrchestratorAgent, BaselineAgent)}
MODES = tuple(AGENTS)


def run(question: str, mode: str = "jev", cfg: Config | None = None,
        ask_user: Callable[[str], str] | None = None,
        on_event: Callable[[dict], None] | None = None, save: bool = True) -> dict:
    """Run one question with the agent registered for `mode`."""
    if mode not in AGENTS:
        raise ValueError(f"mode must be one of {MODES}")
    return AGENTS[mode](cfg).run(question, ask_user=ask_user, on_event=on_event, save=save)


# --- shared by the agents ----------------------------------------------------------
def _guard(action: str, st: RunState, cfg: Config, step: int, can_ask: bool, need_new_evidence: bool):
    """Budget guardrails, applied identically in both modes. Returns (action, reason-if-overridden)."""
    can_search = len(st.searches) < cfg.max_searches
    if step >= cfg.max_steps and action != "answer":
        return "answer", "Step budget exhausted, so the agent answers with what it has."
    if action == "search" and not can_search:
        return "answer", "Search budget exhausted, so the agent answers with what it has."
    if action == "clarify" and (not can_ask or len(st.clarifications) >= cfg.max_clarifications):
        why = "No user is available to ask" if not can_ask else "Clarification budget exhausted"
        return ("search", f"{why}, so the agent searches instead.") if can_search else \
               ("answer", f"{why} and no searches remain, so the agent answers.")
    if action == "answer" and can_search and step < cfg.max_steps:
        if cfg.require_search and not st.searches:
            return "search", "Answers must cite sources, so the agent searches at least once first."
        if need_new_evidence:
            return "search", "The last draft was rejected, so the agent gathers new evidence before redrafting."
    return action, None


def save_run(record: dict, cfg: Config) -> None:
    """Full trace to runs/<id>.json; one metrics line per run to runs/metrics.jsonl."""
    out = Path(cfg.runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    line = {"id": record["id"], "mode": record["mode"], "question": record["question"],
            "error": record["error"], **record["backends"], **record["metrics"]}
    with open(out / "metrics.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
