"""The research agent loop. Identical for both modes; only `parts.controller` differs.

Each step: controller decides -> guardrails -> execute (search / clarify / answer).
The LLM writes all text (queries, source digests, clarifying questions, answers); it makes no
control-flow decisions in Jev mode.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .controllers import BaselineController, Controller, JevOrchestrator
from .jev import JevBackend, JevHTTP, LLMJev, MockJev
from .llm import LLM, AnthropicLLM, MockLLM, OpenAILLM
from .search import Search, make_search
from .state import RunState, Source

MODES = ("jev", "baseline")


@dataclass
class Parts:
    llm: LLM
    search: Search
    controller: Controller
    jev: JevBackend | None = None


def make_llm(cfg: Config) -> LLM:
    if cfg.offline:
        return MockLLM()
    provider, model = cfg.resolved_llm()
    return OpenAILLM(model) if provider == "openai" else AnthropicLLM(model)


def make_jev(cfg: Config) -> JevBackend:
    backend = "mock" if cfg.offline else cfg.jev_backend
    if backend == "auto":
        backend = "http" if os.environ.get("TYPESAFE_API_KEY") else "llm"
    if backend == "http":
        return JevHTTP(cfg.jev_url, cfg.jev_model)
    if backend == "llm":
        return LLMJev(make_llm(cfg))   # separate instance: its tokens are not counted as writer-LLM usage
    return MockJev()


def build(cfg: Config, mode: str) -> Parts:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    llm, search = make_llm(cfg), make_search(cfg)
    if mode == "baseline":
        return Parts(llm, search, BaselineController(llm))
    jev = make_jev(cfg)
    return Parts(llm, search, JevOrchestrator(jev, cfg.min_answer_confidence, cfg.review_threshold), jev)


def describe_backends(cfg: Config) -> dict:
    """What would be used right now (for the UI banner and eval header)."""
    info = {"llm": "mock-llm" if cfg.offline else cfg.resolved_llm()[1]}
    try:
        info["search"] = make_search(cfg).name
    except Exception as exc:
        info["search"] = f"unavailable ({exc})"
    backend = "mock" if cfg.offline else cfg.jev_backend
    if backend == "auto":
        backend = "http" if os.environ.get("TYPESAFE_API_KEY") else "llm"
    info["jev"] = {"http": f"Jev API ({cfg.jev_model})", "llm": "LLM stand-in (no TYPESAFE_API_KEY)",
                   "mock": "mock-jev"}[backend]
    info["jev_is_real"] = backend == "http"
    return info


# --- text generation (the LLM's whole job) ---------------------------------------
def _ctx(snapshot: dict) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=1)


def write_query(llm: LLM, snap: dict) -> str:
    system = ("You write web search queries for a research agent. Given the agent's state, write the single "
              "search query most likely to close its information gaps. Do not repeat a query from "
              "search_history. Reply with the query only.")
    query = llm.text("query", system, _ctx(snap), meta=snap)
    return query.strip().splitlines()[0].strip().strip('"')[:300]


DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "relevant": {"type": "boolean"},
                               "summary": {"type": "string"}},
                "required": ["id", "relevant", "summary"],
                "additionalProperties": False,
            },
        },
        "information_gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sources", "information_gaps"],
    "additionalProperties": False,
}


def digest(llm: LLM, snap: dict, pages: list[dict]) -> dict:
    system = ("You read web pages for a research agent. For each NEW page, write a 2-4 sentence summary of "
              "what it says that bears on the user's question (state facts, figures and who claims them; "
              "note if the page is low-quality or contradicts another source) and mark whether it is "
              "relevant. Then list what is still unknown or unverified for answering the question as short "
              "information gaps, taking the earlier sources into account. Use an empty list if nothing is missing.")
    blocks = "\n\n".join(f"[{p['id']}] {p['title']}\n{p['url']}\n{p['text']}" for p in pages)
    prompt = f"AGENT STATE:\n{_ctx(snap)}\n\nNEW PAGES:\n{blocks}"
    return llm.json("digest", system, prompt, DIGEST_SCHEMA, meta={**snap, "pages": pages})


def write_clarification(llm: LLM, snap: dict) -> str:
    system = ("You write clarifying questions for a research agent. The agent has decided it cannot answer "
              "well without asking the user. Write one short, specific question that resolves the most "
              "important ambiguity. Do not repeat a question from clarifications. Reply with the question only.")
    return llm.text("clarify", system, _ctx(snap), meta=snap).strip()


def write_answer(llm: LLM, snap: dict, style: str, rejected: str | None = None) -> str:
    length = ("Answer in one to three sentences." if style == "concise"
              else "Answer directly first, then give the supporting detail a careful reader needs.")
    system = (
        "You write the final answer for a web research agent. Base factual claims on the listed sources and "
        "cite them inline as [id], using only ids that appear in sources. If the sources conflict, say so. "
        "If they do not support an answer, say what is and is not known rather than guessing. If the question "
        "rests on a false premise or a common misconception, say so plainly. " + length
    )
    prompt = _ctx(snap)
    if rejected:
        prompt += f"\n\nAn earlier draft was rejected as not fully supported or not responsive:\n{rejected}"
    return llm.text("answer", system, prompt, effort="medium", meta=snap)


# --- the loop --------------------------------------------------------------------
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


def run(question: str, mode: str = "jev", cfg: Config | None = None,
        ask_user: Callable[[str], str] | None = None,
        on_event: Callable[[dict], None] | None = None, save: bool = True) -> dict:
    """Run one question. `ask_user(question) -> answer` enables clarification requests."""
    cfg = cfg or Config.from_env()
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

    parts, final, draft, error = None, None, None, None
    searches_since_draft, rejected = 0, False
    try:
        parts = build(cfg, mode)
        for step in range(1, cfg.max_steps + 1):
            m["steps"] = step
            snap = st.snapshot(cfg)
            decision = parts.controller.decide(snap)
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
                query = write_query(parts.llm, snap)
                results = parts.search.search(query, cfg.results_per_search)
                m["searches"] += 1
                searches_since_draft += 1
                known = {s.url for s in st.sources}
                pages = []
                for r in [r for r in results if r["url"] not in known][:cfg.pages_per_search]:
                    text = parts.search.fetch(r["url"], cfg.max_page_chars)
                    if text:
                        m["pages_visited"] += 1
                    src = Source(id=len(st.sources) + 1, title=r["title"], url=r["url"], fetched=bool(text))
                    st.sources.append(src)
                    pages.append({"id": src.id, "title": src.title, "url": src.url, "text": text or r["snippet"]})
                st.searches.append({"query": query, "new_sources": [p["id"] for p in pages]})
                if pages:
                    out = digest(parts.llm, snap, pages)
                    by_id = {s.id: s for s in st.sources}
                    for item in out["sources"]:
                        if item["id"] in by_id:
                            by_id[item["id"]].summary = item["summary"]
                            by_id[item["id"]].relevant = item["relevant"]
                    st.gaps = out["information_gaps"]
                else:
                    st.gaps.append(f"The search '{query}' returned no new sources.")
                emit("search", step=step, query=query, results=len(results), gaps=st.gaps,
                     sources=[vars(s) for s in st.sources if s.id in {p["id"] for p in pages}])

            elif action == "clarify":
                ask = write_clarification(parts.llm, snap)
                m["clarification_requests"] += 1
                emit("clarify_request", step=step, question=ask)
                waited = time.perf_counter()
                reply = (ask_user(ask) or "").strip() or "(no response)"
                m["user_wait_s"] += time.perf_counter() - waited
                st.clarifications.append({"question": ask, "answer": reply})
                emit("clarify_answer", step=step, answer=reply)

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
                    review = parts.controller.review(st.snapshot(cfg), draft)
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
        "id": run_id, "question": question, "mode": mode,
        "backends": {"controller": parts.controller.name, "llm": parts.llm.name,
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


def save_run(record: dict, cfg: Config) -> None:
    """Full trace to runs/<id>.json; one metrics line per run to runs/metrics.jsonl."""
    out = Path(cfg.runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{record['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    line = {"id": record["id"], "mode": record["mode"], "question": record["question"],
            "error": record["error"], **record["backends"], **record["metrics"]}
    with open(out / "metrics.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
