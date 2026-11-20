"""The actions DecisionLoopAgent can take (used by both the baseline and Jev mode).

Each tool does the work and returns what happened; the loop keeps the metrics and emits the events.
Answering needs no tool of its own: the loop calls `prompts.write_answer` and runs the review itself,
because accepting or rejecting a draft is a decision, not an action.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .config import Config
from .llm import LLM
from .prompts import digest, write_clarification, write_query
from .search import Search
from .state import RunState, Source


@dataclass
class SearchOutcome:
    query: str
    n_results: int           # results the search returned
    pages_read: int          # new pages opened and read (a source used from its snippet alone isn't counted)
    new_sources: list[dict]  # the sources this search added, after the digest filled in their summaries


def search(llm: LLM, searcher: Search, st: RunState, cfg: Config, snap: dict) -> SearchOutcome:
    """Write a query, search, open the top new pages, and digest them.

    Adds the new sources, the search and the updated information gaps to `st`.
    """
    query = write_query(llm, snap)
    results = searcher.search(query, cfg.results_per_search)
    known = {s.url for s in st.sources}
    pages, pages_read = [], 0
    for r in [r for r in results if r["url"] not in known][:cfg.pages_per_search]:
        text = searcher.fetch(r["url"], cfg.max_page_chars)
        if text:
            pages_read += 1
        src = Source(id=len(st.sources) + 1, title=r["title"], url=r["url"], fetched=bool(text))
        st.sources.append(src)
        pages.append({"id": src.id, "title": src.title, "url": src.url, "text": text or r["snippet"]})
    st.searches.append({"query": query, "new_sources": [p["id"] for p in pages]})
    if pages:
        out = digest(llm, snap, pages)
        by_id = {s.id: s for s in st.sources}
        for item in out["sources"]:
            if item["id"] in by_id:
                by_id[item["id"]].summary = item["summary"]
                by_id[item["id"]].relevant = item["relevant"]
        st.gaps = out["information_gaps"]
    else:
        st.gaps.append(f"The search '{query}' returned no new sources.")
    new_ids = {p["id"] for p in pages}
    return SearchOutcome(query=query, n_results=len(results), pages_read=pages_read,
                         new_sources=[vars(s) for s in st.sources if s.id in new_ids])


@dataclass
class ClarifyOutcome:
    question: str
    reply: str
    waited_s: float          # time spent waiting on the user


def clarify(llm: LLM, snap: dict, ask_user: Callable[[str], str],
            on_question: Callable[[str], None]) -> ClarifyOutcome:
    """Write a clarifying question and put it to the user.

    `on_question` runs after the question is written and before waiting for the reply, so the UI can
    show the question (the server's `ask_user` blocks until the user answers it).
    """
    question = write_clarification(llm, snap)
    on_question(question)
    started = time.perf_counter()
    reply = (ask_user(question) or "").strip() or "(no response)"
    return ClarifyOutcome(question=question, reply=reply, waited_s=time.perf_counter() - started)
