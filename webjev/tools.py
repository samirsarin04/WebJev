"""The agents' actions. Each tool does the work and returns what happened; the loop that calls it keeps
the metrics and emits the events.

Building blocks, shared by every agent:
  run_query   - the LLM writes a query and it is searched
  read_pages  - open given results, add them as sources, and digest them
  clarify     - the LLM writes a clarifying question and the user is asked

DecisionLoopAgent's search step is `search`: run_query, then read_pages on the top new results. An agent
that lets Jev choose what to read (Option 2) calls run_query and read_pages as separate actions instead.

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


# --- building blocks -----------------------------------------------------------------
@dataclass
class QueryOutcome:
    query: str
    results: list[dict]      # everything the search returned: [{title, url, snippet}]
    new_results: list[dict]  # those whose URL isn't already a source, in result order


def run_query(llm: LLM, searcher: Search, st: RunState, cfg: Config, snap: dict) -> QueryOutcome:
    """The LLM writes a query from the agent's state, and it is searched. Does not change `st`."""
    query = write_query(llm, snap)
    results = searcher.search(query, cfg.results_per_search)
    known = {s.url for s in st.sources}
    return QueryOutcome(query=query, results=results, new_results=[r for r in results if r["url"] not in known])


@dataclass
class ReadOutcome:
    pages_read: int          # pages opened and read (a source used from its snippet alone isn't counted)
    sources: list[Source]    # the sources added, with summaries filled in by the digest


def read_pages(llm: LLM, searcher: Search, st: RunState, cfg: Config, snap: dict,
               results: list[dict]) -> ReadOutcome:
    """Open each result, add it to `st.sources`, and have the LLM digest them.

    A page that can't be read falls back to its search snippet. The digest replaces `st.gaps` with what
    is still unknown. The caller picks `results` and should pass only URLs that aren't sources yet.
    """
    pages, added, pages_read = [], [], 0
    for r in results:
        text = searcher.fetch(r["url"], cfg.max_page_chars)
        if text:
            pages_read += 1
        src = Source(id=len(st.sources) + 1, title=r["title"], url=r["url"], fetched=bool(text))
        st.sources.append(src)
        added.append(src)
        pages.append({"id": src.id, "title": src.title, "url": src.url, "text": text or r["snippet"]})
    if pages:
        out = digest(llm, snap, pages)
        by_id = {s.id: s for s in st.sources}
        for item in out["sources"]:
            if item["id"] in by_id:
                by_id[item["id"]].summary = item["summary"]
                by_id[item["id"]].relevant = item["relevant"]
        st.gaps = out["information_gaps"]
    return ReadOutcome(pages_read=pages_read, sources=added)


# --- DecisionLoopAgent's search step ---------------------------------------------------
@dataclass
class SearchOutcome:
    query: str
    n_results: int           # results the search returned
    pages_read: int          # new pages opened and read (a source used from its snippet alone isn't counted)
    new_sources: list[dict]  # the sources this search added, after the digest filled in their summaries


def search(llm: LLM, searcher: Search, st: RunState, cfg: Config, snap: dict) -> SearchOutcome:
    """Search, then read the top `cfg.pages_per_search` new results.

    Adds the new sources, the search and the updated information gaps to `st`.
    """
    q = run_query(llm, searcher, st, cfg, snap)
    read = read_pages(llm, searcher, st, cfg, snap, q.new_results[:cfg.pages_per_search])
    st.searches.append({"query": q.query, "new_sources": [s.id for s in read.sources]})
    if not read.sources:
        st.gaps.append(f"The search '{q.query}' returned no new sources.")
    return SearchOutcome(query=q.query, n_results=len(q.results), pages_read=read.pages_read,
                         new_sources=[vars(s) for s in read.sources])


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
