"""Text generation: the LLM's whole job. Every prompt the writer LLM sees is here.

The decision prompts (baseline decide/review) live with their decider in controllers.py.
"""
from __future__ import annotations

import json

from .llm import LLM


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
