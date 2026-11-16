"""Web search + page fetching. Backends return [{title, url, snippet}]."""
from __future__ import annotations

import os
import re

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; WebJevResearchBot/0.1)"
# Keep the TruthfulQA dataset itself (and write-ups quoting its answer key) out of the evidence.
BLOCKED = re.compile(r"truthful[\s_-]?qa", re.I)


class SearchError(RuntimeError):
    pass


def fetch_page(url: str, max_chars: int) -> str | None:
    """Return the page's visible text (cut to max_chars), or None if it can't be read."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        return text[:max_chars] if len(text) > 200 else None
    except requests.RequestException:
        return None


class Search:
    name = "search"

    def _raw(self, query: str, k: int) -> list[dict]:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[dict]:
        results = self._raw(query, k + 3)
        keep = [r for r in results if r.get("url") and not BLOCKED.search(r["url"] + " " + r.get("title", ""))]
        return keep[:k]

    def fetch(self, url: str, max_chars: int) -> str | None:
        return fetch_page(url, max_chars)


class DDGSearch(Search):
    """DuckDuckGo via the `ddgs` package. No key needed, but it rate-limits under heavy use."""
    name = "duckduckgo"

    def _raw(self, query, k):
        from ddgs import DDGS

        try:
            hits = DDGS(timeout=15).text(query, max_results=k, backend="duckduckgo") or []
        except Exception as exc:
            # ddgs reports a bot challenge as "No results found", so an empty result is treated as a
            # failure rather than silently burning the agent's search budget.
            raise SearchError(
                f"DuckDuckGo search failed ({exc}). It often blocks automated requests; "
                "set TAVILY_API_KEY in .env for a reliable search backend."
            ) from exc
        return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")} for h in hits]


class TavilySearch(Search):
    name = "tavily"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _raw(self, query, k):
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "max_results": k},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SearchError(f"Tavily search failed: {exc}") from exc
        return [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": h.get("content", "")}
                for h in resp.json().get("results", [])]


class SerperSearch(Search):
    """Google results via serper.dev."""
    name = "serper"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _raw(self, query, k):
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": query, "num": k},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise SearchError(f"Serper search failed: {exc}") from exc
        if resp.status_code in (401, 403):
            raise SearchError("Serper rejected the key. Check SERPER_API_KEY in .env.")
        if resp.status_code != 200:
            raise SearchError(f"Serper returned {resp.status_code}: {resp.text[:200]}")
        return [{"title": h.get("title", ""), "url": h.get("link", ""), "snippet": h.get("snippet", "")}
                for h in resp.json().get("organic", [])]


class MockSearch(Search):
    name = "mock-search"

    def _raw(self, query, k):
        slug = re.sub(r"\W+", "-", query.lower()).strip("-")[:40]
        return [{"title": f"Mock result {i} for '{query}'", "url": f"https://example.com/{slug}/{i}",
                 "snippet": f"Mock snippet {i} about {query}."} for i in range(1, k + 1)]

    def fetch(self, url, max_chars):
        return f"Mock page text for {url}. " * 10


def make_search(cfg) -> Search:
    backend = "mock" if cfg.offline else cfg.search_backend
    tavily_key = os.environ.get("TAVILY_API_KEY")
    serper_key = os.environ.get("SERPER_API_KEY")
    if backend == "auto":
        backend = "serper" if serper_key else "tavily" if tavily_key else "ddg"
    if backend == "mock":
        return MockSearch()
    if backend == "serper":
        if not serper_key:
            raise SearchError("SERPER_API_KEY is not set (add it to .env).")
        return SerperSearch(serper_key)
    if backend == "tavily":
        if not tavily_key:
            raise SearchError("TAVILY_API_KEY is not set (add it to .env).")
        return TavilySearch(tavily_key)
    return DDGSearch()
