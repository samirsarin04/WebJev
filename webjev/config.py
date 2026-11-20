"""Configuration: one dataclass, filled from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {"openai": "gpt-5.6-sol", "anthropic": "claude-opus-5-5"}

# Models that accept the server-side refusal fallback parameter.
FALLBACK_MODELS = {
    "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
    "claude-fable-5-2", "claude-fable-5-1",
}


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader. Empty values are skipped so they never shadow real credentials."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass
class Config:
    # --- backends -----------------------------------------------------------
    offline: bool = False              # use mock LLM / search / Jev (no network, no keys)
    llm_provider: str = "auto"         # auto | openai | anthropic  (auto: OpenAI if OPENAI_API_KEY is set)
    llm_model: str = ""                # empty = the provider's default below
    jev_model: str = "jev-latest"      # pin (e.g. jev-1.13.0) once thresholds are tuned
    jev_url: str = "https://api.typesafe.ai/v1/systemone"
    search_backend: str = "auto"       # auto | serper | ddg | mock  (auto: Serper if SERPER_API_KEY is set)

    # --- agent budgets (identical in both modes, so only the decider differs) ---
    max_steps: int = 8
    max_searches: int = 4
    max_clarifications: int = 2
    max_answer_attempts: int = 2       # drafts; a rejected draft sends the agent back to search
    results_per_search: int = 5
    pages_per_search: int = 2          # new pages opened and read per search
    max_page_chars: int = 8000         # page text passed to the LLM is cut to this length
    require_search: bool = True        # no answer before at least one search (answers must be cited)

    # --- Jev decision thresholds ---------------------------------------------
    min_answer_confidence: float = 0.5  # Jev "answer" below this confidence -> search instead
    review_threshold: float = 0.5       # draft accepted if supported & responsive nouls >= this

    answer_style: str = "full"         # full | concise (concise is used by the TruthfulQA harness)
    runs_dir: str = str(ROOT / "runs")

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_env()
        cfg = cls(
            offline=_env("WEBJEV_OFFLINE", False, bool),
            llm_provider=_env("WEBJEV_LLM_PROVIDER", cls.llm_provider),
            llm_model=_env("WEBJEV_LLM_MODEL", cls.llm_model),
            jev_model=_env("TYPESAFE_DEFAULT_MODEL", cls.jev_model),
            jev_url=_env("WEBJEV_JEV_URL", cls.jev_url),
            search_backend=_env("WEBJEV_SEARCH_BACKEND", cls.search_backend),
            max_steps=_env("WEBJEV_MAX_STEPS", cls.max_steps, int),
            max_searches=_env("WEBJEV_MAX_SEARCHES", cls.max_searches, int),
            max_clarifications=_env("WEBJEV_MAX_CLARIFICATIONS", cls.max_clarifications, int),
            pages_per_search=_env("WEBJEV_PAGES_PER_SEARCH", cls.pages_per_search, int),
            min_answer_confidence=_env("WEBJEV_MIN_ANSWER_CONFIDENCE", cls.min_answer_confidence, float),
            review_threshold=_env("WEBJEV_REVIEW_THRESHOLD", cls.review_threshold, float),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg

    def resolved_llm(self) -> tuple[str, str]:
        """(provider, model) actually in use."""
        provider = self.llm_provider
        if provider == "auto":
            provider = "openai" if os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY") \
                else "anthropic"
        return provider, self.llm_model or DEFAULT_MODELS[provider]

    def public(self) -> dict:
        return asdict(self)
