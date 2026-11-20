"""LLM backends. The agent only needs two primitives: free text and schema-constrained JSON.

`task` names the job (query, digest, clarify, answer, decide, review, judge) so the mock can
answer without a network; `meta` carries the structured inputs for the mock and is ignored otherwise.
"""
from __future__ import annotations

import json

from .config import FALLBACK_MODELS


class LLMError(RuntimeError):
    pass


class LLM:
    name = "llm"

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def text(self, task: str, system: str, prompt: str, effort: str = "low", meta: dict | None = None) -> str:
        raise NotImplementedError

    def json(self, task: str, system: str, prompt: str, schema: dict, effort: str = "low",
             meta: dict | None = None) -> dict:
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model: str):
        super().__init__()
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.name = model
        self.use_fallbacks = model in FALLBACK_MODELS
        try:
            # Resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile.
            self.client = anthropic.Anthropic(max_retries=4)
        except Exception as exc:
            raise LLMError(f"No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env ({exc})") from exc

    def _call(self, system: str, prompt: str, effort: str, schema: dict | None) -> str:
        anthropic = self._anthropic
        output_config: dict = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
        )
        try:
            if self.use_fallbacks:
                # If a safety classifier declines a request, the API re-runs it on a fallback model.
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                resp = self.client.messages.create(**kwargs)
        except TypeError as exc:   # the SDK raises this at request time when no credentials are set
            if "authentication" not in str(exc).lower():
                raise
            raise LLMError("No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the credentials. Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit hit after retries. Lower --workers or try again.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc

        self.calls += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            raise LLMError("The model declined this request (stop_reason=refusal).")
        if resp.stop_reason == "max_tokens":
            raise LLMError("The model's reply was cut off at max_tokens.")
        return "".join(block.text for block in resp.content if block.type == "text").strip()

    def text(self, task, system, prompt, effort="low", meta=None):
        return self._call(system, prompt, effort, None)

    def json(self, task, system, prompt, schema, effort="low", meta=None):
        raw = self._call(system, prompt, effort, schema)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON for task '{task}': {raw[:200]}") from exc


class OpenAILLM(LLM):
    """OpenAI via Chat Completions, with strict JSON-schema output for structured calls."""

    def __init__(self, model: str):
        super().__init__()
        import openai

        self._openai = openai
        self.model = model
        self.name = model
        try:
            self.client = openai.OpenAI(max_retries=4)   # reads OPENAI_API_KEY
        except openai.OpenAIError as exc:
            raise LLMError("No OpenAI credentials found. Set OPENAI_API_KEY in .env.") from exc

    def _call(self, task: str, system: str, prompt: str, schema: dict | None) -> str:
        openai = self._openai
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                  {"role": "user", "content": prompt}])
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": task, "schema": schema, "strict": True}}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the credentials. Check OPENAI_API_KEY in .env.") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"OpenAI does not know the model '{self.model}' (or this key has no access to it). "
                           "Set WEBJEV_LLM_MODEL in .env to a model your account can use.") from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"OpenAI rate limit or quota hit after retries: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Could not reach the OpenAI API: {exc}") from exc

        self.calls += 1
        if resp.usage:
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"The model declined this request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("The model's reply was cut off at its output limit.")
        return (choice.message.content or "").strip()

    def text(self, task, system, prompt, effort="low", meta=None):
        return self._call(task, system, prompt, None)

    def json(self, task, system, prompt, schema, effort="low", meta=None):
        raw = self._call(task, system, prompt, schema)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON for task '{task}': {raw[:200]}") from exc


def mock_policy(snapshot: dict) -> str:
    """Deterministic stand-in policy shared by the mock LLM and mock Jev."""
    if "best" in snapshot["question"].lower().split() and not snapshot["clarifications"]:
        return "clarify"
    if not any(s["relevant"] for s in snapshot["sources"]):
        return "search"
    return "answer"


class MockLLM(LLM):
    """Offline LLM for tests and UI demos. Canned, deterministic output."""
    name = "mock-llm"

    def text(self, task, system, prompt, effort="low", meta=None):
        self.calls += 1
        meta = meta or {}
        question = meta.get("question", "the question")
        if task == "query":
            n = len(meta.get("search_history", []))
            return question if n == 0 else f"{question} evidence {n + 1}"
        if task == "clarify":
            return "What do you mean by 'best' - which criteria matter most to you?"
        if task == "answer":
            cites = "".join(f"[{s['id']}]" for s in meta.get("sources", [])[:2])
            return f"(offline mock answer) Based on the retrieved sources, here is an answer to: {question} {cites}"
        return "(mock text)"

    def json(self, task, system, prompt, schema, effort="low", meta=None):
        self.calls += 1
        meta = meta or {}
        if task == "digest":
            return {
                "sources": [
                    {"id": p["id"], "relevant": True, "summary": f"Mock summary of {p['title']}."}
                    for p in meta.get("pages", [])
                ],
                "information_gaps": [],
            }
        if task == "decide":
            action = mock_policy(meta)
            return {"action": action, "rationale": f"Mock policy chose {action}.",
                    "confidence": 0.8, "evidence_sufficiency": 0.75 if action == "answer" else 0.1}
        if task == "review":
            return {"accept": True, "rationale": "Mock review accepts the draft."}
        if task == "judge":
            return {"truthful": True, "informative": True, "reason": "Mock judge."}
        raise LLMError(f"MockLLM has no canned output for task '{task}'")
