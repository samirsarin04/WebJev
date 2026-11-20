# WebJev

A web research agent where **Jev is the decision layer**. At every step Jev decides whether the agent
should search, ask the user to clarify, or stop and answer. The LLM only writes text: search queries,
source summaries, clarifying questions and the final cited answer. A **baseline mode** (WebGPT-style, the
LLM decides everything) runs the same loop so the two can be compared.

CS 4961 HCI project, Phase 2 prototype.

## Setup

```powershell
cd webjev
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then fill in the keys
```

| Key | Needed for | Without it |
|---|---|---|
| `TYPESAFE_API_KEY` | Jev mode | Jev mode fails with a clear error; baseline mode and `--offline` (mock Jev) still work |
| `OPENAI_API_KEY` | the writer LLM, baseline decisions, the eval judge (default model `gpt-5.6-sol`; change with `WEBJEV_LLM_MODEL`). `ANTHROPIC_API_KEY` works instead | only `--offline` works |
| `SERPER_API_KEY` or `TAVILY_API_KEY` | web search (Serper = Google results; Serper is used if both are set) | DuckDuckGo scraping is tried; it was blocked on the dev machine and is unreliable in general |

## Run

```powershell
.\.venv\Scripts\python -m webjev.server                      # UI at http://127.0.0.1:5000
.\.venv\Scripts\python -m webjev "your question" --mode jev  # CLI; --mode baseline for the baseline
.\.venv\Scripts\python -m webjev.evaluate --n 20             # TruthfulQA, both modes
.\.venv\Scripts\python -m unittest discover tests            # offline tests
```

Add `--offline` to the CLI or eval (or set `WEBJEV_OFFLINE=1` for the UI) to run everything on mocks with
no keys. That checks the plumbing only; the answers are canned.

In the UI, pick **Jev decides**, **Baseline**, or **Both, side by side**. Each step shows the action, why
it was taken (for Jev: the probability of each action, the evidence level and the ambiguity estimate),
what was searched and read, and what is still unknown. Clarifying questions appear inline with an answer box.

## How it works

```
question -> [ controller.decide(state) ] -> search  -> LLM writes query -> search -> read pages -> LLM digests
                     ^                      clarify -> LLM writes question -> user answers
                     |                      answer  -> LLM writes cited draft -> controller.review(draft)
                     +---- updated state <------------------------------- rejected draft goes back to search
```

**What the decider sees each step** (`RunState.snapshot`, `webjev/state.py`): the question, clarifications,
search history, sources with summaries, previous actions, information gaps, current confidence, and the
remaining budget.

**Jev mode** (`JevOrchestrator`, `webjev/controllers.py`) sends that state with three questions in one request:

- `next_action`, a **Choice** over search / clarify / answer
- `evidence`, a **Score** over five levels of how well the sources support an answer
- `ambiguous`, a **Noul**: is the question ambiguous in a way only the user can settle

The Choice picks the action. If Jev picks "answer" with confidence below `WEBJEV_MIN_ANSWER_CONFIDENCE`, the
agent searches again instead. Draft answers are checked with two more Nouls (claims supported by sources;
answers the question), and a rejected draft sends the agent back to search.

**Baseline mode** (`BaselineController`) asks the LLM for the same decision and the same draft review as
free-form judgement. Everything else, including prompts, budgets and guardrails, is shared, so the decider
is the only variable.

### Files

| File | Role |
|---|---|
| `webjev/agent.py` | the `Agent` interface and one subclass per mode (`AGENTS`), the shared loop, guardrails, metrics, run logging, and building the LLM / search / Jev backends |
| `webjev/prompts.py` | every prompt the writer LLM sees: search queries, page digests, clarifying questions, answers |
| `webjev/controllers.py` | the decision layer: baseline and Jev orchestrator |
| `webjev/jev.py` | Jev client (`POST /v1/systemone`) and the offline mock |
| `webjev/llm.py` | OpenAI and Anthropic clients, and the mock |
| `webjev/search.py` | DuckDuckGo / Tavily search, page fetching |
| `webjev/server.py`, `static/index.html` | the UI |
| `webjev/evaluate.py` | TruthfulQA harness |

## Logging and metrics

Every run writes its full trace to `runs/<id>.json` and one line to `runs/metrics.jsonl`:

| Metric | Definition |
|---|---|
| `searches` | web searches issued |
| `pages_visited` | result pages opened and read (a source used from its snippet alone is not counted) |
| `search_answer_cycles` | answer drafts that followed at least one new search. 1 means searched then answered; 2 means the first draft was rejected and the agent searched again |
| `clarification_requests` | questions put to the user |
| `time_to_answer_s` | wall-clock time to the final answer; `agent_time_s` excludes time waiting on the user |
| `decision_time_s` | time spent inside the decider (Jev calls, or the LLM's decision calls) |
| `llm_calls`, `llm_*_tokens`, `jev_calls` | usage |

## TruthfulQA evaluation

`python -m webjev.evaluate --n 20` samples questions (seeded), runs each in every mode, grades the answers
and writes `results.jsonl` and `summary.csv` under `runs/eval-<timestamp>/`. Re-running with the same
`--out` resumes. `--n 0` runs all 817 questions, which is a lot of LLM calls; start small.

- Clarification requests get a fixed reply ("I can't clarify further…"), so they are counted but can't help.
- Grading is an LLM judge against the dataset's reference answers. WebGPT's paper used human raters, so
  treat this as a proxy and spot-check `results.jsonl`.
- Search results that mention TruthfulQA are filtered out so the answer key can't leak into the evidence.

## Switching to "LLM as a tool for Jev"

`Agent` in `webjev/agent.py` is the seam. Each mode is an `Agent` subclass registered in `AGENTS`: the
baseline (`BaselineAgent`) and Jev mode (`JevOrchestratorAgent`) share `DecisionLoopAgent`'s
search / clarify / answer loop and differ only in their controller. Option 2 needs different actions, so it
is a new `Agent` subclass with its own `_loop`: Choice criteria that are explicit tools (new query, refine
last query, open result N, ask user, answer), a Score to rank which search results to open, and a loop that
dispatches on those tool names. Add it to `AGENTS` and it becomes a mode. Run setup, metrics, logging, the UI
and the eval harness come from `Agent.run` and don't change.

## Known limits

- The Jev client follows the public docs but has not been run against the live API yet (no key).
- The Tavily backend is also untested for the same reason.
- Pages are read as plain HTML text, cut to 8,000 characters; PDFs and JavaScript-only pages fall back to
  the search snippet.
