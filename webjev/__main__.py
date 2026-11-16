"""Command line:  python -m webjev "your question" --mode jev|baseline [--offline]"""
from __future__ import annotations

import argparse

from .agent import MODES, run
from .config import Config


def show(event: dict) -> None:
    kind = event["type"]
    if kind == "decision":
        print(f"\n[step {event['step']}] {event['action'].upper()}  (confidence {event['confidence']:.2f}, "
              f"{event['latency_s']}s, {event['decider']})")
        print(f"  why: {event['why']}")
        if event["override"]:
            print(f"  overridden: {event['override']}")
    elif kind == "search":
        print(f"  query: {event['query']}")
        for src in event["sources"]:
            print(f"    [{src['id']}] {src['title']} - {src['url']}")
        for gap in event["gaps"]:
            print(f"    gap: {gap}")
    elif kind == "review":
        print(f"  review: {event['why']}")
    elif kind == "error":
        print(f"\nERROR: {event['message']}")
    elif kind == "final":
        print(f"\n=== Answer ===\n{event['answer']}\n")
        for src in event["sources"]:
            print(f"[{src['id']}] {src['url']}")
        print("\n=== Metrics ===")
        for key, value in event["metrics"].items():
            print(f"  {key}: {value}")


def ask_user(question: str) -> str:
    print(f"\n  The agent asks: {question}")
    return input("  Your answer: ")


def main():
    ap = argparse.ArgumentParser(prog="webjev", description="WebJev research agent")
    ap.add_argument("question")
    ap.add_argument("--mode", choices=MODES, default="jev")
    ap.add_argument("--offline", action="store_true", help="mock LLM, search and Jev (no keys needed)")
    args = ap.parse_args()
    cfg = Config.from_env(offline=True if args.offline else None)
    record = run(args.question, args.mode, cfg, ask_user=ask_user, on_event=show)
    print(f"\nTrace saved to {cfg.runs_dir}\\{record['id']}.json")


if __name__ == "__main__":
    main()
