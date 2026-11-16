"""Offline tests (mock LLM, search and Jev).  Run:  python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webjev import jev as J
from webjev.agent import run
from webjev.config import Config
from webjev.controllers import DECIDE_QUESTIONS, JevOrchestrator


def cfg(**kw):
    return Config(offline=True, runs_dir=tempfile.mkdtemp(), **kw)


class AgentLoop(unittest.TestCase):
    def test_both_modes_search_then_answer(self):
        for mode in ("jev", "baseline"):
            rec = run("Do vaccines cause autism?", mode, cfg())
            self.assertIsNone(rec["error"])
            self.assertTrue(rec["answer"])
            m = rec["metrics"]
            self.assertEqual((m["searches"], m["pages_visited"], m["search_answer_cycles"]), (1, 2, 1))
            self.assertEqual(m["clarification_requests"], 0)
            self.assertEqual([e["action"] for e in rec["events"] if e["type"] == "decision"], ["search", "answer"])
            self.assertEqual(m["jev_calls"] > 0, mode == "jev")

    def test_clarification_round_trip(self):
        asked = []
        rec = run("What is the best laptop?", "jev", cfg(), ask_user=lambda q: asked.append(q) or "for travel")
        self.assertEqual(len(asked), 1)
        self.assertEqual(rec["metrics"]["clarification_requests"], 1)
        self.assertEqual(rec["clarifications"][0]["answer"], "for travel")
        self.assertEqual([e["action"] for e in rec["events"] if e["type"] == "decision"],
                         ["clarify", "search", "answer"])

    def test_clarify_without_user_falls_back_to_search(self):
        rec = run("What is the best laptop?", "baseline", cfg(max_steps=3))
        first = next(e for e in rec["events"] if e["type"] == "decision")
        self.assertEqual((first["chosen"], first["action"]), ("clarify", "search"))
        self.assertEqual(rec["metrics"]["clarification_requests"], 0)
        self.assertTrue(rec["answer"])

    def test_search_budget_forces_answer(self):
        rec = run("What is the best laptop?", "baseline", cfg(max_searches=1, max_steps=6))
        self.assertEqual(rec["metrics"]["searches"], 1)
        self.assertTrue(rec["answer"])


class FixedJev(J.JevBackend):
    def __init__(self, answers):
        super().__init__()
        self.answers = answers

    def ask(self, state, questions):
        return self.answers


class JevLayer(unittest.TestCase):
    def test_concentration_matches_documented_three_option_formula(self):
        self.assertAlmostEqual(J.concentration({"a": 0.8, "b": 0.1, "c": 0.1}), (3 * 0.8 - 1) / 2)
        self.assertAlmostEqual(J.concentration({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), 0)

    def test_normalized_score_handles_either_indexing(self):
        self.assertAlmostEqual(J.normalized_score({"score": 2, "legend": {str(i): "" for i in range(5)}}), 0.5)
        self.assertAlmostEqual(J.normalized_score({"score": 3, "legend": {str(i): "" for i in range(1, 6)}}), 0.5)

    def test_low_confidence_answer_becomes_search(self):
        answers = {
            "next_action": {"choice": "answer", "confidence": 0.2,
                            "probabilities": {"answer": 0.45, "search": 0.4, "clarify": 0.15}},
            "evidence": {"score": 2.0, "confidence": 0.5, "legend": {str(i): f"level {i}" for i in range(5)},
                         "probabilities": {}},
            "ambiguous": {"noul": 0.1},
        }
        d = JevOrchestrator(FixedJev(answers), min_answer_confidence=0.5).decide({})
        self.assertEqual(d.action, "search")
        self.assertEqual(d.details["jev_choice"], "answer")
        self.assertAlmostEqual(d.evidence, 0.5)

    def test_wire_format_of_questions(self):
        self.assertEqual({q["type"] for q in DECIDE_QUESTIONS.values()}, {"choice", "score", "noul"})
        self.assertEqual(set(DECIDE_QUESTIONS["next_action"]["criteria"]), {"search", "clarify", "answer"})


if __name__ == "__main__":
    unittest.main()
