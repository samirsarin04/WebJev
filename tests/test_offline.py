"""Offline tests (mock LLM, search and Jev).  Run:  python -m unittest discover tests"""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webjev import jev as J
from webjev import tools
from webjev.agent import AGENTS, MODES, Agent, DecisionLoopAgent, make_jev, run
from webjev.config import Config
from webjev.controllers import DECIDE_QUESTIONS, JevOrchestrator
from webjev.llm import MockLLM
from webjev.search import MockSearch
from webjev.state import RunState


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


class Agents(unittest.TestCase):
    def test_registry_maps_modes_to_decision_loop_agents(self):
        self.assertEqual(MODES, ("jev", "baseline"))
        for mode, agent in AGENTS.items():
            self.assertEqual(agent.mode, mode)
            self.assertTrue(issubclass(agent, DecisionLoopAgent))
        self.assertEqual(AGENTS["jev"].uses_jev, True)
        self.assertEqual(AGENTS["baseline"].uses_jev, False)

    def test_interfaces_are_abstract(self):
        for abstract in (Agent, DecisionLoopAgent):
            with self.assertRaises(TypeError):
                abstract(cfg())

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            run("q", "nonexistent", cfg())

    def test_backend_failure_is_recorded_not_raised(self):
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "SERPER_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            rec = run("q", "jev", Config(search_backend="serper", runs_dir=tempfile.mkdtemp()))
        self.assertIsNotNone(rec["error"])
        self.assertIsNone(rec["answer"])
        self.assertEqual(rec["backends"], {})

    def test_jev_is_real_or_mock_with_no_fallback(self):
        env = {k: v for k, v in os.environ.items() if k != "TYPESAFE_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(make_jev(Config(offline=True)), J.MockJev)
            with self.assertRaises(J.JevError):      # no key: fail, never silently substitute
                make_jev(Config())
            os.environ["TYPESAFE_API_KEY"] = "test-key"
            self.assertIsInstance(make_jev(Config()), J.Jev)
            self.assertIsInstance(make_jev(Config(offline=True)), J.MockJev)


class UnreadablePages(MockSearch):
    def fetch(self, url, max_chars):
        return None


class Tools(unittest.TestCase):
    def setUp(self):
        self.cfg, self.llm, self.st = cfg(), MockLLM(), RunState(question="Do vaccines cause autism?")

    def snap(self):
        # snapshot() shares RunState's lists; copy it so the mock LLM writes the same query each time
        return copy.deepcopy(self.st.snapshot(self.cfg))

    def test_run_query_drops_known_urls_and_leaves_state_alone(self):
        snap = self.snap()
        first = tools.search(self.llm, MockSearch(), self.st, self.cfg, copy.deepcopy(snap))
        q = tools.run_query(self.llm, MockSearch(), self.st, self.cfg, copy.deepcopy(snap))
        known = {s["url"] for s in first.new_sources}
        self.assertTrue(known)
        self.assertTrue(known <= {r["url"] for r in q.results})   # same mock query, same URLs back
        self.assertFalse(known & {r["url"] for r in q.new_results})
        self.assertEqual(len(self.st.searches), 1)                # run_query recorded nothing

    def test_read_pages_falls_back_to_snippet(self):
        results = MockSearch().search("q", 2)
        read = tools.read_pages(self.llm, UnreadablePages(), self.st, self.cfg, self.snap(), results)
        self.assertEqual(read.pages_read, 0)
        self.assertEqual([s.fetched for s in read.sources], [False, False])
        self.assertEqual(len(self.st.sources), 2)

    def test_search_with_no_new_results_records_a_gap(self):
        self.cfg = cfg(results_per_search=2, pages_per_search=2)   # the first search reads every result
        snap = self.snap()
        tools.search(self.llm, MockSearch(), self.st, self.cfg, copy.deepcopy(snap))
        again = tools.search(self.llm, MockSearch(), self.st, self.cfg, copy.deepcopy(snap))  # same query
        self.assertEqual((again.pages_read, again.new_sources), (0, []))
        self.assertEqual(self.st.searches[-1]["new_sources"], [])
        self.assertIn("returned no new sources", self.st.gaps[-1])


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
