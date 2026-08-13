"""Tests for the retrieval pipeline, with FAKES instead of the network.

This is the concrete gain of having declared contracts: this logic — the most delicate
in the package — could previously only be exercised in an integration test, which nobody
runs while editing. Now it runs in milliseconds, offline, and covers the paths that
matter precisely because they are the ones that degrade in SILENCE.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import retrieval
from core.retrieval import CE, CE_WEAK, DENSE, Policy, fuse_by_id, needs_rerank, two_stage
from tests.fakes import FakeReranker


def hit(id_: str, score: float, text: str = "text"):
    return {"id": id_, "score": score, "document": text}


TEXT = lambda h: h["document"]  # noqa: E731
ID = lambda h: h["id"]           # noqa: E731

# `FakeReranker` comes from `tests.fakes`. This file used to carry a byte-identical copy
# of it, which is the drift that lets a fake and its real counterpart disagree in
# silence — see TestFakeContract below for what now pins them together.


MEMORY_POLICY = Policy(dense_floor=0.45, strict_floor=0.58, min_score=0.10,
                          max_results=6, veto=True)
# Mirrors `core/docs.py`: BOTH floors equal, on purpose. In documents there is no veto
# and the goal is never to return silence, so "go back to the strict cut" has to be a
# no-op — were it 0.58, a cross-lingual collapse (dense ~0.46) would return nothing,
# which is exactly what that policy exists to prevent.
DOCS_POLICY = Policy(dense_floor=0.30, strict_floor=0.30, min_score=0.10,
                       max_results=5, veto=False, order_matters=True)


class TestFusion(unittest.TestCase):
    def test_keeps_the_highest_score_for_each_id(self):
        fused = fuse_by_id([[hit("a", 0.5), hit("b", 0.9)],
                              [hit("a", 0.8), hit("c", 0.3)]], ID)
        by_id = {h["id"]: h["score"] for h in fused}
        self.assertEqual(by_id, {"a": 0.8, "b": 0.9, "c": 0.3},
                         "an id repeated across two angles keeps the HIGHEST score")

    def test_comes_out_sorted_by_score(self):
        fused = fuse_by_id([[hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)]], ID)
        self.assertEqual([h["id"] for h in fused], ["b", "c", "a"])

    def test_empty_batches(self):
        self.assertEqual(fuse_by_id([], ID), [])
        self.assertEqual(fuse_by_id([[], []], ID), [])


class TestWhenToCallTheSecondStage(unittest.TestCase):
    def test_calls_when_there_are_more_candidates_than_slots(self):
        candidates = [hit(str(i), 0.9) for i in range(10)]
        self.assertTrue(needs_rerank(candidates, MEMORY_POLICY))

    def test_calls_when_a_candidate_sits_in_the_permissive_band(self):
        candidates = [hit("a", 0.90), hit("b", 0.50)]  # 0.50 < strict 0.58
        self.assertTrue(needs_rerank(candidates, MEMORY_POLICY),
                        "0.50 only got in because somebody was going to judge it")

    def test_does_NOT_call_when_everything_fits_and_clears_the_strict_floor(self):
        candidates = [hit("a", 0.90), hit("b", 0.70)]
        self.assertFalse(needs_rerank(candidates, MEMORY_POLICY),
                         "reordering would not change what comes out — the call would be wasted")

    def test_empty_list_does_not_call(self):
        self.assertFalse(needs_rerank([], MEMORY_POLICY))
        self.assertFalse(needs_rerank([], DOCS_POLICY))

    def test_a_single_candidate_needs_no_ordering(self):
        self.assertFalse(needs_rerank([hit("a", 0.9)], DOCS_POLICY),
                         "there is nothing to order with a single result")

    def test_when_order_is_the_product_it_calls_even_if_everything_fits(self):
        candidates = [hit("a", 0.90), hit("b", 0.70)]
        self.assertFalse(needs_rerank(candidates, MEMORY_POLICY),
                         "memory injects the whole set: the order is cosmetic")
        self.assertTrue(needs_rerank(candidates, DOCS_POLICY),
                        "a document is read top to bottom: the order is the product")


class TestWithoutTheSecondStage(unittest.TestCase):
    """The most important case: with no judgement, the permissive floor has to go back
    to the strict one, otherwise the mode WITH re-ranking ends up worse than without."""

    def test_absent_reranker_applies_the_strict_cut(self):
        candidates = [hit("a", 0.90), hit("b", 0.60), hit("c", 0.50), hit("d", 0.47)]
        outcome = two_stage(candidates, "q", None, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a", "b"])
        self.assertTrue(all(s.origin == DENSE for s in outcome.scored))
        self.assertFalse(outcome.reranked)

    def test_failing_reranker_applies_the_strict_cut(self):
        candidates = [hit("a", 0.90), hit("b", 0.53), hit("c", 0.49)]
        rr = FakeReranker(ok=False, error="timeout")
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a"],
                         "0.53 and 0.49 were only acceptable with the cross-encoder to clean up")
        self.assertEqual(outcome.rerank_error, "timeout")
        self.assertFalse(outcome.reranked)

    def test_response_without_usable_hits_also_falls_back_to_strict(self):
        rr = FakeReranker(ok=False, error="response with no usable hits")
        outcome = two_stage([hit("a", 0.50)], "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual(outcome.scored, [])


class TestWithVeto(unittest.TestCase):
    """Memory policy: a false positive pollutes the agent's context."""

    def test_eliminates_what_sits_below_the_cutoff(self):
        # a candidate in the permissive band is what forces the call — with everything
        # above the strict floor and fitting the slots, the memory policy would not call
        candidates = [hit("a", 0.9), hit("b", 0.50), hit("c", 0.9)]
        rr = FakeReranker(scores=[0.95, 0.02, 0.40])
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a", "c"],
                         "0.02 is eliminated, not demoted")
        self.assertTrue(all(s.origin == CE for s in outcome.scored))

    def test_the_cross_encoder_overturns_the_dense_verdict(self):
        """Measured against the real archive: the CE killed a dense 0.59 and saved a dense 0.47."""
        candidates = [hit("high_dense", 0.59), hit("low_dense", 0.47)]
        rr = FakeReranker(scores=[0.004, 0.11])
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["low_dense"])

    def test_respects_the_result_ceiling(self):
        candidates = [hit(str(i), 0.9) for i in range(20)]
        rr = FakeReranker(scores=[0.9] * 20)
        outcome = two_stage(candidates, "q", rr, Policy(0.45, 0.58, 0.10, 3), TEXT)
        self.assertEqual(len(outcome.scored), 3)


class TestWithoutVeto(unittest.TestCase):
    """Document policy: whoever asks has already chosen the document, so silence is
    worse than imperfect order."""

    def test_delivers_the_weak_ones_last_and_marked(self):
        candidates = [hit("a", 0.9), hit("b", 0.9)]
        rr = FakeReranker(scores=[0.80, 0.02])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a", "b"])
        self.assertEqual([s.origin for s in outcome.scored], [CE, CE_WEAK])
        self.assertTrue(outcome.scored[1].is_weak)

    def test_never_returns_empty_when_a_candidate_exists(self):
        candidates = [hit("a", 0.9), hit("b", 0.9)]
        rr = FakeReranker(scores=[0.05, 0.03])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertEqual(len(outcome.scored), 2, "with a veto this would become silence")
        self.assertTrue(all(s.is_weak for s in outcome.scored))


class TestCrossLingualCollapse(unittest.TestCase):
    """When the cross-encoder collapses, its ORDER is noise too."""

    def test_detects_collapse_and_falls_back_to_dense_order(self):
        candidates = [hit("best_dense", 0.60), hit("worst_dense", 0.46)]
        # inverted on purpose: the collapsed CE "prefers" the worse dense score
        rr = FakeReranker(scores=[0.0004, 0.0009])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertTrue(outcome.collapsed)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["best_dense", "worst_dense"],
                         "dense order, not the collapsed CE's")
        self.assertTrue(all(s.origin == DENSE for s in outcome.scored))

    def test_collapse_in_memory_RESTORES_the_strict_cut(self):
        """Regression: a collapse discards the judgement, so the permissive floor is once
        again left with nobody to clean up after it. Without this, the mode WITH re-ranking
        returned candidates the mode WITHOUT re-ranking would never return — the defect the
        pipeline exists to prevent. It only shows up under the MEMORY policy, where the two
        floors differ."""
        candidates = [hit("clears", 0.90), hit("permissive", 0.50), hit("permissive2", 0.46)]
        rr = FakeReranker(scores=[0.0004, 0.0009, 0.0002])
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertTrue(outcome.collapsed)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["clears"],
                         "0.50 and 0.46 were only acceptable with the cross-encoder to judge")

    def test_collapse_in_docs_does_NOT_return_silence(self):
        """The other half of the pair: with the floors equal, restoring the strict cut
        removes nothing, and a question in another language still gets answered."""
        candidates = [hit("a", 0.46), hit("b", 0.44)]  # typical cross-lingual band
        rr = FakeReranker(scores=[0.0004, 0.0002])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertTrue(outcome.collapsed)
        self.assertEqual(len(outcome.scored), 2, "silence here would be worse than imperfect order")

    def test_low_score_above_the_threshold_is_not_collapse(self):
        candidates = [hit("a", 0.9), hit("b", 0.9)]
        rr = FakeReranker(scores=[0.05, 0.02])  # 0.05 > COLLAPSE_MAX
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertFalse(outcome.collapsed, "0.05 is low relevance, not a collapse")

    def test_collapse_detection_can_be_turned_off_by_policy(self):
        policy = Policy(0.45, 0.58, 0.10, 5, veto=False, detect_collapse=False)
        rr = FakeReranker(scores=[0.0004, 0.0009])
        outcome = two_stage([hit("a", 0.60), hit("b", 0.46)], "q", rr, policy, TEXT)
        self.assertFalse(outcome.collapsed)


class TestTrace(unittest.TestCase):
    def test_records_the_scale_conversion(self):
        rr = FakeReranker(scores=[0.9, 0.5], was_logit=True)
        outcome = two_stage([hit("a", 0.9), hit("b", 0.9)], "q", rr,
                         Policy(0.45, 0.58, 0.10, 1), TEXT)
        self.assertTrue(outcome.scale_converted)

    def test_records_the_best_dense_score_even_with_no_result(self):
        outcome = two_stage([hit("a", 0.30)], "q", None, MEMORY_POLICY, TEXT)
        self.assertEqual(outcome.scored, [])
        self.assertAlmostEqual(outcome.best_dense, 0.30)

    def test_by_rerank_distinguishes_the_origin(self):
        rr = FakeReranker(scores=[0.9])
        with_ce = two_stage([hit("a", 0.5)], "q", rr, MEMORY_POLICY, TEXT)
        without_ce = two_stage([hit("a", 0.9)], "q", None, MEMORY_POLICY, TEXT)
        self.assertTrue(with_ce.by_rerank)
        self.assertFalse(without_ce.by_rerank)

    def test_the_query_reaches_the_reranker_and_so_do_the_texts(self):
        rr = FakeReranker(scores=[0.9, 0.8])
        two_stage([hit("a", 0.5, "text A"), hit("b", 0.5, "text B")],
                  "my question", rr, MEMORY_POLICY, TEXT)
        query, docs = rr.calls[0]
        self.assertEqual(query, "my question")
        self.assertEqual(docs, ["text A", "text B"])


class TestFloor(unittest.TestCase):
    def test_floor_relaxes_only_when_a_second_stage_exists(self):
        self.assertAlmostEqual(MEMORY_POLICY.floor_for(True), 0.45)
        self.assertAlmostEqual(MEMORY_POLICY.floor_for(False), 0.58)


class TestFakeContract(unittest.TestCase):
    """What ties the fake to the real `Reranker`, beyond the method signature.

    `Protocol` gives structural typing on METHODS, and `rank` returns a dict — so a
    Protocol check says nothing about the dict's KEYS. That is where the real drift
    happened: `era_logit`/`erro`/`descartados` were renamed in the producer while two
    hand-written fakes kept promising the old names, and only a test that read the
    keys caught it.
    """

    def test_the_fake_satisfies_the_protocol_without_inheriting_anything(self):
        from core.ports import RerankModel
        self.assertIsInstance(FakeReranker(), RerankModel)

    @staticmethod
    def _real_info_keys() -> set:
        """The key set the REAL `Reranker.rank` produces, from a real call.

        Port 1 on the loopback refuses instantly, so this needs no server and no DNS
        (measured: ~5ms). The failure path is the right one to sample: `rank` promises
        never to raise, so it has to return the full info dict even when the endpoint is
        unreachable.
        """
        from core.reranking import Reranker
        _, info = Reranker("http://127.0.0.1:1/rerank", "m", timeout=2.0).rank("q", ["a"])
        assert info["ok"] is False, "expected the refused connection to fail, not succeed"

        return set(info)

    def test_the_fake_promises_no_key_the_real_reranker_lacks(self):
        real = self._real_info_keys()
        for label, fake in (("ok", FakeReranker(scores=[0.9])),
                            ("failing", FakeReranker(ok=False, error="timeout"))):
            _, info = fake.rank("q", ["a"])
            extra = set(info) - real
            self.assertEqual(extra, set(),
                             f"the {label} fake invents {extra}, which no consumer can rely on")

    def test_every_key_the_pipeline_reads_is_one_the_reranker_produces(self):
        """Reads the keys out of `two_stage`'s source instead of restating them.

        A restated list would have been renamed alongside the producer and kept agreeing
        with it — the test has to observe the consumer, not mirror it.
        """
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(retrieval.two_stage))
        read = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "info" and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        self.assertTrue(read, "expected two_stage to read info keys; did the shape change?")
        missing = read - self._real_info_keys()
        self.assertEqual(missing, set(),
                         f"two_stage reads {missing}, which Reranker.rank never sets")


if __name__ == "__main__":
    unittest.main(verbosity=2)
