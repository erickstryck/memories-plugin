"""What the stack runs with, and whether it is fast enough for each host.

The fakes stand in at the CONTRACTS the brief names:

- the `diagnose` fake carries the exact keys of the real return value, which the
  test read from `core/setup.py` at `70a9bec` (`"ready"`, `"checks"` as a list of
  `asdict(c)` rows, `"blockers"`, `"warnings"`, `"detected_dim"`,
  `"memory_suggestions"`);
- the embedder fake exposes `embed`/`embed_one`/`detect_dimension`, the three
  methods `core/embedding.py` gives an embedder (its `detect_dimension` answer of
  1024 is `EMBED_DIM`, the catalogue constant for bge-m3);
- the reranker fake returns `rank` pairs shaped like the contract at
  `core/reranking.py` line 151: `(pairs sorted by score desc, info)` with
  `info["ok"]` set, and ranks the right answer first, the way a healthy
  reranker does (the real Paris probe of `core/setup.py` does).

The clock is `time.monotonic` in the shape the code calls it with: a zero-arg
function, so the test advances it by hand.
"""
import ast
import math
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import chunk as core_chunk  # noqa: E402
from core import setup as core_setup  # noqa: E402
from stack import verify  # noqa: E402

PORTS = {"qdrant": 6333, "embed": 8003, "rerank": 8004}  # the catalogue's default ports


class FakeClock:
    """A monotonic clock that advances by a SCRIPTED step per call.

    `time.monotonic` is a zero-arg call returning a monotonically increasing
    float; `calibrate` brackets each measured call with two of them, so the
    step script lists the durations of those brackets in order (a call with no
    step left spends 0).
    """

    def __init__(self, steps: list[float] | None = None):
        self.steps = list(steps or [])
        self.now = 0.0
        self.calls: list[float] = []

    def __call__(self):
        if self.steps:
            self.now += self.steps.pop(0)
        self.calls.append(self.now)
        return self.now


class FakeEmbedder:
    """The `embed`/`embed_one`/`detect_dimension` contract of `core/embedding.py`.

    Every call is recorded, so a test can see the warm-up call BEFORE the
    measured one. `detect_dimension` answers 1024, which is `EMBED_DIM`, the
    catalogue constant for bge-m3: a stack whose embedder answers any other
    dimension is a misinstall, and this fake stands in for a healthy one.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.calls: list = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed", list(texts)))
        return [[0.0] * self.dim for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        self.calls.append(("embed_one", [text]))
        return [0.0] * self.dim

    def detect_dimension(self) -> int:
        return self.dim


class FakeReranker:
    """The `rank` contract of `core/reranking.py`, line 151: it NEVER raises,
    returns `(pairs sorted by score desc, info)` and carries `info["ok"]`.

    The pairs rank document 0 first, which is what a healthy reranker answers
    for the probe `core/setup.py` sends (Paris first). Every call is recorded.
    """

    def __init__(self, docs: int = 1):
        self.docs = docs
        self.calls: list = []

    def rank(self, query: str, documents: list[str]):
        self.calls.append((query, list(documents)))
        pairs = [(i, 1.0 / (i + 1)) for i in range(len(documents))]
        pairs.sort(key=lambda p: -p[1])
        return pairs, {"ok": True, "contract": "jina", "was_logit": False}


def diagnose_fake(rows, *, detected_dim=None):
    """A `diagnose` answer shaped like the real one, which the test read from
    `core/setup.py` at `70a9bec`: the keys `"ready"`, `"checks"` (a list of
    `asdict(c)` rows), `"blockers"`, `"warnings"`, `"detected_dim"` and
    `"memory_suggestions"`."""
    return {
        "ready": True,
        "checks": [asdict(c) for c in rows],
        "blockers": [],
        "warnings": [asdict(c) for c in rows if c.warning],
        "detected_dim": detected_dim,
        "memory_suggestions": [],
    }


class TestStackUrls(unittest.TestCase):
    def test_stack_urls_carry_v1(self):
        # M2: the bare `/embeddings` of llama-server b11382 returns a raw JSON
        # list and crashes `core/embedding.py`; only the `/v1` path has the
        # OpenAI shape. `api_base_url` therefore ends in `/v1` and `embed_url`
        # is empty, so the plugin reaches `/v1/embeddings`.
        self.assertEqual(verify.stack_urls(PORTS), {
            "qdrant_url": "http://127.0.0.1:6333",
            "api_base_url": "http://127.0.0.1:8003/v1",
            "embed_url": "",
            "rerank_url": "http://127.0.0.1:8004/v1/rerank",
        })

    def test_stack_config_ignores_the_users_file_and_env(self):
        # The stack verifies against the stack IT just stood up, not against
        # whatever the user's file or environment points at: an old
        # `embed_url` in the file or a `QDRANT_URL` in the environment must not
        # reach the check (the plan's "Config e ambiente que já apontam para
        # outro lugar"). `load` with an empty `env` reads NO file and NO
        # environment, so the user's `$QCTX_CONFIG` is off the path.
        config_file = Path(tempfile.mkdtemp()) / "config.json"
        config_file.write_text(
            '{"qdrant_url": "http://elsewhere:6333", "embed_url": "http://stale:1/v1"}\n')
        old_config = os.environ.get("QCTX_CONFIG")
        try:
            os.environ["QCTX_CONFIG"] = str(config_file)
            cfg = verify.stack_config(PORTS)
        finally:
            if old_config is None:
                os.environ.pop("QCTX_CONFIG", None)
            else:
                os.environ["QCTX_CONFIG"] = old_config
        self.assertEqual(cfg.qdrant_url, "http://127.0.0.1:6333")
        self.assertEqual(cfg.api_base_url, "http://127.0.0.1:8003/v1")
        self.assertEqual(cfg.embed_url, "")
        self.assertEqual(cfg.rerank_url, "http://127.0.0.1:8004/v1/rerank")
        self.assertEqual(cfg.vector_size, verify.EMBED_DIM)


class TestFunctional(unittest.TestCase):
    def test_functional_keeps_only_the_three_checks(self):
        rows = [
            core_setup.Check("Qdrant", True, "3 collections"),
            core_setup.Check("Embedding", True, "bge-m3 returns 1024 dimensions"),
            core_setup.Check("Re-rank", True, "bge-reranker-v2-m3 answers"),
            core_setup.Check("Context window", False, "no window declared", warning=True),
            core_setup.Check("Ignored settings", True, "none"),
            core_setup.Check("memory_collection", True, "'mem' (will be created)"),
        ]
        fake = lambda cfg: diagnose_fake(rows, detected_dim=1024)  # noqa: E731
        checks, dim = verify.functional(verify.stack_config(PORTS), diagnose=fake)
        self.assertEqual([c.name for c in checks],
                         list(verify.FUNCTIONAL_CHECKS))
        self.assertEqual(dim, 1024)

    def test_a_rerank_warning_fails_the_stack(self):
        # In `diagnose` a Re-rank failure is a WARNING (the reranker is
        # optional), but on a stack the step just stood it up, so here a
        # warning is a failure: `functional_ok` counts it.
        good = [core_setup.Check(name, True, "ok") for name in verify.FUNCTIONAL_CHECKS]
        self.assertTrue(verify.functional_ok(good))
        broken = list(good)
        broken[2] = core_setup.Check("Re-rank", False, "failed",
                                     "check the server", warning=True)
        self.assertFalse(verify.functional_ok(broken))
        # A non-warning failure fails too, through the same `ok` rule.
        blocked = list(good)
        blocked[0] = core_setup.Check("Qdrant", False, "did not answer")
        self.assertFalse(verify.functional_ok(blocked))


class TestCalibrate(unittest.TestCase):
    def test_calibration_warms_up_before_measuring(self):
        # `calibrate` brackets each of the two measurements with two clock
        # calls, so the step script lists the four in order: (embed start,
        # embed end, rerank start, rerank end). The zero at each bracket's
        # start keeps the bracket at zero cost, so the measured durations are
        # the two non-zero steps.
        clock = FakeClock([0.0, 0.3, 0.0, 0.5])
        embedder = FakeEmbedder()
        reranker = FakeReranker(docs=20)
        checks, info = verify.calibrate(
            verify.stack_config(PORTS),
            [verify.Budget("hermes", 2.0, 2.0)],
            embedder=embedder, reranker=reranker,
            clock=clock, memory=lambda: {"embed": 1, "rerank": 1})
        # The embedder's FIRST recorded call is the warm-up, and it has the SAME
        # shape as the measured call (one HARD_MAX_CHARS text): on the GPU the
        # first call of a new shape pays the backend's setup, so a small warm-up
        # left that cost inside the measurement. Measured 2026-10-08 on this
        # host's Intel Vulkan2, freshly loaded server each time, twice: the
        # one-short-text/one-document warm-up gave embed 1.77/1.76 s, rerank
        # 5.33/5.30 s; a same-shape warm-up gave 0.38/0.39 s and 1.98/1.98 s.
        self.assertEqual(embedder.calls[0][0], "embed")
        self.assertEqual(len(embedder.calls[0][1]), 1)
        self.assertEqual(len(embedder.calls[0][1][0]), core_chunk.HARD_MAX_CHARS)
        self.assertEqual(embedder.calls[1][0], "embed")
        self.assertEqual(len(embedder.calls[1][1][0]), core_chunk.HARD_MAX_CHARS)
        # The reranker warms up with the measured pool too: CALIBRATION_RERANK_DOCS
        # documents of TARGET_CHARS, then the same pool is timed, document for
        # document. (On the server the call judges 12 of the 20: the client's
        # max_docs.)
        self.assertEqual(len(reranker.calls[0][1]), verify.CALIBRATION_RERANK_DOCS)
        self.assertEqual(len(reranker.calls[0][1][0]), core_chunk.TARGET_CHARS)
        self.assertEqual(len(reranker.calls[1][1]), verify.CALIBRATION_RERANK_DOCS)
        self.assertEqual(len(reranker.calls[1][1][0]), core_chunk.TARGET_CHARS)
        # The WHOLE warm-up pool is the timed pool (not just its first document).
        self.assertEqual(reranker.calls[0][1], reranker.calls[1][1])
        # One check per host, named after the host, never raised.
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].name, "hermes")
        self.assertTrue(checks[0].ok)
        self.assertEqual(info["hermes"]["embed_s"], 0.3)
        self.assertEqual(info["hermes"]["rerank_s"], 0.5)
        self.assertEqual(info["hermes"]["memory"], {"embed": 1, "rerank": 1})

    def test_a_rerank_that_failed_is_a_warning_not_a_timing(self):
        # `rank` NEVER raises: a failure (the client's 15 s timeout on a slow
        # CPU, a refused connection) comes back as `info["ok"] = False`, which is
        # what `core/reranking.py` line 151 returns. Reading the clock around it
        # measured the failure, not a rerank, and a host whose recall cannot
        # rerank must not read "within budget".
        class FailingReranker(FakeReranker):
            def rank(self, query, documents):
                self.calls.append((query, list(documents)))
                return [], {"ok": False, "contract": "jina", "was_logit": False,
                            "error": "timed out after 15.0s"}
        clock = FakeClock([0.0, 0.3, 0.0, 15.0])
        checks, info = verify.calibrate(
            verify.stack_config(PORTS), [verify.Budget("hermes", 60.0, 120.0)],
            embedder=FakeEmbedder(), reranker=FailingReranker(),
            clock=clock, memory=None)
        self.assertFalse(checks[0].ok)
        self.assertTrue(checks[0].warning, "the stack works; recall degrades")
        self.assertIn("rerank failed", checks[0].detail)
        self.assertIn("timed out after 15.0s", checks[0].detail)
        self.assertNotIn("rerank 15.0 s (budget", checks[0].detail)
        self.assertIsNone(info["hermes"]["rerank_s"], "no rerank was measured")

    def test_no_reranker_is_not_a_failed_rerank(self):
        # A host with no rerank configured (`build_reranker` -> None, e.g. no
        # `rerank_model` on an external-endpoint machine) is not a failed one:
        # `rerank_s` is None, the check is ok, and the detail says the rerank was
        # not measured, not that it failed. `build_reranker` is mocked so the
        # genuinely-absent path is reached (a real config here WOULD build one).
        clock = FakeClock([0.0, 0.3])  # embed bracket only; no rerank clock calls
        with mock.patch.object(verify, "build_reranker", return_value=None):
            checks, info = verify.calibrate(
                verify.stack_config(PORTS), [verify.Budget("hermes", 2.0, 2.0)],
                embedder=FakeEmbedder(), clock=clock, memory=None)
        self.assertTrue(checks[0].ok)
        self.assertNotIn("rerank failed", checks[0].detail)
        self.assertIn("not measured (no rerank configured)", checks[0].detail)
        self.assertIsNone(info["hermes"]["rerank_s"])

    def test_over_budget_is_a_named_warning_never_a_raise(self):
        # Same four-call bracket as above: the measured embed is 0.3 s (under
        # its 2.0 s budget) and the measured rerank is 4.1 s (over its 2.0 s
        # one).
        clock = FakeClock([0.0, 0.3, 0.0, 4.1])
        embedder = FakeEmbedder()
        reranker = FakeReranker(docs=20)
        checks, info = verify.calibrate(
            verify.stack_config(PORTS),
            [verify.Budget("hermes", 2.0, 2.0)],
            embedder=embedder, reranker=reranker,
            clock=clock, memory=lambda: {"embed": 1, "rerank": 1})
        # The measured rerank (4.1 s) is over the 2.0 s budget, the measured
        # embed (0.3 s) is under its own: the result is a named warning, and
        # `calibrate` never raises for it.
        self.assertEqual(len(checks), 1)
        check = checks[0]
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)
        self.assertIn("hermes", check.detail)
        self.assertIn("rerank", check.detail)
        self.assertIn("4.1", check.detail)
        self.assertIn("2.0", check.detail)
        self.assertEqual(info["hermes"]["rerank_s"], 4.1)
        self.assertEqual(info["hermes"]["embed_s"], 0.3)


class TestEnvOverrides(unittest.TestCase):
    def test_env_overrides(self):
        wanted = {"qdrant_url": "http://127.0.0.1:6333", "embed_url": ""}
        with self.subTest("a different value is named"):
            self.assertEqual(
                verify.env_overrides({"QDRANT_URL": "http://elsewhere:6333"}, wanted),
                [("QDRANT_URL", "http://elsewhere:6333")])
        with self.subTest("an equal value is not"):
            self.assertEqual(
                verify.env_overrides({"QDRANT_URL": "http://127.0.0.1:6333"}, wanted),
                [])
        with self.subTest("a blank value is ignored"):
            self.assertEqual(
                verify.env_overrides({"QDRANT_URL": "   "}, wanted),
                [])
        with self.subTest("the canonical beats the legacy"):
            self.assertEqual(
                verify.env_overrides(
                    {"QCTX_QDRANT_URL": "http://127.0.0.1:6333",
                     "QDRANT_URL": "http://elsewhere:6333"}, wanted),
                [])
        with self.subTest("an unset wanted field names its alias"):
            self.assertEqual(
                verify.env_overrides({"RECALL_EMBED_URL": "http://stale:8003"}, wanted),
                [("RECALL_EMBED_URL", "http://stale:8003")])
        with self.subTest("several fields keep the wanted order"):
            out = verify.env_overrides(
                {"QDRANT_URL": "http://elsewhere:6333",
                 "RECALL_EMBED_URL": "http://stale:8003"}, wanted)
            self.assertEqual(
                out,
                [("QDRANT_URL", "http://elsewhere:6333"),
                 ("RECALL_EMBED_URL", "http://stale:8003")])


class TestTheRecallTopK(unittest.TestCase):
    def test_the_rerank_sample_is_the_recall_top_k(self):
        # The calibration reranks the SAME pool the recall hook does: its
        # TOP_K. Reading the default by AST keeps the two numbers from
        # drifting apart silently, which is the failure the spec names ("O
        # resultado é, por host, ok ou aviso" measured over the wrong pool).
        source = (REPO / "hooks" / "recall.py").read_text()
        tree = ast.parse(source)
        top_k = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "TOP_K"
                            for t in node.targets)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "env_num"
                    and node.value.args):
                top_k = node.value.args[2]
                break
        self.assertIsNotNone(top_k, "the TOP_K = env_num(...) assignment in hooks/recall.py")
        self.assertIsInstance(top_k, ast.Constant)
        self.assertEqual(int(top_k.value), verify.CALIBRATION_RERANK_DOCS)


class TestCosine(unittest.TestCase):
    def test_identical_vectors_are_one(self):
        # The norms' squares can miss the dot by an ulp, so the comparison is
        # to within precision, not byte-equality.
        self.assertAlmostEqual(verify.cosine([0.5, -1.0, 2.0], [0.5, -1.0, 2.0]), 1.0,
                               places=12)

    def test_orthogonal_vectors_are_zero(self):
        self.assertEqual(verify.cosine([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0)

    def test_the_known_pair(self):
        # [1, 0] vs [1, 1]: the cosine is 1/sqrt(2) by the definition itself.
        self.assertAlmostEqual(verify.cosine([1.0, 0.0], [1.0, 1.0]), 1 / math.sqrt(2),
                               places=12)

    def test_a_zero_norm_vector_is_a_degenerate_input(self):
        # A zero-norm vector has no direction: the cosine of it is defined as
        # 0.0, deterministically, so a degenerate corpus cannot divide by zero.
        self.assertEqual(verify.cosine([0.0, 0.0], [0.0, 0.0]), 0.0)
        self.assertEqual(verify.cosine([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestNumericalTexts(unittest.TestCase):
    """The fixed corpus of the dzn check: Ruling R1 makes it its own constant
    (the calibration's is synthetic and degenerate: one 6000-char text and 20
    IDENTICAL documents, a corpus in which every similarity is a tie)."""

    def test_the_corpus_is_fixed_distinct_and_english(self):
        # 8 DISTINCT, stable, English texts: a fixed query plus 7 documents.
        # The documents differ in topic AND in length (30 to ~600 chars), so
        # the similarities the models compute over them have real gaps — the
        # gaps the relative deviation is measured against.
        texts = verify.NUMERICAL_TEXTS
        query = verify.NUMERICAL_QUERY
        self.assertEqual(len(texts), 8)
        self.assertNotIn(query, texts, "the query is a separate constant")
        self.assertEqual(len(set(texts)), 8, "the texts must be distinct")
        self.assertEqual(len(set(texts) | {query}), 9, "no text equals the query")
        self.assertTrue(all(t.isascii() and t.strip() for t in [*texts, query]))
        self.assertTrue(all(t == t.lower() for t in texts), "lowercase English")
        self.assertLessEqual(len(query), 200)
        self.assertTrue(30 <= min(len(t) for t in texts)
                        <= max(len(t) for t in texts) <= 800,
                        "the documents span short to a few hundred chars")

    def test_the_query_is_not_one_of_the_documents(self):
        # The query must be its own text: embedding it as a ninth vector and
        # ranking the eight documents against it is what the check compares.
        self.assertNotIn(verify.NUMERICAL_QUERY, verify.NUMERICAL_TEXTS)


def _side(sims):
    """The vectors one side of the comparison embedded: `[1, 0]` for the query
    and `[s, sqrt(1 - s^2)]` for each document, so `cosine(query, doc_i)` is
    EXACTLY `sims[i]` (the dot is `s`, both norms are 1) — a compare over
    these sides measures the deviation of the sims themselves, the order of
    the similarities is the order of the sims, and the gaps are the gaps of
    the sims. Distinct sims keep the documents distinct."""
    vecs = [[1.0, 0.0]]
    for s in sims:
        vecs.append([s, math.sqrt(max(0.0, 1.0 - s * s))])
    return vecs


def _ranks(sims):
    """The similarity order of `sims`: the indices sorted by sim descending —
    the order a healthy reranker of these texts would answer."""
    return [i for i, _ in sorted(enumerate(sims), key=lambda p: -p[1])]


class TestNumericalCompare(unittest.TestCase):
    def test_identical_sims_and_rank_pass(self):
        sims = [0.9, 0.7, 0.5, 0.4, 0.35, 0.2, 0.1, 0.05]
        gpu, cpu = _side(sims), _side(sims)
        ok, reason = verify.numerical_compare(gpu, cpu, _ranks(sims), _ranks(sims))
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_an_order_swap_fails_naming_order(self):
        # The two texts adjacent in similarity TRADE PLACES between the gpu
        # and the cpu answers (the values of documents 2 and 3 are swapped):
        # the ORDER of the similarities differs, which flips a cut — the
        # check (1) failure, reported before any deviation is considered.
        sims = [0.9, 0.7, 0.5, 0.4, 0.35, 0.2, 0.1, 0.05]
        swapped = list(sims)
        swapped[2], swapped[3] = swapped[3], swapped[2]
        gpu, cpu = _side(sims), _side(swapped)
        ok, reason = verify.numerical_compare(gpu, cpu, _ranks(sims), _ranks(sims))
        self.assertFalse(ok)
        self.assertIn("order", reason)

    def test_a_deviation_above_the_min_gap_fails_naming_the_deviation_and_the_gap(self):
        # Same ORDER, but the deviation (0.05 at the first text) is LARGER than
        # the minimum gap between adjacent similarities of the gpu answer
        # (0.02): a deviation that can flip a cut, check (2) — and an ABSOLUTE
        # cosine threshold (0.05 < 0.1) would pass it, the relative rule does
        # not, which is the spec-mestra's own contrast.
        sims = [0.9, 0.7, 0.5, 0.3, 0.28, 0.2, 0.1, 0.05]  # min gap 0.02
        dev = list(sims)
        dev[0] = 0.85  # max |dev - sims| = 0.05, order untouched
        gpu, cpu = _side(dev), _side(sims)
        ok, reason = verify.numerical_compare(gpu, cpu, _ranks(dev), _ranks(sims))
        self.assertFalse(ok)
        self.assertIn("deviation", reason)
        self.assertIn("0.05", reason)
        self.assertIn("0.02", reason)

    def test_a_deviation_below_the_min_gap_passes(self):
        # The same shape, deviation 0.015 < min gap 0.02: the relative rule
        # passes what the spec wants to pass (the cuts cannot flip).
        sims = [0.9, 0.7, 0.5, 0.3, 0.28, 0.2, 0.1, 0.05]  # min gap 0.02
        dev = list(sims)
        dev[0] = 0.885  # deviation 0.015, order untouched
        gpu, cpu = _side(dev), _side(sims)
        ok, reason = verify.numerical_compare(gpu, cpu, _ranks(dev), _ranks(sims))
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_identical_sims_with_a_swapped_rank_fail_naming_rerank(self):
        # Check (3) alone: the similarities agree exactly, but the rerank
        # order of the fixed pairs differs (two indices swapped).
        sims = [0.9, 0.7, 0.5, 0.4, 0.35, 0.2, 0.1, 0.05]
        gpu, cpu = _side(sims), _side(sims)
        rank = list(_ranks(sims))
        rank[1], rank[2] = rank[2], rank[1]  # swap two adjacent pair indices
        ok, reason = verify.numerical_compare(gpu, cpu, rank, _ranks(sims))
        self.assertFalse(ok)
        self.assertIn("rerank", reason)

    def test_a_degenerate_equal_value_corpus_fails_on_any_nonzero_deviation(self):
        # The conservative case: the three documents at 0.5 tie, so the
        # minimum gap is 0, and ANY nonzero deviation is above it (0.0001 > 0)
        # — a corpus with no gap between them cannot be trusted to cut, and
        # the check says so instead of passing on the tie. The order is kept
        # (the bump is inside the tie group's own slot), so the gap check is
        # the one that fails, not the order.
        sims = [0.9, 0.7, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1]  # min gap 0 (the tie)
        dev = list(sims)
        dev[2] = 0.5 + 1e-4
        gpu, cpu = _side(dev), _side(sims)
        ok, reason = verify.numerical_compare(gpu, cpu, _ranks(dev), _ranks(sims))
        self.assertFalse(ok)
        self.assertIn("gap", reason)
        self.assertIn("0", reason)

    def test_identical_degenerate_corpus_passes(self):
        # The same tie everywhere, on BOTH sides: the deviation is 0, which is
        # not above the 0 gap, and no order differs — the tie passes, which is
        # the only reading of the rule that is not vacuous.
        sims = [0.9, 0.7, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1]
        gpu, cpu = _side(sims), _side(sims)
        ok, reason = verify.numerical_compare(gpu, cpu, _ranks(sims), _ranks(sims))
        self.assertTrue(ok)

    def test_a_deviation_equal_to_the_min_gap_fails_and_just_below_passes(self):
        # The boundary the spec pins: the deviation must be STRICTLY below the
        # minimum gap, so a deviation that EQUALS the gap fails — and one just
        # below it passes (proving the fix did not over-tighten). The
        # similarities are dyadic (powers of 1/2), so `dev` and `min_gap` are
        # exact by construction and the test does not depend on libm rounding:
        # `verify.cosine` is stubbed to return them, and the deviation branch
        # (not the cosine) is what is under test.
        axis = object()
        g1, g2, g3 = object(), object(), object()
        eq1, eq2, eq3 = object(), object(), object()
        below1, below2, below3 = object(), object(), object()
        table = {
            g1: 0.5, g2: 0.25, g3: 0.125,             # gpu answer: min gap 0.125
            eq1: 0.5, eq2: 0.25, eq3: 0.0,             # deviation 0.125 == min gap
            below1: 0.5, below2: 0.25, below3: 0.0625,  # deviation 0.0625 < gap
        }
        with mock.patch.object(verify, "cosine",
                               side_effect=lambda _axis, v: table[v]):
            # The vectors are sentinels, not real floats: `cosine` is stubbed
            # to answer from `table`, so the branch under test (deviation vs
            # min gap) runs over the dyadic values with no libm in the path.
            # dev == min_gap (both exactly 0.125) -> FAIL, naming the gap.
            ok, reason = verify.numerical_compare(  # type: ignore[arg-type]
                [axis, g1, g2, g3], [axis, eq1, eq2, eq3], [0, 1, 2], [0, 1, 2])
            self.assertFalse(ok)
            self.assertIn("gap", reason)
            # dev 0.0625 < min gap 0.125 -> PASS.
            ok, reason = verify.numerical_compare(  # type: ignore[arg-type]
                [axis, g1, g2, g3], [axis, below1, below2, below3],
                [0, 1, 2], [0, 1, 2])
            self.assertTrue(ok)
            self.assertEqual(reason, "")

    def test_unequal_vector_counts_refuse_to_align(self):
        # The length guard: a side that embedded a DIFFERENT number of vectors
        # than the other cannot be aligned, so the check refuses with a named
        # reason instead of comparing misaligned lists. (If the guard were
        # removed, the zip in the deviation check would silently truncate and
        # a shorter cpu side could pass - this is the guard the audit found
        # dead: no test fed unequal-length sides.)
        gpu = _side([0.9, 0.7, 0.5])
        cpu = _side([0.9, 0.7])  # one fewer vector
        ok, reason = verify.numerical_compare(gpu, cpu, _ranks([0.9, 0.7, 0.5]), _ranks([0.9, 0.7]))
        self.assertFalse(ok)
        self.assertIn("align", reason)
        self.assertIn("4 and 3", reason)  # gpu side (query+3) vs cpu side (query+2)


if __name__ == "__main__":
    unittest.main()
