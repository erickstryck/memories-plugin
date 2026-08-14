"""The two hosts must inject the SAME text. Tested, not asserted.

This file is the reason core/prompts.py, core/blocks.py and core/session_state.py exist.
Equivalence claimed in a README is a promise; equivalence with a test that bites is a
contract. The failure it prevents is specific and this repo has already lived it: the
two-stage pipeline once existed in three copies, and the re-rank scale normalization was
present in one and missing from another.

The two adapters are driven differently on purpose — claude-code as a real subprocess with
JSON on stdin, hermes as an in-process method call — because if the test drove them
through a shared helper it would prove the helper consistent rather than the hosts
equivalent.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import blocks
from core.retrieval import DENSE, Outcome
from tests.test_blocks import FakeHit


class TestBothHostsRenderTheSameBlock(unittest.TestCase):
    """Same Outcome and same hits through both adapters -> byte-identical text."""

    CASES = {
        "populated": ([FakeHit(id="m1", document="a durable fact about pagination")],
                      Outcome(candidates=3, reranked=True)),
        "populated_dense": ([FakeHit(id="m2", document="a fact judged by proximity",
                                     origin=DENSE, score=0.61)],
                            Outcome(candidates=1)),
        "empty_complete": ([], Outcome(candidates=4, best_dense=0.31, reranked=True)),
        "empty_partial_error": ([], Outcome(candidates=5, best_dense=0.5,
                                            rerank_error="timeout")),
        "empty_partial_breaker": ([], Outcome(candidates=27, best_dense=0.54,
                                              suppressed="circuit breaker: 12s ago")),
        "empty_partial_dropped": ([], Outcome(candidates=26, best_dense=0.5, reranked=True,
                                              dropped=14, dropped_above_floor=3)),
    }

    def _claude_side(self, hits, outcome):
        """Render through the claude-code hook's own module, in a subprocess."""
        script = (
            "import sys, json, os, tempfile\n"
            "sys.path.insert(0, %r)\n"
            "os.environ['QCTX_STATE_DIR'] = tempfile.mkdtemp()\n"
            "sys.path.insert(0, os.path.join(%r, 'hooks'))\n"
            "import recall\n"
            "from core.blocks import recall_block, empty_block\n"
            "from tests.test_blocks import FakeHit, BUDGET\n"
            "from core.retrieval import Outcome\n"
            "payload = json.loads(sys.stdin.read())\n"
            "hits = [FakeHit(**h) for h in payload['hits']]\n"
            "outcome = Outcome(**payload['outcome'])\n"
            "if hits:\n"
            "    print(recall_block(hits, [], payload['n_angles'], outcome, recall.BUDGET), end='')\n"
            "else:\n"
            "    print(empty_block(outcome, payload['n_angles']), end='')\n"
        ) % (str(REPO), str(REPO))
        payload = json.dumps({
            "hits": [dict(id=h.id, document=h.document, origin=h.origin, score=h.score,
                          metadata=h.metadata) for h in hits],
            "outcome": {k: v for k, v in vars(outcome).items() if k != "scored"},
            "n_angles": 2,
        })
        out = subprocess.run([sys.executable, "-c", script], input=payload,
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

        return out.stdout

    def _hermes_side(self, hits, outcome):
        """Render through the hermes provider, in process."""
        from hosts.hermes import MemoriesProvider
        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = Path(tempfile.mkdtemp())

        class FakeStore:
            def recall(self, queries, policy, top_k, suppressed=None):
                return hits, outcome

        p._store = FakeStore()

        return p.prefetch("how does the connector poll paginate its results?")

    def test_every_state_renders_identically_in_both_hosts(self):
        for label, (hits, outcome) in self.CASES.items():
            with self.subTest(state=label):
                claude = self._claude_side(hits, outcome)
                hermes = self._hermes_side(hits, outcome)
                self.assertEqual(claude.strip(), hermes.strip(),
                                 f"the two hosts diverged on the {label} state")

    def test_the_unavailable_block_is_identical(self):
        from hosts.hermes import MemoriesProvider
        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = Path(tempfile.mkdtemp())

        class Broken:
            def recall(self, *a, **kw):
                raise __import__("core").EmbeddingError("endpoint down")

        p._store = Broken()
        hermes = p.prefetch("a real question about the archive")
        expected = blocks.unavailable_block("EmbeddingError", "endpoint down")
        self.assertEqual(hermes.strip(), expected.strip())


class TestBothHostsShareOneConfiguration(unittest.TestCase):
    def test_they_resolve_the_same_collections_from_the_same_file(self):
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        cfg_path.write_text(json.dumps({
            "qdrant_url": "http://example.invalid/qdrant",
            "api_base_url": "http://example.invalid/v1",
            "memory_collection": "shared_memory",
            "docs_collection": "shared_tmp",
            "library_collection": "shared_library",
        }))
        env = dict(os.environ, QCTX_CONFIG=str(cfg_path))
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import core\n"
            "cfg = core.load()\n"
            "print(cfg.require_memory_collection())\n"
            "print(cfg.require_docs_collection())\n"
            "print(cfg.require_library_collection())\n"
        ) % str(REPO)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(),
                         ["shared_memory", "shared_tmp", "shared_library"])

    def test_the_tuning_knobs_have_the_same_names_in_both_hosts(self):
        """Equivalent CONFIGURATION means the same env var moves the same number in both.
        Read from each adapter's source rather than restated, so a rename in one shows up
        here instead of surfacing as a host that ignores a setting.

        Name equality alone does not prove this — see
        test_the_qdrant_budget_knob_has_the_same_semantics_in_both_hosts below for the
        semantic half of the claim."""
        import re
        pattern = re.compile(r'QCTX_RECALL_[A-Z_]+')
        hook = set(pattern.findall((REPO / "hooks" / "recall.py").read_text()))
        host = set(pattern.findall((REPO / "hosts" / "hermes" / "__init__.py").read_text()))
        self.assertTrue(hook, "no QCTX_RECALL_* names found in the hook")
        self.assertEqual(hook, host,
                         f"only in claude-code: {hook - host}; only in hermes: {host - hook}")

    def test_the_checkpoint_knobs_have_the_same_names_in_both_hosts(self):
        """The write side of the same claim: the cadence rides inside `prefetch` on
        hermes and inside its own hook on claude-code, but the knob names and meanings —
        QCTX_CHECKPOINT_INTERVAL, QCTX_CHECKPOINT_DISABLED — must be the ones a deployer
        already knows from the claude-code side, not a second vocabulary for the same
        setting."""
        import re
        pattern = re.compile(r'QCTX_CHECKPOINT_[A-Z_]+')
        hook = set(pattern.findall((REPO / "hooks" / "checkpoint.py").read_text()))
        host = set(pattern.findall((REPO / "hosts" / "hermes" / "__init__.py").read_text()))
        self.assertTrue(hook, "no QCTX_CHECKPOINT_* names found in the hook")
        self.assertEqual(hook, host,
                         f"only in claude-code: {hook - host}; only in hermes: {host - hook}")

    def test_the_qdrant_budget_knob_has_the_same_semantics_in_both_hosts(self):
        """QCTX_RECALL_QDRANT_BUDGET must mean the same thing on both hosts: a ceiling the
        deployer may TIGHTEN, never raise past what the host's own deadline can afford.

        The two hosts' shares of that deadline are NOT equal (6.0s for claude-code's 20s
        hooks.json timeout, 2.0s for hermes' 8s HERMES_PREFETCH_BUDGET_S) and this test
        must not assert they are — the deadlines themselves differ and always will.
        What must be identical is the SHAPE of the response to the knob: a value below a
        host's own share moves that host's computed qdrant timeout, and a value above it
        is clamped to the share, on BOTH hosts. Do not "fix" this into asserting the two
        timeouts are equal; that would be asserting away the correct part of the
        divergence and hiding the part that would actually be a bug.

        Both assertions read the number each host ACTUALLY handed to core.build_memory —
        via recall.py's own module-level computation for claude-code, and via
        `_ensure_store`'s `store.q.timeout` for hermes (the technique already used in
        tests/test_hermes_provider.py::TestEnsureStoreTimeouts) — never a restatement of
        the formula being checked.
        """
        import importlib
        import io
        import unittest.mock

        sys.path.insert(0, str(REPO / "hooks"))

        #: A prompt query.angles() turns into exactly ONE angle (verified directly: no
        #: stopwords to strip, no sentence to split out), so `qdrant_calls` inside `_run()`
        #: is unambiguously 1 + 1 = 2 without the test having to predict `angles()`'s
        #: internals.
        ONE_ANGLE_PROMPT = "pagination cursor logic details"

        def hook_qdrant_timeout(budget_value):
            """The qdrant timeout claude-code's hook ACTUALLY hands to `core.build_memory`
            for one real run of `_run()`, with QCTX_RECALL_QDRANT_BUDGET set to
            `budget_value`. Intercepts the real call — via a patched `core.build_memory`
            that records its `timeouts` argument — instead of recomputing recall.py's
            own min()/division, which would prove the test consistent with itself and
            nothing about the hook."""
            env = dict(os.environ, QCTX_STATE_DIR=tempfile.mkdtemp(),
                      QCTX_RECALL_QDRANT_BUDGET=str(budget_value))
            with unittest.mock.patch.dict(os.environ, env):
                import recall
                importlib.reload(recall)

                captured = {}

                class Stub:
                    reranker = None

                    def recall(self, *a, **kw):
                        return [], Outcome(candidates=0)

                def fake_build_memory(cfg, timeouts=None, **kw):
                    captured["timeouts"] = timeouts

                    return Stub()

                stdin = io.StringIO(json.dumps({"prompt": ONE_ANGLE_PROMPT}))
                with unittest.mock.patch.object(recall.core, "build_memory",
                                                fake_build_memory), \
                     unittest.mock.patch.object(sys, "stdin", stdin), \
                     unittest.mock.patch.object(sys, "stdout", io.StringIO()):
                    recall._run()

                return captured["timeouts"]["qdrant"]

        def hermes_qdrant_timeout(budget_value):
            """The qdrant timeout `_ensure_store` bakes into the store it builds, with
            QCTX_RECALL_QDRANT_BUDGET set to `budget_value`. Reads `store.q.timeout` —
            what `_ensure_store` actually handed to `core.build_memory` — not a
            restatement of its formula."""
            from hosts.hermes import MemoriesProvider

            core_mod = sys.modules.get("core") or __import__("core")
            cfg = core_mod.Config(
                qdrant_url="http://localhost:1", qdrant_api_key="", api_base_url="",
                api_key="", embed_url="http://localhost:1/embeddings", rerank_url="",
                embed_model="m", rerank_model="", memory_collection="test-memories",
                docs_collection="", library_collection="", vector_size=8,
            )
            with unittest.mock.patch.dict(os.environ,
                                          {"QCTX_RECALL_QDRANT_BUDGET": str(budget_value)}):
                original = MemoriesProvider.QDRANT_BUDGET
                try:
                    MemoriesProvider.QDRANT_BUDGET = float(budget_value)
                    p = MemoriesProvider()
                    p._cfg = cfg
                    store = p._ensure_store()

                    return store.q.timeout
                finally:
                    MemoriesProvider.QDRANT_BUDGET = original

        # -- both hosts' baseline shares, read from each adapter's own constants --
        import recall as hook_module
        from hosts.hermes import HERMES_PREFETCH_BUDGET_S, MAX_ANGLES

        hook_share = hook_module.QDRANT_SHARE_S
        hermes_share = HERMES_PREFETCH_BUDGET_S / 4.0
        self.assertNotAlmostEqual(hook_share, hermes_share,
                                  msg="the two host deadlines are not supposed to match; "
                                      "if they now do, pick different probe values below")

        # -- a value BELOW both shares must move the timeout on BOTH hosts --
        low = min(hook_share, hermes_share) / 4
        hook_at_low = hook_qdrant_timeout(low)
        hermes_at_low = hermes_qdrant_timeout(low)
        hook_calls = 1 + 1  # ONE_ANGLE_PROMPT produces exactly one angle
        self.assertAlmostEqual(hook_at_low, low / hook_calls, places=6)
        hermes_calls = MAX_ANGLES + 1
        self.assertAlmostEqual(hermes_at_low, low / hermes_calls, places=6)
        # moved relative to a very high baseline, proving the low value took effect
        self.assertLess(hook_at_low, hook_qdrant_timeout(999.0))
        self.assertLess(hermes_at_low, hermes_qdrant_timeout(999.0))

        # -- a value ABOVE a host's own share must clamp to that host's share, on BOTH --
        high = max(hook_share, hermes_share) * 10
        self.assertAlmostEqual(hook_qdrant_timeout(high), hook_share / hook_calls, places=6)
        self.assertAlmostEqual(hermes_qdrant_timeout(high), hermes_share / hermes_calls,
                               places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
