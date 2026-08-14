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
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import blocks
from core.retrieval import DENSE, Outcome
from tests.test_blocks import FakeHit


def hermes_adapter_source() -> str:
    """EVERY python file of the hermes adapter, concatenated.

    The knob tests below read the environment variable names out of the adapter's source.
    Reading only `__init__.py` was enough while the adapter was one file; now that it has a
    `tools.py`, a knob introduced there would be invisible to the comparison — which is the
    exact failure these tests exist to prevent, one directory deeper.
    """
    return "\n".join(path.read_text()
                     for path in sorted((REPO / "hosts" / "hermes").glob("*.py")))


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


class TestTheInjectedTextNamesNoSingleHostSurface(unittest.TestCase):
    """`core/prompts.py` is injected VERBATIM into both hosts, so a sentence in it that names
    one host's surface is wrong on the other — and unlike a divergence in the block, nothing
    else would catch it: the two hosts inject the same wrong sentence identically.

    The concrete case: the checkpoint procedure ended "the commands are in the memory skill",
    which on hermes points at something the model cannot load. Skills are the claude-code
    surface; hermes gets the 15 tools.
    """

    def texts(self):
        from core import prompts

        return {
            "CHECKPOINT_PROCEDURE": prompts.CHECKPOINT_PROCEDURE,
            "INSTRUCTIONS": prompts.INSTRUCTIONS,
        }

    def test_no_shared_text_promises_a_skill_without_naming_the_tools_too(self):
        for name, text in self.texts().items():
            with self.subTest(text=name):
                if "skill" in text.lower():
                    self.assertIn("tool", text.lower(),
                                  f"{name} sends the model to a skill only — on hermes there "
                                  f"is no loadable memory skill, only the tools")

    def test_no_shared_text_names_a_host_or_its_hook(self):
        for name, text in self.texts().items():
            with self.subTest(text=name):
                lowered = text.lower()
                for word in ("userpromptsubmit", "claude code", "claude-code", "hermes"):
                    self.assertNotIn(word, lowered,
                                     f"{name} names {word!r}, and it is injected into both")


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

    def test_the_hermes_wizard_writes_what_claude_code_actually_loads(self):
        """The claim this task adds: `hermes memory setup` -> `save_config` writes the
        SAME file the test above proves claude-code reads. The test above writes the file
        by hand, which only proves the FILE FORMAT is shared; it says nothing about the
        wizard's own write path. This one drives that path — `MemoriesProvider.save_config`
        — and reads the result back through `core.load()`, the exact function claude-code's
        hook and CLI call.

        Also proves the merge half of the same claim: a field neither wizard call ever
        named (`embed_model`) has to survive, because the user may already be running
        claude-code against this file when they open the hermes wizard, and clobbering it
        would break a session that has nothing to do with the setup flow.

        Runs entirely in ONE subprocess with QCTX_CONFIG set in its environment before
        launch: `core.config.DEFAULT_CONFIG_PATH` is computed once at import time
        (core/config.py:22-25), so setting the variable any later — or in a second,
        separate subprocess without re-passing it — would silently miss the file this test
        just wrote and either fail loudly (nonexistent path) or, worse, land on whatever
        path was frozen in by an earlier import in this process, which could be the
        operator's real config.
        """
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        cfg_path.write_text(json.dumps({"embed_model": "already-configured-model"}))
        env = dict(os.environ, QCTX_CONFIG=str(cfg_path))
        script = (
            "import sys, json; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "MemoriesProvider().save_config({\n"
            "    'qdrant_url': 'http://example.invalid/qdrant',\n"
            "    'api_base_url': 'http://example.invalid/v1',\n"
            "    'memory_collection': 'wizard_shared_memory',\n"
            "    'docs_collection': 'wizard_shared_tmp',\n"
            "    'library_collection': 'wizard_shared_library',\n"
            "}, %r)\n"
            "import core\n"
            "cfg = core.load()\n"
            "print(cfg.require_memory_collection())\n"
            "print(cfg.require_docs_collection())\n"
            "print(cfg.require_library_collection())\n"
            "print(cfg.embed_model)\n"
        ) % (str(REPO), str(cfg_dir))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(),
                         ["wizard_shared_memory", "wizard_shared_tmp",
                          "wizard_shared_library", "already-configured-model"])

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
        host = set(pattern.findall(hermes_adapter_source()))
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
        host = set(pattern.findall(hermes_adapter_source()))
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


#: A prompt `query.angles()` turns into exactly ONE angle, reused by the top_k test below
#: for the same reason the qdrant-budget test uses it: it keeps the arithmetic of the number
#: of store calls out of the assertions.
ONE_ANGLE_PROMPT = "pagination cursor logic details"


class TestTheTopKKnobMeansTheSameThingInBothHosts(unittest.TestCase):
    """`QCTX_RECALL_TOP_K` is CONDITIONAL on the second stage being available, on both hosts.

    The hook computes it after breaker suppression (`hooks/recall.py`: `"20" if
    store.reranker else "8"`), because with no cross-encoder to filter the candidates there
    is no reason to pull 2.5x the Qdrant payload — and the states where it is absent are
    exactly the degraded ones the breaker exists to shed load in. The hermes adapter used the
    unconditional literal 20, so with no cross-encoder configured, or for the 300s after any
    rerank failure, the two hosts pulled different amounts from the same archive and returned
    DIFFERENT memory sets for the same prompt. Measured before the fix:

        cross-encoder PRESENT   claude top_k=20  hermes top_k=20   same
        cross-encoder ABSENT    claude top_k=8   hermes top_k=20   *** DIVERGE ***

    Both sides are read at the store boundary — the `top_k` each host actually hands to
    `MemoryStore.recall` — never from a restatement of the arithmetic. Name equality passed
    over this divergence, and over the two before it; this is the third.

    Both hosts run in SUBPROCESSES with `QCTX_RECALL_TOP_K`/`RECALL_TOP_K` cleared from the
    environment, and that is load-bearing twice over: the hermes values are class attributes
    frozen at import time, and this user exports `RECALL_*` from their `.bashrc`, which would
    otherwise pin both states to one number and make every assertion below vacuous.
    """

    PROMPT = ONE_ANGLE_PROMPT

    def _env(self, state_dir, override=None):
        env = {k: v for k, v in os.environ.items()
               if k not in ("QCTX_RECALL_TOP_K", "RECALL_TOP_K")}
        env["QCTX_STATE_DIR"] = str(state_dir)
        if override is not None:
            env["QCTX_RECALL_TOP_K"] = str(override)

        return env

    def _stub(self, has_reranker: bool) -> str:
        return (
            "class Stub:\n"
            "    reranker = object() if %r else None\n"
            "    def recall(self, queries, policy, top_k, suppressed=None):\n"
            "        captured.append(top_k)\n"
            "        return [], Outcome(candidates=0)\n"
        ) % has_reranker

    def _run(self, script, env):
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        captured = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(len(captured), 1, "the host did not reach the store exactly once")

        return captured[0]

    def _claude_top_k(self, has_reranker, state_dir, override=None):
        """What `hooks/recall.py::_run()` hands to `store.recall`, for one real run.

        `core.load` is patched away as well as `core.build_memory`: the cfg is never used
        once the store is a stub, and depending on the operator's real configuration file
        would make this test pass or fail for a reason that has nothing to do with top_k.
        """
        script = (
            "import sys, json, io, unittest.mock\n"
            "sys.path.insert(0, %r)\n"
            "sys.path.insert(0, %r)\n"
            "import recall\n"
            "from core.retrieval import Outcome\n"
            "captured = []\n"
            + self._stub(has_reranker) +
            "with unittest.mock.patch.object(recall.core, 'build_memory',\n"
            "                                lambda cfg, **kw: Stub()), \\\n"
            "     unittest.mock.patch.object(recall.core, 'load', lambda: object()), \\\n"
            "     unittest.mock.patch.object(sys, 'stdin',\n"
            "                                io.StringIO(json.dumps({'prompt': %r}))), \\\n"
            "     unittest.mock.patch.object(sys, 'stdout', io.StringIO()):\n"
            "    recall._run()\n"
            "print(json.dumps(captured))\n"
        ) % (str(REPO), str(REPO / "hooks"), self.PROMPT)

        return self._run(script, self._env(state_dir, override))

    def _hermes_top_k(self, has_reranker, state_dir, override=None):
        """What `MemoriesProvider.prefetch` hands to `store.recall`, for one real call."""
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from hosts.hermes import MemoriesProvider\n"
            "from core.retrieval import Outcome\n"
            "captured = []\n"
            + self._stub(has_reranker) +
            "p = MemoriesProvider()\n"
            "p._cfg = object()\n"
            "p._state_dir = Path(%r)\n"
            "p._store = Stub()\n"
            "p.prefetch(%r)\n"
            "print(json.dumps(captured))\n"
        ) % (str(REPO), str(state_dir), self.PROMPT)

        return self._run(script, self._env(state_dir, override))

    def _state_dir(self, armed_breaker=False) -> Path:
        """A fresh state directory, with the rerank breaker already armed when asked.

        Armed by writing the file the shared `core.breaker.Breaker` reads, so the
        suppression below is the real one both hosts consult rather than a patched flag.
        """
        d = Path(tempfile.mkdtemp())
        if armed_breaker:
            (d / "rerank-breaker").write_text(str(time.time()))

        return d

    def test_with_a_cross_encoder_both_hosts_ask_for_the_same_top_k(self):
        lenient = self._claude_top_k(True, self._state_dir())
        self.assertEqual(lenient, self._hermes_top_k(True, self._state_dir()),
                         "the two hosts pull different amounts from the same archive")
        self.lenient = lenient

    def test_without_a_cross_encoder_both_hosts_fall_to_the_SAME_stricter_top_k(self):
        claude = self._claude_top_k(False, self._state_dir())
        hermes = self._hermes_top_k(False, self._state_dir())
        self.assertEqual(claude, hermes,
                         "with no second stage the two hosts ask Qdrant for different "
                         "amounts, so they return different memory sets for one prompt")
        with_ce = self._claude_top_k(True, self._state_dir())
        self.assertLess(claude, with_ce,
                        "the strict value has to be strictly smaller, or this test is "
                        "comparing one unconditional number with itself")

    def test_a_suppressed_cross_encoder_falls_to_the_strict_top_k_on_both_hosts(self):
        """The breaker case: a reranker IS configured, and both hosts have just turned it off
        for this invocation. That is the state the breaker exists to shed load in, and it is
        where asking for 2.5x the payload on the tighter deadline is worst."""
        claude = self._claude_top_k(True, self._state_dir(armed_breaker=True))
        hermes = self._hermes_top_k(True, self._state_dir(armed_breaker=True))
        self.assertEqual(claude, hermes)
        self.assertEqual(claude, self._claude_top_k(False, self._state_dir()),
                         "a suppressed cross-encoder must count as an absent one")

    def test_a_malformed_value_costs_neither_host_its_recall(self):
        """`QCTX_RECALL_TOP_K=8x` must fall back on BOTH hosts, and cost neither its search.

        The tolerance itself is held per host (tests/test_hermes_provider.py derives every
        numeric knob from all three source files and imports each with garbage in the
        environment); what is asserted HERE is the half that only a comparison can state: the
        two hosts respond to the same bad value the same way.

        They did not. Measured, with the archive perfectly reachable: hermes fell back and
        kept working, while the hook — the only knob in `hooks/recall.py` read with a bare
        `int(env(...))` — raised ValueError inside `_run`, and `main`'s catch-all turned that
        into "[automatic recall — UNAVAILABLE for this prompt] … the hook failed
        (ValueError)" on every prompt of every session. That block tells the model the archive
        was not consulted, which the model then has to believe, and `env_num`'s explanatory
        note never ran because the value never went through it.

        So this asserts the emitted BLOCK too, not just the number: the number alone would
        pass on a host that fell back and then reported unavailability anyway.
        """
        state_dir = self._state_dir()
        claude = (
            "import sys, json, io, unittest.mock\n"
            "sys.path.insert(0, %r)\n"
            "sys.path.insert(0, %r)\n"
            "import recall\n"
            "from core.retrieval import Outcome\n"
            "captured = []\n"
            + self._stub(True) +
            "buf = io.StringIO()\n"
            "with unittest.mock.patch.object(recall.core, 'build_memory',\n"
            "                                lambda cfg, **kw: Stub()), \\\n"
            "     unittest.mock.patch.object(recall.core, 'load', lambda: object()), \\\n"
            "     unittest.mock.patch.object(sys, 'stdin',\n"
            "                                io.StringIO(json.dumps({'prompt': %r}))), \\\n"
            "     unittest.mock.patch.object(sys, 'stdout', buf):\n"
            "    recall._run()\n"
            "print(json.dumps([captured, buf.getvalue()]))\n"
        ) % (str(REPO), str(REPO / "hooks"), self.PROMPT)
        hermes = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from hosts.hermes import MemoriesProvider\n"
            "from core.retrieval import Outcome\n"
            "captured = []\n"
            + self._stub(True) +
            "p = MemoriesProvider()\n"
            "p._cfg = object()\n"
            "p._state_dir = Path(%r)\n"
            "p._store = Stub()\n"
            "block = p.prefetch(%r)\n"
            "print(json.dumps([captured, block]))\n"
        ) % (str(REPO), str(state_dir), self.PROMPT)

        env = self._env(state_dir, override="8x")
        results = {}
        for label, script in (("claude-code", claude), ("hermes", hermes)):
            out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                 text=True, env=env)
            self.assertEqual(out.returncode, 0,
                             f"{label} died on a malformed knob:\n{out.stderr}")
            captured, block = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(len(captured), 1,
                             f"{label} did not reach the archive exactly once")
            self.assertNotIn("UNAVAILABLE", block,
                             f"{label} reported the archive as not consulted because a "
                             f"KNOB was malformed, while the archive was reachable")
            results[label] = captured[0]

        self.assertEqual(results["claude-code"], results["hermes"],
                         "the two hosts disagree about what a malformed knob falls back to")
        self.assertEqual(results["claude-code"], self._claude_top_k(True, self._state_dir()),
                         "the fallback is not the coded default the knob would have had")

    def test_an_explicit_value_overrides_both_defaults_on_both_hosts(self):
        """The knob still means "this many, whatever the state" when the deployer sets it —
        the same on both hosts, and in both states."""
        for has_reranker in (True, False):
            with self.subTest(cross_encoder=has_reranker):
                self.assertEqual(
                    self._claude_top_k(has_reranker, self._state_dir(), override=5), 5)
                self.assertEqual(
                    self._hermes_top_k(has_reranker, self._state_dir(), override=5), 5)


#: A REAL `core.MemoryStore` over the in-memory fakes, holding three memories that all match
#: the prompt below. Built inside each host's own process, as the store `core.build_memory`
#: would have returned, so what the ceilings below cut is a real retrieval and not a stub's
#: hard-coded answer — the whole point is that `max_memories` is applied as a SLICE.
_REAL_STORE_OVER_FAKES = (
    "from core.memory import MemoryStore\n"
    "from tests.fakes import FakeEmbedder, FakeVectorStore\n"
    "FACTS = [\n"
    "    'the connector poll paginates with a cursor kept in the checkpoint',\n"
    "    'the connector poll retries a paginate cursor failure three times',\n"
    "    'the paginate cursor of the connector poll is opaque and must not be parsed',\n"
    "]\n"
    "def build_store(cfg=None, **kw):\n"
    "    q, emb = FakeVectorStore(), FakeEmbedder()\n"
    "    q.ensure_collection('mem', 8)\n"
    "    store = MemoryStore(q, emb, None, 'mem', 8)\n"
    "    for fact in FACTS:\n"
    "        store.store(fact)\n"
    "    return store\n"
)


class TestAZeroedKnobCannotMakeEitherHostCLAIMAbsence(unittest.TestCase):
    """A knob set to 0 must cost reach, never turn into a claim that the archive is empty.

    The tolerant readers accept `0` and negatives for every integer knob, and `0` meaning
    "unlimited" is a common deployer convention — so this was one plausible typo from a
    permanent, silent lie on every prompt. Measured against three stored memories that all
    match, before the floors: `QCTX_RECALL_MAX_MEMORIES=6` gave 3 hits, `=1` gave 1, `=0`
    gave 0 hits and a block reading "There is no recorded precedent on this subject", and
    `=-1` silently dropped the lowest hit. `QCTX_RECALL_TOP_K=0` asked Qdrant for nothing.

    The ruling is CLAMP, not refuse: a knob that produces a false claim of absence is worse
    than a knob that ignores an absurd value, and refusing to start would trade a silent lie
    for a dead host on a path that fires at every user interaction.

    Asserted on the BLOCK, not on a hit count, and structurally rather than by prose: the
    populated block is the only one carrying the rules of use (`PREVAILS`), and neither the
    empty nor the unavailable block may appear while the archive is answering.
    """

    PROMPT = "how does the connector poll paginate its cursor?"
    FLAT_CLAIM = "There is no recorded precedent"
    EMPTY_MARKER = "no memory above the relevance cutoff"

    #: The floors are neutralised on purpose. They are thresholds, not ceilings — the subject
    #: here is what the CEILINGS do when zeroed — and leaving them at their production values
    #: would make the assertions depend on the fake embedder's cosine scores instead of on
    #: the knob under test.
    def _env(self, state_dir, knob, value):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("QCTX_RECALL_", "RECALL_"))}
        env["QCTX_STATE_DIR"] = str(state_dir)
        env["QCTX_RECALL_STRICT_FLOOR"] = "0.0"
        env["QCTX_RECALL_DENSE_FLOOR"] = "0.0"
        env[knob] = str(value)

        return env

    def _claude_block(self, env) -> str:
        """The context `hooks/recall.py` actually emits, for one real run."""
        script = (
            "import sys, json, io, unittest.mock\n"
            "sys.path.insert(0, %r)\n"
            "sys.path.insert(0, %r)\n"
            "import recall\n"
            + _REAL_STORE_OVER_FAKES +
            "with unittest.mock.patch.object(recall.core, 'build_memory', build_store), \\\n"
            "     unittest.mock.patch.object(recall.core, 'load', lambda: object()), \\\n"
            "     unittest.mock.patch.object(sys, 'stdin',\n"
            "                                io.StringIO(json.dumps({'prompt': %r}))):\n"
            "    recall._run()\n"
        ) % (str(REPO), str(REPO / "hooks"), self.PROMPT)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        emitted = out.stdout.strip().splitlines()
        self.assertTrue(emitted, f"the hook emitted NOTHING at all:\n{out.stderr}")

        return json.loads(emitted[-1])["hookSpecificOutput"]["additionalContext"]

    def _hermes_block(self, env, state_dir) -> str:
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from hosts.hermes import MemoriesProvider\n"
            + _REAL_STORE_OVER_FAKES +
            "p = MemoriesProvider()\n"
            "p._cfg = object()\n"
            "p._state_dir = Path(%r)\n"
            "p._store = build_store()\n"
            "print(json.dumps(p.prefetch(%r)))\n"
        ) % (str(REPO), str(state_dir), self.PROMPT)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)

        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_a_zeroed_ceiling_still_delivers_memories_on_both_hosts(self):
        for knob, value in (("QCTX_RECALL_MAX_MEMORIES", 0),
                            ("QCTX_RECALL_MAX_MEMORIES", -1),
                            ("QCTX_RECALL_TOP_K", 0),
                            ("QCTX_RECALL_MAX_CHARS", 0),
                            ("QCTX_RECALL_MAX_PER_MEM", 0)):
            for host in ("claude-code", "hermes"):
                with self.subTest(knob=knob, value=value, host=host):
                    state_dir = Path(tempfile.mkdtemp())
                    env = self._env(state_dir, knob, value)
                    block = (self._claude_block(env) if host == "claude-code"
                             else self._hermes_block(env, state_dir))
                    self.assertNotIn(self.FLAT_CLAIM, block,
                                     f"{knob}={value} turned an archive that answered into "
                                     f"a claim that nothing is stored")
                    self.assertNotIn(self.EMPTY_MARKER, block,
                                     f"{knob}={value} emptied the result set")
                    self.assertNotIn("UNAVAILABLE", block)
                    self.assertIn("PREVAILS", block,
                                  "the rules of use travel with delivered memories, so "
                                  "their absence means nothing was delivered")

    #: Every knob that can zero the result set, and the attribute each host holds it in. The
    #: two `TOP_K` attributes are one knob with two defaults (lenient with a cross-encoder,
    #: strict without) and BOTH need the floor: the block-level test above runs against a
    #: store with no reranker, so it can only ever witness the strict one.
    CLAMPED = ("QCTX_RECALL_MAX_MEMORIES", "QCTX_RECALL_MAX_CHARS",
               "QCTX_RECALL_MAX_PER_MEM", "QCTX_RECALL_TOP_K")
    CLAMPED_ATTRS = ("MAX_MEMORIES", "MAX_CHARS", "MAX_PER_MEM", "TOP_K", "TOP_K_STRICT")

    def test_every_zeroable_ceiling_is_clamped_on_both_hosts_and_says_so(self):
        """The value-level half, because a block-level test can only witness the ceilings a
        given store's shape actually reaches. Both hosts are read in ONE process with all four
        knobs zeroed at once: the two adapters read the same variable names, and a deployer's
        environment is not scoped per host either.

        The note is required as well as the value. Silence would leave the deployer with a
        knob that does not do what it says — and stderr is the one channel neither host reads
        as data: stdout carries the hook protocol on claude-code and the injected block itself
        on hermes.
        """
        state_dir = Path(tempfile.mkdtemp())
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("QCTX_RECALL_", "RECALL_"))}
        env["QCTX_STATE_DIR"] = str(state_dir)
        for knob in self.CLAMPED:
            env[knob] = "0"
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "sys.path.insert(0, %r)\n"
            "import recall\n"
            "from hosts.hermes import MemoriesProvider as P\n"
            "attrs = %r\n"
            "print(json.dumps({'claude-code': {a: getattr(recall, a) for a in attrs},\n"
            "                  'hermes': {a: getattr(P, a) for a in attrs}}))\n"
        ) % (str(REPO), str(REPO / "hooks"), list(self.CLAMPED_ATTRS))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        values = json.loads(out.stdout.strip().splitlines()[-1])
        for host, attrs in values.items():
            for attr, value in attrs.items():
                with self.subTest(host=host, knob=attr):
                    self.assertEqual(value, 1,
                                     f"{host}: {attr}={value} leaves nothing to return")
        for knob in self.CLAMPED:
            with self.subTest(knob=knob):
                self.assertGreaterEqual(out.stderr.lower().count(knob.lower()), 2,
                                        f"each host has to say it clamped {knob}:\n"
                                        f"{out.stderr}")


#: A prompt that survives `query.skip_reason` and produces a real recall round.
HITS_PROMPT = "how does the connector poll paginate its results?"

#: Two hits, so a state file written by either host carries more than one `seen` entry.
_STUB_STORE = (
    "class Stub:\n"
    "    reranker = None\n"
    "    def recall(self, queries, policy, top_k, suppressed=None):\n"
    "        return ([FakeHit(id='hit-1', document='fact one', origin=CE),\n"
    "                 FakeHit(id='hit-2', document='fact two', origin=CE)],\n"
    "                Outcome(candidates=2, reranked=True))\n"
)


def drive_claude_rounds(state_dir, session: str, rounds: int = 1) -> None:
    """Run `hooks/recall.py::_run()` for real, `rounds` times, against `state_dir`.

    A subprocess and the hook's own `_run`, not a helper shared with the hermes driver
    below: the two hosts are driven through their own entry points on purpose (see this
    module's docstring), and the hook only reads `QCTX_STATE_DIR` at import time.

    `core.load` is patched away along with `core.build_memory` so nothing here depends on
    the operator's real configuration file.
    """
    script = (
        "import sys, json, io, unittest.mock\n"
        "sys.path.insert(0, %r)\n"
        "sys.path.insert(0, %r)\n"
        "import recall\n"
        "from core.retrieval import CE, Outcome\n"
        "from tests.test_blocks import FakeHit\n"
        + _STUB_STORE +
        "for _ in range(%d):\n"
        "    payload = json.dumps({'prompt': %r, 'session_id': %r})\n"
        "    with unittest.mock.patch.object(recall.core, 'build_memory',\n"
        "                                    lambda cfg, **kw: Stub()), \\\n"
        "         unittest.mock.patch.object(recall.core, 'load', lambda: object()), \\\n"
        "         unittest.mock.patch.object(sys, 'stdin', io.StringIO(payload)), \\\n"
        "         unittest.mock.patch.object(sys, 'stdout', io.StringIO()):\n"
        "        recall._run()\n"
    ) % (str(REPO), str(REPO / "hooks"), rounds, HITS_PROMPT, session)
    env = dict(os.environ, QCTX_STATE_DIR=str(state_dir))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env=env)
    if out.returncode != 0:
        raise AssertionError(f"the claude-code hook failed to run:\n{out.stderr}")


def drive_hermes_rounds(state_dir, session: str, rounds: int = 1) -> None:
    """Run `MemoriesProvider.prefetch` for real, `rounds` times, against `state_dir`."""
    from core.retrieval import CE, Outcome
    from hosts.hermes import MemoriesProvider
    p = MemoriesProvider()
    p._cfg = object()
    p._state_dir = Path(state_dir)

    class Stub:
        reranker = None

        def recall(self, queries, policy, top_k, suppressed=None):
            return ([FakeHit(id="hit-1", document="fact one", origin=CE),
                     FakeHit(id="hit-2", document="fact two", origin=CE)],
                    Outcome(candidates=2, reranked=True))

    p._store = Stub()
    for _ in range(rounds):
        p.prefetch(HITS_PROMPT, session_id=session)


#: The session id each host writes its own state under. Different strings on purpose:
#: nothing here may depend on the two hosts sharing a session name.
HOSTS = (("claude", drive_claude_rounds, "claude-live"),
         ("hermes", drive_hermes_rounds, "hermes-live"))


class TestBothHostsSweepDeadSessionState(unittest.TestCase):
    """`core.session_state.purge_dead` exists so a state directory does not grow one file per
    session forever — verbatim what its own docstring says. Spec §4 names dead-session
    purging as content that moved into `core/` SO BOTH HOSTS SHARE IT, and then only the
    claude-code hook ever called it: measured, 60 hermes prefetch rounds against a state dir
    holding a 30-day-old abandoned file left the file exactly where it was.

    The cadence is read from the hook rather than invented (`PURGE_EVERY_ROUNDS`, after the
    state save on a round that found memories), and it is asserted in BOTH directions on both
    hosts: a round that is not a multiple must NOT pay for a glob, and a round that is must
    sweep. A test that only checked "the file eventually disappears" would pass over a host
    that swept on every single prompt.
    """

    def _state_dir(self, session: str, round_no: int) -> tuple:
        """A state dir with one live session at `round_no` and one abandoned 30-day-old file."""
        d = Path(tempfile.mkdtemp())
        (d / f"recall-{session}.json").write_text(json.dumps({"round": round_no, "seen": {}}))
        abandoned = d / "recall-an_abandoned_session.json"
        abandoned.write_text(json.dumps({"round": 3, "seen": {}}))
        stamp = time.time() - 30 * 86400
        os.utime(abandoned, (stamp, stamp))

        return d, abandoned

    def test_a_round_on_the_cadence_sweeps_a_dead_session_on_both_hosts(self):
        from core.session_state import PURGE_EVERY_ROUNDS
        for host, drive, session in HOSTS:
            with self.subTest(host=host):
                # next round IS the cadence
                d, abandoned = self._state_dir(session, PURGE_EVERY_ROUNDS - 1)
                drive(d, session)
                self.assertFalse(abandoned.exists(),
                                 f"{host} never sweeps: the state directory grows one file "
                                 f"per session forever, which is what purge_dead exists to "
                                 f"prevent")
                self.assertTrue((d / f"recall-{session}.json").exists(),
                                "the live session's own state must survive the sweep")

    def test_a_round_off_the_cadence_does_not_pay_for_the_sweep_on_either_host(self):
        from core.session_state import PURGE_EVERY_ROUNDS
        for host, drive, session in HOSTS:
            with self.subTest(host=host):
                # next round is not a multiple of the cadence
                d, abandoned = self._state_dir(session, PURGE_EVERY_ROUNDS - 3)
                drive(d, session)
                self.assertTrue(abandoned.exists(),
                                f"{host} swept off the shared cadence — a glob on every "
                                f"prompt is what the cadence exists to avoid")


class TestBothHostsHealACorruptedSeen(unittest.TestCase):
    """A non-dict `seen` in a state file must be REPLACED on disk, on both hosts.

    Two guards, and ledger ruling F8 kept both: `core.blocks.split_by_budget` degrades a
    non-dict `seen` to "nothing has been seen" so no host can crash on it, and the persisted
    state is HEALED so the corruption does not outlive the round. `split_by_budget` does its
    half on a local substitute on purpose ("the caller owns persistence"), so without the
    second half every later round silently loses dedup: every recalled memory is reinjected
    in full every turn, burning the 14000-char budget on repeats.

    Only claude-code had the healing half. Measured over 3 rounds against
    `{"round": 3, "seen": "corrupted-not-a-dict"}`:

        hermes : {"round": 6, "seen": "corrupted-not-a-dict"}   healed? False
        claude : {"round": 6, "seen": {"hit-1": 4, "hit-2": 4}}  healed? True

    And the healing on the claude side was held by NOTHING — removing the guard left the
    whole suite green, so a future cleanup ("the core already handles this") would have
    deleted a guard that cost F8 a fix round and two rulings to establish. These assertions
    read the state ON DISK, which is the only place the difference between "this round
    survived" and "the corruption is gone" is visible.
    """

    CORRUPT = {"round": 3, "seen": "corrupted-not-a-dict"}

    def _corrupted_state(self, session: str) -> tuple:
        d = Path(tempfile.mkdtemp())
        path = d / f"recall-{session}.json"
        path.write_text(json.dumps(self.CORRUPT))

        return d, path

    def test_the_state_on_disk_is_a_dict_again_after_one_round(self):
        for host, drive, session in HOSTS:
            with self.subTest(host=host):
                d, path = self._corrupted_state(session)
                drive(d, session)
                seen = json.loads(path.read_text()).get("seen")
                self.assertIsInstance(
                    seen, dict,
                    f"{host} left the corruption on disk: every later round loses dedup, so "
                    f"every recalled memory is reinjected in full every turn")

    def test_dedup_actually_comes_back_on_the_round_after_the_corruption(self):
        """The consequence, not just the type: once healed, the ids injected in full are
        recorded, so the next round can tell them apart from fresh ones."""
        for host, drive, session in HOSTS:
            with self.subTest(host=host):
                d, path = self._corrupted_state(session)
                drive(d, session, rounds=2)
                state = json.loads(path.read_text())
                self.assertEqual(sorted(state["seen"]), ["hit-1", "hit-2"],
                                 f"{host} is not recording what it injected")


def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member."""
    import importlib.util
    path = REPO / "cli" / "qctx.py"
    spec = importlib.util.spec_from_file_location("qctx_cli_equiv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def cli_operations() -> set:
    """Every leaf subcommand the CLI offers, as `group_action` names.

    Derived by walking the real parser rather than restated, so an operation added to one
    host and not the other shows up here instead of being noticed in production. The
    dash-to-underscore mapping is the only convention: `qctx memory store-many` is the
    `memory_store_many` tool.
    """
    def leaves(parser, path=()):
        subs = getattr(parser, "_subparsers", None)
        if not subs:
            yield path

            return
        for action in subs._group_actions:
            for name, sub in getattr(action, "choices", {}).items():
                yield from leaves(sub, path + (name,))

    return {"_".join(path).replace("-", "_") for path in leaves(load_cli().build_parser())}


class TestBothHostsOfferTheSameOperations(unittest.TestCase):
    """The INVOCABLE surface, held to the CLI's.

    Recall and the checkpoint cadence are pushed at the model; these are what it can reach
    for. The claim is not that the two surfaces are equal — five operations are deliberately
    out of the model's reach — but that the difference is exactly those five, and that what
    is offered on both does the same thing. Name parity alone would not show the second
    half: see TestTheToolsDoWhatTheCLIDoes below.
    """

    #: Deliberately not tools. Configuration belongs to the operator: a `config set` tool
    #: would let the model point the archive somewhere else mid-conversation, and `setup` is
    #: interactive. The CLI keeps all of them.
    NOT_FOR_THE_MODEL = {"setup", "collections_list", "config_show", "config_set",
                         "config_detect"}

    def setUp(self):
        from hosts.hermes import tools
        self.tool_names = {s["name"] for s in tools.SCHEMAS}
        self.cli_names = cli_operations()

    def test_the_cli_walk_found_the_whole_surface(self):
        """A guard on the derivation itself: if the walk broke, every assertion below would
        pass over an empty set."""
        self.assertGreater(len(self.cli_names), 15)
        self.assertIn("memory_store", self.cli_names)
        self.assertIn("docs_drop", self.cli_names)

    def test_every_tool_is_an_operation_the_cli_also_offers(self):
        """A tool with no CLI equivalent is an operation only one host has — the divergence
        this file exists to prevent, in the other direction."""
        orphans = self.tool_names - self.cli_names
        self.assertEqual(orphans, set(),
                         f"only hermes can do these: {orphans}")

    def test_the_only_cli_operations_withheld_are_the_five_named_ones(self):
        withheld = self.cli_names - self.tool_names
        self.assertEqual(withheld, self.NOT_FOR_THE_MODEL,
                         "the model's surface drifted from the CLI's by something other "
                         "than the configuration commands")

    def test_no_configuration_operation_became_reachable(self):
        for name in self.NOT_FOR_THE_MODEL:
            self.assertNotIn(name, self.tool_names)


class TestTheToolsDoWhatTheCLIDoes(unittest.TestCase):
    """Semantic parity, because name parity is not it.

    A test comparing only the NAMES of things passes over real divergence in what they do —
    the lesson the tuning-knob tests above already encode. So each case here drives the CLI
    handler and the hermes tool over the SAME fake archive and compares what landed in it,
    or what came back, rather than comparing two restatements of the intent.
    """

    def setUp(self):
        from core.docs import DocIndex
        from core.memory import MemoryStore
        from tests.fakes import FakeEmbedder, FakeVectorStore
        self.cli = load_cli()
        self.q, emb = FakeVectorStore(), FakeEmbedder()
        self.q.ensure_collection("mem", emb.dim)
        self.store = MemoryStore(self.q, emb, None, "mem", emb.dim)
        #: What both hosts get when they build their memory access. A test that needs to
        #: observe HOW they call it (the policy case below) swaps this for a recorder — one
        #: object for both hosts, so neither can be observed through a different door.
        self.memory = self.store
        self.idx = DocIndex(self.q, emb, None, "tmp", "lib", emb.dim)
        # A Config that reaches nothing: both hosts get their archive injected below, but a
        # tool call with no configuration at all is refused by design, so the object has to
        # exist.
        self.cfg = core_module().Config(
            qdrant_url="http://localhost:1", qdrant_api_key="", api_base_url="",
            api_key="", embed_url="http://localhost:1/embeddings", rerank_url="",
            embed_model="m", rerank_model="", memory_collection="mem",
            docs_collection="tmp", library_collection="lib", vector_size=emb.dim)

    def _payload_of(self, mid) -> dict:
        p = dict(self.q.get_point("mem", mid)["payload"])
        # The timestamps and the id differ by construction — one call happened after the
        # other, and ids are uuid4. What must match is everything the CALLER decided.
        for volatile in ("created_at", "updated_at"):
            p.pop(volatile, None)

        return p

    def _through_the_cli(self, handler, **args):
        import io
        import unittest.mock
        from contextlib import redirect_stdout

        class Args:
            def __init__(self, **kw):
                self.json = True
                self.json_meta = None
                self.type = self.project = self.area = None
                self.text = self.id = None
                self.__dict__.update(kw)

        out = io.StringIO()
        with unittest.mock.patch.object(self.cli.core, "build_memory",
                                       lambda cfg, **kw: self.memory), \
             unittest.mock.patch.object(self.cli.core, "build_docs", lambda cfg: self.idx), \
             redirect_stdout(out):
            handler(Args(**args), self.cfg)

        return out.getvalue()

    def _through_the_tool(self, name, **args):
        import unittest.mock
        from hosts.hermes import tools
        with unittest.mock.patch.object(core_module(), "build_memory",
                                       lambda cfg, **kw: self.memory), \
             unittest.mock.patch.object(core_module(), "build_docs", lambda cfg: self.idx):
            return json.loads(tools.dispatch(name, args, cfg=self.cfg))

    def test_store_lands_the_same_record_from_both_hosts(self):
        """Same fact, same labels, one written through `qctx memory store` and one through
        the `memory_store` tool. The stored payload has to be indistinguishable."""
        self._through_the_cli(self.cli.cmd_memory_store,
                              text="a durable fact about pagination",
                              json_meta='{"custom": 1}', type="reference", project="p")
        cli_mid = next(iter(self.q.collections["mem"]["points"]))
        tool_mid = self._through_the_tool("memory_store",
                                         information="a durable fact about pagination",
                                         metadata={"custom": 1}, type="reference",
                                         project="p")["id"]
        self.assertNotEqual(cli_mid, tool_mid, "two writes, two records")
        self.assertEqual(self._payload_of(cli_mid), self._payload_of(tool_mid))

    def test_the_shortcut_precedence_is_the_same_on_both_hosts(self):
        """`--type` overriding a `type` already in the metadata object is a rule, and a rule
        settled twice can be settled differently."""
        self._through_the_cli(self.cli.cmd_memory_store, text="a fact",
                              json_meta='{"type": "old", "keep": true}', type="new")
        cli_mid = next(iter(self.q.collections["mem"]["points"]))
        tool_mid = self._through_the_tool("memory_store", information="a fact",
                                         metadata={"type": "old", "keep": True},
                                         type="new")["id"]
        self.assertEqual(self._payload_of(cli_mid)["metadata"],
                         {"type": "new", "keep": True})
        self.assertEqual(self._payload_of(cli_mid), self._payload_of(tool_mid))

    def test_a_text_only_update_keeps_the_labels_on_both_hosts(self):
        """The `{}` versus None trap: an assembled empty object REPLACES the metadata. Both
        hosts have to pass None, or one of them silently strips the labels."""
        for label, write in (("cli", lambda mid: self._through_the_cli(
                                 self.cli.cmd_memory_update, id=mid, text="corrected")),
                             ("tool", lambda mid: self._through_the_tool(
                                 "memory_update", id=mid, information="corrected"))):
            with self.subTest(host=label):
                mid = self.store.store("original", {"type": "reference",
                                                    "project": "p"})["id"]
                write(mid)
                self.assertEqual(self.q.get_point("mem", mid)["payload"]["metadata"],
                                 {"type": "reference", "project": "p"},
                                 f"{label} wiped the labels of a text-only update")

    def test_recall_reports_the_same_two_halves_on_both_hosts(self):
        self.store.store("a durable fact about pagination cursors")
        cli_json = json.loads(self._through_the_cli(
            self.cli.cmd_memory_recall, query="pagination cursors", limit=6,
            dense_floor=0.45, strict_floor=0.58, min_score=0.10, top_k=20))
        tool_json = self._through_the_tool("memory_recall", query="pagination cursors")
        # Both halves asserted, and asserted to be the RIGHT two: `{"info", "hits"}` used to
        # sit in assertEqual's third parameter, which is the message, so the keys themselves
        # were never checked and a rename in both hosts at once would have passed.
        self.assertEqual(set(cli_json), {"info", "hits"})
        self.assertEqual(set(tool_json), {"info", "hits"})
        self.assertEqual(set(cli_json["info"]), set(tool_json["info"]),
                         "the trail a consumer reads must have the same fields on both")
        self.assertEqual([h["document"] for h in cli_json["hits"]],
                         [h["document"] for h in tool_json["hits"]])
        self.assertEqual(set(cli_json["hits"][0]), set(tool_json["hits"][0]))

    def test_recall_runs_the_same_policy_on_both_hosts(self):
        """The CLI's argparse defaults and the hermes tool's floors are the same numbers,
        read from what each host ACTUALLY handed to `MemoryStore.recall` — not from a
        restatement of either. A host that relaxed a floor would return memories the other
        one refuses, which is the divergence that matters and is invisible in the names."""
        captured = {}

        class Recording:
            reranker = None

            def recall(self, queries, policy, top_k, suppressed=None):
                captured[len(captured)] = (policy, top_k)

                return [], Outcome(candidates=0)

        self.memory = Recording()
        self._through_the_cli(self.cli.cmd_memory_recall, query="a topic", limit=6,
                              dense_floor=0.45, strict_floor=0.58, min_score=0.10,
                              top_k=20)
        self._through_the_tool("memory_recall", query="a topic")
        self.assertEqual(len(captured), 2, "both hosts have to have reached the store")
        (cli_policy, cli_top_k), (tool_policy, tool_top_k) = captured[0], captured[1]
        self.assertEqual(cli_policy, tool_policy,
                         "the two hosts search long-term memory with different policies")
        self.assertEqual(cli_top_k, tool_top_k)

    def test_drop_takes_the_same_decision_on_both_hosts(self):
        """Both route through `DocIndex.drop_request`, so the three shapes and the refusal
        cannot be settled differently. Observed through the SHARED method being called with
        the same arguments, plus the archive state afterwards."""
        import unittest.mock
        path = os.path.join(tempfile.mkdtemp(), "doc.md")
        Path(path).write_text("# Title\n\nbody about pagination\n")
        calls = []
        real = self.idx.drop_request

        def recording(*a, **kw):
            calls.append((a, kw))

            return real(*a, **kw)

        with unittest.mock.patch.object(self.idx, "drop_request", recording):
            kept = self.idx.keep_file(path)
            self._through_the_cli(self.cli.cmd_docs_drop, doc_id=kept["doc_id"],
                                  scope="library", purge_tmp=False, expired=False)
            self.assertEqual(len(self.q.collections["lib"]["points"]), 0)
            self.idx.keep_file(path)
            self._through_the_tool("docs_drop", doc_id=kept["doc_id"], scope="library")
            self.assertEqual(len(self.q.collections["lib"]["points"]), 0)
        self.assertEqual(len(calls), 2, "one of the hosts did not route through the core")
        cli_call, tool_call = calls
        self.assertEqual((cli_call[0][0], cli_call[0][1]), (tool_call[0][0], tool_call[0][1]))
        self.assertEqual(cli_call[1], tool_call[1])


def core_module():
    """The `core` module object the tools module resolves — the same one, patched once."""
    import core

    return core




class TestBothHostsDegradeInTheSameWORDS(unittest.TestCase):
    """The suppression reason is not a log line — it reaches the model.

    `Outcome.suppressed` is what the caller uses to TELL the core that the second stage was
    turned off, and the degradation note renders it into the block the model reads. So the
    two hosts describing the same degradation differently is the "same name, different
    meaning" family again, one level down: identical behaviour, divergent explanation.

    Measured while writing this: the string could be changed in the hook alone — from the
    re-rank having FAILED to it merely having been skipped — with the whole suite green.
    That distinction is the one the breaker exists to communicate; a skipped re-rank sounds
    like a choice, a failed one is a degradation the model should weigh.
    """

    REASON = "circuit breaker: the re-rank failed "

    def test_the_breaker_reason_is_identical_in_both_adapters(self):
        hook = (REPO / "hooks" / "recall.py").read_text()
        host = hermes_adapter_source()
        self.assertIn(self.REASON, hook, "the hook stopped naming the breaker as a FAILURE")
        self.assertIn(self.REASON, host, "the hermes adapter no longer matches the hook")

    def test_neither_host_calls_it_merely_skipped(self):
        """A degradation described as a choice invites the model to discount it."""
        for label, text in (("hook", (REPO / "hooks" / "recall.py").read_text()),
                            ("hermes", hermes_adapter_source())):
            self.assertNotIn("the re-rank was skipped", text,
                             f"{label}: 'skipped' hides that the re-rank FAILED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
