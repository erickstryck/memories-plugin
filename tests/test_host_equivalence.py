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
                docs_collection="", library_collection="", repos_collection="repos",
                repos_registry_collection="reg", vector_size=8,
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
            docs_collection="tmp", library_collection="lib", repos_collection="repos",
            repos_registry_collection="reg", vector_size=emb.dim)

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

    #: Text that prints as nothing. The first three were already refused everywhere; the rest
    #: passed `str.strip()` and replaced the document.
    INVISIBLE = ("", "   ", "\xa0", "​", "⁠", "﻿")

    def test_neither_host_lets_an_INVISIBLE_replacement_destroy_a_fact(self):
        """One operation, two answers — the divergence this file exists to catch.

        Measured before the fix: `memory_update(information="")` through the tool returned
        `{"status": "updated", "reembedded": false}` with the text untouched (because `_text`
        mapped `""` to None, i.e. "not sent"), while `qctx memory update --text ""` raised.
        And `information="​"` — one ZERO WIDTH SPACE — was accepted by BOTH: it survives
        `str.strip()`, so it replaced the document and reported success on either surface,
        leaving a record with no readable text that `_flatten_hit` then drops from every
        recall. A fact destroyed, invisibly, by a call that answered "updated".

        Both halves are asserted for both hosts: the call must fail, and the stored fact must
        still be there afterwards.
        """
        for text in self.INVISIBLE:
            for host in ("cli", "tool"):
                with self.subTest(text=repr(text), host=host):
                    mid = self.store.store("the original fact",
                                           {"type": "reference"})["id"]
                    if host == "cli":
                        with self.assertRaises(core_module().CoreError):
                            self._through_the_cli(self.cli.cmd_memory_update,
                                                  id=mid, text=text)
                    else:
                        answer = self._through_the_tool("memory_update", id=mid,
                                                        information=text)
                        self.assertIn("error", answer,
                                      "the tool reported a destructive no-op as success")
                    self.assertEqual(self.q.get_point("mem", mid)["payload"]["document"],
                                     "the original fact", "the fact was destroyed")

    def test_neither_host_clears_the_labels_with_an_empty_object(self):
        """`metadata={}` and `--json-meta '{}'` both read as "no labels sent" — the labels are
        left untouched and the call reports "updated". Held for both hosts because it is the
        documented contract now (see the memory_update schema): a divergence here would make
        one host's documentation false rather than merely surprising.
        """
        for label, write in (("cli", lambda mid: self._through_the_cli(
                                 self.cli.cmd_memory_update, id=mid, json_meta="{}")),
                             ("tool", lambda mid: self._through_the_tool(
                                 "memory_update", id=mid, metadata={}))):
            with self.subTest(host=label):
                mid = self.store.store("a fact", {"type": "reference", "project": "p"})["id"]
                write(mid)
                self.assertEqual(self.q.get_point("mem", mid)["payload"]["metadata"],
                                 {"type": "reference", "project": "p"},
                                 f"{label} cleared the labels with an empty object")

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


# --- The two file-read guards, and where their differences are allowed to live -----------
# `.superpowers/sdd/` is gitignored, so the divergence table that lived in a task report was
# invisible to the next clone. These tests move the burden of proof: the table is in the
# SPEC, and every difference the two adapters actually carry in their source must be NAMED
# there. Derived from the sources, never restated here — a host-only constant added tomorrow
# fails this instead of waiting for somebody to remember the table exists.
import ast   # noqa: E402
import re    # noqa: E402

SPEC = REPO / "docs" / "superpowers" / "specs" / "2026-08-15-big-file-read-guard-design.md"
CLAUDE_GUARD = REPO / "hooks" / "bigfile.py"
HERMES_GUARD = REPO / "hosts" / "hermes" / "bigfile.py"

#: The heading the derivations below read. Named, not guessed: if it is renamed, the guard
#: on the guard below fails loudly instead of every check silently passing over "".
DIVERGENCE_HEADING = "### Divergências declaradas entre os dois adaptadores"


def divergence_section() -> str:
    """The spec's declared-divergence section, heading to the next heading of any level."""
    text = SPEC.read_text()
    start = text.find(DIVERGENCE_HEADING)
    if start < 0:
        return ""
    rest = text[start + len(DIVERGENCE_HEADING):]
    end = re.search(r"^#{1,3} ", rest, re.M)

    return rest[:end.start()] if end else rest


def module_constants(path: Path) -> set:
    """Module-level UPPER_CASE assignments — one adapter's host-forced settings.

    Module level and not every assignment: what a host FORCES on the adapter is decided
    once, at import, and that is exactly the set that must be explainable by the host.
    """
    return {t.id for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name) and t.id.isupper() and not t.id.startswith("_")}


def env_names(path: Path) -> set:
    """Environment variables one adapter READS — from the AST, not from a text scan.

    A text scan reports the names quoted in prose too, and both files explain each other's
    knobs in their docstrings; that would make every divergence look shared.
    """
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        reads_env = (isinstance(f, ast.Name) and f.id in ("env", "env_num")) or (
            isinstance(f, ast.Attribute) and f.attr == "get"
            and isinstance(f.value, ast.Attribute) and f.value.attr == "environ")
        if not reads_env:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and re.fullmatch(r"[A-Z][A-Z0-9_]*", arg.value):
                found.add(arg.value)

    return found


def tool_input_keys(path: Path) -> set:
    """The keys one adapter pulls out of the host's `tool_input`."""
    return set(re.findall(r'tool_input\.get\("([a-z_]+)"\)', path.read_text()))


class TestEveryDivergenceOfTheGuardsIsDeclaredInTheSPEC(unittest.TestCase):
    """The two file-read guards differ, and every difference is FORCED by its host.

    That is a claim, and until now it lived in a task report under `.superpowers/sdd/`,
    which is gitignored — so the one artefact a reader would compare the hosts from does
    not survive a clone. The table moved to the spec, and these tests are what stop it
    from drifting: the DIFFERENCES ARE DERIVED FROM THE TWO SOURCES, and each one must be
    named in the spec's section. Add a ceiling to one adapter and this goes red naming it.

    What this does NOT prove: that a row in the table is still true. A divergence removed
    in code leaves a stale row, and no derivation from source can see prose that describes
    something absent. The direction that matters is covered — silence is what costs the
    next reader, not a leftover line.
    """

    def test_the_spec_has_the_section_at_all(self):
        """The guard on the guard: every check below reads this section, and a section
        that cannot be found is an empty string every `assertIn` passes over."""
        section = divergence_section()
        self.assertTrue(section.strip(), f"{DIVERGENCE_HEADING} is not in {SPEC.name}")
        for name in ("hooks/bigfile.py", "hosts/hermes/bigfile.py"):
            self.assertIn(name, section, "the section does not say which files it compares")

    def test_the_derivations_see_both_adapters(self):
        """The second guard on the guard: a derivation that came back empty would make
        every difference below the empty set, and prove nothing at all."""
        for label, fn in (("constants", module_constants), ("env names", env_names),
                          ("tool_input keys", tool_input_keys)):
            for path in (CLAUDE_GUARD, HERMES_GUARD):
                with self.subTest(derivation=label, source=path.name):
                    self.assertTrue(fn(path), f"the {label} scan went blind on {path}")

    def test_every_host_only_constant_is_explained_in_the_spec(self):
        """`TAIL_BYTES` on one side and `READ_CHAR_CEILING`, `SQLITE_TIMEOUT_S`,
        `BLOCK_EXIT_CODE`, `REPO_ROOT` on the other are not preferences — each exists
        because its host leaves no other way. The spec is where that is said."""
        section = divergence_section()
        claude, hermes = module_constants(CLAUDE_GUARD), module_constants(HERMES_GUARD)
        only = (claude - hermes) | (hermes - claude)
        self.assertTrue(only, "the two adapters suddenly share every constant — check the "
                              "derivation before believing it")
        for name in sorted(only):
            with self.subTest(constant=name):
                self.assertIn(name, section,
                              f"{name} exists in only one guard and the spec never says "
                              f"why the other does not need it")

    def test_every_host_only_environment_variable_is_explained_in_the_spec(self):
        """The knobs must be the SAME on both hosts — that is what a deployer is promised.
        One that is not shared is a promise with an exception, and an undocumented
        exception is the deployer discovering it by having it not work."""
        section = divergence_section()
        claude, hermes = env_names(CLAUDE_GUARD), env_names(HERMES_GUARD)
        self.assertEqual(claude - hermes, set(),
                         "claude-code reads an environment variable hermes does not; if "
                         "that is deliberate it belongs in the table, not in a surprise")
        for name in sorted(hermes - claude):
            with self.subTest(env=name):
                self.assertIn(name, section, f"{name} is read on one host only")

    def test_the_payload_key_the_two_hosts_disagree_on_is_declared(self):
        """The same read arrives as `file_path` on one host and `path` on the other, and a
        guard that reads the wrong key sees no path and allows EVERYTHING — silently."""
        section = divergence_section()
        claude, hermes = tool_input_keys(CLAUDE_GUARD), tool_input_keys(HERMES_GUARD)
        differing = (claude - hermes) | (hermes - claude)
        self.assertTrue(differing, "both guards now read the same tool_input keys — either "
                                   "a host changed or the derivation broke")
        for key in sorted(differing):
            with self.subTest(key=key):
                self.assertIn(f"tool_input.{key}", section,
                              f"the guards disagree on tool_input.{key} and the spec is "
                              f"silent about it")


# --- The equivalence itself, and the one layer at which it is true ------------------------
# The plan for this task asked for "same file and same Budget -> same Verdict on both hosts".
# That is FALSE today, and not by defect: each host now passes its OWN read ceiling into
# `decide`, and hermes' extra byte ceiling makes it price a file of long lines LOWER than
# claude-code does. So the claim is asserted where it holds — same Budget AND same cost ->
# same verdict — and the ceiling difference is tested as the declared divergence it is.
from tests import test_bigfile_claude as claude_fixtures   # noqa: E402
from tests import test_bigfile_hermes as hermes_fixtures   # noqa: E402
from core.bigfile import Budget, decide                    # noqa: E402

#: 171k tokens at the core's 4 chars/token, and no newline anywhere — the file the spec's own
#: worked example is about.
A_BIG_FILE = 4 * 171_000
#: Small enough that BOTH hosts' ceilings fall back to the file itself, which is the regime
#: where the two hosts price identically and the equivalence claim is about something.
A_FILE_UNDER_EVERY_CEILING = 40_000
#: Two ORDINARY text files, and the point is that they are ordinary. Under a 2,000-line
#: ceiling one read pulls 819,200 and 202,271 bytes; under hermes' 100,000-CHARACTER ceiling
#: it pulls 100,000 in both cases. Neither has long lines — 400 and 100 bytes — because the
#: divergence is not about long lines: it is about any read that would pull more than 100 KB,
#: which is 2,000 lines averaging over ~50 bytes, or any big file with few newlines.
A_FILE_ONE_READ_CANNOT_SWALLOW = (4_000, 400)
AN_EVERYDAY_BIG_FILE = (4_000, 100)


def a_flat_file(chars: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as fh:
        fh.write("x" * chars)

    return path


def ceilings_claude_code_applies() -> dict:
    """The read ceilings `hooks/bigfile.py` hands to `decide`, taken from the ADAPTER.

    Captured by running the guard rather than restated here, and that is the whole value of
    it: a test that wrote `read_bytes=None` itself would keep passing with the two hosts'
    ceilings swapped, and would be proving its own literal.
    """
    guard = claude_fixtures.adapter
    seen = []
    real = guard.bigfile.decide

    def recording(path, budget, **kw):
        seen.append(kw)

        return real(path, budget, **kw)

    payload = claude_fixtures.a_read_payload(
        a_flat_file(400), claude_fixtures.a_transcript([claude_fixtures.ASSISTANT]))
    with unittest.mock.patch.object(guard.bigfile, "decide", recording):
        claude_fixtures.run_main(payload, claude_fixtures.Spy(set()))

    return {k: seen[0].get(k) for k in ("read_lines", "read_bytes")}


def ceilings_hermes_applies() -> dict:
    """The same, for `hosts/hermes/bigfile.py`, driven through ITS host's payload and db."""
    guard = hermes_fixtures.adapter
    seen = []
    real = guard.bigfile.decide

    def recording(path, budget, **kw):
        seen.append(kw)

        return real(path, budget, **kw)

    db = hermes_fixtures.a_session_using(400)
    payload = hermes_fixtures.a_read_payload(a_flat_file(400))
    with unittest.mock.patch.object(guard, "state_db_path", lambda: db), \
         unittest.mock.patch.object(guard.bigfile, "decide", recording):
        hermes_fixtures.run_main(payload, hermes_fixtures.Spy(set()))

    return {k: seen[0].get(k) for k in ("read_lines", "read_bytes")}


class TestBothHostsGuardTheSameWay(unittest.TestCase):
    """The same Budget and the same COST must produce the same Verdict on both hosts.

    The hosts MEASURE differently — claude-code exactly, hermes by estimate — and that
    asymmetry is deliberate and documented. What must not differ is the DECISION, and the
    only thing allowed to differ is the wording, which must say the number is a guess.

    Read the qualifier: same Budget AND same cost. The two hosts do not always reach the
    same cost, because each applies its own host's read ceiling — that is the class below,
    and it is a declared divergence, not a break in this one.
    """

    def test_identical_budgets_give_identical_verdicts(self):
        path = a_flat_file(A_BIG_FILE)
        for label, ceilings in (("no ceiling", {}),
                                ("claude-code's", ceilings_claude_code_applies()),
                                ("hermes'", ceilings_hermes_applies())):
            for used, window in ((604_023, 1_000_000), (10_000, 1_000_000),
                                 (190_000, 200_000)):
                exact = decide(path, Budget(window=window, used=used, exact=True), **ceilings)
                approx = decide(path, Budget(window=window, used=used, exact=False),
                                **ceilings)
                with self.subTest(ceilings=label, used=used):
                    self.assertEqual(exact.block, approx.block,
                                     "the decision must not depend on how the number was "
                                     "obtained")
                    self.assertEqual(exact.cost, approx.cost)

    def test_only_the_wording_marks_the_estimate(self):
        """A number that looks precise and is a guess is worse than a guess that admits it."""
        path = a_flat_file(A_BIG_FILE)
        exact = decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        approx = decide(path, Budget(window=1_000_000, used=604_023, exact=False))
        self.assertTrue(exact.block and approx.block, "the fixture stopped blocking")
        self.assertNotIn("≈", exact.reason)
        self.assertIn("≈", approx.reason)
        self.assertNotEqual(exact.reason, approx.reason,
                            "the estimate is being presented as a measurement")

    def test_where_the_ceilings_agree_the_two_hosts_agree_on_everything_but_wording(self):
        """The claim at the host layer, with each adapter's REAL ceilings: on a file below
        the tightest of them both hosts reach the same cost, and there the verdicts must be
        identical — block, allow and the number quoted."""
        path = a_flat_file(A_FILE_UNDER_EVERY_CEILING)
        claude, hermes = ceilings_claude_code_applies(), ceilings_hermes_applies()
        for used, window, expected in ((190_000, 200_000, True), (10_000, 1_000_000, False)):
            c = decide(path, Budget(window=window, used=used, exact=True), **claude)
            h = decide(path, Budget(window=window, used=used, exact=False), **hermes)
            with self.subTest(used=used):
                self.assertEqual(c.cost, h.cost, "the precondition of the claim is gone: "
                                                 "the two hosts priced the same file apart")
                self.assertEqual(c.block, h.block)
                self.assertEqual(c.block, expected, "the fixture stopped exercising what "
                                                    "it was chosen for")


class TestEachHostAppliesItsOwnReadCeiling(unittest.TestCase):
    """The declared divergence, and the reason the equivalence above carries a qualifier.

    One `Read` on claude-code stops at 2,000 lines. One `read_file` on hermes stops at 2,000
    lines OR ~100,000 characters, whichever comes first (`tools/file_tools.py:65`), and the
    second ceiling is the one that bites.

    MEASURED, because the size of this is easy to understate: the two hosts price a file
    identically only while one read stays under 100,000 characters — a file below ~100 KB, or
    lines averaging under ~50 bytes. Past that, hermes is pinned at 25,000 tokens and
    claude-code is not: 4,000 lines of 100 bytes is 202,271 against 100,000, and 4,000 of 400
    is 819,200 against 100,000. So this is not a corner case about exotic files; on nearly
    every file big enough to concern the guard at all, the two hosts may legitimately take
    different decisions.

    Aligning the ceilings would make a "same file, same verdict" test pass, and it would be
    a lie about the host: the guard would be pricing a read that neither host performs.
    """

    def test_each_adapter_hands_decide_its_own_hosts_ceilings(self):
        claude, hermes = ceilings_claude_code_applies(), ceilings_hermes_applies()
        self.assertEqual(claude["read_lines"], hermes["read_lines"],
                         "the LINE ceiling is 2,000 on both hosts and is not a divergence")
        self.assertIsNone(claude["read_bytes"],
                          "claude-code has no readable byte ceiling; passing one prices a "
                          "truncation the host does not perform")
        self.assertEqual(hermes["read_bytes"], hermes_fixtures.adapter.READ_CHAR_CEILING,
                         "hermes truncates by characters and the price must know it")

    #: (shape, window, used). Each budget is chosen so the SIZE-ONLY first pass already
    #: blocks, and that is not tuning: `decide` refines the price only on the rare path it
    #: reaches after deciding to block, so a budget the file sails through never computes
    #: the number this test is about — both hosts would report the size-only price and the
    #: divergence would be invisible.
    STRADDLING = ((A_FILE_ONE_READ_CANNOT_SWALLOW, 1_000_000, 604_023),
                  (AN_EVERYDAY_BIG_FILE, 200_000, 100_000))

    def test_past_the_character_ceiling_the_two_hosts_price_and_decide_apart(self):
        """Both fixtures are ordinary text — 400-byte lines and 100-byte lines — and that
        IS the assertion: same file, same Budget, one host blocks and the other allows, on
        files nobody would call exotic. Declared, and tested so nobody 'fixes' it quietly
        by aligning the two ceilings, which would price a read neither host performs."""
        claude, hermes = ceilings_claude_code_applies(), ceilings_hermes_applies()
        for shape, window, used in self.STRADDLING:
            path = hermes_fixtures.a_file_of_lines(*shape)
            c = decide(path, Budget(window=window, used=used, exact=True), **claude)
            h = decide(path, Budget(window=window, used=used, exact=False), **hermes)
            with self.subTest(lines=shape[0], width=shape[1]):
                self.assertGreater(c.cost, h.cost,
                                   "hermes' character ceiling stopped biting; if the host "
                                   "changed, the spec's divergence table changes with it")
                self.assertTrue(c.block,
                                "the control: without a byte ceiling this read is too big")
                self.assertFalse(h.block, "hermes loads a fraction of it and must not block")


README = REPO / "README.md"
SKILL = REPO / "skills" / "doc-index" / "SKILL.md"
HOOKS_JSON = REPO / "hooks" / "hooks.json"
CUTOVER = REPO / "scripts" / "hermes_cutover.sh"


class TestBothHostsActuallyREGISTERTheGuard(unittest.TestCase):
    """A guard nothing calls is a guard that does not exist, and the two hosts are wired in
    completely different places: a manifest this repo ships, and a line the cutover script
    writes into the user's `config.yaml`.

    The hermes half has its own suite (`tests/test_hermes_cutover.py`) because a script that
    edits a live file has to be driven to be believed. This is the half that had nothing:
    deleting the `PreToolUse` block from `hooks/hooks.json` left every test green, including
    the one that reasons about "the timeout hooks.json gives it".
    """

    def entry(self):
        manifest = json.loads(HOOKS_JSON.read_text())
        for block in manifest.get("hooks", {}).get("PreToolUse", []):
            for hook in block.get("hooks", []):
                if "bigfile.py" in hook.get("command", ""):
                    return block, hook

        return None, None

    def test_claude_code_calls_the_guard_before_a_read(self):
        block, hook = self.entry()
        self.assertIsNotNone(hook, "hooks.json registers no PreToolUse hook for bigfile.py")
        self.assertEqual(block.get("matcher"), "Read",
                         "a wider matcher hands the guard tools it must not block")
        self.assertIn("CLAUDE_PLUGIN_ROOT", hook["command"],
                      "the path has to come from the host, not from whoever wrote it")

    def test_the_timeout_is_the_one_the_rest_of_the_suite_reasons_about(self):
        """`tests/test_bigfile_claude.py` measures the deadline against "the 5s hooks.json
        gives it". That number was in prose on both sides and asserted on neither."""
        _, hook = self.entry()
        self.assertEqual(hook["timeout"], 5)

    def test_both_hosts_get_the_same_budget_and_the_same_tool(self):
        """Same guard, same 5 seconds, and each host's own name for the read tool — `Read`
        against `read_file`. The hermes side is written by the cutover script, so it is read
        out of the script rather than out of a file only an install produces."""
        _, hook = self.entry()
        script = CUTOVER.read_text()
        self.assertIn("matcher: read_file", script)
        self.assertIn(f"timeout: {hook['timeout']}", script)
        self.assertIn("hosts/hermes/bigfile.py", script)

#: Named rather than guessed, for the reason `DIVERGENCE_HEADING` is: a renamed heading has
#: to fail loudly here, not turn every check below into an `assertIn` over "".
GUARD_HEADING = "### The big-file read guard"


def guard_section() -> str:
    """The README's guard section, heading to the next section of the level above."""
    text = README.read_text()
    start = text.find(GUARD_HEADING)
    if start < 0:
        return ""
    rest = text[start + len(GUARD_HEADING):]
    end = re.search(r"^## ", rest, re.M)

    return rest[:end.start()] if end else rest


class TestTheREADMEDescribesTheGuardThatSHIPPED(unittest.TestCase):
    """Documentation is a claim like any other, and this repo has already paid for one: the
    README said "both spellings" of a key the core accepted under three.

    Every number below is DERIVED — from the core's constants, and from the ceilings each
    adapter actually hands to `decide` — so a threshold retuned in code and left stale in
    prose fails here. What it deliberately does not police is the prose itself: a sentence
    can go out of date without a number moving, and no test can see that.
    """

    def test_the_section_is_there_at_all(self):
        self.assertTrue(guard_section().strip(), f"{GUARD_HEADING} is not in README.md")

    def test_it_documents_the_two_thresholds_the_core_decides_with(self):
        from core import bigfile

        section = guard_section()
        for value in (bigfile.FLOOR_PCT, bigfile.SHARE_PCT):
            with self.subTest(default=value):
                self.assertIn(f"`{value:.2f}`", section, "the default is not the one shipped")
                self.assertIn(f"{int(value * 100)}%", section,
                              "the percentage the prose promises is not the one in the code")

    def test_it_names_the_knobs_both_adapters_actually_read(self):
        """Derived from the two adapters, so a knob renamed on one host cannot be documented
        by the other's name — the divergence this repo has caught three times."""
        knobs = {name for name in env_names(CLAUDE_GUARD) & env_names(HERMES_GUARD)
                 if "BIGFILE" in name}
        self.assertTrue(knobs, "the knob derivation went blind; nothing below proves anything")
        section = guard_section()
        for knob in knobs:
            with self.subTest(knob=knob):
                self.assertIn(knob, section)

    def test_the_escape_marker_is_documented_on_both_surfaces(self):
        """The README is for the user, who is the only one who can type it, and the SKILL is
        for the model, which must not believe it can.

        The marker is DERIVED from the two adapters — the only place a live one comes from
        now that it is configurable — and not from a constant in the core, which no longer
        has one: a default kept there would be a second owner of the marker, free to teach
        what the detection rejects. The two adapters must also agree, or the documentation
        would be true on one host only.
        """
        from tests.test_hermes_provider import all_knobs   # where the knob scan lives

        markers = {default for rel in ("hooks/bigfile.py", "hosts/hermes/bigfile.py")
                   for attr, _, _, default in all_knobs(rel) if attr == "ESCAPE_MARKER"}
        self.assertEqual(len(markers), 1, f"the hosts default to different markers: {markers}")
        marker = markers.pop()
        self.assertIn(marker, guard_section())
        self.assertIn(marker, SKILL.read_text())

    def test_it_states_the_per_read_ceilings_the_adapters_apply(self):
        """The "it prices the READ, not the file" claim is only useful with its numbers, and
        those numbers come from the adapters, not from this test."""
        claude, hermes = ceilings_claude_code_applies(), ceilings_hermes_applies()
        section = guard_section()
        self.assertIn(f"{claude['read_lines']:,} lines", section)
        self.assertIn(f"{hermes['read_bytes']:,} characters", section)

    def test_the_divergence_it_quotes_is_the_one_the_two_hosts_produce(self):
        """The pair of numbers that makes "the hosts can decide differently" concrete. Both
        are computed here from the same everyday file the equivalence tests use, so a host
        ceiling that moves takes the README with it instead of leaving a false claim."""
        path = hermes_fixtures.a_file_of_lines(*AN_EVERYDAY_BIG_FILE)
        budget = Budget(window=200_000, used=100_000, exact=True)
        claude = decide(path, budget, **ceilings_claude_code_applies())
        hermes = decide(path, budget, **ceilings_hermes_applies())
        section = guard_section()
        self.assertNotEqual(claude.cost, hermes.cost, "the divergence itself is gone")
        for cost in (claude.cost, hermes.cost):
            with self.subTest(cost=cost):
                self.assertIn(f"{cost:,}", section)

    def test_it_points_at_the_spec_instead_of_copying_the_divergence_table(self):
        """A second copy of that table is a second thing to keep true, and the spec's copy is
        the one with a test on it."""
        section = guard_section()
        self.assertIn(SPEC.name, section)
        self.assertNotIn(DIVERGENCE_HEADING, README.read_text())

    def test_it_says_how_to_declare_the_window_the_hosts_cannot_read(self):
        """The accepted cost of the ceiling table is a guard that sleeps until this is set.
        Accepted only because it is VISIBLE — which is this paragraph and the cutover
        script's report."""
        section = guard_section()
        self.assertIn("context_window", section)
        self.assertIn("QCTX_CONTEXT_WINDOW", section)


#: Every file the guard is made of, on both hosts. The property below is about the guard as
#: a whole, so the list is the whole guard: a call added to any one of them is a call the
#: model never asked for.
GUARD_MODULES = (REPO / "core" / "bigfile.py", REPO / "core" / "inventory.py",
                 CLAUDE_GUARD, HERMES_GUARD)

#: The archive operations the guard is ALLOWED to name. Everything else a store exposes is
#: presumed to act, which is the direction that ages well: a writing method added tomorrow is
#: covered without anyone remembering this list, while a reading one added tomorrow costs one
#: deliberate line here — visible in the diff, which is exactly where it belongs.
#:
#: `list_docs` is on this list because the guard genuinely needs it, and it is NOT free of
#: side effects: it calls `ensure` on both scopes and `sweep` on the temporary one. What that
#: costs the backend is not argued here, it is MEASURED — see
#: `TestABlockedReadIndexesNothing` in each adapter's test file, which records every call the
#: blocked path makes against a fake store.
READING_METHODS = {"search", "list_docs", "find", "recall", "get", "list_page", "count"}


def archive_write_methods() -> set:
    """Everything `DocIndex` and `MemoryStore` can be asked to DO, minus the reads.

    Derived from the classes rather than frozen as a list of names, because a frozen list
    ages: `index_file` and `keep_file` were the two anybody would think to write down, and
    the method that gets added next is precisely the one nobody would.
    """
    from core.docs import DocIndex
    from core.memory import MemoryStore

    return {name for cls in (DocIndex, MemoryStore) for name in dir(cls)
            if not name.startswith("_") and callable(getattr(cls, name))
            and name not in READING_METHODS}


def called_names(path: Path) -> set:
    """Every name this file CALLS — `x.foo()` and `foo()` alike, from the AST.

    From the AST and not a text scan for the reason `env_names` gives: these files explain
    in prose what they must not do, and a text scan would report the explanation."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            found.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            found.add(node.func.id)

    return found


class TestTheGuardDecidesAndDoesNotACT(unittest.TestCase):
    """"The hook decides, it does not act. It indexes nothing." — the spec's Failure modes.

    The requirement was met and held by NOTHING, which is the shape this whole review was
    looking for. It is also the requirement a helpful refactor breaks in one line: blocked
    the read, so index it for the user — and a guard that runs before every file read starts
    firing hundreds of embedding chunks nobody asked for, on a file the user may have been
    about to abandon.

    Two tests, because neither mechanism covers the other. This one is STRUCTURAL and reads
    the sources: the guard may not so much as name a writing operation. The execution half
    lives in each adapter's own file, on the BLOCK path, which is where the temptation to
    index would sit.
    """

    def test_the_derivation_sees_the_operations_that_would_matter(self):
        """The guard on the guard: an empty write set would make every assertion below an
        intersection with nothing, passing for a guard that indexed on every read."""
        writes = archive_write_methods()
        self.assertTrue(writes)
        for named in ("index_file", "keep_file", "store", "delete", "drop"):
            with self.subTest(method=named):
                self.assertIn(named, writes)
        for read in ("list_docs", "search"):
            with self.subTest(method=read):
                self.assertNotIn(read, writes, "a read the guard needs is being forbidden")

    def test_no_file_of_the_guard_calls_anything_that_writes(self):
        writes = archive_write_methods()
        for path in GUARD_MODULES:
            with self.subTest(module=path.name):
                self.assertEqual(sorted(called_names(path) & writes), [],
                                 f"{path} acts on the archive; the guard decides and the "
                                 f"model acts, which is what keeps a blocked read from "
                                 f"costing an indexing run nobody asked for")


if __name__ == "__main__":
    unittest.main(verbosity=2)
