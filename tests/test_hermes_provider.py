"""The hermes-agent host adapter.

It is verified WITHOUT hermes importable, because this repo's suite has to run offline and
hermes lives in its own venv. What the adapter promises is therefore checked structurally:
the method set hermes actually calls, measured from the install rather than assumed from the
published source — the two differ, and the installed one is what runs.

That measurement is taken at RUN time, not written down here, and the plan proved why: the
install moved from v0.20.0 to v0.20.1 mid-task, taking the ABC from 19 public members to 21
— it gained `unavailable_reason` and `recall_status`, the two this adapter had already
implemented while nothing called them. The test below read the new surface and still passed,
because it asks the installed ABC instead of comparing against a list someone typed.

(The five optional hooks further down the adapter — `on_session_switch`, `on_pre_compress`,
`on_delegation`, `on_memory_write`, `backup_paths` — are NOT new in v0.20.1; they were in the
pre-upgrade ABC too, which is what the comment at hosts/hermes/__init__.py:505 says.)
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import core
from hosts.hermes import MemoriesProvider, register

#: Where hermes is installed on this machine. The ABC IS importable from there (measured):
#: `RecallStatus` was absent in v0.20.0 and is present in v0.20.1. Used by the tests below,
#: skipped elsewhere so the suite stays portable.
HERMES_INSTALL = Path.home() / ".hermes" / "hermes-agent"

#: What the adapter implements EXPLICITLY — no reliance on the ABC's defaults, because the
#: suite runs with `_Base = object` and because a default silently absorbs a method hermes
#: renames. Everything here must exist whether or not hermes is importable.
IMPLEMENTED = {
    "name", "is_available", "unavailable_reason", "initialize", "shutdown",
    "prefetch", "queue_prefetch", "recall_status", "system_prompt_block",
    "on_turn_start", "on_session_end", "sync_turn",
    "get_tool_schemas", "handle_tool_call", "get_config_schema", "save_config",
}
HERMES_ABSTRACTS = {"get_tool_schemas", "initialize", "is_available", "name"}


class TestTheContract(unittest.TestCase):
    def test_it_implements_every_abstract(self):
        p = MemoriesProvider()
        for method in HERMES_ABSTRACTS:
            self.assertTrue(hasattr(p, method), f"missing abstract {method}")

    def test_it_defines_its_surface_without_leaning_on_inherited_defaults(self):
        """Explicit, not inherited. Two reasons: this suite runs without hermes, so the
        defaults are not there to lean on; and a default silently absorbs a method a hermes
        upgrade renames, turning a rename into a feature that quietly stops working."""
        p = MemoriesProvider()
        missing = {m for m in IMPLEMENTED if m not in vars(type(p))
                   and not any(m in vars(k) for k in type(p).__mro__[:-1])}
        self.assertEqual(missing, set(), f"declared as implemented but absent: {missing}")

    @unittest.skipUnless(HERMES_INSTALL.exists(), "hermes-agent not installed here")
    def test_it_answers_every_method_the_REAL_installed_abc_declares(self):
        """The honest version of the contract check: read the surface off the install
        instead of trusting a list I typed. Skipped where hermes is absent, so a machine
        without it still runs the rest."""
        import subprocess
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from agent.memory_provider import MemoryProvider as M\n"
            "print(json.dumps([m for m in dir(M) if not m.startswith('_')]))\n"
        ) % str(HERMES_INSTALL)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        surface = set(json.loads(out.stdout))
        p = MemoriesProvider()
        unanswered = {m for m in surface if not hasattr(p, m)}
        self.assertEqual(unanswered, set(),
                         f"hermes calls these and the provider has no answer: {unanswered}")

    def test_the_name_is_the_install_directory_name(self):
        self.assertEqual(MemoriesProvider().name, "memories")

    def test_register_hands_the_provider_to_the_collector(self):
        class Collector:
            provider = None

            def register_memory_provider(self, provider):
                self.provider = provider

        c = Collector()
        register(c)
        self.assertIsInstance(c.provider, MemoriesProvider)


class TestAvailability(unittest.TestCase):
    """is_available gates initialization, so a provider that reports unavailable is never
    initialized — any diagnostic it would log from initialize() is unreachable. The reason
    has to be actionable and it has to come from here."""

    def _clean_env(self):
        keep = {"PATH", "HOME", "LANG", "PYTHONPATH"}

        return {k: v for k, v in os.environ.items() if k in keep}

    def test_unconfigured_is_unavailable_with_an_actionable_reason(self):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "p = MemoriesProvider()\n"
            "print(int(p.is_available())); print(p.unavailable_reason())\n"
        ) % str(REPO)
        env = self._clean_env()
        env["QCTX_CONFIG"] = "/nonexistent/config.json"
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        available, reason = out.stdout.strip().splitlines()[:2]
        self.assertEqual(available, "0")
        self.assertIn("QCTX_", reason, "the reason must say WHICH variable to set")

    def test_is_available_makes_no_network_call(self):
        """Contract: 'Should not make network calls — just check config and installed deps.'
        Pointed at an unroutable address it must still answer fast."""
        import time
        env = dict(os.environ, QCTX_QDRANT_URL="http://10.255.255.1:9/qdrant")
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "MemoriesProvider().is_available()\n"
        ) % str(REPO)
        t0 = time.monotonic()
        subprocess.run([sys.executable, "-c", script], capture_output=True, env=env)
        self.assertLess(time.monotonic() - t0, 3.0, "is_available reached the network")


class TestSymlinkInstall(unittest.TestCase):
    def test_it_finds_the_repo_when_installed_through_a_symlink(self):
        """Measured trap: through a symlink, abspath(__file__) resolves to the SYMLINK's
        directory ($HERMES_HOME/plugins), not the repo. The path pattern hooks/recall.py
        uses would fail here; only realpath finds core/."""
        home = Path(tempfile.mkdtemp()) / "plugins"
        home.mkdir(parents=True)
        link = home / "memories"
        link.symlink_to(REPO / "hosts" / "hermes")

        script = (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('u.memories', %r,\n"
            "    submodule_search_locations=[%r])\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(m.REPO_ROOT)\n"
            "print(m.MemoriesProvider().name)\n"
        ) % (str(link / "__init__.py"), str(link))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        root, name = out.stdout.strip().splitlines()[:2]
        self.assertEqual(Path(root).resolve(), REPO.resolve())
        self.assertEqual(name, "memories")


class TestTheInstallPathTheLoaderActuallyReads(unittest.TestCase):
    """Where the symlink has to go, asked of the INSTALLED loader rather than assumed.

    `scripts/hermes_cutover.sh` installs into `$HERMES_HOME/plugins/memories`, and the
    claude-code cutover's own comments record what it costs to guess a path: it checked
    `~/.mcp.json` while the live configuration was in `~/.claude.json` and printed "ok" for
    a state it had never verified. So this drives `plugins/memory/__init__.py` itself,
    against a temp HERMES_HOME, and requires it to find the provider at that path and NOT
    at the one-level-deeper layout the third-party provider on this machine uses.
    """

    @unittest.skipUnless(HERMES_INSTALL.exists(), "hermes-agent not installed here")
    def test_the_loader_finds_the_provider_at_hermes_home_plugins_name(self):
        import shutil

        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, True)
        (home / "plugins" / "memory").mkdir(parents=True)
        (home / "plugins" / "memories").symlink_to(REPO / "hosts" / "hermes")
        # The same adapter, one level deeper — the layout `$HERMES_HOME/plugins/memory/
        # <name>/` that the qdrant provider uses on this machine.
        (home / "plugins" / "memory" / "deeper").symlink_to(REPO / "hosts" / "hermes")

        script = (
            "import json, sys\n"
            "sys.path.insert(0, %r)\n"
            "from plugins.memory import find_provider_dir, load_memory_provider\n"
            "flat = find_provider_dir('memories')\n"
            "deep = find_provider_dir('deeper')\n"
            "p = load_memory_provider('memories')\n"
            "print(json.dumps([str(flat), str(deep), getattr(p, 'name', None)]))\n"
        ) % str(HERMES_INSTALL)
        env = dict(os.environ, HERMES_HOME=str(home))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        flat, deep, name = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(Path(flat), home / "plugins" / "memories")
        self.assertEqual(deep, "None", "the one-level-deeper layout is discovered after all")
        self.assertEqual(name, "memories", "the loader did not instantiate the provider")


class TestManifest(unittest.TestCase):
    def test_the_yaml_declares_what_the_loader_reads(self):
        text = (REPO / "hosts" / "hermes" / "plugin.yaml").read_text()
        for key in ("name: memories", "category: memory", "kind: exclusive"):
            self.assertIn(key, text)

    def test_the_init_contains_the_string_discovery_greps_for(self):
        """_is_memory_provider_dir is a cheap text scan of the first 8192 bytes for
        'register_memory_provider' or 'MemoryProvider'. Without one of those the directory
        is not even considered a provider."""
        head = (REPO / "hosts" / "hermes" / "__init__.py").read_text()[:8192]
        self.assertTrue("register_memory_provider" in head or "MemoryProvider" in head)


class TestPrefetch(unittest.TestCase):
    """The read direction, through the hermes entry point.

    The provider is driven with fakes injected into its store slot, so these run offline.
    """

    def _provider(self, hits, outcome):
        from core.retrieval import Outcome  # noqa: F401  (documents the shape below)
        p = MemoriesProvider()
        p._cfg = object()

        class FakeStore:
            def recall(self, queries, policy, top_k, suppressed=None):
                return hits, outcome

        p._store = FakeStore()
        p._state_dir = Path(tempfile.mkdtemp())

        return p

    def test_a_populated_result_carries_the_rules_and_the_memory(self):
        from core.retrieval import CE, Outcome
        from tests.test_blocks import FakeHit
        p = self._provider([FakeHit(id="m1", document="a durable fact", origin=CE)],
                           Outcome(candidates=1, reranked=True))
        out = p.prefetch("how does the poll paginate?")
        self.assertIn("a durable fact", out)
        self.assertIn("m1", out)
        self.assertIn("PREVAILS", out, "the rules of use travel with the memory")

    def test_an_empty_complete_result_says_there_is_no_precedent(self):
        from core.retrieval import Outcome
        p = self._provider([], Outcome(candidates=4, best_dense=0.31, reranked=True))
        self.assertIn("There is no recorded precedent", p.prefetch("an absent subject"))

    def test_a_partial_result_withdraws_the_claim(self):
        from core.retrieval import Outcome
        p = self._provider([], Outcome(candidates=27, best_dense=0.54,
                                       suppressed="circuit breaker: 12s ago"))
        out = p.prefetch("a subject that might have history")
        self.assertNotIn("There is no recorded precedent", out)
        self.assertIn("not evidence that no precedent exists", out)

    def test_a_failure_reaches_the_MODEL_and_not_only_the_log(self):
        """The central contract: silent to the user, never to the model."""
        p = MemoriesProvider()
        p._cfg = object()

        class Broken:
            def recall(self, *a, **kw):
                raise core.EmbeddingError("endpoint down")

        p._store = Broken()
        p._state_dir = Path(tempfile.mkdtemp())
        out = p.prefetch("a real question about the archive")
        self.assertIn("UNAVAILABLE", out)
        self.assertIn("was not consulted", out)

    def test_an_unexpected_exception_also_reaches_the_model(self):
        p = MemoriesProvider()
        p._cfg = object()

        class Exploding:
            def recall(self, *a, **kw):
                raise RuntimeError("something nobody predicted")

        p._store = Exploding()
        p._state_dir = Path(tempfile.mkdtemp())
        self.assertIn("UNAVAILABLE", p.prefetch("a real question about the archive"))

    def test_a_trivial_prompt_costs_nothing(self):
        """hermes has its own is_trivial_prompt, but it is ENGLISH ONLY — measured: it does
        not match "ok, pode continuar". The plugin's composed Portuguese filter runs too."""
        p = MemoriesProvider()
        p._cfg = object()

        class Counting:
            calls = 0

            def recall(self, *a, **kw):
                Counting.calls += 1

                return [], None

        p._store = Counting()
        p._state_dir = Path(tempfile.mkdtemp())
        for prompt in ("ok", "sim, pode ser", "ok, pode continuar", "/status"):
            self.assertEqual(p.prefetch(prompt), "", prompt)
        self.assertEqual(Counting.calls, 0, "a trivial prompt reached the network")

    def test_the_system_prompt_block_is_static_and_not_the_recall(self):
        block = MemoriesProvider().system_prompt_block()
        self.assertIn("qctx memory", block)
        self.assertNotIn("── ", block, "recalled memories go through prefetch, not here")

    def test_an_unwritable_state_directory_does_not_cost_the_search(self):
        """The third appearance of 'safe direction, wrong message' in this plan:
        `_state_path` returns None when it cannot create the state directory, and that
        None used to reach `Breaker(None, ...)`, which raised TypeError before
        `store.recall` ever ran — an archive that was perfectly reachable got reported as
        not consulted. `core.breaker.Breaker` now tolerates `path=None` the same way
        `session_state` tolerates a missing state directory, so the search must still run
        and a REAL block (not the unavailability one) must come back.
        """
        from core.retrieval import Outcome
        p = MemoriesProvider()
        p._cfg = object()

        calls = []

        class Counting:
            def recall(self, *a, **kw):
                calls.append(1)

                return [], Outcome(candidates=0, best_dense=0.0)

        p._store = Counting()
        # A plain FILE where a directory is expected: base.mkdir(...) fails with
        # FileExistsError (an OSError), so _state_path degrades to None.
        blocker = Path(tempfile.mkdtemp()) / "blocked"
        blocker.write_text("not a directory")
        p._state_dir = blocker

        out = p.prefetch("a real question the archive should be reachable for")
        self.assertEqual(calls, [1], "the search must run even without a state directory")
        self.assertNotIn("UNAVAILABLE", out, "the archive was reachable; it must not be "
                         "reported as unconsulted because a directory could not be made")

    def test_recall_disabled_by_env_costs_nothing(self):
        """The hook honours QCTX_RECALL_DISABLED / RECALL_DISABLED; the adapter must too —
        a user who disabled recall expects it disabled in both hosts, and Task 6's
        equivalence test extracts this exact name from both files."""
        p = MemoriesProvider()
        p._cfg = object()

        class Counting:
            calls = 0

            def recall(self, *a, **kw):
                Counting.calls += 1

                return [], None

        p._store = Counting()
        p._state_dir = Path(tempfile.mkdtemp())
        with unittest.mock.patch.dict(os.environ, {"QCTX_RECALL_DISABLED": "1"}):
            self.assertEqual(p.prefetch("a real question about the archive"), "")
        self.assertEqual(Counting.calls, 0, "prefetch reached the network while disabled")

        Counting.calls = 0
        with unittest.mock.patch.dict(os.environ, {"RECALL_DISABLED": "1"}):
            self.assertEqual(p.prefetch("a real question about the archive"), "")
        self.assertEqual(Counting.calls, 0, "the legacy name must disable it too")


class TestTheBreakerDegradesONETurnAndNotTheWholeSession(unittest.TestCase):
    """A rerank failure must cost the turns the breaker is open for, and not one more.

    This host is the only one that can get this wrong, and it did. The claude-code hook is a
    fresh process per prompt, so writing `store.reranker = None` cannot outlive the prompt
    that wrote it. Here the store is cached for the whole session (`_ensure_store`, for the
    connection the 8s prefetch budget is sized around), so the suppression outlived the
    breaker. Measured across three prefetches with the breaker armed only for the first:

        turn 1 (breaker OPEN)  top_k=8  reranker=False  suppressed='circuit breaker: …'
        turn 2 (breaker cold)  top_k=8  reranker=False  suppressed=None
        turn 3 (breaker cold)  top_k=8  reranker=False  suppressed=None

    Two failures in one, and the second is the serious one: the pipeline stayed degraded
    for the rest of the session, AND `suppressed` went back to None, so the degradation
    note vanished while the degradation did not — an empty result then printed "There is no
    recorded precedent on this subject". A flat claim of absence produced by a crippled
    search is exactly what the four block states exist to prevent.

    Nothing drove a SECOND turn before this: `grep breaker tests/test_hermes_provider.py`
    found comments only, and one-turn tests cannot see a bug whose whole shape is what the
    first turn leaves behind.
    """

    #: Read as instance attributes rather than trusting the class defaults: `TOP_K` and
    #: `TOP_K_STRICT` are frozen at import from `QCTX_RECALL_TOP_K`/`RECALL_TOP_K`, and this
    #: user exports `RECALL_*` from `.bashrc` — which would pin both states to one number
    #: and make the top_k half of every assertion below vacuous. This test is about the
    #: breaker; the knob's own defaults are held by test_host_equivalence.py.
    LENIENT, STRICT = 20, 8

    def _provider(self, state_dir):
        from core.retrieval import Outcome

        turns = []

        class Reranker:
            """Never actually asked to rank — its PRESENCE is the whole subject."""

            def rank(self, query, documents):  # pragma: no cover — see the docstring
                return [], {"ok": False, "error": "not exercised here"}

        class Store:
            reranker = Reranker()

            def recall(self, queries, policy, top_k, suppressed=None):
                turns.append({"top_k": top_k,
                              "reranker": getattr(self, "reranker", None) is not None,
                              "suppressed": suppressed})
                # What `core.retrieval.two_stage` really does with the parameter: it goes
                # into the Outcome, which is what `blocks` reads to pick the conclusion.
                return [], Outcome(candidates=3, best_dense=0.54, suppressed=suppressed)

        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = state_dir
        p._store = Store()
        p.TOP_K, p.TOP_K_STRICT = self.LENIENT, self.STRICT
        p.BREAKER_SECONDS = 300.0

        return p, turns

    PROMPT = "how does the connector poll paginate its results?"
    FLAT_CLAIM = "There is no recorded precedent"

    def test_a_cooled_breaker_gives_the_cross_encoder_back_on_the_very_next_turn(self):
        import time

        state_dir = Path(tempfile.mkdtemp())
        breaker_file = state_dir / "rerank-breaker"
        p, turns = self._provider(state_dir)

        breaker_file.write_text(str(time.time()))                 # failed just now
        blocks_out = [p.prefetch(self.PROMPT)]
        breaker_file.write_text(str(time.time() - 600))           # cooldown 300s: expired
        blocks_out.append(p.prefetch(self.PROMPT))
        blocks_out.append(p.prefetch(self.PROMPT))

        self.assertEqual(len(turns), 3, "the store was not reached once per turn")

        # Turn 1: the breaker is open, so this is what a DEGRADED turn has to look like.
        self.assertEqual(turns[0]["top_k"], self.STRICT)
        self.assertFalse(turns[0]["reranker"], "the open breaker must hold the re-rank back")
        self.assertIn("circuit breaker", turns[0]["suppressed"] or "")
        self.assertNotIn(self.FLAT_CLAIM, blocks_out[0],
                         "a degraded search may never claim the archive holds nothing")

        # Turns 2 and 3: the breaker cooled, so two-stage retrieval comes BACK — and stays
        # back. Turn 3 is not redundant: the bug was permanent, and a fix that restored the
        # reranker only on the first cooled turn would pass with turn 2 alone.
        for i in (1, 2):
            with self.subTest(turn=i + 1):
                self.assertTrue(turns[i]["reranker"],
                                "the cross-encoder never came back: one rerank failure "
                                "degraded the rest of the session")
                self.assertEqual(turns[i]["top_k"], self.LENIENT,
                                 "top_k stayed at the strict floor with a reranker present")
                self.assertIsNone(turns[i]["suppressed"],
                                  "nothing was suppressed on this turn")

        self.assertIsNotNone(getattr(p._store, "reranker", None),
                             "the session's store lost its cross-encoder permanently")

    def test_no_turn_may_claim_absence_while_the_pipeline_is_degraded(self):
        """The invariant behind the numbers above, stated once over all three turns.

        The old code broke it on turns 2 and 3: the reranker was gone from a store that has
        one, `suppressed` was None so no note was rendered, and the block asserted absence
        anyway. A turn may claim the archive holds nothing ONLY if the search behind that
        claim was whole.
        """
        import time

        state_dir = Path(tempfile.mkdtemp())
        breaker_file = state_dir / "rerank-breaker"
        p, turns = self._provider(state_dir)

        breaker_file.write_text(str(time.time()))
        out = [p.prefetch(self.PROMPT)]
        breaker_file.write_text(str(time.time() - 600))
        out += [p.prefetch(self.PROMPT), p.prefetch(self.PROMPT)]

        for i, (turn, block) in enumerate(zip(turns, out), 1):
            degraded = not turn["reranker"] or turn["top_k"] == self.STRICT
            with self.subTest(turn=i):
                if degraded:
                    self.assertNotIn(self.FLAT_CLAIM, block,
                                     f"turn {i} ran degraded ({turn}) and still told the "
                                     f"model there is no precedent")
                    self.assertIsNotNone(turn["suppressed"],
                                         f"turn {i} ran degraded and said nothing about it")


def _dummy_cfg():
    """A Config that satisfies build_memory's validation without ever touching the
    network — Qdrant/Embedder/Reranker only open connections lazily, on first use."""
    return core.Config(
        qdrant_url="http://localhost:1", qdrant_api_key="", api_base_url="",
        api_key="", embed_url="http://localhost:1/embeddings", rerank_url="",
        embed_model="m", rerank_model="", memory_collection="test-memories",
        docs_collection="", library_collection="", vector_size=8,
    )


class TestEnsureStoreTimeouts(unittest.TestCase):
    """`_ensure_store` derives the qdrant timeout from HERMES_PREFETCH_BUDGET_S and
    QCTX_RECALL_QDRANT_BUDGET. Both must have a REAL effect: a knob that is read and then
    ignored would still pass Task 6's name-equality check while doing nothing, which
    converts a real divergence between the two hosts into documented assurance.
    """

    def setUp(self):
        from hosts.hermes import HERMES_PREFETCH_BUDGET_S, MAX_ANGLES
        self.share = HERMES_PREFETCH_BUDGET_S / 4.0
        self.qdrant_calls = MAX_ANGLES + 1
        self._original_budget = MemoriesProvider.QDRANT_BUDGET

    def tearDown(self):
        MemoriesProvider.QDRANT_BUDGET = self._original_budget

    def test_the_default_budget_is_sized_for_the_worst_case_angle_count(self):
        """Regression for the bug: the store used to be sized from whichever prompt's
        angle count built it first, so a 1-angle prompt baked in a timeout a later
        3-angle prompt would then multiply past the ceiling. `_ensure_store` no longer
        takes an angle count at all — it must always assume MAX_ANGLES."""
        p = MemoriesProvider()
        p._cfg = _dummy_cfg()
        store = p._ensure_store()
        expected = min(self._original_budget, self.share) / self.qdrant_calls
        self.assertAlmostEqual(store.q.timeout, expected, places=6)

    def test_the_budget_knob_can_tighten_the_qdrant_timeout(self):
        MemoriesProvider.QDRANT_BUDGET = 0.1   # far below the derived share
        p = MemoriesProvider()
        p._cfg = _dummy_cfg()
        store = p._ensure_store()
        expected = 0.1 / self.qdrant_calls
        self.assertAlmostEqual(store.q.timeout, expected, places=6)

    def test_the_budget_knob_cannot_exceed_this_hosts_own_derived_share(self):
        """The hook's default (5.0s) was sized for its own, more generous host deadline.
        Taking it literally here — on an 8s ceiling shared with embed and rerank — would
        blow past it on the default alone, so the knob is a ceiling that can only
        tighten, never loosen past what this host can afford."""
        MemoriesProvider.QDRANT_BUDGET = 999.0  # far above the derived share
        p = MemoriesProvider()
        p._cfg = _dummy_cfg()
        store = p._ensure_store()
        expected = self.share / self.qdrant_calls
        self.assertAlmostEqual(store.q.timeout, expected, places=6)


class TestCheckpointCadence(unittest.TestCase):
    """The write nudge. It rides along in prefetch, and the reason is structural:
    on_turn_start runs every turn and CARRIES the number, but returns None — so it cannot
    inject. The only injection points hermes offers are prefetch and system_prompt_block.
    """

    def _provider(self, interval):
        from core.retrieval import Outcome
        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = Path(tempfile.mkdtemp())
        p.CHECKPOINT_INTERVAL = interval

        class FakeStore:
            def recall(self, queries, policy, top_k, suppressed=None):
                return [], Outcome(candidates=1, best_dense=0.2, reranked=True)

        p._store = FakeStore()

        return p

    def test_the_procedure_appears_on_the_interval_and_only_there(self):
        p = self._provider(3)
        seen = []
        for turn in range(1, 8):
            p.on_turn_start(turn, "a real question about the archive")
            out = p.prefetch("a real question about the archive")
            if "memory checkpoint" in out:
                seen.append(turn)
        self.assertEqual(seen, [3, 6])

    def test_the_procedure_renders_with_the_turn_number(self):
        p = self._provider(1)
        p.on_turn_start(1, "a real question about the archive")
        out = p.prefetch("a real question about the archive")
        self.assertIn("Interaction 1 of this conversation (every 1)", out)
        self.assertNotIn("{count}", out)

    def test_an_interval_of_zero_disables_it(self):
        p = self._provider(0)
        for turn in range(1, 6):
            p.on_turn_start(turn, "a real question about the archive")
            self.assertNotIn("memory checkpoint", p.prefetch("a real question about the archive"))

    def test_the_recall_block_still_comes_first(self):
        """Order matters: the memories and their rules, then the write nudge."""
        from core.retrieval import CE, Outcome
        from tests.test_blocks import FakeHit
        p = self._provider(1)
        p._store = type("S", (), {"recall": lambda self, *a, **kw: (
            [FakeHit(id="m1", document="a durable fact", origin=CE)],
            Outcome(candidates=1, reranked=True))})()
        p.on_turn_start(1, "a real question about the archive")
        out = p.prefetch("a real question about the archive")
        self.assertLess(out.index("a durable fact"), out.index("memory checkpoint"))

    def test_a_provider_never_told_a_turn_number_is_never_due(self):
        """The equivalence test and several TestPrefetch cases drive `prefetch` directly,
        without ever calling `on_turn_start` first — exactly what a provider looks like on
        its very first read. `0 % interval == 0` for any positive interval, so without a
        guard against turn 0 every one of those callers would get a spurious checkpoint
        the claude-code side never renders, breaking host equivalence. `on_turn_start`
        hands out turns starting at 1 in production; turn 0 only ever means "never told."
        """
        p = self._provider(1)
        out = p.prefetch("a real question about the archive")
        self.assertNotIn("memory checkpoint", out)

    def test_the_env_switch_disables_it_even_when_due(self):
        """Same name and meaning as the claude-code hook's QCTX_CHECKPOINT_DISABLED — a
        user who turned checkpointing off expects it off on both hosts."""
        p = self._provider(1)
        p.on_turn_start(1, "a real question about the archive")
        with unittest.mock.patch.dict(os.environ, {"QCTX_CHECKPOINT_DISABLED": "1"}):
            out = p.prefetch("a real question about the archive")
        self.assertNotIn("memory checkpoint", out)


class TestCheckpointIntervalIsRobust(unittest.TestCase):
    """`CHECKPOINT_INTERVAL` is read at import time, before any per-call guard runs, and it
    now feeds the SAME call that produces the recall block. A malformed value here is
    worse than in the claude-code hook: it would take recall down with it instead of just
    itself.
    """

    def _read_it(self, env):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "print(MemoriesProvider().CHECKPOINT_INTERVAL)\n"
        ) % str(REPO)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)

        return out.stdout.strip()

    def test_a_malformed_interval_falls_back_to_the_default_instead_of_raising(self):
        env = dict(os.environ, QCTX_CHECKPOINT_INTERVAL="5x")
        env.pop("REMEMBER_INTERVAL", None)
        self.assertEqual(self._read_it(env), "5")

    def test_the_legacy_alias_still_works(self):
        env = dict(os.environ, REMEMBER_INTERVAL="9")
        env.pop("QCTX_CHECKPOINT_INTERVAL", None)
        self.assertEqual(self._read_it(env), "9")


#: Every numeric knob a host reads at import time, DERIVED FROM ITS SOURCE rather than
#: listed here: `ATTR = env_num("QCTX_...", "LEGACY", "<default>"[, kind][, …])`, with or
#: without the leading underscore, and the bare `ATTR = int(env("QCTX_...", "LEGACY",
#: "<numeric default>"))` form too — which is exactly the form that has to be caught,
#: because that is how a knob skips the tolerant helper.
#:
#: `^\s*` and not `^\s{4}`: the hermes knobs sit in a class body and the two hooks' sit at
#: module level, and the whole point of scanning all three is that indentation must not
#: decide what gets checked.
_KNOB = re.compile(
    r'^\s*(?P<attr>[A-Z][A-Z0-9_]*)\s*=\s*(?:(?P<cast>int|float)\()?'
    r'_?env(?:_num)?\(\s*"(?P<name>[A-Z][A-Z0-9_]*)"\s*,\s*"(?P<legacy>[A-Z][A-Z0-9_]*)"\s*,'
    r'\s*"(?P<default>[^"]*)"(?:\s*,\s*(?P<kind>int|float))?[^)\n]*\)',
    re.M)

#: The files that read numeric knobs at import time, and the statement that imports each so
#: its values can be read back. All THREE hosts' files, not just the hermes adapter: while
#: this scan covered `hosts/hermes/__init__.py` alone, `hooks/recall.py` kept the one bare
#: `int(env(...))` in the repo for three reviews — the ledger named the blind spot and the
#: scan still could not see into it. A knob is checked because a test reads the FILE it
#: lives in, so the list of files is the coverage.
KNOB_SOURCES = {
    "hosts/hermes/__init__.py": "from hosts.hermes import MemoriesProvider as M\n",
    "hooks/recall.py": "import recall as M\n",
    "hooks/checkpoint.py": "import checkpoint as M\n",
}


def numeric_knobs(rel_path: str) -> list:
    """(attribute, canonical env name, legacy env name, coded default) for one file's knobs.

    Read out of the source so a knob added later is covered without anyone remembering to
    add it here — which is the whole point twice over: the knob that broke was the one
    nobody thought to list, and the one that stayed broken was in a file nobody scanned.
    """
    found = []
    for m in _KNOB.finditer((REPO / rel_path).read_text()):
        try:
            float(m.group("default"))
        except ValueError:
            continue                      # not a numeric knob
        found.append((m.group("attr"), m.group("name"), m.group("legacy"),
                      m.group("default")))

    return found


def bare_env_casts(rel_path: str) -> list:
    """`int(env(...))`-shaped reads — a raw environment value cast where it is read.

    An AST walk and not a regex, because this file, `hooks/recall.py`, `hooks/checkpoint.py`
    and `hosts/hermes/__init__.py` all QUOTE the offending shape in prose to explain why it
    is forbidden — a text scan reports every one of those comments and nothing else.
    """
    tree = ast.parse((REPO / rel_path).read_text())
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("int", "float") and node.args):
            continue
        inner = node.args[0]
        if not isinstance(inner, ast.Call):
            continue
        read = inner.func
        # `env("NAME", ...)` / `_env(...)` — the tolerant helpers are `env_num`/`_env_num`,
        # which cast on their own and are never wrapped — or `os.environ.get(...)` direct.
        raw = (isinstance(read, ast.Name) and read.id in ("env", "_env")) or (
            isinstance(read, ast.Attribute) and read.attr == "get"
            and isinstance(read.value, ast.Attribute) and read.value.attr == "environ")
        if raw:
            found.append(f"{rel_path}:{node.lineno}")

    return found


class TestEveryNumericKnobToleratesAMalformedValue(unittest.TestCase):
    """No numeric knob may take a host down at import time, on EITHER host.

    The measured failure on hermes: `TOP_K = int(_env("QCTX_RECALL_TOP_K", ...))` — the one
    numeric knob that skipped `_env_num` — turned `QCTX_RECALL_TOP_K=8x` into a ValueError
    while the class body was executing. hermes' loader swallows that at `logger.debug`, and
    `agent/agent_init.py` only warns when the provider is NOT None, so a None provider says
    NOTHING: no recall, no checkpoint, no tools, no message.

    The measured failure on claude-code, found later and by the same knob: `hooks/recall.py`
    read it with a bare `int(env(...))` too, so `QCTX_RECALL_TOP_K=8x` made the hook emit
    "[automatic recall — UNAVAILABLE for this prompt] … the hook failed (ValueError)" on
    every prompt of every session while the archive was perfectly reachable — and `env_num`'s
    explanatory note never ran, because the value never went through it.

    Not hypothetical either: `QCTX_RECALL_MAX_CHARS=14k` is the mistake that made the
    tolerant read exist in the first place, and `RECALL_*` variables are exported from this
    user's `.bashrc`.

    The knob list is DERIVED from each file's source, so a future knob written with a bare
    `int(...)` fails this test instead of waiting for a typo in production.
    """

    def test_the_derivation_found_knobs_in_every_file_it_scans(self):
        """A guard on the guard, per file: if the regex stopped matching one of them,
        everything below would pass over an empty list for that file and prove nothing."""
        per_file = {rel: {attr for attr, *_ in numeric_knobs(rel)} for rel in KNOB_SOURCES}
        for rel, attrs in per_file.items():
            self.assertTrue(attrs, f"no numeric knob derived from {rel}: the scan went "
                                   f"blind on that file")
        self.assertGreaterEqual(len(per_file["hosts/hermes/__init__.py"]), 9,
                                f"only found {per_file['hosts/hermes/__init__.py']}")
        for expected in ("TOP_K", "MAX_CHARS", "STRICT_FLOOR", "CHECKPOINT_INTERVAL"):
            self.assertIn(expected, per_file["hosts/hermes/__init__.py"])
        # The hook's own knobs, TOP_K above all: it is the knob this scan was extended for.
        for expected in ("TOP_K", "TOP_K_STRICT", "MAX_CHARS", "BREAKER_SECONDS"):
            self.assertIn(expected, per_file["hooks/recall.py"],
                          "the hook's numeric knobs are not being scanned")
        self.assertIn("INTERVAL", per_file["hooks/checkpoint.py"])

    def test_no_knob_reads_the_environment_without_the_tolerant_helper(self):
        """The complement of the derivation: a knob the scan cannot SEE is not covered by it.

        The derivation above only looks at assignments, so a cast written inside a function —
        which is precisely where the hook's `int(env("QCTX_RECALL_TOP_K", ...))` hid for three
        reviews — would be invisible to it while still killing the host. This forbids the
        shape itself, anywhere in the file.
        """
        for rel in KNOB_SOURCES:
            with self.subTest(source=rel):
                self.assertEqual(bare_env_casts(rel), [],
                                 f"{rel} casts a raw environment read instead of going "
                                 f"through the tolerant helper")

    def _malformed_env(self):
        """Every knob of every scanned file set to garbage, at once.

        All of them together rather than one file at a time: the three files share knob
        names (QCTX_RECALL_TOP_K is read by two of them), and a deployer's environment is
        not scoped per file either.
        """
        env = dict(os.environ)
        env["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        for rel in KNOB_SOURCES:
            for _, name, legacy, _ in numeric_knobs(rel):
                env[name] = "not-a-number"
                env.pop(legacy, None)

        return env

    def test_a_malformed_value_falls_back_instead_of_killing_the_import(self):
        env = self._malformed_env()
        for rel, importer in KNOB_SOURCES.items():
            knobs = numeric_knobs(rel)
            with self.subTest(source=rel):
                script = (
                    "import sys, json\n"
                    "sys.path.insert(0, %r)\n"
                    "sys.path.insert(0, %r)\n"
                    + importer +
                    "print(json.dumps({a: getattr(M, a) for a in %r}))\n"
                ) % (str(REPO), str(REPO / "hooks"), [attr for attr, *_ in knobs])
                out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                     text=True, env=env)
                self.assertEqual(out.returncode, 0,
                                 f"a malformed knob took {rel} down at import time, before "
                                 f"any guard could report it:\n" + out.stderr)
                values = json.loads(out.stdout.strip().splitlines()[-1])
                for attr, _, _, default in knobs:
                    self.assertAlmostEqual(float(values[attr]), float(default), places=6,
                                           msg=f"{rel}: {attr} did not fall back to its "
                                               f"coded default")


class TestCheckpointFailureDoesNotCostRecall(unittest.TestCase):
    """The checkpoint and the recall now share one return value. A failure inside the
    checkpoint half must be worth at most one skipped nudge — never the loss of a recall
    block that had already been built successfully.
    """

    def test_a_broken_cadence_check_still_returns_the_recall_block(self):
        from core.retrieval import CE, Outcome
        from tests.test_blocks import FakeHit
        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = Path(tempfile.mkdtemp())
        p.CHECKPOINT_INTERVAL = 1

        class FakeStore:
            def recall(self, queries, policy, top_k, suppressed=None):
                return ([FakeHit(id="m1", document="a durable fact", origin=CE)],
                       Outcome(candidates=1, reranked=True))

        p._store = FakeStore()
        p.on_turn_start(1, "a real question about the archive")

        with unittest.mock.patch("hosts.hermes.session_state.due",
                                 side_effect=RuntimeError("boom")):
            out = p.prefetch("a real question about the archive")

        self.assertIn("a durable fact", out, "the recall must survive a broken checkpoint")
        self.assertNotIn("UNAVAILABLE", out, "a checkpoint failure is not a recall failure")
        self.assertNotIn("memory checkpoint", out)


class TestConfigSchema(unittest.TestCase):
    """`hermes memory setup` walks this, and it must write where claude-code reads.

    The two write-path tests below run the whole thing in a FRESH SUBPROCESS rather than
    mutating `os.environ["QCTX_CONFIG"]` in-process, and that is not a style preference.
    `core.config.DEFAULT_CONFIG_PATH` (core/config.py:22-25) is computed ONCE, at module
    import time, from whatever `QCTX_CONFIG` says at that moment — it is a frozen
    constant, never re-read per call. `core` is already imported at the top of this test
    FILE, before any test method runs, so setting the env var inside a test method has
    ZERO effect on it (measured). `save_config` calls `core.save(patch)` with no explicit
    path, so it falls back to that already-frozen `DEFAULT_CONFIG_PATH` — which, absent an
    ambient `QCTX_CONFIG` for the whole test run, IS the operator's real
    `~/.config/memories-plugin/config.json`. Every other QCTX_CONFIG-dependent test in
    this suite (TestAvailability above, TestCheckpointIntervalIsRobust, and the two other
    test modules) already sets the variable in a subprocess's `env=` for exactly this
    reason; these two follow the same idiom instead of introducing a second, unsafe one.
    """

    def test_the_two_api_keys_are_declared_secret(self):
        by_key = {f["key"]: f for f in MemoriesProvider().get_config_schema()}
        for key in ("qdrant_api_key", "api_key"):
            self.assertTrue(by_key[key]["secret"],
                            f"{key} in a config file is a leaked secret")
            self.assertTrue(by_key[key]["env_var"].startswith("QCTX_"),
                            "a secret needs the env var name it will be read from")

    def test_no_non_secret_field_is_marked_secret(self):
        by_key = {f["key"]: f for f in MemoriesProvider().get_config_schema()}
        self.assertFalse(by_key["memory_collection"]["secret"])
        self.assertFalse(by_key["qdrant_url"]["secret"])

    def test_the_schema_covers_every_configurable_field(self):
        from core.config import Config
        from dataclasses import fields as dc_fields
        declared = {f["key"] for f in MemoriesProvider().get_config_schema()}
        self.assertEqual({f.name for f in dc_fields(Config)}, declared,
                         "a field missing here is a setting the hermes wizard cannot set")

    def _clean_env(self, cfg_path):
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        env["QCTX_CONFIG"] = str(cfg_path)

        return env

    def test_save_config_writes_where_claude_code_reads(self):
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "MemoriesProvider().save_config({'memory_collection': 'shared_memory'}, %r)\n"
            "import core\n"
            "print(json.dumps(core.load().memory_collection))\n"
        ) % (str(REPO), str(cfg_dir))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=self._clean_env(cfg_path))
        self.assertEqual(out.returncode, 0, out.stderr)
        # Both halves checked: the file itself (the artifact claude-code reads from disk)
        # AND core.load() (the actual read path claude-code uses) — a divergence between
        # the two would mean the file is right but the loader disagrees, or vice versa.
        self.assertEqual(json.loads(cfg_path.read_text())["memory_collection"],
                         "shared_memory")
        self.assertEqual(json.loads(out.stdout.strip()), "shared_memory")

    def test_save_config_refuses_a_secret_even_if_hermes_passes_one(self):
        """core.save already refuses; this asserts the adapter does not route around it."""
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "MemoriesProvider().save_config({'api_key': 'SUPERSECRET'}, %r)\n"
            "print('done')\n"
        ) % (str(REPO), str(cfg_dir))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=self._clean_env(cfg_path))
        self.assertEqual(out.returncode, 0, out.stderr)
        written = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.assertNotIn("api_key", written)
        self.assertNotIn("SUPERSECRET", json.dumps(written))

    def test_save_config_merges_and_does_not_clobber_an_unrelated_existing_field(self):
        """The user may already be running claude-code against a live config. A wizard run
        through hermes must not truncate it: a field the wizard never asked about — here,
        `embed_model`, chosen because it is untouched by either write below — has to
        survive both the secret-refusal write and the ordinary write that follows it."""
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        cfg_path.write_text(json.dumps({"embed_model": "already-configured-model",
                                        "memory_collection": "old_collection"}))
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "p = MemoriesProvider()\n"
            "p.save_config({'api_key': 'SUPERSECRET'}, %r)\n"
            "p.save_config({'memory_collection': 'new_collection'}, %r)\n"
            "print('done')\n"
        ) % (str(REPO), str(cfg_dir), str(cfg_dir))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=self._clean_env(cfg_path))
        self.assertEqual(out.returncode, 0, out.stderr)
        written = json.loads(cfg_path.read_text())
        self.assertEqual(written["embed_model"], "already-configured-model",
                         "a field the wizard did not touch must survive the save")
        self.assertEqual(written["memory_collection"], "new_collection")
