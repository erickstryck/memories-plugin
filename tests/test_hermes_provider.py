"""The hermes-agent host adapter.

It is verified WITHOUT hermes importable, because this repo's suite has to run offline and
hermes lives in its own venv. What the adapter promises is therefore checked structurally:
the method set hermes v0.20.0 actually calls, measured from the install rather than assumed
from the published source — the two differ, and the installed one is what runs.
"""
import json
import os
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

#: Where hermes is installed on this machine. The ABC IS importable from there (measured);
#: only `RecallStatus` is absent in v0.20.0. Used by the test below, skipped elsewhere so
#: the suite stays portable.
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
