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


if __name__ == "__main__":
    unittest.main(verbosity=2)
