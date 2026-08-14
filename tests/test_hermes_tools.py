"""The 15 tools the model may call, and the 5 it may not.

Configuration is the operator's, not the model's: exposing `config set` as a tool would let
the model point the archive somewhere else mid-conversation. `setup` is interactive and
wants a TTY. The CLI keeps all 20 in both hosts — the restriction is only about what the
MODEL can reach on its own.

Two properties are checked from OUTSIDE the tools module, not by importing it:
the provider must expose the schemas (a tools module nothing reaches is dead weight a test
importing it directly would still pass over), and `handle_tool_call` must return a JSON
STRING — a dict there fails at the host boundary, which no test of the function's logic
would notice.
"""
import json
import os
import subprocess
import sys
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import core
from core.docs import DocIndex
from core.memory import MemoryStore
from hosts.hermes import MemoriesProvider, tools
from tests.fakes import FakeEmbedder, FakeVectorStore

EXPECTED = {
    "memory_store", "memory_store_many", "memory_find", "memory_recall", "memory_get",
    "memory_update", "memory_delete", "memory_list", "memory_search_collections",
    "docs_index", "docs_keep", "docs_search", "docs_list", "docs_refresh", "docs_drop",
}
FORBIDDEN = {"setup", "config_set", "config_detect", "config_show", "collections"}

#: Where hermes is installed on this machine. The real ABC and the real schema
#: normalization live there; tests that read them are skipped elsewhere so the suite stays
#: portable.
HERMES_INSTALL = Path.home() / ".hermes" / "hermes-agent"


#: Started in `setUpModule`, stopped in `tearDownModule`. See the docstring there.
_CONFIG_GUARD = None


def setUpModule():
    """Make any test that reaches the OPERATOR'S REAL CONFIG fail loudly and immediately.

    This is a guard, not a reminder, because the danger is concrete and one line deep:
    `core.load()` defaults `library_collection` to `memories_docs_library`
    (core/config.py:54-55), a PRODUCTION archive, and `docs_refresh` has no required
    arguments. A provider left without an injected config — one deleted `p._cfg = ...` line
    in a test that walks all 15 tools — would reindex the user's permanent library from a
    unit test, with the suite green.

    Every test here therefore injects a `Config`. The one behaviour that must call
    `core.load` (`_config()`'s fallback) re-patches it locally with a recorder, which is
    both allowed and visible.
    """
    global _CONFIG_GUARD

    def forbidden(*a, **kw):
        raise AssertionError(
            "a test tried to read the operator's real configuration via core.load(); "
            "inject a Config instead — this suite must never reach a production collection")

    _CONFIG_GUARD = unittest.mock.patch.object(core, "load", forbidden)
    _CONFIG_GUARD.start()


def tearDownModule():
    _CONFIG_GUARD.stop()


def a_config(**over) -> core.Config:
    """A Config that reaches nothing. The store is injected in these tests, so the only
    field that has to be right is `vector_size`, which has to match `FakeEmbedder`."""
    values = dict(qdrant_url="http://localhost:1", qdrant_api_key="",
                  api_base_url="http://localhost:1/v1", api_key="", embed_url="",
                  rerank_url="", embed_model="m", rerank_model="",
                  memory_collection="mem", docs_collection="tmp",
                  library_collection="lib", vector_size=8)
    values.update(over)

    return core.Config(**values)


class TestSchemas(unittest.TestCase):
    def test_exactly_the_fifteen_are_exposed(self):
        self.assertEqual({s["name"] for s in tools.SCHEMAS}, EXPECTED)

    def test_no_configuration_tool_is_reachable_by_the_model(self):
        names = {s["name"] for s in tools.SCHEMAS}
        self.assertEqual(names & FORBIDDEN, set(),
                         "config is the operator's; a tool here could redirect the archive")

    def test_every_schema_has_the_three_required_keys(self):
        for s in tools.SCHEMAS:
            self.assertEqual({"name", "description", "parameters"} - set(s), set(), s["name"])
            self.assertEqual(s["parameters"]["type"], "object", s["name"])
            self.assertIn("properties", s["parameters"], s["name"])

    def test_every_description_says_when_to_use_it(self):
        for s in tools.SCHEMAS:
            self.assertGreater(len(s["description"]), 40,
                               f"{s['name']}: a description the model cannot act on is a stub")

    def test_the_provider_exposes_them(self):
        self.assertEqual({s["name"] for s in MemoriesProvider().get_tool_schemas()}, EXPECTED)

    def test_required_parameters_are_declared(self):
        by_name = {s["name"]: s for s in tools.SCHEMAS}
        self.assertIn("information", by_name["memory_store"]["parameters"]["required"])
        self.assertIn("query", by_name["memory_recall"]["parameters"]["required"])
        self.assertIn("path", by_name["docs_index"]["parameters"]["required"])
        self.assertIn("id", by_name["memory_get"]["parameters"]["required"])

    def test_every_required_name_is_a_declared_property(self):
        """A `required` entry with no property is a schema the model cannot satisfy: it is
        told to send an argument that is never described."""
        for s in tools.SCHEMAS:
            params = s["parameters"]
            missing = set(params.get("required", [])) - set(params["properties"])
            self.assertEqual(missing, set(), f"{s['name']} requires undescribed {missing}")

    def test_every_property_carries_a_description(self):
        """The property descriptions are what the model reads to fill an argument in. A
        bare `{"type": "string"}` makes it guess."""
        for s in tools.SCHEMAS:
            for prop, spec in s["parameters"]["properties"].items():
                self.assertTrue(spec.get("description"), f"{s['name']}.{prop} has none")

    def test_the_drop_scope_says_what_omitting_it_does(self):
        """`docs_drop` is the one tool where omitting an argument REMOVES something.

        `scope` defaults to `all`, matching the CLI, so a `doc_id` drop reaches the
        PERMANENT library as well as the temporary archive. The default stays — the two hosts
        must agree — but a model must not discover it by touching the permanent archive, so
        the schema has to say it. The shared scope description used by search and list does
        not, which is why this reads the drop tool's own.
        """
        by_name = {s["name"]: s for s in tools.SCHEMAS}
        scope = by_name["docs_drop"]["parameters"]["properties"]["scope"]["description"]
        self.assertIn("default", scope.lower(), "the default has to be stated")
        self.assertIn("all", scope, "and it has to say WHICH default")
        self.assertIn("library", scope,
                      "and that the default reaches the permanent archive")
        self.assertIn("tmp", scope, "and how to avoid it")

    def test_the_schemas_survive_the_wire(self):
        """hermes serializes these into a chat-completions request. Anything that does not
        JSON-encode takes the WHOLE toolset down, not just this provider's part."""
        json.dumps(tools.SCHEMAS)

    def test_the_provider_hands_out_copies(self):
        """One caller mutating what it got must not edit the module constant. hermes
        normalizes and wraps these schemas, and a second session would inherit the edit."""
        got = MemoriesProvider().get_tool_schemas()
        got.pop()
        got[0]["parameters"]["properties"].clear()
        self.assertEqual({s["name"] for s in tools.SCHEMAS}, EXPECTED)
        self.assertTrue(all(s["parameters"]["properties"] for s in tools.SCHEMAS))


class TestDispatch(unittest.TestCase):
    def test_every_declared_tool_is_routed(self):
        """A schema the dispatcher does not know is a tool the model calls and gets an
        error for — worse than not offering it."""
        for name in EXPECTED:
            self.assertIn(name, tools.ROUTES, f"{name} is declared but not routed")

    def test_every_route_is_declared(self):
        """The other direction: a routed tool with no schema is unreachable code that
        looks like a feature."""
        self.assertEqual(set(tools.ROUTES), {s["name"] for s in tools.SCHEMAS})

    def test_an_unknown_tool_returns_json_not_an_exception(self):
        out = tools.dispatch("memory_teleport", {}, cfg=None)
        self.assertEqual(json.loads(out)["error"], "unknown tool: memory_teleport")

    def test_a_failure_comes_back_as_json_not_an_exception(self):
        """handle_tool_call must return a JSON string. A raise would surface to the user as
        a crashed turn instead of a result the model can react to."""
        cfg = core.Config(qdrant_url="", qdrant_api_key="", api_base_url="", api_key="",
                          embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                          memory_collection="", docs_collection="d", library_collection="l",
                          vector_size=1024)
        out = tools.dispatch("memory_recall", {"query": "anything"}, cfg=cfg)
        payload = json.loads(out)
        self.assertIn("error", payload)

    def test_a_missing_required_argument_is_reported_not_raised(self):
        out = tools.dispatch("memory_store", {}, cfg=None)
        self.assertIn("error", json.loads(out))

    def test_the_missing_argument_is_NAMED_so_the_model_can_retry(self):
        """An error that does not say which argument is missing costs another turn."""
        for name, missing in (("memory_store", "information"), ("memory_get", "id"),
                              ("memory_recall", "query"), ("docs_index", "path"),
                              ("docs_search", "query"), ("memory_find", "query"),
                              ("memory_update", "id"), ("memory_delete", "id"),
                              ("docs_keep", "path"), ("memory_store_many", "items"),
                              ("memory_search_collections", "query")):
            out = json.loads(tools.dispatch(name, {}, cfg=None))
            self.assertIn("error", out, name)
            self.assertIn(missing, out["error"], f"{name} does not name {missing}")

    def test_a_wrong_typed_argument_is_reported_not_raised(self):
        """An integer the model sent as text that cannot BE an integer: named, not raised.

        The other half of the claim — that `"3"` is accepted and honoured — is in
        TestMemoryTools, where a real archive can show the limit taking effect. It used to
        live here as `assertNotIn("error", ok if isinstance(ok, dict) else {})`, and
        `memory_find` returns a LIST, so that line asserted `"error" not in {}` and could
        never fail.
        """
        with unittest.mock.patch.object(core, "build_memory", lambda cfg, **kw: _store()[0]):
            bad = json.loads(tools.dispatch("memory_find", {"query": "x", "limit": "many"},
                                            cfg=a_config()))
        self.assertIn("error", bad)
        self.assertIn("limit", bad["error"])
        self.assertIn("must be an integer", bad["error"],
                      "the message has to say what was wrong, not only which argument")

    def test_an_unexpected_exception_is_still_a_json_error(self):
        """The failure the `except Exception` exists for: not a core error, not a bad
        argument, just something nobody predicted. It must reach the model as a result and
        not the host as a raise, or the turn dies."""
        class Exploding:
            def find(self, *a, **kw):
                raise RuntimeError("something nobody predicted")

        with unittest.mock.patch.object(core, "build_memory", lambda cfg, **kw: Exploding()):
            out = tools.dispatch("memory_find", {"query": "x"}, cfg=a_config())
        self.assertIsInstance(out, str)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn("RuntimeError", payload["error"],
                      "the model can only report a failure it is told the shape of")

    def test_no_arguments_at_all_does_not_explode(self):
        for args in (None, {}, ""):
            out = tools.dispatch("memory_list", args, cfg=None)
            self.assertIsInstance(out, str)
            self.assertIn("error", json.loads(out))

    def test_arguments_arriving_as_a_json_string_are_accepted(self):
        """Measured shape of the problem: some models emit the arguments object as a
        string. Refusing it costs a turn for nothing."""
        with unittest.mock.patch.object(core, "build_memory", lambda cfg, **kw: _store()[0]):
            out = json.loads(tools.dispatch("memory_store",
                                            '{"information": "a fact through a string"}',
                                            cfg=a_config()))
        self.assertEqual(out["status"], "created")

    def test_unparseable_arguments_say_what_was_expected(self):
        """A JSON parse error escaping to the catch-all reads "Expecting value: line 1
        column 1", which says nothing about the shape the model should have sent."""
        for junk in ("not json at all", "{unclosed", "information='a fact'"):
            out = json.loads(tools.dispatch("memory_store", junk, cfg=a_config()))
            self.assertIn("error", out, junk)
            self.assertIn("must be an object", out["error"])
            self.assertNotIn("Expecting value", out["error"])

    def test_arguments_that_parse_to_something_other_than_an_object_are_named(self):
        for not_an_object in ("[1, 2]", "42", '"a string"'):
            out = json.loads(tools.dispatch("memory_store", not_an_object, cfg=a_config()))
            self.assertIn("must be an object", out["error"], not_an_object)

    def test_an_unconfigured_host_says_so_instead_of_failing_on_None(self):
        out = json.loads(tools.dispatch("memory_find", {"query": "x"}, cfg=None))
        self.assertIn("error", out)
        self.assertNotIn("NoneType", out["error"],
                         "an AttributeError on None tells the model nothing it can act on")

    def test_the_provider_routes_through_dispatch(self):
        p = MemoriesProvider()
        p._cfg = a_config()          # see setUpModule: reaching the real config is fatal
        out = p.handle_tool_call("memory_teleport", {})
        self.assertIn("error", json.loads(out))

    def test_handle_tool_call_returns_a_STRING(self):
        """The ABC's contract (agent/memory_provider.py, :182 in v0.20.0 and :232 in the
        v0.20.1 now installed): "Must return a JSON string". A dict here fails at the host
        boundary, past every test of the handler's own logic."""
        p = MemoriesProvider()
        p._cfg = a_config()
        out = p.handle_tool_call("memory_teleport", {})
        self.assertIsInstance(out, str)
        json.loads(out)

    def test_every_tool_returns_a_string_even_when_it_fails(self):
        p = MemoriesProvider()
        p._cfg = a_config(qdrant_url="")      # every handler fails on the config
        for name in EXPECTED:
            out = p.handle_tool_call(name, {})
            self.assertIsInstance(out, str, name)
            self.assertIsInstance(json.loads(out), (dict, list), name)


class TestTheConfigAToolCallRunsAgainst(unittest.TestCase):
    """`MemoriesProvider._config()` — where a tool call gets its configuration.

    `is_available()` caches it and hermes calls that before initializing, so the fallback to
    a fresh `core.load()` covers a host that dispatches a tool without having asked. Untested,
    it could be deleted with the suite green, and the symptom would be every tool call
    answering "not configured" while recall kept working — a feature that looks like a model
    that stopped bothering.
    """

    def setUp(self):
        self.provider = MemoriesProvider()
        self.loads = []

    def _loading(self, cfg):
        """A `core.load` that records its calls. Overrides the module-level guard for this
        test only — deliberately, since this is the one behaviour that must call it."""
        def load():
            self.loads.append(1)

            return cfg

        return unittest.mock.patch.object(core, "load", load)

    def test_a_tool_call_with_no_cached_config_loads_it_itself(self):
        store, q = _store()
        with self._loading(a_config()), \
             unittest.mock.patch.object(core, "build_memory", lambda cfg, **kw: store):
            store.store("a fact that was already there")
            out = json.loads(self.provider.handle_tool_call("memory_list", {}))
        self.assertEqual(self.loads, [1], "the provider never read the configuration")
        self.assertNotIn("error", out)
        self.assertEqual(out["count"], 1)

    def test_the_configuration_is_read_once_and_cached(self):
        store, _ = _store()
        with self._loading(a_config()), \
             unittest.mock.patch.object(core, "build_memory", lambda cfg, **kw: store):
            self.provider.handle_tool_call("memory_list", {})
            self.provider.handle_tool_call("memory_list", {})
        self.assertEqual(self.loads, [1], "one config read per provider, not per tool call")

    def test_an_unconfigured_install_says_what_the_OPERATOR_has_to_do(self):
        """A ConfigError here is not the model's fault and it cannot fix it, so the message
        has to name the operator's action instead of the model's argument."""
        def raising():
            raise core.ConfigError("memory_collection is not configured (QCTX_...)")

        with unittest.mock.patch.object(core, "load", raising):
            out = json.loads(self.provider.handle_tool_call("memory_find", {"query": "x"}))
        self.assertIn("error", out)
        self.assertIn("qctx setup", out["error"])
        self.assertNotIn("NoneType", out["error"],
                         "an AttributeError on None tells the model nothing it can act on")
        self.assertIn("QCTX_", self.provider.unavailable_reason(),
                      "the operator's actual reason has to survive for whoever asks")


def _store(collection="mem"):
    q, emb = FakeVectorStore(), FakeEmbedder()
    q.ensure_collection(collection, emb.dim)

    return MemoryStore(q, emb, None, collection, emb.dim), q


class TestMemoryTools(unittest.TestCase):
    """The nine memory tools, against the real core with a fake store underneath.

    They route to the same `MemoryStore` methods the CLI's `cmd_memory_*` handlers call —
    these tests exist to catch a tool wired to the wrong one, which a schema test cannot
    see.
    """

    def setUp(self):
        self.store, self.q = _store()
        patch = unittest.mock.patch.object(core, "build_memory",
                                          lambda cfg, **kw: self.store)
        patch.start()
        self.addCleanup(patch.stop)
        self.cfg = a_config()

    def call(self, name, **args):
        out = tools.dispatch(name, args, cfg=self.cfg)
        self.assertIsInstance(out, str, f"{name} did not return a string")

        return json.loads(out)

    def test_store_writes_the_fact_and_returns_its_id(self):
        res = self.call("memory_store", information="a durable fact about pagination")
        self.assertEqual(res["status"], "created")
        point = self.q.get_point("mem", res["id"])
        self.assertEqual(point["payload"]["document"], "a durable fact about pagination")

    def test_store_assembles_metadata_from_the_object_and_the_shortcuts(self):
        res = self.call("memory_store", information="a fact",
                        metadata={"type": "old", "custom": 1}, type="reference",
                        project="memories-plugin")
        self.assertEqual(self.q.get_point("mem", res["id"])["payload"]["metadata"],
                         {"type": "reference", "custom": 1, "project": "memories-plugin"})

    def test_an_empty_fact_is_refused_by_the_core_and_reported_as_json(self):
        self.assertIn("error", self.call("memory_store", information="   "))

    def test_store_many_is_one_call_and_all_or_nothing(self):
        res = self.call("memory_store_many",
                        items=[{"information": "fact one"},
                               {"information": "fact two", "metadata": {"type": "x"}}])
        self.assertEqual(res["count"], 2)
        self.assertEqual(len(self.q.collections["mem"]["points"]), 2)

    def test_store_many_refuses_a_bad_batch_without_writing_half_of_it(self):
        self.assertIn("error", self.call("memory_store_many",
                                        items=[{"information": "ok"}, {"information": " "}]))
        self.assertEqual(len(self.q.collections["mem"]["points"]), 0)

    def test_store_many_accepts_the_array_as_a_json_string(self):
        res = self.call("memory_store_many", items='[{"information": "through a string"}]')
        self.assertEqual(res["count"], 1)

    def test_store_many_refuses_something_that_is_not_a_list(self):
        """The guard has to produce an ACTIONABLE message, not just an error.

        Asserting only `"error" in …` could not tell the guard from what happens without it:
        the core iterates the dict's keys and raises `AttributeError: 'str' object has no
        attribute 'get'`, which tells the model nothing about the shape it should have sent.
        """
        out = self.call("memory_store_many", items={"information": "not in an array"})
        self.assertIn("error", out)
        self.assertIn("must be an array", out["error"])
        self.assertIn("dict", out["error"], "say what arrived, so the model can see the gap")
        self.assertNotIn("AttributeError", out["error"])

    def test_find_returns_the_dense_hits(self):
        self.call("memory_store", information="pagination cursor in the connector poll")
        hits = self.call("memory_find", query="pagination cursor")
        self.assertTrue(hits)
        self.assertIn("pagination", hits[0]["document"])

    def test_find_honours_its_limit(self):
        for i in range(4):
            self.call("memory_store", information=f"pagination fact number {i}")
        self.assertEqual(len(self.call("memory_find", query="pagination fact", limit=2)), 2)

    def test_an_integer_sent_as_a_string_is_honoured_and_not_merely_tolerated(self):
        """The accepting half of the coercion claim, over a real archive: `"2"` has to
        actually LIMIT the search, not just avoid an error. Asserted against the unlimited
        result, so a coercion that silently fell back to the default would show."""
        for i in range(4):
            self.call("memory_store", information=f"pagination fact number {i}")
        self.assertEqual(len(self.call("memory_find", query="pagination fact")), 4)
        hits = self.call("memory_find", query="pagination fact", limit="2")
        self.assertIsInstance(hits, list)
        self.assertEqual(len(hits), 2, "the string '2' was not honoured as the limit")

    def test_recall_reports_the_trail_next_to_the_hits(self):
        """The tool answers with the same two halves the CLI's `--json` does: the hits and
        the trail that says whether they were judged or merely near."""
        self.call("memory_store", information="a durable fact about pagination cursors")
        res = self.call("memory_recall", query="pagination cursors")
        self.assertEqual(set(res), {"info", "hits"})
        self.assertIn("best_dense", res["info"])
        self.assertIsNone(res["info"]["scored"], "the items travel under 'hits'")

    def test_get_returns_the_record_and_a_missing_id_is_not_an_error(self):
        mid = self.call("memory_store", information="a fact to fetch")["id"]
        self.assertEqual(self.call("memory_get", id=mid)["document"], "a fact to fetch")
        self.assertEqual(self.call("memory_get", id="00000000-0000-0000-0000-000000000000")
                         ["status"], "not_found")

    def test_update_rewrites_the_text_and_reembeds(self):
        mid = self.call("memory_store", information="the original text")["id"]
        res = self.call("memory_update", id=mid, information="the corrected text")
        self.assertEqual(res["status"], "updated")
        self.assertTrue(res["reembedded"])
        self.assertEqual(self.q.get_point("mem", mid)["payload"]["document"],
                         "the corrected text")

    def test_update_without_metadata_does_not_wipe_it(self):
        """The trap the CLI already avoids with `or None`: an assembled `{}` means "set the
        metadata to nothing", while None means "leave it alone". A tool call that only
        fixes the text must not strip the labels."""
        mid = self.call("memory_store", information="text",
                        metadata={"type": "reference", "project": "p"})["id"]
        self.call("memory_update", id=mid, information="corrected text")
        self.assertEqual(self.q.get_point("mem", mid)["payload"]["metadata"],
                         {"type": "reference", "project": "p"})

    def test_update_can_fix_a_label_without_touching_the_text(self):
        mid = self.call("memory_store", information="text that must survive",
                        metadata={"type": "wrong"})["id"]
        res = self.call("memory_update", id=mid, type="reference")
        self.assertFalse(res["reembedded"], "unchanged text must not pay an embedding call")
        payload = self.q.get_point("mem", mid)["payload"]
        self.assertEqual(payload["document"], "text that must survive")
        self.assertEqual(payload["metadata"]["type"], "reference")

    def test_delete_removes_the_record(self):
        mid = self.call("memory_store", information="a fact that will go")["id"]
        self.assertEqual(self.call("memory_delete", id=mid)["status"], "deleted")
        self.assertIsNone(self.q.get_point("mem", mid))

    def test_list_pages_the_archive(self):
        for i in range(3):
            self.call("memory_store", information=f"fact {i}")
        res = self.call("memory_list", limit=2)
        self.assertEqual(res["count"], 2)
        self.assertEqual(len(res["memories"]), 2)

    def test_search_collections_is_read_only_and_reports_what_it_skipped(self):
        """It searches OTHER systems' archives. A collection built by another embedding
        model is skipped and reported, because reading it returns plausible nonsense."""
        self.q.ensure_collection("someone_elses", 8)
        self.q.upsert("someone_elses", [{"id": 1, "vector": [1.0] * 8,
                                         "payload": {"text": "a foreign document"}}])
        self.q.ensure_collection("wrong_dimension", 16)
        with unittest.mock.patch.object(core, "build_qdrant", lambda cfg, **kw: self.q), \
             unittest.mock.patch.object(core, "build_embedder",
                                        lambda cfg, **kw: FakeEmbedder()):
            res = self.call("memory_search_collections", query="a foreign document")
        self.assertIn("someone_elses", res["searched"])
        self.assertTrue(any(s["collection"] == "wrong_dimension" for s in res["skipped"]),
                        "a mismatched dimension has to be reported, never read silently")

    def test_search_collections_refuses_a_collections_argument_that_is_not_a_list(self):
        """`collections` reaches `qdrant.list_collections()`'s place in the core, which
        iterates it: a dict would search collections named after its KEYS and a bare string
        one letter at a time, both reporting "not found" for archives nobody asked about.
        Named here instead, before anything is searched."""
        for bad in ({"name": "x"}, "42"):        # a dict, and a JSON scalar in a string
            out = self.call("memory_search_collections", query="x", collections=bad)
            self.assertIn("error", out, bad)
            self.assertIn("must be an array", out["error"])
        # A single name in a proper array still works — the guard cannot pass by refusing all.
        with unittest.mock.patch.object(core, "build_qdrant", lambda cfg, **kw: self.q), \
             unittest.mock.patch.object(core, "build_embedder",
                                        lambda cfg, **kw: FakeEmbedder()):
            self.call("memory_store", information="a fact to find across archives")
            ok = self.call("memory_search_collections", query="a fact to find",
                           collections=["mem"])
        self.assertEqual(ok["searched"], ["mem"])


class TestDocsTools(unittest.TestCase):
    """The six document tools, against the real `DocIndex` with a fake store underneath."""

    def setUp(self):
        q, emb = FakeVectorStore(), FakeEmbedder()
        self.q = q
        self.idx = DocIndex(q, emb, None, "tmp", "lib", emb.dim)
        patch = unittest.mock.patch.object(core, "build_docs", lambda cfg: self.idx)
        patch.start()
        self.addCleanup(patch.stop)
        self.cfg = a_config()
        self.path = os.path.join(tempfile.mkdtemp(), "document.md")
        Path(self.path).write_text("# Title\n\nthe body of the document, about pagination\n")

    def call(self, name, **args):
        out = tools.dispatch(name, args, cfg=self.cfg)
        self.assertIsInstance(out, str, f"{name} did not return a string")

        return json.loads(out)

    def test_index_is_temporary_and_carries_the_expiry_it_was_given(self):
        res = self.call("docs_index", path=self.path, ttl="30m")
        self.assertEqual(res["scope"], "tmp")
        self.assertIsNotNone(res["expires_at"])
        point = list(self.q.collections["tmp"]["points"].values())[0]
        self.assertIn("expires_at_ts", point["payload"])
        self.assertEqual(point["payload"]["metadata"]["ttl_seconds"], 1800,
                         "the TTL asked for has to be the TTL stored")

    def test_index_defaults_to_the_same_24h_the_cli_defaults_to(self):
        res = self.call("docs_index", path=self.path)
        point = list(self.q.collections["tmp"]["points"].values())[0]
        self.assertEqual(point["payload"]["metadata"]["ttl_seconds"], 24 * 3600)
        self.assertIsNotNone(res["expires_at"])

    def test_an_invalid_ttl_is_reported_not_raised(self):
        out = self.call("docs_index", path=self.path, ttl="soon")
        self.assertIn("error", out)
        self.assertIn("soon", out["error"])

    def test_keep_is_permanent(self):
        res = self.call("docs_keep", path=self.path)
        self.assertEqual(res["scope"], "library")
        self.assertIsNone(res["expires_at"], "the library does not expire, by construction")

    def test_a_missing_file_is_reported_not_raised(self):
        out = self.call("docs_keep", path="/nonexistent/nothing.md")
        self.assertIn("error", out)

    def test_search_returns_the_chunk_and_the_trail(self):
        self.call("docs_keep", path=self.path)
        res = self.call("docs_search", query="pagination body document")
        self.assertEqual(set(res), {"info", "hits"})
        self.assertTrue(res["hits"])
        self.assertIn("pagination", res["hits"][0]["text"])
        self.assertEqual(res["hits"][0]["scope"], "library")

    def test_search_can_be_narrowed_to_one_document(self):
        kept = self.call("docs_keep", path=self.path)
        other = os.path.join(tempfile.mkdtemp(), "other.md")
        Path(other).write_text("# Other\n\nan unrelated document about pagination too\n")
        self.call("docs_keep", path=other)
        res = self.call("docs_search", query="pagination body document",
                        doc_id=kept["doc_id"])
        self.assertTrue(res["hits"])
        self.assertTrue(all(h["path"] == kept["path"] for h in res["hits"]))

    def test_list_shows_what_is_indexed_in_both_archives(self):
        self.call("docs_keep", path=self.path)
        self.call("docs_index", path=self.path, ttl="1h")
        scopes = {d["scope"] for d in self.call("docs_list")}
        self.assertEqual(scopes, {"tmp", "library"})
        self.assertEqual([d["scope"] for d in self.call("docs_list", scope="library")],
                         ["library"])

    def test_refresh_reindexes_what_changed_on_disk(self):
        self.call("docs_keep", path=self.path)
        self.assertEqual([r["action"] for r in self.call("docs_refresh")], ["ok"])
        Path(self.path).write_text("# Title\n\na genuinely different body about cursors\n")
        self.assertEqual([r["action"] for r in self.call("docs_refresh")], ["reindexed"])

    def test_drop_removes_one_document(self):
        kept = self.call("docs_keep", path=self.path)
        res = self.call("docs_drop", doc_id=kept["doc_id"], scope="library")
        self.assertEqual(res["status"], "removed")
        self.assertEqual(len(self.q.collections["lib"]["points"]), 0)

    def test_drop_with_no_target_is_refused_rather_than_a_silent_no_op(self):
        self.call("docs_keep", path=self.path)
        self.assertIn("error", self.call("docs_drop"))
        self.assertTrue(self.q.collections["lib"]["points"], "nothing was to be removed")

    def test_purge_tmp_leaves_the_library_alone(self):
        self.call("docs_keep", path=self.path)
        self.call("docs_index", path=self.path, ttl="1h")
        res = self.call("docs_drop", purge_tmp=True)
        self.assertEqual(res["status"], "purged")
        self.assertNotIn("tmp", self.q.collections)
        self.assertTrue(self.q.collections["lib"]["points"],
                        "the library is out of reach of any cleanup, by construction")

    def test_a_boolean_sent_as_a_string_is_understood(self):
        self.call("docs_index", path=self.path, ttl="1h")
        self.assertEqual(self.call("docs_drop", purge_tmp="true")["status"], "purged")

    def test_a_boolean_that_means_nothing_is_reported(self):
        """Asserted on the REASON, not on the argument name.

        Naming the argument was not enough to pin this: with `_bool` swallowing `"maybe"` as
        False, `docs_drop` gets no target at all and `drop_request` refuses with "give a
        doc_id, purge_tmp or expired" — which contains "purge_tmp" and satisfied the old
        assertion. The model would have been told to supply an argument it did supply.
        """
        out = self.call("docs_drop", purge_tmp="maybe")
        self.assertIn("error", out)
        self.assertIn("must be true or false", out["error"])
        self.assertIn("maybe", out["error"], "quote the value back; the model sent it")

    def test_a_scope_the_archive_does_not_have_is_reported_with_the_valid_ones(self):
        """Driven with the fake archive in place, so the ONLY thing that can produce an
        error is the guard being tested. Pointed at an unreachable Qdrant instead, this
        would pass on the connection failure and prove nothing."""
        self.assertEqual([d["scope"] for d in self.call("docs_list", scope="tmp")], [])
        out = self.call("docs_list", scope="everything")
        self.assertIn("error", out)
        for valid in ("tmp", "library"):
            self.assertIn(valid, out["error"], "an invalid choice must list the valid ones")

    def test_refresh_refuses_the_scope_the_CLI_refuses(self):
        """`docs refresh` takes library or tmp, never `all`: `DocIndex.refresh` BRANCHES on
        the scope and would reindex a LIBRARY document as a temporary one, silently giving
        a permanent document an expiry. argparse's `choices` guards the CLI; this guards the
        tool. Both accepted scopes are exercised too, so the guard cannot pass by refusing
        everything."""
        self.call("docs_keep", path=self.path)
        self.assertEqual([r["action"] for r in self.call("docs_refresh", scope="library")],
                         ["ok"])
        self.assertEqual(self.call("docs_refresh", scope="tmp"), [])
        out = self.call("docs_refresh", scope="all")
        self.assertIn("error", out)
        self.assertIn("scope", out["error"])
        self.assertIsNone(list(self.q.collections["lib"]["points"].values())[0]["payload"]
                          .get("expires_at_ts"),
                          "a library document must never come back with an expiry")

    def test_expired_sweeps_the_temporary_archive(self):
        self.call("docs_index", path=self.path, ttl="1h")
        res = self.call("docs_drop", expired=True)
        self.assertEqual(res["status"], "swept")


class TestTheWiringFromOutside(unittest.TestCase):
    """Proof that the tools are reachable through the door hermes actually opens.

    A test that imports `hosts.hermes.tools` proves the module works; it does not prove
    anything reaches it. These drive the PROVIDER, loaded through the REAL loader for a
    user-installed memory provider, against a real install symlink.

    WHICH LOADER, because getting this wrong is how a broken production load passed here
    once already. `plugin.yaml` declares `kind: exclusive`, and
    `hermes_cli/plugins.py` explicitly SKIPS exclusive plugins ("handled by category
    discovery" — :1415-1428 in v0.20.0, :3907-3917 in the v0.20.1 now installed), so
    `_load_local_module` — which an earlier version of this test emulated — never loads a
    memory provider. The real path is
    `plugins/memory/__init__.py::_load_provider_from_dir` (v0.20.0 lines 218-327, v0.20.1
    lines 419-542), and it
    does one thing the other does not: it PRE-EXECS every sibling `*.py` before
    `__init__.py`, registering each in `sys.modules` first and swallowing failures at
    `logger.debug`. Measured consequence when `tools.py` could not import `core` on its own:
    the broken shell stayed in `sys.modules`, the package's `from . import tools` succeeded
    and returned it, `bind_tuning` raised AttributeError, and the provider did not load at
    all — no recall, no checkpoint, one debug line.
    """

    #: Mirrors `_load_provider_from_dir` for a USER-installed provider: synthetic parent
    #: package (`_register_synthetic_package`, :42-57), the module registered in
    #: `sys.modules` before exec, then the sibling pre-exec loop (:277-297), then
    #: `exec_module` on `__init__.py`, then `register(ctx)` (:305-311). Two deliberate
    #: departures, both toward strictness: the sibling failures are RECORDED in
    #: `SIBLING_ERRORS` instead of being swallowed, and the `__init__.py` failure is left to
    #: raise instead of being logged — the swallowing is precisely what made the production
    #: breakage invisible, so a test must not reproduce it.
    USER_NAMESPACE = "_hermes_user_memory"

    #: Everything up to and including the sibling pre-exec. Split out so the sibling test
    #: can stop HERE and report which sibling failed, instead of inheriting the
    #: AttributeError that the failure causes two steps later in `__init__.py`.
    LOADER_PRE = (
        "import importlib.machinery, importlib.util, json, pathlib, sys\n"
        "USER_NS = %(ns)r\n"
        "provider_dir = %(dir)r\n"
        "module_name = USER_NS + '.memories'\n"
        "ns_spec = importlib.machinery.ModuleSpec(USER_NS, None, is_package=True)\n"
        "ns_spec.submodule_search_locations = []\n"
        "sys.modules[USER_NS] = importlib.util.module_from_spec(ns_spec)\n"
        "spec = importlib.util.spec_from_file_location(module_name,\n"
        "    provider_dir + '/__init__.py', submodule_search_locations=[provider_dir])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[module_name] = module\n"
        "SIBLING_ERRORS = {}\n"
        "SIBLINGS_SEEN = []\n"
        "for sub in sorted(pathlib.Path(provider_dir).glob('*.py')):\n"
        "    if sub.name == '__init__.py':\n"
        "        continue\n"
        "    full = module_name + '.' + sub.stem\n"
        "    if full in sys.modules:\n"
        "        continue\n"
        "    SIBLINGS_SEEN.append(sub.name)\n"
        "    sub_spec = importlib.util.spec_from_file_location(full, str(sub))\n"
        "    sub_mod = importlib.util.module_from_spec(sub_spec)\n"
        "    sys.modules[full] = sub_mod\n"
        "    try:\n"
        "        sub_spec.loader.exec_module(sub_mod)\n"
        "    except Exception as exc:\n"
        "        SIBLING_ERRORS[sub.name] = '%%s: %%s' %% (type(exc).__name__, exc)\n"
    )

    LOADER_INIT = (
        "spec.loader.exec_module(module)\n"
        "class Collector:\n"
        "    provider = None\n"
        "    def register_memory_provider(self, p):\n"
        "        self.provider = p\n"
        "collector = Collector()\n"
        "module.register(collector)\n"
        "provider = collector.provider\n"
    )

    def _run(self, body: str, *, siblings_only: bool = False) -> str:
        """Load the adapter through the real loader in a subprocess, then run `body`.

        `provider`, `module` and `SIBLING_ERRORS` are in scope for the body — all but
        `SIBLING_ERRORS` only when `siblings_only` is false.

        The subprocess inherits no PYTHONPATH and runs from an EMPTY directory. Both matter:
        the adapter has to find `core` by itself, from the symlink, and `python3 -c` puts the
        working directory on `sys.path` — so running from the repo root made `import core`
        succeed for free and this very test passed with the bootstrap deleted (measured).
        hermes runs from wherever the user launched it, never from this repo.

        The child gets `QCTX_CONFIG` pointing at a THROWAWAY config. `setUpModule`'s guard
        against `core.load()` is in-process and cannot reach here, so without this the child
        resolves the operator's real config — measured: it loaded `claude_memory` and the
        permanent `memories_docs_library`. Nothing in these bodies touches an archive today,
        but `docs_refresh` takes no required arguments, so one future body naming a real tool
        would reindex the user's permanent library with the suite green. That is the exact
        hole the in-process guard exists to close, so it gets closed on both sides.
        """
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        home = root / "plugins" / "memory"
        home.mkdir(parents=True)
        link = home / "memories"
        link.symlink_to(REPO / "hosts" / "hermes")
        elsewhere = Path(tempfile.mkdtemp())      # nothing importable in here
        self.addCleanup(shutil.rmtree, elsewhere, ignore_errors=True)
        throwaway = root / "config.json"
        throwaway.write_text(json.dumps({
            "qdrant_url": "http://127.0.0.1:1",
            "memory_collection": "throwaway_never_real",
            "docs_collection": "throwaway_never_real_tmp",
            "library_collection": "throwaway_never_real_library",
        }))
        loader = self.LOADER_PRE if siblings_only else self.LOADER_PRE + self.LOADER_INIT
        script = loader % {"ns": self.USER_NAMESPACE, "dir": str(link)} + body
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        env["QCTX_CONFIG"] = str(throwaway)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                             cwd=str(elsewhere), env=env)
        self.assertEqual(out.returncode, 0, out.stderr)

        return out.stdout

    def test_the_real_loader_produces_a_provider_at_all(self):
        """The regression this class exists for: `_load_provider_from_dir` returning None
        means hermes has NO memory provider, and says so only at debug level."""
        out = self._run("print(type(provider).__name__)\n"
                        "print(provider.name)\n")
        kind, name = out.strip().splitlines()[:2]
        self.assertEqual(kind, "MemoriesProvider")
        self.assertEqual(name, "memories")

    def test_no_sibling_module_fails_its_pre_exec(self):
        """Each sibling is exec'd on its own, BEFORE `__init__.py` has bootstrapped
        anything. A sibling that cannot stand alone leaves a broken shell in `sys.modules`
        that the package's own relative import then picks up in silence.

        Asserts what was SEEN as well as what failed. `SIBLING_ERRORS == {}` alone cannot
        tell "every sibling exec'd cleanly" from "there was no sibling to exec" — measured:
        against a directory holding only `__init__.py`, the same loop prints `{}` and the
        empty-dict assertion passes. So the test would survive `tools.py` disappearing,
        which is the one thing it is here to notice.
        """
        out = self._run("print(json.dumps(SIBLING_ERRORS))\n"
                        "print(json.dumps(SIBLINGS_SEEN))\n", siblings_only=True)
        errors, seen = (json.loads(line) for line in out.strip().splitlines()[:2])
        self.assertIn("tools.py", seen,
                      "the pre-exec loop saw no tools.py — an empty SIBLING_ERRORS below "
                      "would then mean 'nothing was tried', not 'nothing failed'")
        self.assertEqual(errors, {},
                         "a sibling failed to exec standalone; the shell it left behind is "
                         "what the package's `from . import` will hand back")

    def test_a_provider_loaded_the_way_hermes_loads_it_offers_the_fifteen(self):
        out = self._run("print(json.dumps([s['name'] for s in provider.get_tool_schemas()]))\n")
        self.assertEqual(set(json.loads(out)), EXPECTED)

    def test_the_setup_wizard_surface_survives_the_real_loader_too(self):
        """The config wizard's two methods, driven through the loader hermes really uses.

        They are covered elsewhere by direct import, which proves they work — it does not
        prove hermes can reach them. That distinction is not academic here: the Critical this
        class exists for was code that behaved correctly on import and could not be loaded by
        `_load_provider_from_dir` at all, and the tool tests above would not have caught the
        same breakage in `get_config_schema`/`save_config` because they never call them.

        `save_config` writes through `core.save` to whatever `core.load` reads, so this would
        target the operator's REAL config were it not for `_run` handing the child a
        throwaway `QCTX_CONFIG`. The round-trip is asserted from `core.load()` inside the
        same child, which is the only way to prove the two halves agree on the file.
        """
        out = self._run(
            "keys = [f['key'] for f in provider.get_config_schema()]\n"
            "secret = sorted(f['key'] for f in provider.get_config_schema()\n"
            "                if f.get('secret'))\n"
            "provider.save_config({'memory_collection': 'loader_probe_collection'},\n"
            "                     hermes_home=provider_dir)\n"
            "import core\n"
            "print(json.dumps({'keys': keys, 'secret': secret,\n"
            "                  'read_back': core.load().memory_collection}))\n")
        got = json.loads(out)
        self.assertIn("memory_collection", got["keys"],
                      "the wizard cannot offer a field it does not expose")
        self.assertEqual(got["secret"], ["api_key", "qdrant_api_key"],
                         "the two API keys must still be the secret ones through this loader")
        self.assertEqual(got["read_back"], "loader_probe_collection",
                         "save_config wrote somewhere core.load() does not read — the two "
                         "hosts would then have separate configurations")

    def test_it_dispatches_through_the_provider_and_answers_with_a_json_string(self):
        out = self._run(
            "r = provider.handle_tool_call('memory_teleport', {})\n"
            "print(json.dumps({'type': type(r).__name__, 'payload': json.loads(r)}))\n")
        answer = json.loads(out)
        self.assertEqual(answer["type"], "str", "the host boundary wants a JSON string")
        self.assertIn("error", answer["payload"])

    def test_the_tools_module_is_reached_through_the_package_not_by_luck(self):
        """`tools` has to be an attribute of the loaded provider module, carrying the real
        schemas. If the wiring were missing, `get_tool_schemas` could still be hand-written
        and the module would sit there dead — which is the failure this asserts against."""
        out = self._run("print(module.tools.__name__)\n"
                        "print(len(module.tools.SCHEMAS))\n"
                        "print(module.tools.__file__)\n")
        name, count, file_ = out.strip().splitlines()[:3]
        self.assertIn("tools", name)
        self.assertEqual(int(count), 15)
        self.assertTrue(file_.endswith("tools.py"),
                        "an empty shell registered by the pre-exec has no file of its own")


class TestTheRealHostAcceptsThem(unittest.TestCase):
    """Read the acceptance rules off the INSTALL, not off the published documentation.

    Both checks here guard SILENT drops. hermes normalizes every schema and skips the ones
    it cannot resolve a name from; it also refuses any tool whose name shadows a built-in,
    logging a warning nobody reads and leaving the model without the tool. Neither failure
    is visible from inside this repo, so the rules are imported from the install.
    """

    @unittest.skipUnless(HERMES_INSTALL.exists(), "hermes-agent not installed here")
    def test_the_hosts_own_normalizer_resolves_every_schema(self):
        script = (
            "import json, sys\n"
            "sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "from agent.memory_manager import normalize_tool_schema\n"
            "from hosts.hermes import tools\n"
            "print(json.dumps([normalize_tool_schema(s) is not None and\n"
            "                  normalize_tool_schema(s)['name'] for s in tools.SCHEMAS]))\n"
        ) % (str(REPO), str(HERMES_INSTALL))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(set(json.loads(out.stdout)), EXPECTED,
                         "a schema the host cannot resolve a name from is skipped silently")

    @unittest.skipUnless(HERMES_INSTALL.exists(), "hermes-agent not installed here")
    def test_no_tool_name_shadows_a_built_in(self):
        """A provider tool named like a core tool is rejected at registration (#40466 in
        the host's own comment): built-ins always win, and the model never sees ours."""
        script = (
            "import json, sys\n"
            "sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "from toolsets import _HERMES_CORE_TOOLS\n"
            "from hosts.hermes import tools\n"
            "names = {s['name'] for s in tools.SCHEMAS}\n"
            "print(json.dumps(sorted(names & set(_HERMES_CORE_TOOLS))))\n"
        ) % (str(REPO), str(HERMES_INSTALL))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), [],
                         "these names collide with hermes' own tools and would be dropped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
