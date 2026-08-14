"""Tests for configuration and for the re-rank score scale.

Both things here are SILENT-failure traps — they raise nothing, they just deliver a
worse result. That is why they have tests: it is the only way to notice.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as cfgmod
from core.reranking import normalize_scores, sigmoid


class TestPrecedence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file_path = Path(self.tmp.name) / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_when_there_is_nothing(self):
        cfg = cfgmod.load(self.file_path, env={})
        self.assertEqual(cfg.embed_model, "bge-m3")
        self.assertEqual(cfg.vector_size, 1024)
        self.assertEqual(cfg.memory_collection, "", "memory starts out empty on purpose")

    def test_file_beats_default(self):
        self.file_path.write_text(json.dumps({"embed_model": "another-model"}))
        self.assertEqual(cfgmod.load(self.file_path, env={}).embed_model, "another-model")

    def test_env_beats_file(self):
        self.file_path.write_text(json.dumps({"embed_model": "from-file"}))
        cfg = cfgmod.load(self.file_path, env={"QCTX_EMBED_MODEL": "from-env"})
        self.assertEqual(cfg.embed_model, "from-env")

    def test_legacy_alias_is_accepted(self):
        cfg = cfgmod.load(self.file_path, env={"SERVER_BASE_URL": "http://x/v1",
                                             "QDRANT_SERVICE_API_KEY": "k"})
        self.assertEqual(cfg.api_base_url, "http://x/v1")
        self.assertEqual(cfg.qdrant_api_key, "k")

    def test_canonical_name_beats_the_legacy_one(self):
        cfg = cfgmod.load(self.file_path, env={"QCTX_QDRANT_URL": "canonical",
                                             "QDRANT_URL": "legacy"})
        self.assertEqual(cfg.qdrant_url, "canonical")

    def test_save_preserves_the_other_keys(self):
        cfgmod.save({"embed_model": "a"}, self.file_path)
        cfgmod.save({"memory_collection": "b"}, self.file_path)
        data = json.loads(self.file_path.read_text())
        self.assertEqual(data["embed_model"], "a")
        self.assertEqual(data["memory_collection"], "b")

    def test_save_refuses_an_unknown_key(self):
        with self.assertRaises(cfgmod.ConfigError):
            cfgmod.save({"made_up_key": 1}, self.file_path)

    def test_save_refuses_a_secret_and_points_at_the_environment(self):
        """A secret in a text file ends up in backups and in dotfile sync."""
        for field in ("qdrant_api_key", "api_key"):
            with self.assertRaises(cfgmod.ConfigError) as ctx:
                cfgmod.save({field: "secret-value"}, self.file_path)
            self.assertIn("QCTX_", str(ctx.exception), "the message has to say WHERE to put it")
            self.assertNotIn("secret-value", str(ctx.exception), "the value must not appear even in the error")
        self.assertFalse(self.file_path.exists(), "nothing was written")

    def test_save_of_an_ordinary_field_still_works(self):
        cfgmod.save({"memory_collection": "x"}, self.file_path)
        self.assertEqual(cfgmod.read_file(self.file_path)["memory_collection"], "x")


class TestTheProductionCollectionDefaultsArePinned(unittest.TestCase):
    """HAZARD, stated so it cannot be changed by accident: with an EMPTY config and no
    `QCTX_*` in the environment, the document tools reach the user's REAL archives.
    `docs_collection` defaults to `memories_docs_tmp` and `library_collection` to
    `memories_docs_library` — both live collections on this machine, the permanent one
    holding kept reference material.

    That IS the intended default: the two hosts share one archive by the user's decision, so
    nothing here changes it. What was missing is that nothing recorded it — mutating either
    name survived the whole suite, so neither the names nor the fact that an un-injected
    Config reaches production was held anywhere. Every test that walks the tool surface has to
    inject a Config for exactly this reason — see `tests/test_hermes_tools.py::setUpModule`,
    which guards it.

    The no-required-arguments path to a production collection is `docs_refresh`: its `scope`
    DEFAULTS to "library", so a bare call reads and rewrites the operator's permanent archive
    — and it is the tool carrying the parked delete-before-embed hazard, which makes it the
    worst one to reach by accident. NOT `docs_drop`, as an earlier version of this docstring
    said: measured, a bare `docs_drop` raises `DocsError("nothing to drop: give a doc_id,
    purge_tmp or expired")` and touches nothing. Its danger is a different one, already
    documented on the schema — a `doc_id` with no `scope` defaults to "all" and so reaches
    the library too.

    `memory_collection` is the counter-example and it must stay one: no default, and a read
    that raises with instructions instead of silently searching some other collection.
    """

    #: A path that does not exist, so `read_file` returns {} and only the DEFAULTS remain.
    #: `env={}` closes the other door — the real environment is never consulted here.
    def _bare(self):
        missing = Path(tempfile.mkdtemp()) / "no-config-here.json"

        return cfgmod.load(missing, env={})

    def test_the_document_defaults_are_the_users_real_production_archives(self):
        cfg = self._bare()
        self.assertEqual(cfg.docs_collection, "memories_docs_tmp")
        self.assertEqual(cfg.library_collection, "memories_docs_library")
        # And they resolve, i.e. the defaults are usable as-is: this is what makes an
        # un-injected Config reach production instead of failing.
        self.assertEqual(cfg.require_docs_collection(), "memories_docs_tmp")
        self.assertEqual(cfg.require_library_collection(), "memories_docs_library")

    def test_the_memory_collection_has_no_default_and_says_what_to_do(self):
        cfg = self._bare()
        self.assertEqual(cfg.memory_collection, "",
                         "a defaulted memory collection would make a misconfigured install "
                         "search the wrong archive and report no precedent")
        with self.assertRaises(cfgmod.ConfigError) as ctx:
            cfg.require_memory_collection()
        self.assertIn("config set memory-collection", str(ctx.exception))

    def test_the_defaults_table_and_a_bare_load_cannot_drift_apart(self):
        cfg = self._bare()
        for field in ("docs_collection", "library_collection", "memory_collection"):
            self.assertEqual(getattr(cfg, field), cfgmod.DEFAULTS[field], field)


class TestDerivedUrl(unittest.TestCase):
    def _config(self, **kw):
        base = dict(qdrant_url="", qdrant_api_key="", api_base_url="", api_key="",
                    embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                    memory_collection="", docs_collection="d", library_collection="l",
                    vector_size=1024)
        base.update(kw)

        return cfgmod.Config(**base)

    def test_full_url_takes_priority(self):
        cfg = self._config(embed_url="http://direct/v1/embeddings", api_base_url="http://base/v1")
        self.assertEqual(cfg.resolved_embed_url(), "http://direct/v1/embeddings")

    def test_derives_from_the_base_when_there_is_no_full_url(self):
        cfg = self._config(api_base_url="http://base/v1/")
        self.assertEqual(cfg.resolved_embed_url(), "http://base/v1/embeddings")
        self.assertEqual(cfg.resolved_rerank_url(), "http://base/v1/rerank")

    def test_raises_when_neither_is_set(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config().resolved_embed_url()


class TestCollectionCollision(unittest.TestCase):
    """Every collision degrades silently, so the guard has to be hard."""

    def _config(self, mem, docs, lib):
        return cfgmod.Config(qdrant_url="q", qdrant_api_key="", api_base_url="b", api_key="",
                             embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                             memory_collection=mem, docs_collection=docs,
                             library_collection=lib, vector_size=1024)

    def test_documents_in_the_memory_collection_is_refused(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config("same", "same", "lib").require_docs_collection()

    def test_library_in_the_temporary_collection_is_refused(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config("mem", "identical", "identical").require_library_collection()

    def test_library_in_the_memory_collection_is_refused(self):
        with self.assertRaises(cfgmod.ConfigError):
            self._config("identical", "docs", "identical").require_library_collection()

    def test_three_distinct_names_pass(self):
        cfg = self._config("mem", "docs", "lib")
        self.assertEqual(cfg.require_docs_collection(), "docs")
        self.assertEqual(cfg.require_library_collection(), "lib")
        self.assertEqual(cfg.require_memory_collection(), "mem")

    def test_unconfigured_memory_raises_with_instructions(self):
        with self.assertRaises(cfgmod.ConfigError) as ctx:
            self._config("", "docs", "lib").require_memory_collection()
        self.assertIn("collections list", str(ctx.exception))


class TestScoreScale(unittest.TestCase):
    """The same model returns a sigmoid on one server and a raw logit on another. A
    cutoff calibrated on one scale is inert on the other, with no error appearing."""

    def test_sigmoid_passes_through_untouched(self):
        pairs = [(0, 0.9), (1, 0.05), (2, 0.5)]
        output, was_logit = normalize_scores(list(pairs))
        self.assertFalse(was_logit)
        self.assertEqual(output, pairs)

    def test_logit_is_converted(self):
        # -11.04 is logit(1.6e-05): the value measured for the same irrelevant document
        # on the two servers.
        output, was_logit = normalize_scores([(0, 5.5), (1, -11.04)])
        self.assertTrue(was_logit)
        self.assertAlmostEqual(output[1][1], 1.6e-05, places=6)
        self.assertGreater(output[0][1], 0.99)

    def test_cutoff_equivalence(self):
        """sigmoid 0.10 <=> logit -2.197: the calibrated cutoff has to carry over."""
        self.assertAlmostEqual(sigmoid(-2.1972), 0.10, places=4)

    def test_a_single_negative_is_enough_to_detect_a_logit(self):
        output, was_logit = normalize_scores([(0, 0.8), (1, -0.5)])
        self.assertTrue(was_logit, "a range outside [0,1] on ANY pair means logits")

    def test_sigmoid_does_not_overflow_on_an_extreme_value(self):
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0, places=10)
        self.assertAlmostEqual(sigmoid(1000.0), 1.0, places=10)

    def test_empty_list(self):
        self.assertEqual(normalize_scores([]), ([], False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
