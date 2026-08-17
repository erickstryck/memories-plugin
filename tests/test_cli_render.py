"""Tests for the CLI's rendering, over the REAL producers' output shapes.

`cli/qctx.py` had zero coverage, and that is how two defects survived a mechanical
rename: `_render_check` read `c["nome"]`/`c["aviso"]`/`c["correcao"]` after the `Check`
dataclass fields had become `name`/`warning`/`fix_hint`, and the refresh render read
`r["acao"]` after the report key had become `action`. Both are pure dict reads, so
neither needed the network to be caught — there simply was no test.

The rule these tests follow: NEVER hand-write the dict being rendered. Build it with
`asdict(Check(...))`, or from what `DocIndex._write()` and `refresh()` actually return.
A hand-written fixture would have been renamed alongside the reader and kept agreeing
with it while both drifted from the producer — which is exactly the failure it has to
catch.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, fields
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
from core.docs import DocIndex
from core.setup import Check
from tests.fakes import FakeEmbedder, FakeVectorStore


def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "cli" / "qctx.py"
    spec = importlib.util.spec_from_file_location("qctx_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def rendered(fn, *args, **kw) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        fn(*args, **kw)

    return out.getvalue()


class TestRenderCheck(unittest.TestCase):
    """`_render_check` consumes `asdict(Check)`, so the field names are the contract."""

    def setUp(self):
        self.cli = load_cli()

    def test_reads_every_field_the_dataclass_declares(self):
        # The assertion that would have caught the bug: whatever `Check` is called
        # today, the renderer has to read those names and no others.
        declared = {f.name for f in fields(Check)}
        self.assertEqual(declared, {"name", "ok", "detail", "fix_hint", "warning"},
                         "if Check's fields change, _render_check has to change with them")

    def test_renders_a_passing_check(self):
        out = rendered(self.cli._render_check,
                       asdict(Check("Embedding", True, "bge-m3 returns 1024 dimensions")))
        self.assertIn("ok", out)
        self.assertIn("Embedding", out)
        self.assertIn("1024 dimensions", out)

    def test_a_blocking_failure_shows_the_fix(self):
        out = rendered(self.cli._render_check,
                       asdict(Check("Qdrant", False, "did not answer", "check the URL")))
        self.assertIn("FAIL", out)
        self.assertIn("-> check the URL", out,
                      "a diagnostic that says 'failed' without saying what to do is useless")

    def test_a_warning_is_not_rendered_as_a_failure(self):
        out = rendered(self.cli._render_check,
                       asdict(Check("Re-rank", False, "not configured (optional)",
                                    "export QCTX_RERANK_URL=…", warning=True)))
        self.assertIn("warn", out)
        self.assertNotIn("FAIL", out,
                        "re-ranking is optional; rendering it as FAIL says the package is broken")

    def test_no_fix_hint_renders_without_an_empty_arrow(self):
        out = rendered(self.cli._render_check, asdict(Check("X", False, "broke", None)))
        self.assertNotIn("->", out)


class TestRenderDiagnose(unittest.TestCase):
    """The whole `setup` render path, over the real `diagnose()`.

    `diagnose()` needs no network when nothing is configured: every check fails fast on
    the missing URL. So the key assembly — which is where the rename broke — is testable
    offline, contrary to what test_setup_logic's docstring assumed.
    """

    def setUp(self):
        self.cli = load_cli()
        self.cfg = core.Config(
            qdrant_url="", qdrant_api_key="", api_base_url="", api_key="",
            embed_url="", rerank_url="", embed_model="m", rerank_model="r",
            memory_collection="", docs_collection="d", library_collection="l",
            repos_collection="repos", repos_registry_collection="reg",
            vector_size=1024)

    def test_diagnose_returns_the_keys_the_cli_reads(self):
        rel = core.setup.diagnose(self.cfg)
        self.assertEqual(set(rel), {"ready", "checks", "blockers", "warnings",
                                    "detected_dim", "memory_suggestions"})

    def test_nothing_configured_is_not_ready_and_says_why(self):
        rel = core.setup.diagnose(self.cfg)
        self.assertFalse(rel["ready"])
        self.assertTrue(rel["blockers"], "an unconfigured install has to report blockers")
        self.assertTrue(all(not c["ok"] for c in rel["blockers"]))

    def test_the_optional_rerank_is_a_warning_not_a_blocker(self):
        rel = core.setup.diagnose(self.cfg)
        warned = {c["name"] for c in rel["warnings"]}
        blocked = {c["name"] for c in rel["blockers"]}
        self.assertIn("Re-rank", warned)
        self.assertNotIn("Re-rank", blocked)

    def test_the_full_setup_render_survives_a_failing_diagnose(self):
        class Args:
            json = False
            check = True
        out = rendered(self.cli.cmd_setup, Args(), self.cfg)
        self.assertIn("diagnostics:", out)
        self.assertIn("block use", out)
        for name in ("Qdrant", "Embedding", "Re-rank", "memory_collection"):
            self.assertIn(name, out)


class TestRenderDocs(unittest.TestCase):
    """The docs renders, over what `DocIndex` actually returns."""

    def setUp(self):
        self.cli = load_cli()
        q, emb = FakeVectorStore(), FakeEmbedder()
        self.idx = DocIndex(q, emb, None, "tmp", "lib", emb.dim)
        d = tempfile.mkdtemp()
        self.path = os.path.join(d, "doc.md")
        Path(self.path).write_text("# Title\n\nbody of the document here\n")

    def test_report_write_renders_the_real_index_result(self):
        res = self.idx.index_file(self.path, ttl_seconds=3600)
        out = rendered(self.cli._report_write, res, False)
        self.assertIn("indexed (temporary)", out)
        self.assertIn("doc.md", out)
        self.assertIn("expires at", out)
        self.assertIn(res["doc_id"], out)

    def test_report_write_distinguishes_the_library(self):
        res = self.idx.keep_file(self.path)
        out = rendered(self.cli._report_write, res, False)
        self.assertIn("kept in the library", out)
        self.assertIn("no expiry", out,
                      "the library has no TTL; saying it expires would be a lie")

    def test_refresh_report_keys_match_what_the_cli_renders(self):
        """The bug: refresh() produced "acao" while the CLI read "action".

        `cmd_docs_refresh` builds its own DocIndex from config, so it cannot be called
        with a fake. What is checkable without the network is the pair that broke: the
        key the producer emits, and the mark table the renderer looks it up in. Read the
        table out of the CLI's source rather than restating it here — a restated copy
        would agree with itself while drifting from the code.
        """
        self.idx.keep_file(self.path)
        report = self.idx.refresh(scope="library")
        self.assertTrue(report)

        import inspect
        import re
        src = inspect.getsource(self.cli.cmd_docs_refresh)
        m = re.search(r'mark = (\{.*?\})\.get\(r\["(\w+)"\]', src, re.S)
        self.assertIsNotNone(m, "the mark table moved; this test has to follow it")
        marks, key_read = eval(m.group(1)), m.group(2)  # noqa: S307 — our own source

        for row in report:
            self.assertIn(key_read, row,
                          f"the CLI reads r[{key_read!r}] and refresh() does not emit it")
            self.assertIn(row[key_read], marks,
                          "an unmapped action renders blank and hides a missing file")

    def test_an_unchanged_file_reports_ok_and_a_changed_one_reindexes(self):
        self.idx.keep_file(self.path)
        self.assertEqual([r["action"] for r in self.idx.refresh(scope="library")], ["ok"])
        Path(self.path).write_text("# Title\n\na genuinely different body here\n")
        self.assertEqual([r["action"] for r in self.idx.refresh(scope="library")], ["reindexed"])

    def test_a_deleted_file_reports_missing(self):
        self.idx.keep_file(self.path)
        os.unlink(self.path)
        self.assertEqual([r["action"] for r in self.idx.refresh(scope="library")], ["missing"])


class TestRenderDrop(unittest.TestCase):
    """`cmd_docs_drop` renders what `DocIndex.drop_request` decided.

    The decision moved into the core so the hermes `docs_drop` tool could route through
    the same one. Two things have to stay true here: each status the core can return still
    renders as text a person can read — an unmapped one would print nothing and look like a
    no-op that removed something — and the `--json` shape a script may already parse is
    unchanged.
    """

    def setUp(self):
        self.cli = load_cli()
        q, emb = FakeVectorStore(), FakeEmbedder()
        self.idx = DocIndex(q, emb, None, "tmp", "lib", emb.dim)
        self.path = os.path.join(tempfile.mkdtemp(), "doc.md")
        Path(self.path).write_text("# Title\n\nbody of the document here\n")

    class Args:
        def __init__(self, **kw):
            self.doc_id = None
            self.scope = "all"
            self.purge_tmp = False
            self.expired = False
            self.json = False
            self.__dict__.update(kw)

    def _run(self, args):
        import unittest.mock
        with unittest.mock.patch.object(self.cli.core, "build_docs", lambda cfg: self.idx):
            return rendered(self.cli.cmd_docs_drop, args, None)

    def test_a_doc_id_renders_the_removal(self):
        res = self.idx.keep_file(self.path)
        out = self._run(self.Args(doc_id=res["doc_id"], scope="library"))
        self.assertIn(res["doc_id"], out)
        self.assertIn("removed", out)

    def test_purge_says_the_library_was_left_alone(self):
        self.idx.index_file(self.path, ttl_seconds=60)
        out = self._run(self.Args(purge_tmp=True))
        self.assertIn("library untouched", out,
                      "the one thing a person needs told when a collection is deleted")

    def test_expired_renders_the_sweep(self):
        self.idx.index_file(self.path, ttl_seconds=60)
        out = self._run(self.Args(expired=True))
        self.assertIn("expired entries removed", out)

    def test_every_status_the_core_can_return_renders_something(self):
        """A branch the renderer does not know prints an empty line, which reads as
        success. Read the statuses out of `drop_request`'s source so a fourth one added
        later shows up here."""
        import inspect
        import re
        statuses = set(re.findall(r'"status": "(\w+)"',
                                  inspect.getsource(self.idx.drop_request)))
        self.assertTrue(statuses, "drop_request stopped declaring its statuses inline")
        rendered_statuses = set(re.findall(r'"status"\] == "(\w+)"',
                                          inspect.getsource(self.cli.cmd_docs_drop)))
        unhandled = statuses - rendered_statuses - {"removed"}   # `removed` is the else
        self.assertEqual(unhandled, set(),
                         f"cmd_docs_drop renders nothing for {unhandled}")

    def test_the_json_shape_of_a_doc_id_drop_is_the_one_already_published(self):
        """`qctx docs drop <id> --json` has printed exactly these three keys since it
        existed, and a user's script may parse them. Moving the decision into the core was
        not a reason to rename one of them."""
        res = self.idx.keep_file(self.path)
        out = self._run(self.Args(doc_id=res["doc_id"], scope="library", json=True))
        payload = json.loads(out)
        self.assertEqual(set(payload), {"doc_id", "scope", "status"})
        self.assertEqual(payload["status"], "removed")
        self.assertEqual(payload["doc_id"], res["doc_id"])
        self.assertEqual(payload["scope"], "library")

    def test_the_purge_and_expired_branches_now_honour_json(self):
        """These two printed human text and ignored `--json` entirely. Nothing could have
        been parsing what did not exist, so they carry the core's own shape."""
        self.idx.index_file(self.path, ttl_seconds=60)
        purged = json.loads(self._run(self.Args(purge_tmp=True, json=True)))
        self.assertEqual(purged["status"], "purged")
        self.assertEqual(purged["collection"], "tmp")
        swept = json.loads(self._run(self.Args(expired=True, json=True)))
        self.assertEqual(swept["status"], "swept")

    def test_no_target_exits_2_rather_than_reporting_success(self):
        from contextlib import redirect_stderr
        with self.assertRaises(SystemExit) as caught, redirect_stderr(io.StringIO()):
            self._run(self.Args())
        self.assertEqual(caught.exception.code, 2,
                         "2 means 'you called it wrong', distinct from a core failure")


class TestParser(unittest.TestCase):
    def setUp(self):
        self.cli = load_cli()

    def test_json_reaches_every_leaf_subcommand(self):
        """`--json` is documented as usable anywhere; it used to exist only at the top."""
        def leaves(parser, path=()):
            subs = getattr(parser, "_subparsers", None)
            if not subs:
                yield path, parser
                return
            for action in subs._group_actions:
                for name, sub in getattr(action, "choices", {}).items():
                    yield from leaves(sub, path + (name,))

        found = 0
        for path, leaf in leaves(self.cli.build_parser()):
            found += 1
            opts = {o for a in leaf._actions for o in a.option_strings}
            self.assertIn("--json", opts, f"`qctx {' '.join(path)}` cannot take --json")
        self.assertGreater(found, 15, "the walk should reach every leaf, not a couple")


if __name__ == "__main__":
    unittest.main(verbosity=2)
