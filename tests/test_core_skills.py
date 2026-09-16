"""The skills catalogue, driven directly rather than through whichever host happens to read it.

WHY THIS FILE EXISTS. `core/skills.py` owns one fact for several callers -- the hermes
adapter registers from it, and the tests that assert about skills read from it. A module that
owns a rule for N callers earns tests that drive the rule, or the rule is only ever checked
incidentally, through whatever its callers happened to assert. That is exactly how the
published-file MODE went wrong in `core/statefile.py` while the suite stayed green.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import skills  # noqa: E402


class TestItDescribesTheREALDirectory(unittest.TestCase):
    """The catalogue's whole purpose is to agree with the tree it claims to describe."""

    def test_it_finds_the_three_shipped_skills(self):
        self.assertEqual(skills.names(), ["doc-index", "memory", "repo-index"])

    def test_every_name_it_reports_has_a_readable_SKILL_md(self):
        for skill in skills.find():
            with self.subTest(skill=skill.name):
                self.assertTrue(os.path.isfile(skill.path), skill.path)
                self.assertTrue(os.path.getsize(skill.path) > 0, "empty SKILL.md")

    def test_the_path_is_the_one_the_directory_layout_promises(self):
        """`<repo>/skills/<name>/SKILL.md`. Hosts resolve it relative to the plugin root."""
        for skill in skills.find():
            with self.subTest(skill=skill.name):
                self.assertEqual(
                    Path(skill.path),
                    REPO / "skills" / skill.name / "SKILL.md")

    def test_it_survives_the_symlinked_install_hermes_uses(self):
        """The hermes development install symlinks the repo root into `$HERMES_HOME/plugins`.

        This pins the install shape, NOT the choice of `realpath` over `abspath`: measured,
        both resolve this case, because the path through the link reaches the same `skills`
        directory. The case that distinguishes them is the next test.
        """
        box = Path(tempfile.mkdtemp())
        link = box / "memories"
        try:
            os.symlink(REPO, link)
        except OSError:
            self.skipTest("this filesystem does not do symlinks")

        self.assertEqual(self._names_from(link), ["doc-index", "memory", "repo-index"],
                         "the catalogue did not survive the symlinked install")

    def test_it_resolves_THIS_FILE_through_a_link_into_a_tree_with_no_skills(self):
        """What `realpath` actually buys, and the reason it is not `abspath`.

        A staging tree that links individual modules rather than the whole root: `abspath`
        resolves `skills/` relative to the STAGING directory, where it does not exist, and
        `find()` returns nothing at all -- an empty catalogue, silently, because a missing
        directory is tolerated by design. Every skill would go unregistered with no error
        anywhere. `realpath` resolves through the link to the real package.
        """
        box = Path(tempfile.mkdtemp())
        staged_core = box / "core"
        staged_core.mkdir()
        try:
            os.symlink(REPO / "core" / "skills.py", staged_core / "skills.py")
        except OSError:
            self.skipTest("this filesystem does not do symlinks")
        (staged_core / "__init__.py").write_text("", encoding="utf-8")
        self.assertFalse((box / "skills").exists(), "the staging tree must have no skills/")

        self.assertEqual(
            self._names_from(box), ["doc-index", "memory", "repo-index"],
            "resolved relative to the staging tree instead of the real package, which "
            "yields an EMPTY catalogue and silently registers nothing")

    def _names_from(self, root) -> list:
        """`skills.names()` as imported from *root*, in a clean interpreter.

        A subprocess because `SKILLS_DIR` is resolved at import: re-importing in this process
        would hand back the already-imported module and measure nothing.

        It imports `core.skills` DIRECTLY rather than `from core import skills`, so that a
        staging tree holding only this module needs no working `core/__init__.py`. The first
        version of this test linked two files and let the package's real `__init__` run: it
        died on `No module named core.config` and reported that as the assertion failing,
        which made a surviving mutation look caught. An import error is not evidence.
        """
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys, importlib; sys.path.insert(0, sys.argv[1]); "
             "m = importlib.import_module('core.skills'); print(','.join(m.names()))",
             str(root)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, f"import failed: {out.stderr.strip()}")
        return [n for n in out.stdout.strip().split(",") if n]


class TestItToleratesAnIncompleteTree(unittest.TestCase):
    """These callers are plugins being loaded. A broken accessory must not abort a host."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _skill(self, name, body="# x"):
        os.makedirs(os.path.join(self.dir, name))
        with open(os.path.join(self.dir, name, "SKILL.md"), "w", encoding="utf-8") as fh:
            fh.write(body)

    def test_a_directory_without_a_SKILL_md_is_skipped_not_raised(self):
        """Both hosts treat it as a silent no-op; a half-created directory is not an error."""
        self._skill("real")
        os.makedirs(os.path.join(self.dir, "empty"))

        self.assertEqual(skills.names(self.dir), ["real"])

    def test_a_FILE_sitting_among_the_skill_directories_is_skipped(self):
        """`skills/README.md` is not a skill, and `os.path.isfile` on `<file>/SKILL.md`
        answers False rather than raising NotADirectoryError."""
        self._skill("real")
        with open(os.path.join(self.dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write("not a skill")

        self.assertEqual(skills.names(self.dir), ["real"])

    def test_a_DIRECTORY_named_SKILL_md_is_not_a_skill(self):
        """`isfile`, not `exists` -- the one shape that tells the two apart.

        Found by review, as a mutation the suite did not kill: `isfile(path)` ->
        `exists(path)` left all 1644 tests green, because the case above (a plain file among
        the directories) is caught by `exists` too -- the join `README.md/SKILL.md` simply
        does not exist. A DIRECTORY named `SKILL.md` exists and is not readable as a skill.

        What the mutant does in production: hands the directory to `ctx.register_skill`,
        whose own check is `path.exists()` (hermes_cli/plugins.py), so it registers happily
        and the failure surfaces later, as a `skill_view` that cannot read its file --
        instead of being skipped here, silently and correctly, at scan time.
        """
        self._skill("real")
        os.makedirs(os.path.join(self.dir, "trap", "SKILL.md"))

        self.assertEqual(skills.names(self.dir), ["real"])

    def test_every_shipped_skill_name_is_one_a_host_can_REGISTER(self):
        """The catalogue's filter is "has a SKILL.md"; hosts add a name rule on top.

        hermes refuses a bare name that does not match `[a-zA-Z0-9_-]+`
        (`agent/skill_utils.py::_NAMESPACE_RE`, raised from `PluginContext.register_skill`),
        and the adapter's registration loop swallows that refusal so it cannot cost the
        memory provider. MEASURED: a `skills/repo.index/` directory ships, is counted
        everywhere, satisfies every other test, and reaches no user -- the same
        skill-on-disk-that-no-host-can-load defect this catalogue was written to fix, one
        layer down.

        The rule is asserted HERE and not enforced in `core/skills.py` on purpose: it is a
        host's rule, and the catalogue stays host-agnostic. This fails at CI time instead.
        """
        offenders = [n for n in skills.names() if not re.match(r"^[a-zA-Z0-9_-]+$", n)]
        self.assertEqual(offenders, [],
                         "these skill directories cannot be registered on hermes and would "
                         "be dropped in silence")

    def test_a_MISSING_skills_directory_yields_nothing(self):
        self.assertEqual(skills.names(os.path.join(self.dir, "nope")), [])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root lists a mode-000 directory anyway, so the probe would prove nothing")
    def test_an_UNREADABLE_skills_directory_yields_nothing(self):
        """Distinct from missing: the directory exists and refuses to be listed."""
        self._skill("real")
        os.chmod(self.dir, 0o000)
        self.addCleanup(os.chmod, self.dir, 0o700)

        self.assertEqual(skills.names(self.dir), [])

    def test_the_order_is_stable_and_sorted(self):
        """Callers compare this against a sorted list; readdir order is not sorted."""
        for name in ("zulu", "alpha", "mike"):
            self._skill(name)

        self.assertEqual(skills.names(self.dir), ["alpha", "mike", "zulu"])


class TestTheTwoShapesAgree(unittest.TestCase):
    """`find`, `paths` and `names` are three views of one answer, not three answers."""

    def test_paths_and_find_report_the_same_skills(self):
        self.assertEqual(
            skills.paths(), {s.name: s.path for s in skills.find()})

    def test_names_and_find_report_the_same_skills(self):
        self.assertEqual(skills.names(), [s.name for s in skills.find()])

    def test_all_three_HONOUR_the_directory_they_are_given(self):
        """Found by mutation: `paths()` calling `find()` with no argument passed the whole
        suite, because every other test that passes a directory goes through `names()`. A
        view that silently ignores its argument answers about the wrong tree."""
        box = tempfile.mkdtemp()
        os.makedirs(os.path.join(box, "only-one"))
        with open(os.path.join(box, "only-one", "SKILL.md"), "w", encoding="utf-8") as fh:
            fh.write("# x")

        self.assertEqual([s.name for s in skills.find(box)], ["only-one"])
        self.assertEqual(skills.names(box), ["only-one"])
        self.assertEqual(sorted(skills.paths(box)), ["only-one"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
