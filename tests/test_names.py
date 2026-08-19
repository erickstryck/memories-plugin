"""One owner for turning an outside name into a filename.

Whole-project review, finding 8: this expression existed FIVE times byte-identically -- in
`core/jobs.py`, `core/lease.py`, `hosts/hermes/__init__.py`, and inline in `hooks/recall.py` and
`hooks/checkpoint.py`. Two of those derive the state-file name for the SAME session on the two
hosts, so a divergence between them would silently split one session's recall state into two
files: the symptom is a session that keeps re-injecting memories it already showed, with nothing
pointing at the cause. Identical copies are not a shared decision, they are a coincidence that
holds until someone edits one.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import names  # noqa: E402


class TestNoPathCanEscape(unittest.TestCase):
    def test_a_traversal_attempt_keeps_no_separators_and_no_dots(self):
        got = names.safe("../../etc/passwd")
        self.assertNotIn("/", got)
        self.assertNotIn("..", got)
        self.assertEqual(got, "______etc_passwd")

    def test_an_absolute_path_cannot_survive_as_one(self):
        self.assertNotIn("/", names.safe("/etc/shadow"))

    def test_a_windows_separator_is_removed_too(self):
        self.assertNotIn("\\", names.safe("a\\b"))

    def test_an_absent_name_becomes_default_and_never_empty(self):
        """An empty filename is not a filename -- the write fails, or it hits the directory."""
        for absent in ("", None, 0, False):
            self.assertEqual(names.safe(absent), "default", f"{absent!r} produced no name")

    def test_a_name_of_only_separators_still_produces_something(self):
        self.assertEqual(names.safe("///"), "___")


class TestItIsNotAnAsciiTransliteration(unittest.TestCase):
    """Deliberate, and worth pinning: `str.isalnum()` is true for letters in any script. The
    goal is a path that cannot escape or collide, not a name made of ASCII -- and quietly
    turning accented names into underscores would make two different repositories share a
    file."""

    def test_accented_and_non_latin_names_pass_through(self):
        self.assertEqual(names.safe("ção"), "ção")
        self.assertEqual(names.safe("проект"), "проект")

    def test_two_names_that_differ_only_in_accent_stay_different(self):
        self.assertNotEqual(names.safe("acao"), names.safe("ação"))


class TestBothHostsAgreeOnTheSameSessionsFile(unittest.TestCase):
    """The failure this consolidation exists to prevent. The hermes provider and the claude-code
    recall hook each derive the recall state filename for a session; if they ever disagree, one
    session gets two state files and its recall history silently resets."""

    def test_the_job_and_cancel_files_of_one_repo_share_one_stem(self):
        """`jobs` derives two filenames for one repository. They have to agree, or a cancel
        lands beside a job that never reads it."""
        import tempfile
        from core import jobs
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        self.addCleanup(os.environ.pop, "QCTX_STATE_DIR", None)
        job_file = str(jobs._job_path("re/po 1") if hasattr(jobs, "_job_path")
                       else jobs.dir() / f"{names.safe('re/po 1')}.json")
        self.assertIn(names.safe("re/po 1"), job_file,
                      f"the job filename does not use the shared rule: {job_file}")

    def test_a_lease_file_uses_the_shared_rule_for_its_session_id(self):
        import json
        import tempfile
        from core import lease
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        self.addCleanup(os.environ.pop, "QCTX_STATE_DIR", None)
        lease.write("sess/ion 42", "claude", pid=os.getpid())
        written = [f for f in os.listdir(lease.dir()) if f.endswith(".json")]
        self.assertEqual(written, [f"{names.safe('sess/ion 42')}.json"],
                         f"the lease filename does not use the shared rule: {written}")

    def test_the_hosts_use_the_shared_function_and_not_a_copy(self):
        """A grep-level guard: a reintroduced local copy passes every behavioural test above,
        because it would start out identical. What must not come back is a SECOND owner."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []
        for folder in ("core", "hooks", "hosts", "cli"):
            for dirpath, _, filenames in os.walk(os.path.join(root, folder)):
                if "__pycache__" in dirpath:
                    continue
                for filename in filenames:
                    if not filename.endswith(".py"):
                        continue
                    full = os.path.join(dirpath, filename)
                    if os.path.abspath(full) == os.path.abspath(names.__file__):
                        continue
                    with open(full, encoding="utf-8") as handle:
                        if 'isalnum() or c in "-_"' in handle.read():
                            offenders.append(os.path.relpath(full, root))
        self.assertEqual(offenders, [],
                         f"the filename rule was copied again into: {offenders}. "
                         f"Call core.names.safe instead -- two owners of one filename is how "
                         f"the two hosts end up disagreeing about a session's state file.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
