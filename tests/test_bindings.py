import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bindings  # noqa: E402
from core.repos import RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


def a_git_repo(remote: str | None = None) -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    if remote:
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=root, check=True)

    return root


class TestTheLocalBinding(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_an_unbound_checkout_reads_as_None(self):
        self.assertIsNone(bindings.get("/home/me/never-seen"))

    def test_binding_survives_a_fresh_read(self):
        bindings.bind("/home/me/alpha", "alpha")
        self.assertEqual(bindings.get("/home/me/alpha"), "alpha")

    def test_two_checkouts_may_bind_to_the_same_repo(self):
        bindings.bind("/home/me/alpha", "alpha")
        bindings.bind("/home/me/alpha-2", "alpha")
        self.assertEqual(bindings.get("/home/me/alpha-2"), "alpha")

    def test_forgetting_a_repo_unbinds_every_checkout_of_it_and_names_them(self):
        """Dropping a repo must invalidate the bindings, or a checkout keeps claiming to
        belong to a repo that no longer exists and the next index writes to a phantom."""
        bindings.bind("/home/me/alpha", "alpha")
        bindings.bind("/home/me/alpha-2", "alpha")
        bindings.bind("/home/me/beta", "beta")
        freed = bindings.forget_repo("alpha")
        self.assertEqual(sorted(freed), ["/home/me/alpha", "/home/me/alpha-2"])
        self.assertIsNone(bindings.get("/home/me/alpha"))
        self.assertEqual(bindings.get("/home/me/beta"), "beta")

    def test_a_corrupt_state_file_reads_as_no_bindings_instead_of_raising(self):
        with open(os.path.join(bindings.state_dir(), bindings.FILENAME), "w") as fh:
            fh.write("{not json")
        self.assertIsNone(bindings.get("/home/me/alpha"))


class TestGitFacts(unittest.TestCase):
    def test_the_root_of_a_subdirectory_is_the_repo_root(self):
        root = a_git_repo()
        deep = os.path.join(root, "a", "b")
        os.makedirs(deep)
        self.assertEqual(os.path.realpath(bindings.git_root(deep)), os.path.realpath(root))

    def test_a_plain_directory_has_no_root(self):
        self.assertIsNone(bindings.git_root(tempfile.mkdtemp()))

    def test_remotes_are_read_and_a_repo_without_one_is_fine(self):
        self.assertEqual(bindings.remotes_of(a_git_repo()), [])
        self.assertEqual(bindings.remotes_of(a_git_repo("git@host:me/alpha.git")),
                         ["git@host:me/alpha.git"])

    def test_the_same_repo_over_ssh_and_https_normalizes_alike(self):
        """Otherwise the join offer misses, and the user is asked to name a repo that is
        already registered under the other URL form."""
        self.assertEqual(bindings.normalize_remote("git@github.com:me/alpha.git"),
                         bindings.normalize_remote("https://github.com/me/alpha"))

    def test_a_non_standard_ssh_port_is_not_a_directory(self):
        """A port is transport, not identity. Folding `:2222` into the path is precisely the
        miss this function exists to prevent: the same self-hosted repository reached over a
        non-standard SSH port would be offered as a new repo instead of a join."""
        self.assertEqual(bindings.normalize_remote("ssh://git@github.com:2222/me/alpha.git"),
                         "github.com/me/alpha")
        self.assertEqual(bindings.normalize_remote("ssh://git@github.com:2222/me/alpha.git"),
                         bindings.normalize_remote("git@github.com:me/alpha.git"))

    def test_a_numeric_first_path_segment_survives_the_scp_form(self):
        """scp-style `user@host:path` has NO port syntax in git's URL grammar — everything
        after the colon is the path. A repository under a numeric directory (Gerrit and
        several self-hosted layouts do this) must keep that segment, or the scp form stops
        matching the https form of the same repository: the port fix arriving from the other
        side."""
        self.assertEqual(bindings.normalize_remote("git@host:1234/repo.git"), "host/1234/repo")
        self.assertEqual(bindings.normalize_remote("git@host:1234/repo.git"),
                         bindings.normalize_remote("https://host/1234/repo"))

    def test_a_port_is_still_dropped_when_a_scheme_said_it_was_one(self):
        """The other side of the same coin, and the reason the scheme is the disambiguator: to
        name a port you MUST write `ssh://user@host:port/path`, so here `1234` really is a
        port and folding it into the path would be the original miss again."""
        self.assertEqual(bindings.normalize_remote("ssh://git@host:1234/repo.git"), "host/repo")
        self.assertEqual(bindings.normalize_remote("ssh://git@host:1234/repo.git"),
                         bindings.normalize_remote("git@host:repo.git"))

    def test_an_upper_case_scheme_normalizes_like_a_lower_case_one(self):
        """A case-sensitive scheme strip leaves `HTTPS://` in place, and then the scp-style
        colon rule fires on the scheme's own colon and yields `https///github.com/me/alpha`.
        The value is asserted, not just the equality: two forms agreeing on garbage would
        satisfy an equality alone."""
        self.assertEqual(bindings.normalize_remote("HTTPS://GitHub.com/Me/Alpha"),
                         "github.com/me/alpha")
        self.assertEqual(bindings.normalize_remote("HTTPS://GitHub.com/Me/Alpha"),
                         bindings.normalize_remote("https://github.com/me/alpha"))

    def test_the_slug_is_stable_and_filesystem_shaped(self):
        self.assertEqual(bindings.slug_for("My Repo!"), "my-repo")
        self.assertEqual(bindings.slug_for("awesome-cv3"), "awesome-cv3")


class TestTheChoiceOffered(unittest.TestCase):
    def setUp(self):
        a_state_dir()
        self.ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)

    def test_a_matching_remote_is_offered_to_JOIN(self):
        self.ix.register("alpha", "Alpha", ["git@github.com:me/alpha.git"], "/home/me/alpha")
        out = self.ix.candidates_for("/home/me/alpha-2", ["https://github.com/me/alpha"])
        self.assertEqual([c["repo"] for c in out["join"]], ["alpha"])

    def test_with_no_match_the_suggestion_is_the_directory_name(self):
        out = self.ix.candidates_for("/home/me/brand-new", [])
        self.assertEqual(out["join"], [])
        self.assertEqual(out["suggest"], "brand-new")

    def test_an_already_bound_checkout_reports_its_binding_and_asks_nothing(self):
        self.ix.register("alpha", "Alpha", [], "/home/me/alpha")
        bindings.bind("/home/me/alpha", "alpha")
        self.assertEqual(self.ix.candidates_for("/home/me/alpha", [])["bound"], "alpha")

    def test_a_binding_to_a_repo_that_no_longer_exists_is_NOT_reported_as_bound(self):
        """The healing path. A stale binding must behave as unbound, or the checkout writes
        into a repo the registry does not know."""
        bindings.bind("/home/me/alpha", "deleted-repo")
        self.assertIsNone(self.ix.candidates_for("/home/me/alpha", [])["bound"])

    def test_a_taken_slug_is_reported_as_a_conflict_and_not_silently_joined(self):
        """Merging on slug collision would decide identity by an accident of naming, which is
        what the declared-identity decision exists to reject."""
        self.ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        out = self.ix.candidates_for("/home/me/some-other/alpha", [])
        self.assertEqual(out["suggest"], "alpha")
        self.assertTrue(out["taken"], "a suggestion that already exists must be flagged")

    def test_a_free_slug_is_NOT_flagged(self):
        """The other direction, or `taken` could be hard-coded true and still pass above —
        and the host would then report a conflict for every brand-new repository."""
        self.ix.register("alpha", "Alpha", [], "/home/me/alpha")
        out = self.ix.candidates_for("/home/me/brand-new", [])
        self.assertEqual(out["suggest"], "brand-new")
        self.assertFalse(out["taken"])

    def test_a_collision_is_reported_as_a_conflict_and_NOT_as_a_join_offer(self):
        """The two are different answers and the host must not confuse them: `join` means
        "this IS that repository, by remote"; `taken` means "the name you would be offered
        already belongs to something else". Same directory name, no shared remote."""
        self.ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        out = self.ix.candidates_for("/home/me/some-other/alpha",
                                     ["git@host:someone-else/alpha.git"])
        self.assertEqual(out["join"], [], "a different remote is not the same repository")
        self.assertTrue(out["taken"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
