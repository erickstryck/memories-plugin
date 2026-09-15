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

    def test_concurrent_writers_do_not_CRASH_each_other(self):
        """Five of the six atomic writers in `core/` name their temp file after the pid; this
        one used a fixed `.tmp`, and `core/windowcache.py:110-116` names the gap in its own
        comment without closing it.

        With one name, two processes open the SAME temp file: the first `os.replace` renames
        it out from under the second, whose next write lands on a path that no longer exists.
        Measured with six concurrent processes binding 40 checkouts each: 85 raised
        `FileNotFoundError` before, 0 after. The credit goes to writing the temporary in one
        `write_text` instead of holding it open across a `json.dump` — verified by removing
        the pid from the name, which alone does NOT bring the crash back.

        `os.replace` makes PUBLICATION atomic, not read-modify-write, so entries are still
        lost to the last writer winning — that needs a lock and is a different change. What
        must not happen is an exception reaching the caller."""
        import multiprocessing

        def writer(n, out):
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from core import bindings as b

            errors = 0
            for i in range(40):
                try:
                    b.bind(f"/checkout/{n}/{i}", f"repo{n}")
                except Exception:                       # noqa: BLE001 — that is the assertion
                    errors += 1
            out.put(errors)

        queue = multiprocessing.Queue()
        procs = [multiprocessing.Process(target=writer, args=(n, queue)) for n in range(6)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)

        raised = sum(queue.get() for _ in procs)
        self.assertEqual(raised, 0, f"{raised} writes crashed on a shared temp file")

    def test_no_temp_files_are_left_behind(self):
        bindings.bind("/checkout/a", "alpha")

        leftovers = [p for p in os.listdir(os.path.dirname(bindings._path()))
                     if p.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_a_write_that_FAILS_is_not_reported_as_success(self):
        """`_save` going through a shared writer must not lose the raise its callers need.

        `repos._forget_bindings` catches OSError here to say "the archive WAS deleted, but its
        local bindings could not be cleared" — a partial failure the operator has to know
        about. A review measured the version that swallowed it: on a read-only state dir,
        `repos drop` removed the chunks and the registry entry and then answered
        `{'unbound': ['/tmp/alpha'], 'already_gone': False}`, with the checkout still bound on
        disk to a repo the registry no longer knows. Running it again said `already_gone: True`
        and left the binding exactly where it was."""
        d = a_state_dir()
        bindings.bind("/checkout/alpha", "alpha")
        os.chmod(d, 0o500)
        try:
            with self.assertRaises(OSError, msg="a failed write was reported as success"):
                bindings.bind("/checkout/beta", "beta")
            with self.assertRaises(OSError):
                bindings.forget_repo("alpha")
        finally:
            os.chmod(d, 0o700)
        self.assertEqual(bindings._load(), {"/checkout/alpha": "alpha"},
                         "nothing should have reached the disk")

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

    def test_the_free_name_it_offers_is_actually_FREE(self):
        """Every candidate must be tested before it is offered, including the last one.

        Written as an assignment inside a loop that breaks on the check, the final candidate
        escaped untested. A review measured the threshold: with 99 `alpha*` names registered,
        the method offered `alpha-99` — which was taken — and `repos init` printed it as the
        primary advice, so following it merged two unrelated checkouts into one entry."""
        names = ["alpha"] + [f"alpha-{n}" for n in range(2, 100)]
        for n in names:
            self.ix.register(n, n, [], f"/home/them/{n}")
        found = self.ix.candidates_for("/home/me/mine/alpha", [])
        self.assertNotIn(found["free"], names,
                         f"it offered {found['free']!r}, which is already registered")

    def test_it_says_NOTHING_rather_than_offer_a_taken_name(self):
        """Past the bound there is no free name, and inventing one is worse than saying so."""
        names = ["alpha"] + [f"alpha-{n}" for n in range(2, 101)]
        for n in names:
            self.ix.register(n, n, [], f"/home/them/{n}")
        found = self.ix.candidates_for("/home/me/mine/alpha", [])
        self.assertIsNone(found["free"],
                          "with every name in use it still claimed one was free")

    def test_a_taken_slug_comes_with_a_FREE_alternative(self):
        """Naming the conflict is necessary and not sufficient: the host still has to tell the
        user what to type. Without an alternative the only name on offer is the taken one, and
        the CLI printed exactly that — "the name 'alpha' already belongs to another
        repository" followed by "index this working copy as: qctx repos add-all alpha",
        which accumulates this checkout and its remote into the OTHER repository's entry.

        Measured before this: an `alpha` with remote github.com/other/alpha ended up holding
        checkouts from two unrelated projects and remotes from two different forges — the
        merge by accident of naming that the declared-identity decision exists to refuse."""
        self.ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")

        out = self.ix.candidates_for("/home/me/some-other/alpha", [])

        self.assertTrue(out["taken"], "precondition: the name is taken")
        self.assertTrue(out["free"], "no alternative name was offered")
        self.assertNotEqual(out["free"], out["suggest"], "the alternative is the taken name")
        self.assertNotIn(out["free"], {r["repo"] for r in self.ix.list_repos()},
                         "the alternative is itself taken")

    def test_the_alternative_keeps_looking_until_it_finds_a_free_one(self):
        """One suffix is not enough: a machine with `alpha`, `alpha-2` and `alpha-3` must
        still get a name it can use."""
        for name in ("alpha", "alpha-2", "alpha-3"):
            self.ix.register(name, name, [], f"/home/me/{name}")

        out = self.ix.candidates_for("/home/me/other/alpha", [])

        self.assertNotIn(out["free"], {"alpha", "alpha-2", "alpha-3"})

    def test_a_FREE_slug_offers_itself_as_the_alternative(self):
        """When nothing is taken there is nothing to work around, and the host must not invent
        a suffix for a brand-new repository."""
        out = self.ix.candidates_for("/home/me/brand-new", [])

        self.assertFalse(out["taken"])
        self.assertEqual(out["free"], "brand-new")

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
