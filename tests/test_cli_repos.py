"""The repository verbs on the CLI: that they exist, and that what they print is true.

The argument wiring is checked through `--help` in a SUBPROCESS, because those calls must
never reach Qdrant. Everything else is driven in-process over a REAL `RepoIndex` on the
fakes, never over a hand-written dict — the rule `tests/test_cli_render.py` states: a
fixture written next to its reader keeps agreeing with it while both drift away from the
producer, which is precisely the drift a render test exists to catch.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import bindings  # noqa: E402
from core import daemon as core_daemon  # noqa: E402
from core.repos import RepoIndex  # noqa: E402
from tests.fakes import (FakeEmbedder, FakeVectorStore, make_divergent,  # noqa: E402
                         make_emptied)


def run_cli(*args, env_extra=None):
    env = dict(os.environ)
    env["QCTX_STATE_DIR"] = env.get("QCTX_STATE_DIR") or tempfile.mkdtemp()
    env.update(env_extra or {})

    return subprocess.run([sys.executable, os.path.join(ROOT, "cli", "qctx.py"), *args],
                          capture_output=True, text=True, cwd=ROOT, env=env)


class TestTheVerbsExist(unittest.TestCase):
    """Argument wiring only: these must not reach Qdrant, so they are checked by --help."""

    def test_repos_has_the_five_verbs(self):
        out = run_cli("repos", "--help")
        self.assertEqual(out.returncode, 0, out.stderr)
        for verb in ("list", "search", "add", "drop", "register"):
            self.assertIn(verb, out.stdout)

    def test_register_takes_a_name_and_an_optional_label(self):
        out = run_cli("repos", "register", "--help")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--label", out.stdout)

    def test_search_takes_repo_and_all(self):
        out = run_cli("repos", "search", "--help")
        self.assertIn("--repo", out.stdout)
        self.assertIn("--all", out.stdout)

    def test_drop_requires_confirmation_by_default(self):
        """Deleting a permanent archive on a bare command would be a footgun."""
        out = run_cli("repos", "drop", "--help")
        self.assertIn("--yes", out.stdout)


# ---- in-process rendering ---------------------------------------------------


def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "cli" / "qctx.py"
    spec = importlib.util.spec_from_file_location("qctx_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def a_file(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


class Args:
    """What argparse would have produced. `json` is False unless a case asks for it."""

    def __init__(self, **kw):
        self.json = False
        self.repo = None
        self.label = None
        self.across = False
        self.limit = 8
        self.yes = False
        self.__dict__.update(kw)


class CLICase(unittest.TestCase):
    """Every case drives the real handler over a real RepoIndex on the fakes."""

    def setUp(self):
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        self.cli = load_cli()
        self.ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
        self.cfg = None
        self.cli.core.build_repos = lambda cfg: self.ix

    def rendered(self, handler, **kw) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            handler(Args(**kw), self.cfg)

        return out.getvalue()


class TestTheDropRenderTellsTheTwoOutcomesApart(CLICase):
    """`drop_repo` reports THREE keys, and `already_gone` is not decoration.

    Finishing a half-done drop deletes no archive — the archive was already gone — and it
    clears the stale bindings that outlived it. Printing the same sentence for both states
    tells the user something that did not happen, and hides the state that did.
    """

    def _populated(self):
        self.ix.register("alpha", "Alpha", [], "/tmp/alpha")
        self.ix.add_files("alpha", [a_file("invoice total = 1\n")])
        bindings.bind("/tmp/alpha", "alpha")

    def test_a_confirmed_drop_says_the_archive_was_deleted(self):
        self._populated()
        text = self.rendered(self.cli.cmd_repos_drop, repo="alpha", yes=True)
        self.assertIn("alpha", text)
        self.assertIn("dropped", text.lower())
        self.assertNotIn("already", text.lower())

    def test_finishing_a_half_done_drop_does_not_claim_to_have_deleted_an_archive(self):
        """The registry entry is already gone; only stale bindings remain. Saying "dropped"
        here is a false report of a deletion that happened in some earlier, failed run."""
        bindings.bind("/tmp/orphan", "orphan")
        text = self.rendered(self.cli.cmd_repos_drop, repo="orphan", yes=True)
        self.assertIn("already gone", text.lower())
        self.assertIn("1", text, "it has to say how many stale bindings it cleared")

    def test_an_unconfirmed_drop_deletes_nothing_and_names_the_confirmation(self):
        self._populated()
        with self.assertRaises(Exception) as caught:
            self.rendered(self.cli.cmd_repos_drop, repo="alpha")
        self.assertIn("--yes", str(caught.exception))
        self.assertIsNotNone(self.ix.get_repo("alpha"), "it must still be there")

    def test_the_json_form_is_actually_printed(self):
        """`main()` DISCARDS what a handler returns, so a handler that returns the payload
        under --json prints nothing at all. The silent branch is the one to pin."""
        self._populated()
        text = self.rendered(self.cli.cmd_repos_drop, repo="alpha", yes=True, json=True)
        self.assertEqual(json.loads(text)["repo"], "alpha")


class TestDeclaringBeforeIndexing(CLICase):
    """`repos add` on a name nobody declared would write chunks no listing can reach.

    That is the state `divergent_repos` exists to report, produced by the FIRST command a
    user would type. `repos register` is the manual declaration that makes `add` legal, and
    the refusal names it.
    """

    def test_register_declares_the_name_and_says_so(self):
        text = self.rendered(self.cli.cmd_repos_register, repo="alpha", label=None)
        self.assertIn("alpha", text)
        self.assertEqual([r["repo"] for r in self.ix.list_repos()], ["alpha"])

    def test_register_keeps_a_label_that_was_given(self):
        text = self.rendered(self.cli.cmd_repos_register, repo="alpha",
                             label="Alpha, the first")
        self.assertIn("Alpha, the first", text)

    def test_the_json_form_of_register_is_actually_printed(self):
        text = self.rendered(self.cli.cmd_repos_register, repo="alpha", label=None, json=True)
        self.assertEqual(json.loads(text)["repo"], "alpha")

    def test_add_refuses_a_repository_that_was_never_declared(self):
        with self.assertRaises(Exception) as caught:
            self.rendered(self.cli.cmd_repos_add, repo="alpha", paths=[a_file("x = 1\n")])
        self.assertIn("repos register alpha", str(caught.exception))
        self.assertEqual(list(self.ix.q.scroll_all("c")), [], "it indexed it anyway")
        self.assertEqual(self.ix.divergent_repos(), [],
                         "the ordinary path manufactured the divergence it denounces")

    def test_add_works_the_moment_the_repository_is_declared(self):
        """The refusal is only honest if the remedy it names actually unblocks the call."""
        self.rendered(self.cli.cmd_repos_register, repo="alpha", label=None)
        text = self.rendered(self.cli.cmd_repos_add, repo="alpha", paths=[a_file("x = 1\n")])
        self.assertIn("1 file(s)", text)


class TestOneLimitMeansTheSameThingToTheUser(CLICase):
    """`--limit` is ONE user-facing number over `search`'s two honest knobs.

    `limit` trims groups, `group_size` caps hits inside one. A scoped search has exactly one
    group, so `limit` does nothing there and `--limit 8` would deliver at most the default 3.
    """

    def _two_repos(self):
        body = "".join(f"invoice line {i} charge total billing\n" for i in range(400))
        for name in ("alpha", "beta"):
            self.ix.register(name, name.title(), [], f"/tmp/{name}")
            self.ix.add_files(name, [a_file(body)])

    def test_a_scoped_search_delivers_the_number_the_user_asked_for(self):
        self._two_repos()
        text = self.rendered(self.cli.cmd_repos_search, query="invoice", repo="alpha",
                             limit=6, json=True)
        hits = json.loads(text)["groups"][0]["hits"]
        self.assertEqual(len(hits), 6,
                         "--limit reached `limit`, which trims groups, and a scoped search "
                         "has one group — so it capped nothing")

    def test_a_hit_arrives_as_an_OBJECT_and_not_as_a_dataclass_repr(self):
        """`search_payload` exists so both hosts serialise hits into JSON objects. Nothing
        held it: every assertion here read `len(hits)` or the repo names, and both survive
        `default=str` rendering each hit as the string "RepoHit(score=0.41, path=...)".
        Reading one FIELD is what pins the shape — a length is blind to it."""
        self._two_repos()
        text = self.rendered(self.cli.cmd_repos_search, query="invoice", repo="alpha",
                             limit=6, json=True)
        hit = json.loads(text)["groups"][0]["hits"][0]
        self.assertIsInstance(hit, dict, "a hit reached the caller as a repr, not an object")
        self.assertTrue(hit["path"], "the hit carries no path to read the current file from")
        self.assertIn("score", hit)

    def test_an_across_search_still_reaches_every_repository(self):
        self._two_repos()
        text = self.rendered(self.cli.cmd_repos_search, query="invoice", across=True,
                             limit=8, json=True)
        out = json.loads(text)
        self.assertEqual({g["repo"] for g in out["groups"]}, {"alpha", "beta"})
        self.assertFalse(out["truncated"])


class TestAnEmptyAnswerIsPrintedAndNotSwallowed(CLICase):
    """`--all` with nothing above the cut printed an EMPTY STRING.

    The loop over `groups` renders nothing when there are none, and the `truncated` line
    answers a different question (were there more groups than `--limit` allowed). So the
    one sentence the spec demands of this feature — never affirm absence — reached neither
    host. The core produces it; here it must actually be printed.
    """

    def _two_registered_repos_with_nothing_indexed(self):
        for name in ("alpha", "beta"):
            self.ix.register(name, name.title(), [], f"/tmp/{name}")

    def test_the_human_render_says_something_rather_than_nothing(self):
        self._two_registered_repos_with_nothing_indexed()
        text = self.rendered(self.cli.cmd_repos_search, query="invoice", across=True)
        self.assertTrue(text.strip(), "an empty search printed an empty string")
        self.assertIn("above", text.lower())

    def test_the_json_form_carries_it_too_so_a_model_reads_it(self):
        """A model reading `{"groups": []}` concludes absence. It has to read the sentence
        instead of inferring from an empty array."""
        self._two_registered_repos_with_nothing_indexed()
        text = self.rendered(self.cli.cmd_repos_search, query="invoice", across=True,
                             json=True)
        out = json.loads(text)
        self.assertEqual(out["groups"], [])
        self.assertTrue(out["note"])


class TestSearchRefusesInsteadOfWideningSilently(CLICase):
    def test_outside_an_indexed_repository_it_names_the_two_remedies(self):
        """Falling back to a broad search would be the noise the scoped default exists to
        avoid, and it would be silent, which is worse than wrong."""
        with self.assertRaises(Exception) as caught:
            self.rendered(self.cli.cmd_repos_search, query="invoice")
        message = str(caught.exception)
        self.assertIn("--repo", message)
        self.assertIn("--all", message)


class TestTheListReportsWhatWasIndexed(CLICase):
    """The listing is where the registry's counts become an answer to a person.

    The spec's failure table says an entry with no chunks is marked BY THE LISTING, and
    nothing marked it. It is also the render that has to tell a declared-but-never-indexed
    repository from an indexed one, which is the distinction the counts bought.
    """

    def test_an_indexed_repo_shows_ITS_SIZE_and_not_its_last_batch(self):
        """RENAMED AND RE-AIMED. It used to assert "1 file(s)", from the registry's own
        `files`/`chunks` -- which are "as of the last `add_files` that wrote something". The
        daemon indexes in batches of 8, so a 20-file `add-all` ended with a 4-file batch and
        this line printed "4 file(s)" for a 20-file repository; a later one-file `refresh` made
        it 1. The line now prints the count the ARCHIVE reports, which is what a reader of a
        column in this position takes it to mean."""
        self.ix.register("alpha", "Alpha", [], "/tmp/alpha")
        self.ix.add_files("alpha", [a_file("invoice total = 1\n")])
        text = self.rendered(self.cli.cmd_repos_list)
        self.assertIn("chunk(s)", text)
        self.assertNotIn("file(s)", text,
                         "the per-batch file count is back, presented as a repository size")

    def test_the_size_shown_is_the_ARCHIVE_count_not_the_last_batch(self):
        """The measurement that motivated the change: index in two batches, the way the daemon
        does, and the line must report the total rather than the final batch."""
        self.ix.register("alpha", "Alpha", [], "/tmp/alpha")
        self.ix.add_files("alpha", [a_file("first file = 1\n"), a_file("second file = 2\n")])
        self.ix.add_files("alpha", [a_file("third file = 3\n")])
        text = self.rendered(self.cli.cmd_repos_list)
        self.assertIn("3 chunk(s)", text,
                      f"the listing reported the last batch, not the repository: {text!r}")

    def test_a_declared_but_never_indexed_repo_says_so_instead_of_showing_a_date(self):
        """It used to show the registration time under a column called "last indexed"."""
        self.ix.register("alpha", "Alpha", [], "/tmp/alpha")
        self.assertIn("never indexed", self.rendered(self.cli.cmd_repos_list))

    def test_an_entry_whose_chunks_are_gone_is_MARKED(self):
        make_emptied(self.ix, "gamma", a_file("invoice\n"))
        text = self.rendered(self.cli.cmd_repos_list)
        self.assertIn("gamma", text)
        self.assertIn("archive has none", text)

    def test_the_json_listing_carries_both_divergences(self):
        make_emptied(self.ix, "gamma", a_file("invoice\n"))
        make_divergent(self.ix, "ghost", a_file("invoice\n"))
        out = json.loads(self.rendered(self.cli.cmd_repos_list, json=True))
        self.assertEqual(out["emptied"], ["gamma"])
        self.assertEqual(out["divergent"], ["ghost"])


class TestTheListNamesWhatCannotBeListed(CLICase):
    def test_a_divergent_repo_is_printed_even_though_it_has_no_entry(self):
        """A repo with chunks and no registry entry cannot be listed, therefore cannot be
        dropped by name. Printing it is what makes it fixable."""
        self.ix.register("alpha", "Alpha", [], "/tmp/alpha")
        self.ix.add_files("alpha", [a_file("invoice\n")])
        make_divergent(self.ix, "ghost", a_file("invoice\n"))
        text = self.rendered(self.cli.cmd_repos_list)
        self.assertIn("alpha", text)
        self.assertIn("ghost", text)


def a_git_repo() -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True,
                   timeout=60)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True, timeout=60)

    return root


def track(root: str, name: str, text: str = "x = 1\n") -> str:
    path = os.path.join(root, name)
    with open(path, "w") as fh:
        fh.write(text)
    subprocess.run(["git", "-C", root, "add", name], check=True, timeout=60)

    return path


class TestAddAllRegistersAndBindsBeforeQueueing(CLICase):
    """Whole-branch review findings 1 and 2 — the primary documented path (`repos init` then
    `repos add-all`) DID NOT WORK: `add-all` never registered the repository, so
    `RepoIndex.add_files` refused the first batch and every job landed FAILED; and it never
    bound the checkout, so `repos init` kept offering to index a working copy that was
    already fully indexed, forever.

    `daemon.start` is mocked in every case here: a live lease is written so the CODE PATH that
    would call it is actually reached, but the real one launches a detached subprocess, which
    "no test may start a real daemon" forbids.
    """

    def setUp(self):
        super().setUp()
        from core import lease
        lease.write("s1", "claude", pid=os.getpid())
        patcher = unittest.mock.patch.object(
            core_daemon, "start", return_value={"action": "started", "pid": 999})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_add_all_registers_the_repo_before_enqueueing(self):
        root = a_git_repo()
        track(root, "a.py")
        self.rendered(self.cli.cmd_repos_add_all, repo="newrepo", path=root)
        self.assertIsNotNone(self.ix.get_repo("newrepo"),
                             "add-all queued work under a name that was never registered")

    def test_a_job_queued_by_add_all_actually_indexes_instead_of_failing(self):
        """Reproduces the Critical exactly: before the fix, `add_files` raised RepoError on
        the first batch because the name was never registered, and every job landed FAILED —
        run through `daemon._run_one`, the same code the real daemon uses to run a job and
        set its final state, so a raised RepoError shows up here exactly as `failed`."""
        root = a_git_repo()
        track(root, "a.py")
        self.rendered(self.cli.cmd_repos_add_all, repo="newrepo", path=root)
        from core import daemon, indexer, jobs
        daemon._run_one(jobs.load("newrepo"), indexer.work(index=self.ix))
        job = jobs.load("newrepo")
        self.assertEqual(job["state"], jobs.DONE,
                         f"the job could not run: {job.get('error')}")
        self.assertGreater(self.ix.get_repo("newrepo")["files"], 0)

    def test_add_all_binds_the_checkout_so_init_stops_offering(self):
        root = a_git_repo()
        track(root, "a.py")
        self.rendered(self.cli.cmd_repos_add_all, repo="newrepo", path=root)
        from core import bindings
        self.assertEqual(bindings.get(root), "newrepo",
                         "the checkout was never bound — repos init would keep offering it")

    def test_a_daemon_that_fails_to_start_does_not_hide_the_queued_job(self):
        """`enqueue` runs BEFORE `daemon.start()` and raises rather than lying, so by the time
        a start can fail the work is already on disk. Letting `DaemonError` travel to the top
        from there printed only "could not start the daemon", and the user would reasonably
        conclude nothing had been queued and re-run the command -- while the job sat waiting
        for the next session. A failure that reads as "there is nothing" is the one outcome
        this project refuses, so the queue must be reported even on this path."""
        from core import daemon, jobs
        root = a_git_repo()
        track(root, "a.py")
        with unittest.mock.patch.object(
                core_daemon, "start",
                side_effect=daemon.DaemonError("state directory unavailable")):
            out = self.rendered(self.cli.cmd_repos_add_all, repo="newrepo", path=root)
        self.assertIsNotNone(jobs.load("newrepo"),
                             "the job was not queued at all, so this tests the wrong thing")
        self.assertIn("queued under", out,
                      "the user was told the daemon failed but not that the work survived")
        self.assertIn("state directory unavailable", out,
                      "the reason the daemon could not start never reached the user")

    def test_registration_failure_is_reported_and_nothing_is_queued(self):
        """A taken/invalid name must fail LOUD, not queue work that can never run."""
        root = a_git_repo()
        track(root, "a.py")
        with self.assertRaises(Exception):
            self.rendered(self.cli.cmd_repos_add_all, repo="Not A Valid Slug", path=root)
        from core import jobs
        self.assertIsNone(jobs.load("Not A Valid Slug"))

    def test_a_second_checkout_of_an_already_registered_repo_joins_its_checkouts(self):
        first_root = a_git_repo()
        track(first_root, "a.py")
        self.rendered(self.cli.cmd_repos_add_all, repo="shared", path=first_root)
        second_root = a_git_repo()
        track(second_root, "b.py")
        self.rendered(self.cli.cmd_repos_add_all, repo="shared", path=second_root)
        entry = self.ix.get_repo("shared")
        self.assertIn(first_root, entry["checkouts"])
        self.assertIn(second_root, entry["checkouts"])


class TestAddAllRespectsTheLeaseGate(CLICase):
    """Whole-branch review finding 3: from a bare terminal (no claude or hermes session),
    `add-all` used to print "daemon started (pid N)" for a daemon that would exit on its own
    first cycle (`daemon.run` checks `lease.live()` first) — a promise it could not keep, with
    nothing telling the user why `status` then showed the job stuck `pending`.

    THE CHOSEN ANSWER: queue the work for real either way, but only claim the daemon started
    when a lease is actually live; otherwise say plainly that it starts once a session opens
    (see `cmd_repos_add_all`'s docstring in cli/qctx.py for the full reasoning, and why this
    is now true rather than aspirational: the lease write points in both hosts now also call
    `daemon.start()`)."""

    def test_with_no_live_lease_the_job_is_still_queued_but_daemon_is_not_started(self):
        root = a_git_repo()
        track(root, "a.py")
        with unittest.mock.patch.object(core_daemon, "start") as started:
            text = self.rendered(self.cli.cmd_repos_add_all, repo="newrepo", path=root)
        started.assert_not_called()
        from core import jobs
        self.assertIsNotNone(jobs.load("newrepo"), "the job was not queued at all")
        self.assertIn("no claude or hermes session", text.lower())

    def test_with_a_live_lease_daemon_start_IS_called(self):
        from core import lease
        lease.write("s1", "claude", pid=os.getpid())
        root = a_git_repo()
        track(root, "a.py")
        with unittest.mock.patch.object(
                core_daemon, "start",
                return_value={"action": "started", "pid": 999}) as started:
            text = self.rendered(self.cli.cmd_repos_add_all, repo="newrepo", path=root)
        started.assert_called_once()
        self.assertIn("daemon started", text.lower())


class TestStatusReapsBeforeRendering(CLICase):
    """Whole-branch review finding 5: `reap` only ever ran from inside `daemon.run`'s own
    loop, so `status` — the ONLY reader with no guarantee a daemon is alive to reap for it —
    printed a frozen `running` job forever once the daemon that owned it died. The spec's own
    testing section names this exact case, and there was previously no test file for
    `cmd_repos_status` at all."""

    def test_a_job_running_under_a_dead_daemon_is_reaped_before_status_renders(self):
        from core import jobs
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, daemon_pid=4_000_000)  # not a live pid
        text = self.rendered(self.cli.cmd_repos_status)
        self.assertEqual(jobs.load("alpha")["state"], jobs.FAILED,
                         "status did not reap a job left running by a dead daemon")
        self.assertRegex(text, r"alpha\s+failed")

    def test_a_job_of_a_still_LIVE_daemon_is_left_alone(self):
        from core import jobs
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, daemon_pid=os.getpid())  # this process: alive
        self.rendered(self.cli.cmd_repos_status)
        self.assertEqual(jobs.load("alpha")["state"], jobs.RUNNING,
                         "status reaped a job whose daemon is genuinely still alive")

    def test_with_no_daemon_and_no_jobs_it_says_so_plainly(self):
        text = self.rendered(self.cli.cmd_repos_status)
        self.assertIn("not running", text.lower())
        self.assertIn("no indexing jobs", text.lower())

    def test_the_json_form_reflects_the_reap_too(self):
        from core import jobs
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, daemon_pid=4_000_000)
        out = json.loads(self.rendered(self.cli.cmd_repos_status, json=True))
        self.assertEqual(out["jobs"][0]["state"], jobs.FAILED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
