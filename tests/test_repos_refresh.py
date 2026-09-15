"""Reindexing what changed on disk, for a repository archive.

`docs` has had `refresh` since the library existed, and for the same reason: an archive that
never expires holds chunks of a file as it was, so a file edited later returns text that no
longer exists there. Search already SAYS so — every hit carries a `stale` reason — and that
warning was the whole repair story for repositories. This is the other half.

WHAT CALLS IT: the daemon's watcher, on a poll cycle that sees a file changed on disk (see
`core/indexer.py`'s `watcher`), or a person running `qctx repos refresh` directly. An earlier
version of this docstring named a git `post-commit` hook as the trigger and pointed at a
`TestTheHookScript` in this file — both gone: the hook (`core/githook.py`, `repos
install-hook`, `tests/test_githook.py`) was removed once the daemon covered everything it did
and more (see `core/repos.py`'s `refresh` docstring for the fuller history of that claim).
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding import EmbeddingError  # noqa: E402
from core.repos import RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


def an_index(*declared: str) -> RepoIndex:
    ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), CHUNKS, REG, 8)
    for name in declared:
        ix.register_request(name)

    return ix


def a_file(text: str, suffix: str = ".py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def rewrite(path: str, text: str) -> None:
    """Changes the file's CONTENT. The digest is what `source_changed` trusts, so a rewrite is
    detected even when mtime and size happen to match."""
    with open(path, "w") as fh:
        fh.write(text)


class TestAnOutageDoesNotDESTROYWhatWasIndexed(unittest.TestCase):
    """`_write_one` deleted the old chunks BEFORE embedding the new ones, so an embedding
    outage in between left the file with no chunks at all — and nothing brought them back.

    The file's own comment already named this consequence; the fix that let infrastructure
    errors propagate changed only WHO is blamed, not the ordering that loses the data."""

    def test_a_failed_embed_leaves_the_previous_chunks_in_place(self):
        ix = an_index("alpha")
        path = a_file("def a():\n    return 1\n")
        ix.add_files("alpha", [path])
        before = len(ix.q.collections[CHUNKS]["points"])
        self.assertTrue(before, "precondition: the file was indexed")

        rewrite(path, "def a():\n    return 2\n")

        def refused(texts):
            raise EmbeddingError("connection refused")

        ix.embedder.embed = refused
        with self.assertRaises(EmbeddingError):
            ix.refresh("alpha")

        self.assertEqual(len(ix.q.collections[CHUNKS]["points"]), before,
                         "an outage deleted the chunks of a file that is fine on disk")

    def test_the_file_is_still_reachable_after_the_outage(self):
        """The half that makes it unrecoverable: with the chunks gone, `poll` reports the
        file as neither indexed nor changed, so no later cycle ever re-indexes it."""
        ix = an_index("alpha")
        path = a_file("def a():\n    return 1\n")
        ix.add_files("alpha", [path])
        rewrite(path, "def a():\n    return 2\n")

        def refused(texts):
            raise EmbeddingError("connection refused")

        ix.embedder.embed = refused
        with self.assertRaises(EmbeddingError):
            ix.refresh("alpha")

        self.assertIn(path, ix.poll("alpha")["indexed"],
                      "the archive forgot a file that is on disk, with no way back")


class TestItReindexesOnlyWhatCHANGED(unittest.TestCase):
    def test_an_untouched_file_is_reported_ok_and_NOT_reindexed(self):
        """Re-embedding an unchanged file is pure network cost for an identical result, and the
        hook that calls this runs on every commit."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        before = len(ix.q.collections[CHUNKS]["points"])
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["ok"])
        self.assertEqual(len(ix.q.collections[CHUNKS]["points"]), before)

    def test_a_changed_file_is_reindexed(self):
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        rewrite(path, "x = 1\ny = 2\nz = 3\n")
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["reindexed"])
        self.assertEqual(report[0]["path"], path)
        self.assertGreater(report[0]["chunks"], 0)

    def test_the_reindex_REPLACES_the_old_chunks_instead_of_adding_to_them(self):
        """The point of the repair. Two copies of one file would double its weight in every
        search and answer with text from both versions."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        rewrite(path, "y = 2\n")
        ix.refresh("alpha")
        # `.strip()` because the chunker drops the trailing newline — an assertion that
        # depends on it is testing the chunker, not the refresh.
        texts = [p["payload"]["document"].strip()
                 for p in ix.q.collections[CHUNKS]["points"].values()]
        self.assertIn("y = 2", texts)
        self.assertNotIn("x = 1", texts, "the old version survived the refresh")

    def test_a_file_that_is_GONE_is_reported_and_NOT_deleted(self):
        """Deletion in this plugin is explicit and permanent — `repos drop` demands --yes — so a
        refresh does not quietly remove an archive. The hits stay, and they stay MARKED, which is
        the visible state a user can act on; silently dropping them would be a deletion nobody
        asked for."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        before = len(ix.q.collections[CHUNKS]["points"])
        os.unlink(path)
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["missing"])
        self.assertEqual(len(ix.q.collections[CHUNKS]["points"]), before,
                         "refresh deleted chunks for a file that vanished")

    def test_one_unreadable_file_does_not_abort_the_others(self):
        """The same rule `add_files` follows: a batch stops being usable the moment one bad
        member can end it."""
        ix = an_index("alpha")
        good, bad = a_file("x = 1\n"), a_file("y = 2\n")
        ix.add_files("alpha", [good, bad])
        rewrite(good, "x = 11\n")
        os.unlink(bad)
        actions = {r["path"]: r["action"] for r in ix.refresh("alpha")}
        self.assertEqual(actions[good], "reindexed")
        self.assertEqual(actions[bad], "missing")


class TestItStaysInsideITSOwnRepository(unittest.TestCase):
    def test_a_changed_file_of_ANOTHER_repo_is_left_alone(self):
        """The archive is one collection keyed by `repo`, so a refresh that forgot the filter
        would reindex every repository on the machine — and charge the user for it."""
        ix = an_index("alpha", "beta")
        mine, theirs = a_file("x = 1\n"), a_file("y = 2\n")
        ix.add_files("alpha", [mine])
        ix.add_files("beta", [theirs])
        rewrite(mine, "x = 11\n")
        rewrite(theirs, "y = 22\n")
        report = ix.refresh("alpha")
        self.assertEqual([r["path"] for r in report], [mine])
        texts = [p["payload"]["document"].strip()
                 for p in ix.q.collections[CHUNKS]["points"].values()]
        self.assertIn("y = 2", texts, "beta was reindexed by alpha's refresh")

    def test_refreshing_an_unregistered_repository_is_refused(self):
        from core.repos import RepoError
        with self.assertRaises(RepoError):
            an_index().refresh("never-declared")

    def test_a_repository_with_nothing_indexed_reports_nothing(self):
        self.assertEqual(an_index("alpha").refresh("alpha"), [])


class TestEachPathIsJudgedONCE(unittest.TestCase):
    def test_a_file_of_many_chunks_is_reindexed_once_not_once_per_chunk(self):
        """A file is stored as N chunks that all carry the same source metadata. Judging each
        chunk would re-embed the file N times and report it N times."""
        ix = an_index("alpha")
        # 900 lines is ~13 KB, which the chunker splits into six; 400 lines fits in ONE and
        # the first version of this fixture measured nothing. Verified below rather than
        # assumed, because the threshold is the chunker's business and can move.
        path = a_file("\n".join(f"line_{i} = {i}" for i in range(900)) + "\n")
        ix.add_files("alpha", [path])
        self.assertGreater(len(ix.q.collections[CHUNKS]["points"]), 1, "fixture is not multi-chunk")
        rewrite(path, "small = 1\n")
        report = ix.refresh("alpha")
        self.assertEqual(len(report), 1, f"the file was judged {len(report)} times")


class TestChangedPathsIsMetadataOnly(unittest.TestCase):
    """Whole-branch review finding 7: `changed_paths` used to call `source_changed`, which
    reads and SHA-1s the whole file whenever a digest was recorded — which is always, for
    anything this class indexed — turning a 5-second poll into "hash every tracked byte of
    the repository, forever". It must compare `mtime`/`size` ONLY, leaving the digest check to
    `refresh`."""

    def test_a_SAME_SIZE_content_change_with_the_ORIGINAL_mtime_restored_is_MISSED(self):
        """The proof that no hash runs here any more: `cp -p`-style tampering (same mtime,
        same size, different bytes) is EXACTLY what a digest check exists to catch, and
        `changed_paths` — metadata only, by design now — must NOT catch it. `refresh`, which
        still compares the digest before re-embedding, must catch it regardless: the
        watcher's cheap half missing something never means the archive stays wrong forever,
        only that this poll cycle did not notice."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        recorded_mtime = ix._indexed_sources("alpha")[path]["src_mtime"]
        with open(path, "w") as fh:
            fh.write("x = 2\n")                        # same byte length as "x = 1\n"
        os.utime(path, (recorded_mtime, recorded_mtime))  # restore the exact recorded mtime
        self.assertEqual(ix.changed_paths("alpha"), [],
                         "changed_paths caught a content-only change — it is reading the file")
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["reindexed"],
                         "refresh's digest check did not catch what changed_paths missed")

    def test_an_mtime_only_touch_with_UNCHANGED_content_is_a_SAFE_false_positive(self):
        """The other side of the trade the docstring makes: metadata alone CAN be fooled the
        other way too (a `touch` with no edit), and that is fine — `refresh` re-checks the
        digest and reports `ok`, spending nothing on an embedding call."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        far_future = time.time() + 10_000
        os.utime(path, (far_future, far_future))
        self.assertEqual(ix.changed_paths("alpha"), [path],
                         "a real mtime change was not flagged by changed_paths")
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["ok"],
                         "refresh re-embedded a file whose content never changed")

    def test_a_size_change_is_caught(self):
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        with open(path, "a") as fh:
            fh.write("y = 2\n")
        self.assertEqual(ix.changed_paths("alpha"), [path])

    def test_a_file_that_vanished_is_caught(self):
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        os.unlink(path)
        self.assertEqual(ix.changed_paths("alpha"), [path])

    def test_an_untouched_file_is_not_flagged(self):
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        self.assertEqual(ix.changed_paths("alpha"), [])


class TestRefreshRespectsShouldStop(unittest.TestCase):
    """Whole-branch review finding 9: cancellation was never checked inside `refresh`, so a
    watcher-queued refresh job that got cancelled mid-run would re-embed everything that
    changed and STILL end up reported `cancelled` afterward — a claim about work that, by
    then, had fully happened. `should_stop` gives `refresh` the same per-item boundary the
    "index" job's batch loop already has."""

    def test_should_stop_halts_before_reaching_every_changed_file(self):
        ix = an_index("alpha")
        p1, p2 = a_file("a = 1\n"), a_file("b = 1\n")
        ix.add_files("alpha", [p1, p2])
        rewrite(p1, "a = 11\n")
        rewrite(p2, "b = 11\n")
        calls = []

        def stop_after_one():
            calls.append(1)

            return len(calls) > 1

        report = ix.refresh("alpha", should_stop=stop_after_one)
        self.assertEqual(len(report), 1, "should_stop did not halt the loop early")

    def test_should_stop_TRUE_from_the_start_reindexes_nothing(self):
        ix = an_index("alpha")
        p1, p2 = a_file("a = 1\n"), a_file("b = 1\n")
        ix.add_files("alpha", [p1, p2])
        rewrite(p1, "a = 11\n")
        rewrite(p2, "b = 11\n")
        report = ix.refresh("alpha", should_stop=lambda: True)
        self.assertEqual(report, [])

    def test_files_NOT_reached_before_should_stop_are_still_refreshed_on_the_NEXT_call(self):
        """What was skipped is not lost, only deferred — the next unrestricted `refresh` picks
        it up, the same way a repository that changes every cycle eventually catches up."""
        ix = an_index("alpha")
        p1, p2 = a_file("a = 1\n"), a_file("b = 1\n")
        ix.add_files("alpha", [p1, p2])
        rewrite(p1, "a = 11\n")
        rewrite(p2, "b = 11\n")
        first = ix.refresh("alpha", should_stop=lambda: True)
        self.assertEqual(first, [])
        second = ix.refresh("alpha")
        self.assertEqual(len(second), 2)
        self.assertEqual({r["action"] for r in second}, {"reindexed"})

    def test_should_stop_defaults_to_off(self):
        """`refresh` called the way every existing caller (`qctx repos refresh`, and every
        test above this class) already calls it must behave exactly as before: unconditional,
        judging every path."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        rewrite(path, "x = 11\n")
        report = ix.refresh("alpha")
        self.assertEqual([r["action"] for r in report], ["reindexed"])


class TestThePollDoesNotPullTheWholeArchiveOverTheWire(unittest.TestCase):
    """Whole-project review, Critical 2. To answer "did anything change?", the watcher scrolled
    every chunk of the repository WITH ITS FULL PAYLOAD -- and the payload holds each chunk's
    TEXT. So the cheap half of the change check was dragging the repository's entire indexed
    content across the network every ~5 s, per repository, for as long as a session stayed open.
    Measured on the payload shape this code writes: 1358 B per point against 129 B for the one
    field actually read, ~10.5x, or 13.6 MB against 1.3 MB per cycle for a 10k-chunk archive.

    The fix asks Qdrant for `metadata` only. These tests hold BOTH halves: that the projection
    is requested, and that the code still works when it is honoured -- `FakeVectorStore` drops
    the unrequested keys exactly as a real server does, because a fake that returned everything
    would let a caller read a field it no longer asks for and still pass.
    """

    def _recording(self):
        """An index whose store records the `payload_fields` of every scroll."""
        asked = []
        store = FakeVectorStore()
        original = store.scroll_all

        def recording(name, filter_=None, with_vector=False, payload_fields=None):
            asked.append(payload_fields)

            return original(name, filter_=filter_, with_vector=with_vector,
                            payload_fields=payload_fields)

        store.scroll_all = recording
        ix = RepoIndex(store, FakeEmbedder(dim=8), CHUNKS, REG, 8)
        ix.register_request("alpha")

        return ix, asked

    def test_the_change_check_asks_for_METADATA_ONLY(self):
        ix, asked = self._recording()
        ix.add_files("alpha", [a_file("x = 1\n")])
        asked.clear()
        ix.changed_paths("alpha")
        self.assertTrue(asked, "no scroll happened, so this asserts nothing")
        for fields in asked:
            self.assertEqual(fields, ["metadata"],
                             f"the change check pulled {fields!r} -- the full payload carries "
                             f"every chunk's text, and this runs every few seconds")

    def test_it_still_detects_a_change_with_the_projection_in_place(self):
        """The other half: a projection that broke detection would be worse than the cost."""
        ix = an_index("alpha")
        path = a_file("x = 1\n")
        ix.add_files("alpha", [path])
        self.assertEqual(ix.changed_paths("alpha"), [], "unchanged file reported as changed")
        with open(path, "w") as fh:
            fh.write("x = 1\ny = 2\nz = 3\n")
        self.assertEqual(ix.changed_paths("alpha"), [path],
                         "the change was missed -- the projection dropped a field it needs")

    def test_poll_reads_the_archive_ONCE_for_both_answers(self):
        """`changed_paths` and the newly-tracked-file diff each used to fetch the same source
        metadata, so one watch cycle scrolled the archive twice for two halves of one question."""
        ix, asked = self._recording()
        ix.add_files("alpha", [a_file("x = 1\n")])
        asked.clear()
        out = ix.poll("alpha")
        self.assertEqual(len(asked), 1, f"poll cost {len(asked)} archive reads, not 1")
        self.assertIn("changed", out)
        self.assertIn("indexed", out)


class TestASkippedFileIsReportedNotRaised(unittest.TestCase):
    """`add_files` reports a skip as the TUPLE `(path, reason)` — `cli/qctx.py` reads it that
    way. `refresh` read it as a dict, so the first unindexable file raised AttributeError,
    `daemon._run_one` marked the whole job FAILED, and every remaining changed file in that
    refresh was never reindexed."""

    def test_a_file_that_became_unindexable_is_reported_as_skipped(self):
        ix = an_index("alpha")
        path = a_file("real content here\n")
        ix.add_files("alpha", [path])
        # Emptying it makes `_write_one` raise "nothing indexable", which is the exact
        # condition 20 of the 22 looping files on the user's machine are in.
        rewrite(path, "")
        report = ix.refresh("alpha")
        entry = next(r for r in report if r["path"] == path)
        self.assertEqual(entry["action"], "skipped")
        self.assertIn("nothing indexable", entry["reason"])


class TestAnOutageIsNotAPropertyOfTheFile(unittest.TestCase):
    """`add_files` used to catch `CoreError`, the ROOT of the hierarchy, so an unreachable
    embedding endpoint arrived in `skipped` in exactly the same shape as "nothing indexable".
    Whoever remembered skips then recorded a network failure as a permanent fact about a file,
    keyed by an mtime an outage does not change. Worse on the refresh path: `_write_one`
    deletes before it embeds, so the file's existing chunks went too, with no way back."""

    def _an_index_whose_endpoint_is_down(self):
        from core.embedding import EmbeddingError

        ix = an_index("alpha")

        def refuse(texts):
            raise EmbeddingError("network failure on http://127.0.0.1:8003/v1/embeddings")

        ix.embedder.embed = refuse

        return ix

    def test_an_embedding_outage_raises_instead_of_reporting_a_skip(self):
        ix = self._an_index_whose_endpoint_is_down()
        from core.embedding import EmbeddingError

        with self.assertRaises(EmbeddingError):
            ix.add_files("alpha", [a_file("real content\n")])

    def test_a_qdrant_outage_raises_too(self):
        from core.qdrant import QdrantError

        ix = an_index("alpha")

        def refuse(*a, **kw):
            raise QdrantError("could not reach the archive")

        ix.q.upsert = refuse
        with self.assertRaises(QdrantError):
            ix.add_files("alpha", [a_file("real content\n")])

    def test_a_file_level_failure_is_still_reported_as_a_skip(self):
        """The distinction is the point: an empty file IS a property of the file, so it must
        keep landing in `skipped` rather than failing the whole batch."""
        ix = an_index("alpha")
        out = ix.add_files("alpha", [a_file("")])
        self.assertEqual(out["files"], 0)
        self.assertIn("nothing indexable", out["skipped"][0][1])

    def test_one_unreadable_file_does_not_stop_the_batch(self):
        ix = an_index("alpha")
        good = a_file("x = 1\n")
        out = ix.add_files("alpha", [a_file(""), good])
        self.assertEqual(out["files"], 1, "a per-file skip must not abort its neighbours")
        self.assertEqual(len(out["skipped"]), 1)


class TestResumingDoesNotPayForWhatDidNotChange(unittest.TestCase):
    """The spec's own justification for cancelling being safe: "E retomar é barato: um novo
    trabalho pula os arquivos cujo digest não mudou, comparação que refresh já implementa."
    `docs/usage.md:68` repeats the promise to the user — "running `add-all` again skips the
    files that did not change".

    It did not skip anything. `add_files` embedded every path it was given, so resuming a
    cancelled `add-all` of 1,800 files re-embedded all 1,800 — against the same shared
    endpoint the quarantine work exists to keep from being hammered. Measured on a 3-file
    repo: pass 1 = 3 embed calls, pass 2 with the disk untouched = 3 more, while `refresh`
    over the identical state cost 0.
    """

    def test_a_second_add_files_over_an_UNCHANGED_disk_embeds_nothing(self):
        ix = an_index("alpha")
        paths = [a_file("def a():\n    return 1\n"), a_file("def b():\n    return 2\n")]
        ix.add_files("alpha", paths)
        self.assertTrue(ix.embedder.calls, "precondition: the first pass did embed")

        ix.embedder.calls.clear()
        out = ix.add_files("alpha", paths)

        self.assertEqual(ix.embedder.calls, [],
                         "resuming re-embedded files whose content never changed")
        self.assertEqual(out["files"], len(paths),
                         "a skipped file is still a file this repo holds, not a failure")

    def test_a_file_that_DID_change_is_still_re_embedded(self):
        """The other direction, so the fix cannot be "never index twice"."""
        ix = an_index("alpha")
        path = a_file("def a():\n    return 1\n")
        ix.add_files("alpha", [path])
        ix.embedder.calls.clear()

        rewrite(path, "def a():\n    return 999\n")
        ix.add_files("alpha", [path])

        self.assertTrue(ix.embedder.calls, "a changed file was skipped")

    def test_a_file_the_archive_never_held_is_indexed(self):
        """And a genuinely new path is not mistaken for an unchanged one."""
        ix = an_index("alpha")
        ix.add_files("alpha", [a_file("def a():\n    return 1\n")])
        ix.embedder.calls.clear()

        ix.add_files("alpha", [a_file("def brand_new():\n    return 2\n")])

        self.assertTrue(ix.embedder.calls, "a file the archive never saw was skipped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
