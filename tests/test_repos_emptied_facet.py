"""A repo whose chunks are all gone is GONE, however the archive still indexes its name.

WHY THIS FILE EXISTS. `_chunks_per_repo` answers `{repo: count}` from a Qdrant facet, and the
real server keeps a value in its keyword index after the last point carrying it is deleted —
reporting it at `count: 0`. Both divergence reads derive their answer from the KEYS of that
mapping and never look at the counts, so a repo with no chunks at all reads to them as a repo
that holds chunks.

MEASURED on the live archive after dropping every chunk of one repo:

    {"value": "awesome-cv3", "count": 28019}
    ...
    {"value": "validacao-descartavel", "count": 0}      <- no points, still indexed

and `qctx repos list` then printed

    validacao-descartavel    (chunks with no registry entry — run `repos drop ...`)

while the command it advises answered `error: repository '...' is not indexed`. Advice that
cannot be followed is worse than silence: it sends a user to a command that refuses, with
nothing else offered.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bindings  # noqa: E402
from core.repos import RepoIndex  # noqa: E402
from tests.fakes import (FakeEmbedder, FakeVectorStore, make_divergent,  # noqa: E402
                         make_emptied)


def a_file(text: str = "content = 1\n") -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def an_index() -> RepoIndex:
    os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()

    return RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)


def a_populated_index() -> RepoIndex:
    ix = an_index()
    for name in ("alpha", "beta"):
        ix.register(name, name.title(), [], f"/tmp/{name}")
        ix.add_files(name, [a_file()])
        bindings.bind(f"/tmp/{name}", name)

    return ix


class TestAnEmptiedFacetValueIsNotARepo(unittest.TestCase):
    """Count 0 means no chunks. It must not read as "this repo holds chunks"."""

    def test_the_double_reports_an_emptied_value_the_way_the_server_does(self):
        """The premise, asserted first: without this the rest proves nothing.

        A double that counted only live points could never produce the row this file is
        about, so every assertion below would pass on a fiction.
        """
        ix = an_index()
        make_divergent(ix, "ghost", a_file())
        ix.q.delete_by_filter(ix.chunks_name, {"must": [{"key": "repo",
                                                         "match": {"value": "ghost"}}]})

        rows = {h["value"]: h["count"] for h in ix.q.facet(ix.chunks_name, "repo", 1000)}

        self.assertEqual(rows.get("ghost"), 0,
                         "the fake dropped the value instead of keeping it at count 0")

    def test_a_repo_whose_chunks_are_all_gone_is_not_reported_as_divergent(self):
        """The bug, stated as the user meets it.

        `divergent_repos` exists to name a repo that HOLDS chunks under a name the registry
        cannot see, precisely so a human can drop it. A name with no chunks fails that
        description in the only way that matters: `drop_repo` refuses it.
        """
        ix = an_index()
        make_divergent(ix, "ghost", a_file())
        self.assertEqual(ix.divergent_repos(), ["ghost"], "setup: it must start divergent")

        ix.q.delete_by_filter(ix.chunks_name, {"must": [{"key": "repo",
                                                         "match": {"value": "ghost"}}]})

        self.assertEqual(ix.divergent_repos(), [],
                         "a repo with zero chunks was reported as holding chunks")

    def test_a_REAL_divergence_is_still_named(self):
        """The guard must not become "never report" — the failure in the other direction.

        MY FIRST TWO VERSIONS OF THIS TEST ASSERTED A FALSE CONTRACT, and reading `drop_repo`
        is what corrected them. That method refuses a name with no registry entry AND no
        bindings ON PURPOSE (`core/repos.py:836-847`): "chunks without an entry are a
        divergence this method must not silently absorb". `make_divergent` builds exactly that
        state — `register_request` records no binding — so "everything named is droppable" was
        never the rule and asserting it was testing my assumption, not the design.

        What the product actually promises, and what is asserted here: a repo that genuinely
        holds orphaned chunks is NAMED, so a human can see it. The phantom that holds none is
        asserted absent above. Naming and dropping are separate contracts.
        """
        ix = an_index()
        make_divergent(ix, "ghost", a_file())

        self.assertEqual(ix.divergent_repos(), ["ghost"])
        self.assertEqual(ix._chunks_per_repo().get("ghost"), 1,
                         "it must be named BECAUSE it holds chunks")

    def test_a_repo_that_STILL_HOLDS_chunks_is_reported(self):
        """The guard must not become "never report".

        Silencing the whole read would hide the state it exists to surface, which is the
        failure in the other direction and the more expensive one: chunks nobody can name.
        """
        ix = an_index()
        make_divergent(ix, "ghost", a_file())

        self.assertEqual(ix.divergent_repos(), ["ghost"])

    def test_an_emptied_value_does_not_silence_the_EMPTIED_report_either(self):
        """The same ghost corrupts the other direction, in the opposite way.

        `emptied_repos` names registry entries that CLAIM chunks over an archive holding
        none, and it decides with `r["repo"] not in seen`. The ghost value puts the name INTO
        `seen`, so a repo that really did lose its chunks stops being reported — the listing
        goes quiet about the state it exists to announce.
        """
        ix = an_index()
        make_emptied(ix, "gamma", a_file())

        self.assertEqual(ix.emptied_repos(), ["gamma"],
                         "an emptied repo stopped being reported because its name survived "
                         "in the facet")

    def test_a_healthy_archive_reports_neither(self):
        ix = a_populated_index()

        self.assertEqual((ix.divergent_repos(), ix.emptied_repos()), ([], []))


class TestTheListingAgreesWithItself(unittest.TestCase):
    """`list_request` feeds both reads from one archive read; the ghost must not leak in."""

    def test_a_dropped_repo_disappears_from_every_part_of_the_listing(self):
        """The user's own sequence: index it, drop it, list it. Nothing should remain.

        This is the end-to-end shape of the defect as it was met in the field — a repo
        registered, indexed, and dropped by the ordinary path, whose name then lingered in
        the listing with advice that refused.
        """
        ix = an_index()
        ix.register("temp", "Temp", [], "/tmp/temp")
        ix.add_files("temp", [a_file()])
        self.assertEqual(ix._chunks_per_repo().get("temp"), 1, "setup: it must hold a chunk")

        ix.drop_repo("temp")
        out = ix.list_request()

        self.assertEqual(out["divergent"], [], "the dropped name lingered as a phantom")
        self.assertEqual(out["emptied"], [])
        self.assertNotIn("temp", [r["repo"] for r in out["repos"]])


if __name__ == "__main__":
    unittest.main()
