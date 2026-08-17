import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore, make_divergent  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


def an_index(*declared: str) -> RepoIndex:
    """An empty archive. Any names given are DECLARED first.

    Registering is not decoration in a fixture: `add_files` refuses a repository the registry
    does not know, so a test that indexes into an undeclared name is testing a path the
    product no longer has. See `TestIndexingRequiresADeclaredRepository`.
    """
    ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), CHUNKS, REG, 8)
    for name in declared:
        ix.register_request(name)

    return ix


def an_index_that_declared(name: str) -> dict:
    """The registry entry `register_request` produced, for asserting on it."""
    return an_index().register_request(name)


def a_file(text: str, suffix: str = ".py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


class TestWriting(unittest.TestCase):
    def test_every_chunk_carries_the_repo(self):
        ix = an_index("alpha")
        ix.add_files("alpha", [a_file("def one():\n    return 1\n")])
        points = list(ix.q.scroll_all(CHUNKS))
        self.assertTrue(points)
        self.assertEqual({p["payload"]["repo"] for p in points}, {"alpha"})

    def test_the_repo_is_top_level_and_not_buried_in_metadata(self):
        """`group_by` and the payload index address a top-level key, exactly as `doc_id`
        already does. Burying it under metadata would work for reading and break both."""
        ix = an_index("alpha")
        ix.add_files("alpha", [a_file("x = 1\n")])
        payload = next(iter(ix.q.scroll_all(CHUNKS)))["payload"]
        self.assertIn("repo", payload)
        self.assertNotIn("repo", payload.get("metadata", {}))

    def test_reindexing_the_same_file_replaces_instead_of_accumulating(self):
        """Without this the old version and the new one coexist and one search mixes chunks
        from two states of the same file."""
        ix = an_index("alpha")
        path = a_file("first content here\n")
        ix.add_files("alpha", [path])
        before = len(list(ix.q.scroll_all(CHUNKS)))
        with open(path, "w") as fh:
            fh.write("second content, entirely different\n")
        ix.add_files("alpha", [path])
        after = list(ix.q.scroll_all(CHUNKS))
        self.assertEqual(len(after), before)
        self.assertIn("second", " ".join(p["payload"]["document"] for p in after))

    def test_a_file_that_shrinks_leaves_no_orphan_chunks_behind(self):
        """The replacement above is invisible when the chunk count does not change: point ids
        are derived from `(doc_id, chunk index)`, so re-upserting a file that still needs the
        same N chunks overwrites the same N ids and looks clean with no deletion at all.

        Only a file that now needs FEWER chunks exposes the orphans, and they are the real
        damage: chunks of a version that no longer exists on disk, still answering searches
        and still reporting themselves as that file's lines.
        """
        ix = an_index("alpha")
        path = a_file("\n\n".join(f"def f{i}():\n    return {i}\n" + "# padding line here\n" * 40
                                 for i in range(6)))
        ix.add_files("alpha", [path])
        self.assertGreater(len(list(ix.q.scroll_all(CHUNKS))), 1)
        with open(path, "w") as fh:
            fh.write("x = 1\n")
        ix.add_files("alpha", [path])
        self.assertEqual(len(list(ix.q.scroll_all(CHUNKS))), 1)

    def test_an_empty_file_is_skipped_and_reported_not_raised(self):
        """One unindexable file in a list of eight hundred must not abort the other 799."""
        ix = an_index("alpha")
        out = ix.add_files("alpha", [a_file("   \n\n"), a_file("real = 1\n")])
        self.assertEqual(out["files"], 1)
        self.assertEqual(len(out["skipped"]), 1)

    def test_an_unreadable_file_is_skipped_and_reported(self):
        ix = an_index("alpha")
        gone = os.path.join(tempfile.mkdtemp(), "never-existed.py")
        out = ix.add_files("alpha", [gone, a_file("real = 1\n")])
        self.assertEqual(out["files"], 1)
        self.assertEqual([p for p, _ in out["skipped"]], [gone])

    def test_writing_under_an_empty_repo_name_is_refused(self):
        """Chunks whose `repo` is empty can never be filtered to and never be dropped: the
        one operation that would remove them is the one that needs the name. Refusing at the
        door is the only point where the mistake is still cheap."""
        ix = an_index("alpha")
        with self.assertRaises(RepoError):
            ix.add_files("", [a_file("x = 1\n")])
        self.assertEqual(list(ix.q.scroll_all(CHUNKS)), [])

    def test_the_digest_is_stored_so_staleness_can_be_judged_later(self):
        """`source_changed` compares by digest because mtime and size both lie: cp -p,
        rsync --times and any restore preserve mtime, and a one-character edit preserves
        size. Storing it is what lets the watcher (sub-project E) exist at all."""
        ix = an_index("alpha")
        ix.add_files("alpha", [a_file("content = 1\n")])
        md = next(iter(ix.q.scroll_all(CHUNKS)))["payload"]["metadata"]
        self.assertTrue(md["src_digest"])
        self.assertIn("src_mtime", md)
        self.assertIn("src_size", md)


class TestIndexingRequiresADeclaredRepository(unittest.TestCase):
    """The registry is authoritative over WHICH REPOS EXIST — the module docstring says so,
    and until this guard existed it was simply not true: `add_files` wrote any name it was
    handed.

    The harm is not abstract. `qctx repos add alpha file.py` on a name nobody declared writes
    chunks that no listing can reach and that `divergent_repos` then reports as a defect — so
    the ordinary first command a user types manufactures the exact state this feature has a
    function to denounce.
    """

    def test_indexing_an_unregistered_repository_is_refused(self):
        ix = an_index()
        with self.assertRaisesRegex(RepoError, "register"):
            ix.add_files("nunca-registrado", [a_file("x = 1\n")])

    def test_the_refusal_writes_no_chunks_at_all(self):
        """A refusal that had already written half the batch would leave the divergence it
        exists to prevent, and blame the caller for it."""
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.add_files("nunca-registrado", [a_file("x = 1\n"), a_file("y = 2\n")])
        self.assertEqual(list(ix.q.scroll_all(CHUNKS)), [])
        self.assertEqual(ix.divergent_repos(), [])

    def test_it_refuses_instead_of_registering_the_name_itself(self):
        """Auto-registration would turn a typo into a permanent archive: `repos add alpah …`
        would create `alpah`, and repositories in this design are removed only by hand. The
        refusal catches the typo; auto-creating buries it."""
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.add_files("alpah", [a_file("x = 1\n")])
        self.assertEqual(ix.list_repos(), [], "it invented an entry for a mistyped name")

    def test_a_refused_add_DECLARES_nothing_either(self):
        """The other half of the ruling, pinned apart from the refusal itself.

        A guard that registered the name and then indexed it would still refuse nothing, and
        the typo would still become a permanent archive — so `assertRaises` alone cannot tell
        "refused" from "auto-registered": it short-circuits before any assertion about the
        registry can run. Measured: removing the guard entirely and making it auto-register
        failed exactly the same six tests until this one existed. The error is swallowed here
        on purpose, so the state is asserted whether or not the call raised.
        """
        ix = an_index()
        try:
            ix.add_files("alpah", [a_file("x = 1\n")])
        except RepoError:
            pass
        self.assertEqual(ix.list_repos(), [], "it declared a mistyped name")

    def test_the_message_names_the_command_that_fixes_it(self):
        ix = an_index()
        with self.assertRaises(RepoError) as caught:
            ix.add_files("alpha", [a_file("x = 1\n")])
        self.assertIn("repos register alpha", str(caught.exception))


class TestDeclaringARepository(unittest.TestCase):
    """`register_request` is the MANUAL path: it records the name that was declared.

    Deliberately plain. Detecting the working copy, matching it against known remotes and
    offering to join an existing repository is `candidates_for`'s job and belongs to the
    interactive flow that will wrap this one. This is what that flow will call, not a
    competitor to it.
    """

    def test_declaring_a_name_makes_it_listable_and_indexable(self):
        ix = an_index()
        ix.register_request("alpha")
        self.assertEqual([r["repo"] for r in ix.list_repos()], ["alpha"])
        self.assertEqual(ix.add_files("alpha", [a_file("x = 1\n")])["files"], 1)

    def test_the_label_defaults_to_the_name(self):
        self.assertEqual(an_index_that_declared("alpha")["label"], "alpha")

    def test_a_label_that_was_given_is_kept(self):
        ix = an_index()
        self.assertEqual(ix.register_request("alpha", "Alpha, the first")["label"],
                         "Alpha, the first")

    def test_it_records_NO_checkout_and_NO_remotes_because_none_were_declared(self):
        """Inferring them from the working directory is precisely the derivation this
        design rejected: identity here is declared, so a manual declaration records what was
        declared and nothing more."""
        entry = an_index_that_declared("alpha")
        self.assertEqual(entry["checkouts"], [])
        self.assertEqual(entry["remotes"], [])

    def test_declaring_the_same_name_twice_updates_it_instead_of_making_a_twin(self):
        ix = an_index()
        ix.register_request("alpha", "First")
        ix.register_request("alpha", "Second")
        self.assertEqual([r["repo"] for r in ix.list_repos()], ["alpha"])
        self.assertEqual(ix.get_repo("alpha")["label"], "Second")

    def test_a_nameless_declaration_is_refused(self):
        with self.assertRaises(RepoError):
            an_index().register_request("")


class TestTheNameMustAlreadyBeItsOwnSlug(unittest.TestCase):
    """The spec says the id is a SLUG derived from the chosen name, and `slug_for` exists to
    produce it. It was not enforced: `register_request("My Repo!")` stored `'My Repo!'`.

    REFUSED, NOT SILENTLY SLUGIFIED, and the second reason is the decisive one. Transforming
    here means `repos register "My Repo"` stores `my-repo` while `repos add "My Repo" f.py`
    then looks up `My Repo` and is refused — the two paths disagree unless every entry point
    transforms identically, which is four places that must never drift. Refusing with the
    remedy named is what every other refusal in this feature does, and it costs the user one
    retyped name.
    """

    def test_a_name_that_is_not_a_slug_is_refused(self):
        with self.assertRaises(RepoError):
            an_index().register_request("My Repo!")

    def test_the_refusal_names_the_slug_to_use(self):
        with self.assertRaises(RepoError) as caught:
            an_index().register_request("My Repo!")
        self.assertIn("my-repo", str(caught.exception))

    def test_it_does_NOT_store_the_slug_it_suggested(self):
        """Silent transformation is the failure this guard exists to avoid, so the refusal
        must leave the registry empty — not quietly hold what it recommended."""
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.register_request("My Repo!")
        self.assertEqual(ix.list_repos(), [])

    def test_a_name_that_is_already_a_slug_passes_through_untouched(self):
        for name in ("alpha", "awesome-cv3", "repo2"):
            with self.subTest(name=name):
                self.assertEqual(an_index().register_request(name)["repo"], name)

    def test_the_primitive_refuses_too_and_not_only_the_request_wrapper(self):
        """Same placement argument as `add_files`' registry guard: the interactive flow will
        call `register` DIRECTLY, with the remotes and the checkout it detected, so a guard
        that lived only in the wrapper would leave the hole open for the caller most likely
        to fill the registry through it."""
        with self.assertRaises(RepoError):
            an_index().register("My Repo!", "My Repo!", [], "/home/me/x")

    def test_a_declaration_that_SUCCEEDS_is_always_indexable_under_the_SAME_name(self):
        """The decisive reason for refusing rather than slugifying, as a round trip.

        Had `register` quietly stored `my-repo` for `My Repo!`, the declaration would report
        success and the very next command — `repos add "My Repo!" f.py` — would look up
        `My Repo!`, find nothing, and be refused for a repository the user had just declared.
        Each half looks correct alone; only the round trip catches the two paths disagreeing.

        Measured, and the reason this test exists: removing the guard and MAKING IT SLUGIFY
        failed exactly the same five tests, because every one of them asserts on the refusal
        and none asserted on what a SUCCESSFUL declaration promises. A refusal is skipped
        here on purpose — it promised nothing, so there is nothing to honour.
        """
        ix = an_index()
        for name in ("My Repo!", "alpha", "Awesome CV3", "awesome-cv3"):
            with self.subTest(name=name):
                try:
                    declared = ix.register_request(name)["repo"]
                except RepoError:
                    continue
                self.assertEqual(declared, name,
                                 "it stored a name the caller never gave, so the read path "
                                 "will look up a repository that is not there")
                self.assertEqual(ix.add_files(name, [a_file("x = 1\n")])["files"], 1)

    def test_a_name_made_only_of_separators_is_refused_rather_than_becoming_repo(self):
        """`slug_for` falls back to the literal `repo` for a name with nothing usable in it.
        Accepting that would register an archive called `repo` for someone who typed `---`."""
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.register_request("---")
        self.assertEqual(ix.list_repos(), [])


class TestTheRegistry(unittest.TestCase):
    def test_registering_makes_the_repo_listable(self):
        ix = an_index()
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        self.assertEqual([r["repo"] for r in ix.list_repos()], ["alpha"])

    def test_the_registry_is_authoritative_over_WHICH_repos_exist(self):
        """Chunks own content; the registry owns existence. A repo with chunks and no entry
        is a divergence, not a repo — and it must be visible as one."""
        ix = an_index()
        make_divergent(ix, "ghost", a_file("x = 1\n"))
        self.assertEqual(ix.list_repos(), [])
        self.assertIsNone(ix.get_repo("ghost"))

    def test_registering_the_same_repo_twice_accumulates_checkouts_without_duplicating(self):
        ix = an_index()
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha-2")
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        entry = ix.get_repo("alpha")
        self.assertEqual(sorted(entry["checkouts"]), ["/home/me/alpha", "/home/me/alpha-2"])
        self.assertEqual(len(ix.list_repos()), 1)

    def test_the_registry_never_lands_in_the_chunk_collection(self):
        ix = an_index()
        ix.register("alpha", "Alpha", [], "/home/me/alpha")
        self.assertEqual(list(ix.q.scroll_all(CHUNKS)), [])

    def test_an_empty_repo_name_is_refused(self):
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.register("", "Alpha", [], "/home/me/alpha")


class TestTheEntryRecordsWHATWasIndexedAndWHEN(unittest.TestCase):
    """The spec's entry holds "contagem de arquivos e chunks, e quando foi indexado".

    It held neither: `register` wrote five keys and `add_files` never touched the registry,
    so the counts did not exist and `indexed_at` recorded the last REGISTRATION. Three
    things were false because of it — the tool description telling the model the listing
    shows when each repo was last indexed, the `divergent_repos` docstring arguing that the
    data model the spec already specified would have to be invented, and the spec's own
    failure table promising the listing marks an entry with no chunks.
    """

    def test_a_fresh_registration_claims_nothing_indexed(self):
        entry = an_index_that_declared("alpha")
        self.assertEqual((entry["files"], entry["chunks"]), (0, 0))

    def test_a_fresh_registration_has_no_indexed_at_because_it_was_never_indexed(self):
        """`?` in the listing is the truth here. A timestamp would say a repository with no
        chunks in it was indexed a second ago."""
        self.assertIsNone(an_index_that_declared("alpha").get("indexed_at"))

    def test_indexing_records_the_counts_it_wrote(self):
        ix = an_index("alpha")
        wrote = ix.add_files("alpha", [a_file("invoice line\n" * 400)])
        entry = ix.get_repo("alpha")
        self.assertEqual(entry["files"], wrote["files"])
        self.assertEqual(entry["chunks"], wrote["chunks"])
        self.assertGreater(entry["chunks"], 0, "the fixture has to write more than nothing")

    def test_indexing_stamps_when_it_happened(self):
        ix = an_index("alpha")
        ix.add_files("alpha", [a_file("x = 1\n")])
        self.assertTrue(ix.get_repo("alpha")["indexed_at"])

    def test_registering_again_does_NOT_reset_when_it_was_last_indexed(self):
        """A second working copy is declared through `register`, which accumulates it. If
        registering stamped `indexed_at`, declaring a checkout would claim an indexing that
        never happened — which is exactly what the field did before."""
        ix = an_index()
        ix.register("alpha", "Alpha", [], "/home/me/alpha")
        ix.add_files("alpha", [a_file("invoice\n")])
        # A COPY, and not the entry itself. `FakeVectorStore` stores the payload it is
        # handed, so the "before" picture is the very dict the next write mutates: read
        # without copying, this assertion compares an object with itself and holds nothing.
        indexed = dict(ix.get_repo("alpha"))
        ix.register("alpha", "Alpha", [], "/home/me/alpha-2")
        after = ix.get_repo("alpha")
        self.assertEqual(after["indexed_at"], indexed["indexed_at"])
        self.assertEqual((after["files"], after["chunks"]),
                         (indexed["files"], indexed["chunks"]))
        self.assertEqual(len(after["checkouts"]), 2, "it still accumulated the checkout")

    def test_a_batch_that_indexed_NOTHING_claims_no_indexing(self):
        """Every path skipped means nothing was written. Overwriting the counts with zeros
        would erase a real claim — and with it the only evidence that tells a half-deleted
        archive from a fresh registration."""
        ix = an_index("alpha")
        ix.add_files("alpha", [a_file("invoice\n")])
        before = dict(ix.get_repo("alpha"))          # a COPY — see the test above
        result = ix.add_files("alpha", ["/no/such/file/at/all.py"])
        after = ix.get_repo("alpha")
        self.assertEqual(result["files"], 0)
        self.assertEqual(result["skipped"] and len(result["skipped"]), 1)
        self.assertEqual((after["files"], after["chunks"], after["indexed_at"]),
                         (before["files"], before["chunks"], before["indexed_at"]))

    def test_the_counts_are_AS_OF_THE_LAST_indexing_and_not_a_running_total(self):
        """Stated because it is a limit and not an accident: the registry is authoritative
        over WHICH repos exist, never over how big the archive is right now. Only a scroll
        knows that, and scrolling the archive for a metadata question is the cost this
        design refuses to pay."""
        ix = an_index("alpha")
        ix.add_files("alpha", [a_file("one\n"), a_file("two\n")])
        ix.add_files("alpha", [a_file("three\n")])
        self.assertEqual(ix.get_repo("alpha")["files"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
