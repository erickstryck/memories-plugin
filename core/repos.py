"""Repository archive: code chunks grouped by repo, with a registry of what exists.

WHY A THIRD ARCHIVE. `core/docs.py` keeps documents out of the memory collection because a
file chunk competing with a curated fact wins on volume and drowns it. One level down, the
same argument: the library is hand-picked reference material, and a repository is tens of
thousands of automatic chunks. It gets its own collection or it drowns the library.

WHY A REGISTRY, AND WHY IN ITS OWN COLLECTION. `ports.VectorStore` has no facet, distinct or
count, so answering "which repos do I have" from the chunks would scroll every point in the
archive. The registry answers it from a handful of rows. It sits in a separate collection so
that no search has to filter registry rows out: a filter forgotten once turns a registry row
into a search hit, and a guard that only exists by vigilance gets deleted by a cleanup that
passes CI.

WHO OWNS WHAT. The registry is authoritative over WHICH REPOS EXIST, their labels and their
checkouts. The chunks are authoritative over CONTENT, and the `repo` on a chunk is derived
from the registry, never the source of it. Chunks without an entry are a DIVERGENCE, and
`list_repos` deliberately does not invent an entry for them — a repo you cannot list is a
repo you cannot drop, and inventing one would hide that.

WHAT THIS MODULE DOES NOT DO. It never discovers files. `add_files` indexes exactly the paths
it is handed; walking a repository, honouring .gitignore and skipping minified bundles belong
to the bulk pipeline, which is a separate concern with a separate failure mode (scale).
"""
import os
import time
from dataclasses import dataclass

from . import ports
from .chunk import chunk_text, mode_for_suffix
from .docs import _iso, _point_id, _read_source, content_digest, doc_id_for, source_changed
from .errors import CoreError


class RepoError(CoreError):
    """Something about a repository archive operation could not be done."""


#: The registry stores no meaning in its vector — it is a key-value table that happens to
#: live in Qdrant, read by scroll and never by similarity. Size 1 says so out loud, and a
#: unit vector avoids the zero-norm that Cosine has no answer for.
REGISTRY_VECTOR_SIZE = 1
REGISTRY_VECTOR = [1.0]


@dataclass
class RepoHit:
    score: float
    repo: str
    path: str
    start_line: int
    end_line: int
    mode: str
    text: str
    indexed_at: str
    stale: str | None    # the reason, when the file changed since it was indexed


def search_payload(result: dict) -> dict:
    """The JSON form of a `search` result: hits as objects, never as their repr.

    Both hosts need it and neither should invent it. `RepoHit` is a dataclass, so a plain
    `json.dumps(..., default=str)` — which is what both hosts' JSON writers do with an object
    they do not know — renders a hit as `RepoHit(score=0.41, repo='alpha', …)`: a string no
    caller can address a file by, and a shape no consumer can parse. One conversion, in the
    core, so the two hosts cannot answer the same search with two different JSON documents.
    """
    return {**result,
            "groups": [{"repo": g["repo"], "hits": [vars(h) for h in g["hits"]]}
                       for g in result.get("groups", [])]}


class RepoIndex:
    def __init__(self, qdrant: ports.VectorStore, embedder: ports.EmbeddingModel,
                 repos_collection: str, registry_collection: str, vector_size: int):
        self.q = qdrant
        self.embedder = embedder
        self.chunks_name = repos_collection
        self.registry_name = registry_collection
        self.vector_size = vector_size

    # ---- collections -------------------------------------------------------

    def ensure(self) -> None:
        if self.q.ensure_collection(self.chunks_name, self.vector_size):
            self.q.ensure_payload_index(self.chunks_name, "repo", "keyword")
            self.q.ensure_payload_index(self.chunks_name, "doc_id", "keyword")
        if self.q.ensure_collection(self.registry_name, REGISTRY_VECTOR_SIZE):
            self.q.ensure_payload_index(self.registry_name, "repo", "keyword")

    # ---- writing -----------------------------------------------------------

    def add_files(self, repo: str, paths: list[str]) -> dict:
        """Indexes exactly `paths` under `repo`. Never raises for one bad file.

        A list of eight hundred paths with one empty file in it must index the other 799:
        aborting the batch for an unindexable member would make the bulk pipeline's job
        impossible, so the failure is REPORTED per path instead.

        It catches CoreError and not a list of leaf types: `_read_source` raises DocsError
        for a missing file AND for a binary one, and enumerating leaf classes is how a
        catch silently stops covering the case it was written for.

        A BINARY FILE IS THEREFORE ALREADY SKIPPED AND REPORTED, for free: `_read_source`
        refuses it. The bulk pipeline does not have to detect binaries to avoid poisoning
        the archive — it only has to avoid paying to open them.
        """
        if not repo:
            raise RepoError("a repository name is required")
        self.ensure()
        files = chunks = 0
        skipped: list = []
        for path in paths:
            try:
                added = self._write_one(repo, path)
            except (CoreError, OSError, ValueError) as exc:
                skipped.append((path, str(exc)))
                continue
            files += 1
            chunks += added

        return {"repo": repo, "files": files, "chunks": chunks, "skipped": skipped}

    def _write_one(self, repo: str, path: str) -> int:
        path, st, content = _read_source(path)
        pieces = chunk_text(content)
        if not pieces:
            raise RepoError("nothing indexable (empty file, or whitespace only)")
        doc_id = doc_id_for(path)
        digest = content_digest(content)
        mode = mode_for_suffix(os.path.splitext(path)[1])
        now = time.time()

        # Reindexing REPLACES: without this the old version and the new one coexist and one
        # search mixes chunks from two states of the same file.
        self.q.delete_by_filter(self.chunks_name,
                                {"must": [{"key": "doc_id", "match": {"value": doc_id}}]})

        vectors = self.embedder.embed([p.text for p in pieces])
        points = []
        for ix, (piece, vector) in enumerate(zip(pieces, vectors)):
            points.append({
                "id": _point_id(doc_id, ix),
                "vector": vector,
                "payload": {
                    "document": piece.text,
                    "doc_id": doc_id,
                    # Top level, like doc_id: the payload index and group_by address it.
                    "repo": repo,
                    "metadata": {
                        "path": path, "start_line": piece.start_line,
                        "end_line": piece.end_line, "mode": mode,
                        "chunk_ix": ix, "n_chunks": len(pieces),
                        "indexed_at": _iso(now),
                        "src_mtime": st.st_mtime, "src_size": st.st_size,
                        "src_digest": digest,
                    },
                },
            })
        self.q.upsert(self.chunks_name, points)

        return len(points)

    # ---- registry ----------------------------------------------------------

    def register(self, repo: str, label: str, remotes: list[str], checkout: str) -> dict:
        """Creates or updates the entry. Checkouts and remotes ACCUMULATE without repeating:
        the same repository legitimately lives in several working copies at once."""
        if not repo:
            raise RepoError("a repository name is required")
        self.ensure()
        entry = self.get_repo(repo) or {"repo": repo, "label": label or repo,
                                        "remotes": [], "checkouts": []}
        entry["label"] = label or entry.get("label") or repo
        for value, key in ((checkout, "checkouts"), *[(r, "remotes") for r in remotes or []]):
            if value and value not in entry[key]:
                entry[key].append(value)
        entry["indexed_at"] = _iso(time.time())
        self.q.upsert(self.registry_name, [{"id": _point_id(f"registry:{repo}", 0),
                                            "vector": list(REGISTRY_VECTOR),
                                            "payload": entry}])

        return entry

    def get_repo(self, repo: str) -> dict | None:
        point = self.q.get_point(self.registry_name, _point_id(f"registry:{repo}", 0))

        return (point or {}).get("payload") if point else None

    def list_repos(self) -> list[dict]:
        """Every registered repo, from the REGISTRY and never from the chunks.

        Deriving this from the chunks would mean scrolling the whole archive, and it would
        also invent entries for divergent chunks — hiding a repo that cannot be dropped.
        """
        try:
            rows = [p.get("payload") or {} for p in self.q.scroll_all(self.registry_name)]
        except Exception as exc:                       # noqa: BLE001
            raise RepoError(f"the repository registry could not be read: {exc}") from exc

        return sorted((r for r in rows if r.get("repo")), key=lambda r: r["repo"])

    def candidates_for(self, root: str, remotes: list[str]) -> dict:
        """The choice to offer for this working copy. Presenting it is the host's job.

        `bound` is set only when the binding points at a repo the registry still knows: a
        stale binding must behave as unbound, or this checkout writes into a phantom repo.
        """
        from . import bindings

        known = self.list_repos()
        by_name = {r["repo"] for r in known}
        bound = bindings.get(root)
        if bound and bound not in by_name:
            bound = None
        wanted = {bindings.normalize_remote(r) for r in remotes or [] if r}
        join = [r for r in known
                if wanted & {bindings.normalize_remote(x) for x in r.get("remotes") or []}]

        return {"bound": bound, "join": join,
                "suggest": bindings.slug_for(os.path.basename(os.path.realpath(root)))}

    # ---- searching ---------------------------------------------------------

    def search(self, query: str, repo: str | None = None, across: bool = False,
               limit: int = 8, group_size: int = 3) -> dict:
        """Grouped hits: one repo by default, every repo when `across`.

        FAILURE DOES NOT OPEN HERE, and that is the inverse of the read guard on purpose. A
        search that cannot reach the archive and returns [] is indistinguishable from "there
        is nothing about this", and a caller would conclude absence from an outage. It raises.

        THE INVARIANT IS `CoreError`, NOT `RepoError`: failure surfaces as some CoreError and
        never as an empty result. The embedder is deliberately NOT wrapped here — it already
        raises `EmbeddingError`, which is a CoreError that the CLI boundary catches — and
        wrapping it would diverge from `docs.py` and `memory.py`, which call it bare too.

        `across` asks for as many groups as the registry knows, which is exactly why the
        registry exists: without it, how many groups to ask for would be a guess. It remains
        best-effort over what the search reaches — never a proof of absence.
        """
        known = {r["repo"] for r in self.list_repos()}
        if across:
            if not known:
                raise RepoError("no repository is indexed yet")
            group_limit, filter_ = len(known), None
        else:
            if not repo:
                raise RepoError("name a repository, or pass across=True")
            if repo not in known:
                raise RepoError(f"repository {repo!r} is not indexed")
            group_limit = 1
            filter_ = {"must": [{"key": "repo", "match": {"value": repo}}]}

        vector = self.embedder.embed_one(query)
        try:
            raw = self.q.search_groups(self.chunks_name, vector, group_by="repo",
                                       limit=group_limit, group_size=group_size,
                                       filter_=filter_)
        except RepoError:
            raise
        except Exception as exc:                        # noqa: BLE001
            raise RepoError(f"the repository archive could not be searched: {exc}") from exc

        # The PARSING is guarded too, and not only the call. A response whose shape drifted,
        # or a chunk written with a wrong field type, would otherwise leave `search` as an
        # unhandled AttributeError from inside a comprehension — a crash the caller cannot
        # tell from a bug in the search itself. Malformed is an error, never a dropped hit:
        # silently skipping one would under-report, which is this task's harm in miniature.
        try:
            groups = []
            for group in raw[:limit]:
                hits = [self._to_hit(h) for h in group.get("hits", [])]
                if hits:
                    groups.append({"repo": group.get("id"), "hits": hits})
            truncated = len(raw) > limit
        except Exception as exc:                        # noqa: BLE001
            raise RepoError(
                f"the repository archive returned a result that could not be read: {exc}"
            ) from exc

        return {"scope": "across" if across else "repo", "repo": None if across else repo,
                "groups": groups, "truncated": truncated}

    def search_request(self, query: str, repo: str | None = None, across: bool = False,
                       limit: int = 8, cwd: str | None = None) -> dict:
        """The search BOTH HOSTS call: resolve the scope, translate the limit, then search.

        It lives here rather than in the two adapters for the reason `DocIndex.drop_request`
        does: everything below is a DECISION, and a decision taken twice is a decision the
        two hosts are free to take differently — which is the divergence this plugin's
        equivalence requirement exists to forbid.

        THE SCOPE IS RESOLVED FROM THE WORKING DIRECTORY, and REFUSED when there is none.
        Falling back to a search across every repository would be exactly the noise the
        scoped default was chosen to avoid, and it would be silent, which is worse than
        wrong. The refusal names both remedies, because either one may be what was meant.

        ONE LIMIT, TWO KNOBS. `search` keeps both of its honest knobs — `limit` trims GROUPS,
        `group_size` caps hits inside a group — because both are real and the core should not
        pretend otherwise. A caller has ONE number in mind. On a scoped search there is only
        ever one group, so `limit` trims nothing and the answer stops at `group_size` however
        large a number was typed; the caller's number therefore becomes `group_size` there.
        Across, it stays `limit`, which is how many repositories come back.
        """
        if not repo and not across:
            from . import bindings

            root = bindings.git_root(cwd if cwd is not None else os.getcwd())
            repo = (bindings.get(root) if root else None) or None
            if not repo:
                raise RepoError(
                    "not inside an indexed repository — name one with --repo <name> "
                    "(repo=\"<name>\" as a tool argument), or search every one with --all "
                    "(across=true)")
        knob = {"limit": limit} if across else {"group_size": limit}

        return self.search(query, repo=repo, across=across, **knob)

    def _to_hit(self, raw: dict) -> RepoHit:
        payload = raw.get("payload") or {}
        md = payload.get("metadata") or {}
        path = md.get("path", "?")

        return RepoHit(
            score=float(raw.get("score") or 0.0),
            repo=payload.get("repo", "?"),
            path=path,
            start_line=int(md.get("start_line") or 0),
            end_line=int(md.get("end_line") or 0),
            mode=md.get("mode", "snapshot"),
            text=payload.get("document", ""),
            indexed_at=md.get("indexed_at", "?"),
            # Reported, never hidden: a chunk from an older state of the file must degrade to
            # "this changed" instead of answering as if it were current.
            stale=source_changed(path, md.get("src_mtime"), md.get("src_size"),
                                 md.get("src_digest")),
        )

    def list_request(self) -> dict:
        """What a LISTING is, on either host: the registered repos AND the divergent ones.

        The two calls are one operation because leaving the second one out is precisely what
        makes a divergent repo invisible — chunks, no entry, and therefore no name to drop it
        by. Shared so that one host cannot quietly answer with half of it.
        """
        return {"repos": self.list_repos(), "divergent": self.divergent_repos()}

    # ---- deleting -----------------------------------------------------------

    def drop_request(self, repo: str, confirmed: bool) -> dict:
        """The guarded deletion BOTH HOSTS call: refuses unless it was confirmed.

        The archive this removes is permanent by design and there is no sweep to undo it, so
        the confirmation is the only thing between a mistyped name and a lost index. It sits
        here, next to the deletion, and not in each adapter, because a guard that lives in
        the adapters is a guard the second host can be written without — and the tool surface
        is the host where a MODEL, not a person, chooses the arguments.
        """
        if not repo:
            raise RepoError("a repository name is required")
        if not confirmed:
            raise RepoError(
                f"this permanently deletes the archive of {repo!r} and there is no undo. "
                f"Confirm it explicitly: --yes on the CLI, \"yes\": true as a tool argument.")

        return self.drop_repo(repo)

    def drop_repo(self, repo: str) -> dict:
        """Deletes a repository: chunks, then the entry, then the local bindings.

        THE ORDER IS LOAD-BEARING. Registry first with the chunks failing leaves chunks no
        listing can reach, which still compete in an `across` search — unreachable garbage.
        Chunks first with the registry failing leaves an entry pointing at zero chunks:
        visible, and a second run finishes the job. Prefer the recoverable remainder.

        EACH STEP REPORTS WHAT IT ACTUALLY DID, because the order is only half the promise: a
        failure that misnames which steps completed sends the caller looking for damage that
        is not there, or leaves them unaware of damage that is.

        AND THE PROMISE HOLDS FOR ALL THREE STEPS, which is why a repo with no entry but with
        local bindings is FINISHED here instead of refused. Reviewed 2026-08-17: a rerun after
        a step-1 or step-2 failure finishes the job because the entry is still there to find,
        but a step-3 failure used to be terminal — the entry is correctly gone, so the rerun
        answered "is not indexed" and the stray binding was unreachable forever, `forget_repo`
        having no other caller. "Break toward the state a rerun can fix" has to be true of the
        step it was written for. Dropping a repository means making it not exist ANYWHERE, so
        finishing a half-done drop IS the request, not a different one.
        """
        if not self.get_repo(repo):
            # No entry. Either a half-done drop to finish, or a repo that never existed — and
            # the bindings are what tell them apart, so they are cleared BEFORE deciding. When
            # nothing was bound, nothing was cleared and the refusal below is unchanged.
            stale = self._forget_bindings(repo)
            if not stale:
                raise RepoError(f"repository {repo!r} is not indexed")

            # Deliberately NOT touching the chunks: chunks without an entry are a divergence
            # this method must not silently absorb. `list_repos` refuses to invent an entry for
            # them and `divergent_repos` names them for a human — see its docstring.
            return {"repo": repo, "unbound": stale, "already_gone": True}
        try:
            self.q.delete_by_filter(self.chunks_name,
                                    {"must": [{"key": "repo", "match": {"value": repo}}]})
        except RepoError:
            raise
        except Exception as exc:                        # noqa: BLE001
            raise RepoError(f"the chunks of {repo!r} could not be deleted, so its entry was "
                            f"kept for a second attempt: {exc}") from exc
        # Deliberately bare, unlike its neighbours: the store adapter raises `QdrantError`,
        # which is already a `CoreError` the CLI boundary catches, and the recoverable
        # remainder this ordering exists to produce is exactly what a failure here leaves.
        self.q.delete_points(self.registry_name, [_point_id(f"registry:{repo}", 0)])
        # Last, and only once the archive agrees: a checkout must never claim to belong to a
        # repo that is gone, because the next index would write into a phantom.
        unbound = self._forget_bindings(repo)

        return {"repo": repo, "unbound": unbound, "already_gone": False}

    def _forget_bindings(self, repo: str) -> list[str]:
        """Clears the local bindings of `repo`, turning a write failure into a true report.

        Shared by both paths through `drop_repo` rather than copied into each, and that is not
        only tidiness: the finishing path exists BECAUSE this step can fail, so if the two
        described the same failure differently, the second attempt would contradict the first.
        """
        from . import bindings

        try:
            return bindings.forget_repo(repo)
        except RepoError:
            raise
        except Exception as exc:                        # noqa: BLE001
            # STILL A FAILURE — three things were asked for and two were done — but an
            # ACCURATE one. `bindings._save` writes a file, so a read-only or full state dir
            # arrives here as a bare OSError, which reads as "the deletion failed" when the
            # deletion succeeded; a message that misnames which steps completed sends the
            # caller looking for damage that is not there. Say what is true instead — and the
            # rerun it points at is a real path, not a hope: see `drop_repo`.
            raise RepoError(
                f"the archive of {repo!r} WAS deleted, but its local bindings could not be "
                f"cleared: {exc}. Nothing points into a phantom, since a binding naming a "
                f"repo the registry no longer knows already reads as unbound, and dropping "
                f"{repo!r} again clears the remainder."
            ) from exc

    def divergent_repos(self) -> list[str]:
        """Repos with chunks and no registry entry.

        This exists because the design has TWO sources of truth about which repos exist: the
        registry, which is authoritative, and the `repo` field on every chunk, which is
        derived. Naming the owner is half the defence; this is the other half — without it the
        copy is free to diverge unobserved.

        IT DETECTS ONE DIRECTION AND ONLY ONE: chunks with no entry. The opposite divergence —
        an ENTRY POINTING AT ZERO CHUNKS, which is exactly what a failure of `drop_repo`'s
        registry step leaves — is NOT reported here, and that is a decision rather than an
        omission (review, 2026-08-17).

        WHY IT CANNOT BE DETECTED, as opposed to merely not being implemented yet: an entry
        with zero chunks is BYTE-FOR-BYTE the state of a repository that was just registered
        and has not been indexed yet, which is the normal path: `register` and `add_files` are
        separate calls by design, and every repo passes through that state on its way in.
        Nothing in the data distinguishes "half-deleted" from "brand new", so reporting it
        would cry wolf on every fresh registration, and a divergence report that fires on the
        happy path is one users learn to ignore — which costs more than the case it catches.

        WHAT IT WOULD TAKE, if it is ever wanted: the registry would have to record a chunk
        count at index time and `add_files` would have to maintain it, so that "expected N, has
        zero" became a statement about a promise instead of a guess. That is a data-model
        change with its own drift problem (a stored count is a third copy of a fact), and it is
        deliberately not this method's to invent.
        """
        known = {r["repo"] for r in self.list_repos()}
        seen = {(p.get("payload") or {}).get("repo")
                for p in self.q.scroll_all(self.chunks_name)}

        return sorted(r for r in seen if r and r not in known)
