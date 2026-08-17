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

        groups = []
        for group in raw[:limit]:
            hits = [self._to_hit(h) for h in group.get("hits", [])]
            if hits:
                groups.append({"repo": group.get("id"), "hits": hits})

        return {"scope": "across" if across else "repo", "repo": None if across else repo,
                "groups": groups, "truncated": len(raw) > limit}

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
