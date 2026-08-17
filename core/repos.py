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

from . import ports
from .chunk import chunk_text, mode_for_suffix
from .docs import _iso, _point_id, _read_source, content_digest, doc_id_for
from .errors import CoreError


class RepoError(CoreError):
    """Something about a repository archive operation could not be done."""


#: The registry stores no meaning in its vector — it is a key-value table that happens to
#: live in Qdrant, read by scroll and never by similarity. Size 1 says so out loud, and a
#: unit vector avoids the zero-norm that Cosine has no answer for.
REGISTRY_VECTOR_SIZE = 1
REGISTRY_VECTOR = [1.0]


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
