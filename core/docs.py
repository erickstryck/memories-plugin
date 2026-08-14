"""Document index, in two archives with different lifecycles.

The problem: answering a question about 40 lines of an 8,000-line file should not cost
the whole file in context. The file is read from disk by this process, sliced,
embedded and stored; the search returns only the chunks that answer — and, when the
file is re-readable, it returns the LOCATION so the consumer reads the current content
instead of a snapshot.

TWO ARCHIVES, and the separation is structural, not a convention:

  - TEMPORARY (`tmp`): a document opened for one task. Each chunk carries
    `expires_at_ts` and every operation sweeps what expired. Qdrant has no native
    expiry, so the TTL is exactly this: a delete-by-filter on each call, no daemon.
    This archive is DESTROYABLE by construction — there is a command that deletes the
    collection.

  - LIBRARY (`library`): a document worth keeping for reference. No TTL, never swept,
    and out of reach of any cleanup command. It lives in its own collection precisely
    so the temporary archive's destroyability cannot reach it.

Neither of them is the MEMORY collection: a file chunk competing with a curated fact in
a recall search wins on volume and drowns the archive that matters most.
"""
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from . import ports, retrieval
from .errors import CoreError
from .chunk import chunk_text, is_probably_binary, mode_for_suffix

DEFAULT_TTL_SECONDS = 24 * 3600
DENSE_TOP_K = 20
RERANK_MIN_SCORE = 0.10
#: First-stage floor in document search. More generous than in memory because here the
#: second stage does not veto — the goal is not to lose a candidate.
DENSE_FLOOR = 0.30

# The cross-lingual collapse threshold lives in `retrieval`, next to the logic that
# uses it — here we only pick the policy.

SCOPES = ("all", "tmp", "library")


class DocsError(CoreError):
    pass


@dataclass
class Hit:
    score: float
    origin: str          # "CE" when the cross-encoder judged it, "dense" when it did not
    scope: str           # tmp | library
    path: str
    start_line: int
    end_line: int
    mode: str            # locator | snapshot
    text: str
    indexed_at: str
    stale: str | None    # the reason, when the file changed since indexing


def parse_ttl(spec: str) -> float:
    """Accepts `30m`, `24h`, `7d` or bare seconds."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", str(spec).strip().lower())
    if not m:
        raise DocsError(f"invalid TTL: {spec!r} (use 30m, 24h, 7d or seconds)")
    mult = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]

    return float(m.group(1)) * mult


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def doc_id_for(path: str) -> str:
    """An id derived from the absolute path, so reindexing the same file REPLACES the
    previous index instead of accumulating two competing versions."""
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]


def _point_id(doc_id: str, ix: int) -> int:
    """A stable, deterministic numeric id (Qdrant accepts an integer or a UUID)."""
    return int(hashlib.sha1(f"{doc_id}:{ix}".encode()).hexdigest()[:15], 16)


MTIME_TOLERANCE = 0.001

#: Reason returned when the source file is gone. It is a constant because `refresh`
#: BRANCHES on it — comparing against a literal spelled out in two places breaks
#: silently the day one of them is reworded.
GONE = "file no longer exists"


def content_digest(content: str) -> str:
    """What the file WAS when we indexed it, independent of any filesystem metadata."""
    return hashlib.sha1(content.encode("utf-8", "replace")).hexdigest()[:16]


def source_changed(path: str, src_mtime, src_size, src_digest=None) -> str | None:
    """The staleness reason, or None if the file matches what was indexed.

    The digest is the answer when we have one, because it is the only comparison that
    cannot be fooled. Metadata can: `cp -p`, `rsync --times`, `touch -r` and any tar or
    backup restore preserve the mtime, and an edit that swaps one character preserves the
    size — so a genuinely changed file reports itself unchanged. The file is fully read at
    index time anyway, so hashing it costs nothing we were not already paying.

    Metadata remains the fallback for documents indexed before the digest existed, and
    there it COMPARES WITH A TOLERANCE, which is not sloppiness: `st_mtime` is a float and
    the JSON round trip loses the last bits (measured: 1786646270.9956777 written against
    1786646270.9956775 on disk, a 2.4e-7 difference). Exact equality produced a false
    positive on EVERY search of EVERY document — the warning became ignorable noise and
    `refresh` reindexed the whole archive on each run, paying for embeddings for nothing.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return GONE
    if src_digest:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                if content_digest(fh.read()) != src_digest:
                    return "file contents changed since indexing"

                return None
        except OSError:
            return GONE
    if src_size is not None and st.st_size != src_size:
        return "file size changed since indexing"
    if src_mtime is not None and abs(st.st_mtime - float(src_mtime)) > MTIME_TOLERANCE:
        return "file changed since indexing"

    return None


def _read_source(path: str) -> tuple[str, os.stat_result, str]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise DocsError(f"not a file: {path}")
    st = os.stat(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    if is_probably_binary(content[:8192]):
        raise DocsError("the file looks binary — convert it to text before indexing")

    return path, st, content


class DocIndex:
    def __init__(self, qdrant: ports.VectorStore, embedder: ports.EmbeddingModel,
                 reranker: ports.RerankModel | None, tmp_collection: str,
                 library_collection: str, vector_size: int):
        self.q = qdrant
        self.embedder = embedder
        self.reranker = reranker
        self.collections = {"tmp": tmp_collection, "library": library_collection}
        self.vector_size = vector_size

    # ---- maintenance -------------------------------------------------------

    def _collection(self, scope: str) -> str:
        if scope not in self.collections:
            raise DocsError(f"invalid scope: {scope!r} (use tmp or library)")

        return self.collections[scope]

    def ensure(self, scope: str) -> None:
        name = self._collection(scope)
        if self.q.ensure_collection(name, self.vector_size):
            self.q.ensure_payload_index(name, "doc_id", "keyword")
            if scope == "tmp":
                self.q.ensure_payload_index(name, "expires_at_ts", "float")

    def sweep(self) -> None:
        """Deletes what expired — in the temporary archive only. The library is never
        swept."""
        self.ensure("tmp")
        self.q.delete_by_filter(
            self._collection("tmp"),
            {"must": [{"key": "expires_at_ts", "range": {"lt": time.time()}}]},
        )

    # ---- indexing ----------------------------------------------------------

    def index_file(self, path: str, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                   doc_id: str | None = None) -> dict:
        """Indexes as TEMPORARY, with a TTL."""
        return self._write(path, "tmp", ttl_seconds, doc_id)

    def keep_file(self, path: str, doc_id: str | None = None) -> dict:
        """Keeps it in the LIBRARY, with no expiry."""
        return self._write(path, "library", None, doc_id)

    def _write(self, path: str, scope: str, ttl_seconds: float | None,
               doc_id: str | None) -> dict:
        path, st, content = _read_source(path)
        chunks = chunk_text(content)
        if not chunks:
            raise DocsError("nothing indexable (empty file, or whitespace only)")

        doc_id = doc_id or doc_id_for(path)
        mode = mode_for_suffix(os.path.splitext(path)[1])
        now_ts = time.time()
        name = self._collection(scope)

        self.ensure(scope)
        if scope == "tmp":
            self.sweep()
        # Reindexing replaces: without this, the old version and the new one coexist and
        # the search mixes chunks from two states of the same file.
        self.drop(doc_id, scope)

        vectors = self.embedder.embed([t.text for t in chunks])
        points = []
        for ix, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "document": chunk.text,
                "doc_id": doc_id,
                "metadata": {
                    "path": path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "mode": mode,
                    "scope": scope,
                    # Read by nobody in this package, kept on purpose: when a search
                    # returns a chunk that looks wrong, "which slice of how many" is the
                    # first question, and it is unanswerable after the fact without this.
                    "chunk_ix": ix,
                    "n_chunks": len(chunks),
                    "indexed_at": _iso(now_ts),
                    "src_mtime": round(st.st_mtime, 3),
                    "src_size": st.st_size,
                    "src_digest": content_digest(content),
                },
            }
            if ttl_seconds is not None:
                expires_at = now_ts + ttl_seconds
                payload["expires_at_ts"] = expires_at
                payload["metadata"]["expires_at"] = _iso(expires_at)
                # Store the DURATION, not just the instant: without it `refresh` has no
                # way to know you asked for 1 hour, and reindexed with the 24h default,
                # silently stretching the deadline you chose yourself.
                payload["metadata"]["ttl_seconds"] = ttl_seconds
            points.append({"id": _point_id(doc_id, ix), "vector": vector, "payload": payload})
        self.q.upsert(name, points)

        return {
            "doc_id": doc_id, "path": path, "scope": scope, "collection": name,
            "chunks": len(chunks), "lines": content.count("\n") + 1,
            "chars": len(content), "mode": mode,
            "expires_at": _iso(now_ts + ttl_seconds) if ttl_seconds is not None else None,
        }

    def refresh(self, scope: str = "library") -> list[dict]:
        """Reindexes the documents whose file changed since it was indexed.

        It exists because of the permanent archive: a document stored in August whose
        file changed in October returns a chunk that no longer exists. The warning on
        each hit alerts you; this fixes it.
        """
        report = []
        for doc in self.list_docs(scope):
            path = doc["path"]
            reason = source_changed(path, doc.get("src_mtime"), doc.get("src_size"),
                                    doc.get("src_digest"))
            if reason == GONE:
                report.append({"doc_id": doc["doc_id"], "path": path, "action": "missing"})
                continue
            if reason is None:
                report.append({"doc_id": doc["doc_id"], "path": path, "action": "ok"})
                continue
            if scope == "library":
                res = self.keep_file(path, doc["doc_id"])
            else:
                # Reuse the original DURATION, and deliberately restart it: a document
                # that was just re-read is freshly relevant, so it earns its full window
                # again. What must not happen is falling back to the 24h default, which
                # would silently stretch the deadline the user chose at index time.
                res = self.index_file(path, doc.get("ttl_seconds") or DEFAULT_TTL_SECONDS,
                                      doc["doc_id"])
            report.append({"doc_id": doc["doc_id"], "path": path,
                              "action": "reindexed", "chunks": res["chunks"]})

        return report

    # ---- search ------------------------------------------------------------

    def search(self, query: str, scope: str = "all", doc_id: str | None = None,
               limit: int = 5, min_score: float = RERANK_MIN_SCORE
               ) -> tuple[list[Hit], retrieval.Outcome]:
        """Searches the requested archives and reranks the UNION.

        Reranking the union, rather than each archive separately, is what makes the
        scores comparable: the cross-encoder judges every candidate against the same
        query, so a library chunk and a temporary one compete for the same slot on equal
        footing.

        The policy here differs from memory in two ways, both deliberate: NO VETO,
        because whoever asks has already chosen the document and silence is worse than
        imperfect order; and THE ORDER IS THE PRODUCT, because the result is a list read
        top to bottom — so reranking is worth it even when everything fits.
        """
        if scope not in SCOPES:
            raise DocsError(f"invalid scope: {scope!r} (use {', '.join(SCOPES)})")
        scopes = ("tmp", "library") if scope == "all" else (scope,)

        # The two floors are EQUAL, on purpose: with no veto, "go back to the strict cut"
        # when the judgement does not happen has to be a no-op. If they differed, a
        # cross-lingual collapse (dense in the 0.46 band) would return silence —
        # exactly what this policy exists to prevent.
        policy = retrieval.Policy(
            dense_floor=DENSE_FLOOR, strict_floor=DENSE_FLOOR, min_score=min_score,
            max_results=limit, veto=False, detect_collapse=True, order_matters=True,
        )
        vector = self.embedder.embed_one(query)
        candidates: list[dict] = []
        for sc in scopes:
            for raw in self._search_scope(sc, vector, doc_id):
                candidates.append(raw)
        candidates.sort(key=lambda b: -(b.get("score") or 0.0))

        outcome = retrieval.two_stage(candidates, query, self.reranker, policy,
                                  text_of=lambda b: b["payload"]["document"])

        return [self._to_hit(s.item, s.score, s.origin) for s in outcome.scored], outcome

    def _search_scope(self, scope: str, vector: list[float],
                      doc_id: str | None) -> list[dict]:
        """The first stage within one archive. Only the temporary one filters on expiry."""
        self.ensure(scope)
        must = []
        if scope == "tmp":
            self.sweep()
            must.append({"key": "expires_at_ts", "range": {"gt": time.time()}})
        if doc_id:
            must.append({"key": "doc_id", "match": {"value": doc_id}})
        filter_ = {"must": must} if must else None
        raw_points = self.q.search(self._collection(scope), vector, DENSE_TOP_K, filter_)
        for b in raw_points:
            b["_scope"] = scope

        return raw_points

    def _to_hit(self, raw: dict, score: float, origin: str) -> Hit:
        """Translates the raw hit + the pipeline's verdict into the presentation shape."""
        p = raw.get("payload", {})
        md = p.get("metadata", {})
        path = md.get("path", "?")
        reason = source_changed(path, md.get("src_mtime"), md.get("src_size"),
                                md.get("src_digest"))
        stale = f"{reason} ({md.get('indexed_at')})" if reason else None

        return Hit(
            score=score, origin=origin, scope=raw.get("_scope", md.get("scope", "?")),
            path=path, start_line=md.get("start_line", 0), end_line=md.get("end_line", 0),
            mode=md.get("mode", "snapshot"), text=p.get("document", ""),
            indexed_at=md.get("indexed_at", "?"), stale=stale,
        )

    # ---- inventory and removal ---------------------------------------------

    def list_docs(self, scope: str = "all") -> list[dict]:
        if scope not in SCOPES:
            raise DocsError(f"invalid scope: {scope!r}")
        scopes = ("tmp", "library") if scope == "all" else (scope,)
        by_scope_doc: dict[tuple[str, str], dict] = {}
        for sc in scopes:
            self.ensure(sc)
            if sc == "tmp":
                self.sweep()
            for point in self.q.scroll_all(self._collection(sc)):
                p = point.get("payload", {})
                md = p.get("metadata", {})
                key = (sc, p.get("doc_id", "?"))
                d = by_scope_doc.setdefault(key, {
                    "doc_id": p.get("doc_id", "?"), "scope": sc, "chunks": 0,
                    "path": md.get("path", "?"), "mode": md.get("mode", "?"),
                    "indexed_at": md.get("indexed_at", "?"),
                    "expires_at_ts": p.get("expires_at_ts"),
                    "src_mtime": md.get("src_mtime"), "src_size": md.get("src_size"),
                    "src_digest": md.get("src_digest"),
                    "ttl_seconds": md.get("ttl_seconds"),
                })
                d["chunks"] += 1

        return sorted(by_scope_doc.values(), key=lambda d: (d["scope"], d["path"]))

    def drop(self, doc_id: str, scope: str = "all") -> None:
        scopes = ("tmp", "library") if scope == "all" else (scope,)
        for sc in scopes:
            self.ensure(sc)
            self.q.delete_by_filter(self._collection(sc),
                                    {"must": [{"key": "doc_id", "match": {"value": doc_id}}]})

    def drop_all_tmp(self) -> str:
        """Deletes the entire TEMPORARY collection.

        It exists for the temporary archive only. The library has no equivalent on
        purpose: a permanent archive is pruned document by document, with the id in hand.
        """
        name = self._collection("tmp")
        self.q.delete_collection(name)

        return name
