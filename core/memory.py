"""Long-term semantic memory.

It replaces a hand-made MCP server without migrating a single record: the PAYLOAD
FORMAT is deliberately identical to what is already stored —
`{document, metadata, created_at, updated_at}` — so an existing archive stays
readable and writable by this module with no conversion, no reindexing and no window
of risk.

Two directions, and the reading one is the one usually forgotten:
  - `recall` / `find`: search BEFORE asserting, so as not to re-decide what has
    already been decided nor contradict what has already been measured.
  - `store` / `update`: persist a durable fact, one per record.

`recall` implements the TWO-GATE pipeline: the dense stage relaxes the floor to gain
reach and the cross-encoder supplies the precision. The asymmetry matters — if the
second gate does not run, the first one's permissive floor MUST go back to the strict
value, otherwise the mode with re-ranking ends up worse than the mode without, which
is the opposite of the intent.
"""
import uuid
from dataclasses import dataclass

from . import ports, retrieval
from .errors import CoreError
from datetime import datetime, timezone


class MemoryStoreError(CoreError):
    """Named after `MemoryStore`, like every other error here is named after its subject.

    It was `MemoryError_` — the trailing underscore dodging the builtin `MemoryError`,
    which means something entirely different and would have been a genuinely confusing
    thing to shadow. The escape character was the only one in the package.
    """


@dataclass
class Recalled:
    id: str
    score: float
    origin: str            # "CE" or "dense"
    dense_score: float
    document: str
    metadata: dict
    updated_at: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


#: The metadata shortcuts every host offers by name, declared once so no caller retypes
#: them. They are conventions, not a schema — `metadata` accepts any keys.
METADATA_FIELDS = ("type", "project", "area")


def metadata_from(meta=None, **fields) -> dict:
    """Assemble a memory's metadata from a base object plus the named shortcuts.

    One definition because there are two callers with the same rule: the CLI's
    `--json-meta` with `--type/--project/--area`, and the hermes `memory_store` /
    `memory_update` tools' `metadata` object with the same three. The precedence is that
    an explicit FIELD WINS over the base, so `--type reference` fixes one label without
    the caller having to rewrite the whole object.

    A `None` field is absent, not empty: writing `{"project": None}` reads back later as
    a project named nothing rather than as no project at all.

    `meta` may be a dict or a JSON object STRING, because the CLI passes a string and a
    model asked for an object emits one often enough that refusing it would spend a tool
    call on syntax. Anything else raises — including a JSON scalar or array, since
    silently dropping metadata the caller believes it wrote is the worse failure: the
    record lands unlabelled and nothing says so.
    """
    if meta is None or meta == "":
        base = {}
    elif isinstance(meta, dict):
        base = dict(meta)
    elif isinstance(meta, str):
        import json
        try:
            parsed = json.loads(meta)
        except ValueError as exc:
            raise MemoryStoreError(f"metadata is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            raise MemoryStoreError(
                f"metadata must be a JSON object, got {type(parsed).__name__}")
        base = parsed
    else:
        raise MemoryStoreError(
            f"metadata must be an object or a JSON object string, got {type(meta).__name__}")
    for key, value in fields.items():
        if value is not None:
            base[key] = value

    return base


class MemoryStore:
    def __init__(self, qdrant: ports.VectorStore, embedder: ports.EmbeddingModel,
                 reranker: ports.RerankModel | None, collection: str, vector_size: int):
        self.q = qdrant
        self.embedder = embedder
        self.reranker = reranker
        self.collection = collection
        self.vector_size = vector_size

    def ensure(self) -> None:
        """Ensures the collection exists. For the WRITE path only."""
        self.q.ensure_collection(self.collection, self.vector_size)

    def require_existing(self) -> None:
        """Requires the collection to exist. For the READ path.

        Reading must not CREATE: with a typo in the name, `ensure` created an empty
        collection, the search returned zero hits and the consumer concluded "there is
        no recorded precedent on this subject". That is, a configuration typo turned
        into the most dangerous statement this system can make. Better to fail loudly —
        and the hook turns that into an explicit unavailability warning.
        """
        if self.q.collection_info(self.collection) is None:
            raise MemoryStoreError(
                f"memory collection {self.collection!r} does not exist. Check the name with "
                f"`collections list`; nothing is created by a read."
            )

    # ---- writing -----------------------------------------------------------

    def store(self, information: str, metadata: dict | None = None) -> dict:
        if not information or not information.strip():
            raise MemoryStoreError("an empty memory is not writable")
        self.ensure()
        mid = str(uuid.uuid4())
        vector = self.embedder.embed_one(information)
        now_ts = _now()
        self.q.upsert(self.collection, [{
            "id": mid,
            "vector": vector,
            "payload": {"document": information, "metadata": metadata or {},
                        "created_at": now_ts, "updated_at": now_ts},
        }])

        return {"status": "created", "id": mid}

    def store_many(self, items: list[dict]) -> dict:
        """A batch that embeds before it writes, all-or-nothing.

        The vectors are generated BEFORE any write: a timeout halfway through the batch
        does not leave half of it stored, which is the worst possible state for an archive
        where a partial duplicate is indistinguishable from a new fact.

        It used to claim ONE trip to the embeddings endpoint, which is true only up to
        `EMBED_BATCH` (32) items — measured, 40 items cost 2 requests. The all-or-nothing
        guarantee does not depend on it and survives: every request happens before the
        first write. It does weaken past `Qdrant.upsert`'s own batch of 256 points, where
        the write itself becomes several calls; nothing here writes batches that large,
        and saying so is better than implying a guarantee that stops at a size nobody
        states.
        """
        if not items:
            return {"status": "noop", "ids": [], "count": 0}
        texts = []
        for i, item in enumerate(items):
            info = (item or {}).get("information")
            if not isinstance(info, str) or not info.strip():
                raise MemoryStoreError(f"items[{i}] needs 'information' (a non-empty string)")
            texts.append(info)
        self.ensure()
        vectors = self.embedder.embed(texts)
        now_ts = _now()
        points, ids = [], []
        for item, text, vector in zip(items, texts, vectors):
            mid = str(uuid.uuid4())
            ids.append(mid)
            points.append({
                "id": mid,
                "vector": vector,
                "payload": {"document": text, "metadata": (item or {}).get("metadata") or {},
                            "created_at": now_ts, "updated_at": now_ts},
            })
        self.q.upsert(self.collection, points)

        return {"status": "created", "ids": ids, "count": len(ids)}

    def update(self, mid: str, information: str | None = None,
               metadata: dict | None = None) -> dict:
        point = self.q.get_point(self.collection, mid)
        if point is None:
            return {"status": "not_found", "id": mid}
        previous = point.get("payload", {})
        previous_doc = previous.get("document", "")
        new_doc = information if information is not None else previous_doc
        new_meta = metadata if metadata is not None else previous.get("metadata", {})
        payload = {"document": new_doc, "metadata": new_meta,
                   "created_at": previous.get("created_at", _now()), "updated_at": _now()}

        # Unchanged text means an unchanged vector: recomputing it would pay a network
        # round trip to get the same number back. Worse, it made fixing a label
        # impossible while the embedding endpoint was down — an operation that does not
        # depend on it.
        # Compared against the SAME default the fallback used. Reading it twice with
        # different defaults meant a payload with no `document` key produced "" on one
        # side and None on the other: they compared unequal, so a metadata-only update
        # took the re-embed branch, embedded the empty string and REPLACED the point's
        # vector. It corrupted rather than failed, which is the worse of the two.
        if new_doc == previous_doc:
            self.q.set_payload(self.collection, mid, payload)

            return {"status": "updated", "id": mid, "reembedded": False}

        point_with_vector = {"id": mid, "vector": self.embedder.embed_one(new_doc),
                             "payload": payload}
        self.q.upsert(self.collection, [point_with_vector])

        return {"status": "updated", "id": mid, "reembedded": True}

    def delete(self, mid: str) -> dict:
        self.q.delete_points(self.collection, [mid])

        return {"status": "deleted", "id": mid}

    # ---- reading -----------------------------------------------------------

    def find(self, query: str, limit: int = 5) -> list[dict]:
        """Pure dense search, no re-ranking. Cheap; use it when the order among the
        relevant results does not matter (for example to dedupe before writing)."""
        self.require_existing()
        vector = self.embedder.embed_one(query)
        output = []
        for hit in self.q.search(self.collection, vector, limit):
            p = hit.get("payload", {})
            output.append({
                "id": hit.get("id"),
                "score": round(hit.get("score", 0.0), 4),
                "document": p.get("document"),
                "metadata": p.get("metadata", {}),
                "updated_at": p.get("updated_at"),
            })

        return output

    def recall(self, queries: list[str], policy: retrieval.Policy, top_k: int,
               suppressed: str | None = None) -> tuple[list[Recalled], retrieval.Outcome]:
        """Retrieves memories for several ANGLES on the same question.

        This class handles the FIRST stage — embeddings, vector search and fusion. The
        second stage and the selection policy live in `retrieval`, shared with document
        search: while they were two separate pieces of code, a fix landed in one and not
        the other.

        The angles go out in a single embeddings call, because the endpoint accepts
        `input` as an array. Fusion is by id keeping the HIGHEST score: a record that
        shows up in two angles should not be penalized by the worse of the two.
        """
        self.require_existing()
        vectors = self.embedder.embed(queries)
        batches = [[self._flatten_hit(h) for h in self.q.search(self.collection, v, top_k)]
                 for v in vectors]
        batches = [[h for h in batch if h is not None] for batch in batches]
        fused = retrieval.fuse_by_id(batches, id_of=lambda h: h["id"])

        floor = policy.floor_for(self.reranker is not None)
        candidates = [h for h in fused if h["score"] >= floor]
        outcome = retrieval.two_stage(candidates, queries[0], self.reranker, policy,
                                   text_of=lambda h: h["document"], suppressed=suppressed)
        # The pipeline's `best_dense` only sees the candidates; to say "nothing cleared
        # the cut, the best was X" the useful number is the best of ALL the hits.
        if fused:
            outcome.best_dense = fused[0]["score"]

        return [self._to_recalled(s) for s in outcome.scored], outcome

    @staticmethod
    def _flatten_hit(hit: dict) -> dict | None:
        """Flattens the store's hit into the shape the pipeline consumes.

        Returns None for a record with no usable document: a vector without text can
        neither be judged by the cross-encoder nor presented to anyone.
        """
        p = hit.get("payload", {}) or {}
        doc = p.get("document")
        if not isinstance(doc, str) or not doc.strip():
            return None

        return {
            "id": str(hit.get("id")),
            "score": float(hit.get("score") or 0.0),
            "document": doc.strip(),
            "metadata": p.get("metadata") or {},
            "updated_at": p.get("updated_at"),
        }

    @staticmethod
    def _to_recalled(s: retrieval.Scored) -> Recalled:
        h = s.item

        return Recalled(id=h["id"], score=s.score, origin=s.origin,
                        dense_score=h["score"], document=h["document"],
                        metadata=h["metadata"], updated_at=h.get("updated_at"))

    def get(self, mid: str) -> dict:
        # `require_existing` here for the same reason `find` and `recall` have it, and it
        # was missing on exactly the entry point the memory skill tells the model to use
        # when following a pointer. With a typo in the collection name a real, existing id
        # came back as `not_found` — which reads as "that memory was deleted", not "you
        # are looking in the wrong place". A read must never turn a configuration mistake
        # into a statement about the archive's contents.
        self.require_existing()
        point = self.q.get_point(self.collection, mid)
        if point is None:
            return {"status": "not_found", "id": mid}
        p = point.get("payload", {})

        return {"id": point.get("id"), "document": p.get("document"),
                "metadata": p.get("metadata", {}), "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at")}

    def list_page(self, limit: int = 20, offset=None) -> dict:
        self.require_existing()
        points, next_offset = self.q.scroll(self.collection, limit=limit, offset=offset)
        memories = [{
            "id": pt.get("id"),
            "document": pt.get("payload", {}).get("document"),
            "metadata": pt.get("payload", {}).get("metadata", {}),
            "updated_at": pt.get("payload", {}).get("updated_at"),
        } for pt in points]

        return {"count": len(memories), "memories": memories, "next_offset": next_offset}

    def count(self) -> int | None:
        """Points in the collection, or None when it does not exist yet.

        NOT `require_existing`: None here is a real answer with a real use — the wizard
        and `setup --check` call this to decide whether a collection is worth suggesting,
        before anything has been written. The caller distinguishes None from 0.
        """
        info = self.q.collection_info(self.collection)

        return info.get("points") if info else None


def search_collections(qdrant, embedder, query: str, collections: list[str] | None,
                       vector_size: int, limit: int = 5,
                       max_results: int = 25) -> dict:
    """READ-ONLY search across arbitrary collections.

    It serves to query archives belonging to other systems that share the same
    embedding model. A collection with a different dimension is SKIPPED and reported —
    reading silently from an archive built by another model returns random neighbours
    with a plausible score, which is worse than returning nothing.
    """
    vector = embedder.embed_one(query)
    targets = collections if collections else qdrant.list_collections()
    searched, skipped_cols, results = [], [], []
    for name in targets:
        info = qdrant.collection_info(name)
        if info is None or info.get("size") is None:
            skipped_cols.append({"collection": name, "reason": "not found / named vector"})
            continue
        if info["size"] != vector_size:
            skipped_cols.append({"collection": name, "reason": f"dimension {info['size']} ≠ {vector_size}"})
            continue
        try:
            hits = qdrant.search(name, vector, limit)
        except Exception as exc:
            skipped_cols.append({"collection": name, "reason": f"error: {type(exc).__name__}"})
            continue
        searched.append(name)
        for hit in hits:
            p = hit.get("payload", {}) or {}
            doc = None
            for key in ("document", "text", "content", "page_content", "chunk", "body"):
                if isinstance(p.get(key), str) and p[key]:
                    doc = p[key]
                    break
            results.append({
                "collection": name, "id": hit.get("id"),
                "score": round(hit.get("score", 0.0), 4), "document": doc,
                "metadata": p.get("metadata", {}),
                "payload": None if doc is not None else p,
            })
    results.sort(key=lambda h: -h["score"])

    return {"searched": sorted(searched), "skipped": skipped_cols,
            "total_found": len(results), "results": results[:max_results]}
