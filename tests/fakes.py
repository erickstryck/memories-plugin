"""In-memory fakes for the `core.ports` contracts.

They exist because the most important property of the package — the stored payload
being byte-compatible with that of the server it replaces — was verified ONLY against
live infrastructure. A test that requires Qdrant and a GPU does not run while you edit,
and what does not run does not protect.

No fake inherits from anything: the contracts are `Protocol`s, so having the methods is
enough. `FakeVectorStore` keeps the points as they are, normalizing nothing, precisely
so a test can assert on the payload KEYS.
"""
import math
import zlib


class FakeVectorStore:
    """A vector store in a dict. Real similarity (cosine), not simulated."""

    def __init__(self):
        self.collections: dict[str, dict] = {}   # name -> {"size", "points": {id: point}}
        self.calls: list[tuple] = []

    # ---- collections ----
    def list_collections(self) -> list[str]:
        return list(self.collections)

    def collection_info(self, name: str) -> dict | None:
        c = self.collections.get(name)
        if c is None:
            return None

        return {"size": c["size"], "distance": "Cosine", "points": len(c["points"])}

    def ensure_collection(self, name: str, size: int, distance: str = "Cosine") -> bool:
        if name in self.collections:
            if self.collections[name]["size"] != size:
                raise ValueError(f"dimension {self.collections[name]['size']} != {size}")

            return False
        self.collections[name] = {"size": size, "points": {}}
        self.calls.append(("ensure_collection", name))

        return True

    def ensure_payload_index(self, name: str, field: str, schema: str) -> None:
        self.calls.append(("ensure_payload_index", name, field))

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    # ---- points ----
    def upsert(self, name: str, points: list[dict], batch: int = 256) -> int:
        self.ensure_collection(name, self.collections.get(name, {}).get("size", 4))
        for p in points:
            self.collections[name]["points"][p["id"]] = p

        return len(points)

    def get_point(self, name: str, point_id):
        return self.collections.get(name, {}).get("points", {}).get(point_id)

    def set_payload(self, name: str, point_id, payload: dict) -> None:
        point = self.collections.get(name, {}).get("points", {}).get(point_id)
        if point is not None:
            point["payload"] = payload
        self.calls.append(("set_payload", name, point_id))

    def delete_points(self, name: str, ids: list) -> None:
        for i in ids:
            self.collections.get(name, {}).get("points", {}).pop(i, None)

    def delete_by_filter(self, name: str, filter_: dict) -> None:
        points = self.collections.get(name, {}).get("points", {})
        for pid in [p for p, v in points.items() if _matches_filter(v.get("payload", {}), filter_)]:
            points.pop(pid)

    def search(self, name: str, vector: list[float], limit: int,
               filter_: dict | None = None, with_payload: bool = True) -> list[dict]:
        self.calls.append(("search", name, limit))
        output = []
        for pid, p in self.collections.get(name, {}).get("points", {}).items():
            if filter_ and not _matches_filter(p.get("payload", {}), filter_):
                continue
            output.append({"id": pid, "score": _cosine(vector, p["vector"]),
                          "payload": p.get("payload", {})})
        output.sort(key=lambda h: -h["score"])

        return output[:limit]

    def search_groups(self, name: str, vector: list[float], group_by: str, limit: int,
                      group_size: int, filter_: dict | None = None,
                      with_payload: bool = True) -> list[dict]:
        """Real grouping over the real cosine ranking, so the shadowing test means
        something. Over-fetches deliberately: grouping the top-K is the defect this method
        exists to avoid, so the fake must not reproduce it."""
        ranked = self.search(name, vector, limit=len(self.collections.get(name, {}).get("points", {})),
                             filter_=filter_, with_payload=True)
        groups: dict = {}
        for hit in ranked:
            key = (hit.get("payload") or {}).get(group_by)
            if key is None:
                continue
            groups.setdefault(key, []).append(hit)
        out = [{"id": key, "hits": hits[:group_size]} for key, hits in groups.items()]
        out.sort(key=lambda g: g["hits"][0]["score"], reverse=True)

        return out[:limit]

    def scroll(self, name: str, limit: int = 256, offset=None,
               with_vector: bool = False, filter_: dict | None = None):
        items = [{"id": pid, "payload": p.get("payload", {})}
                 for pid, p in self.collections.get(name, {}).get("points", {}).items()
                 if not filter_ or _matches_filter(p.get("payload", {}), filter_)]

        return items[:limit], None

    def scroll_all(self, name: str, filter_: dict | None = None, with_vector: bool = False,
                   payload_fields: list[str] | None = None):
        """HONOURS `payload_fields` BY ACTUALLY DROPPING THE OTHERS, rather than merely
        accepting the argument. A fake that returns the whole payload where production returns
        one projected field hides the exact bug the projection can introduce: a caller that
        goes on to read a key it no longer asked for works here and fails against a real
        Qdrant. The fake has to be as poor as the real thing.
        """
        items, _ = self.scroll(name, limit=10_000, filter_=filter_)
        for item in items:
            if payload_fields:
                keep = {k: v for k, v in (item.get("payload") or {}).items()
                        if k in payload_fields}
                item = dict(item, payload=keep)
            yield item


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0

    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _matches_filter(payload: dict, filter_: dict) -> bool:
    """Supports the two shapes the package uses: `match.value` and `range`."""
    for cond in filter_.get("must", []):
        key = cond.get("key")
        value = payload.get(key)
        if "match" in cond:
            if value != cond["match"].get("value"):
                return False
        elif "range" in cond:
            range_ = cond["range"]
            if value is None:
                return False
            if "gt" in range_ and not value > range_["gt"]:
                return False
            if "lt" in range_ and not value < range_["lt"]:
                return False

    return True


class FakeEmbedder:
    """Deterministic vectors derived from the text.

    Deterministic rather than random so the similarity is PREDICTABLE: texts that share
    words end up close together, which makes it possible to test real retrieval instead
    of testing the fake.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))

        return [self._vector_for(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def detect_dimension(self) -> int:
        return self.dim

    def _vector_for(self, text: str) -> list[float]:
        # `crc32` and NOT `hash()`: Python's string hash is salted per process, so the
        # same text lands in different positions on each run. A fake whose similarity
        # changes between runs produces a test that passes today and fails tomorrow with
        # nothing having changed — worse than having no fake at all.
        vec = [0.0] * self.dim
        for word in text.lower().split():
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        if not any(vec):
            vec[0] = 1.0

        return vec


class FailingFakeEmbedder:
    """Raises the domain error. For exercising the degradation path."""

    def __init__(self, error):
        self.error = error

    def embed(self, texts):
        raise self.error

    def embed_one(self, text):
        raise self.error


class FakeReranker:
    """Returns the scores it was given, against the order of the documents received."""

    def __init__(self, scores=None, ok=True, error=None, was_logit=False):
        self.scores = scores
        self.ok = ok
        self.error = error
        self.was_logit = was_logit
        self.calls: list[tuple] = []

    def rank(self, query, documents):
        self.calls.append((query, list(documents)))
        if not self.ok:
            return [], {"ok": False, "error": self.error or "failed", "was_logit": False}
        scores = self.scores if self.scores is not None else [1.0] * len(documents)
        pairs = sorted(enumerate(scores[:len(documents)]), key=lambda p: -p[1])

        return pairs, {"ok": True, "error": None, "was_logit": self.was_logit}


class RecordingVectorStore:
    """A `FakeVectorStore` that remembers WHICH operations were asked of it.

    Derived, not enumerated: `__getattr__` wraps whatever is asked for, so an operation
    added to the contract tomorrow is recorded without this class being touched. That is the
    point — the question it answers is "what did this code path do to the backend", and a
    hand-written list of methods could only ever answer "did it do one of the things I
    thought of".

    `FakeVectorStore` keeps a `calls` list of its own, but only for the four operations its
    other users assert on; a path that upserted would leave no trace in it.
    """

    def __init__(self):
        self.inner = FakeVectorStore()
        self.ops: list[str] = []

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def recorded(*args, **kwargs):
            self.ops.append(name)

            return attr(*args, **kwargs)

        return recorded

    def points(self) -> int:
        """How many points exist anywhere in it — zero is "nothing was indexed"."""
        return sum(len(c["points"]) for c in self.inner.collections.values())


# ---- fixtures the product can no longer build by itself ----------------------


def make_divergent(ix, repo: str, path: str) -> None:
    """Leave CHUNKS WITH NO REGISTRY ENTRY: the state a half-done drop leaves behind.

    Built the way it actually happens — declare, index, lose the entry — and NOT by indexing
    an undeclared name, because `RepoIndex.add_files` refuses that on purpose: the ordinary
    path must not be able to manufacture the divergence `divergent_repos` exists to denounce.
    A fixture taking the forbidden shortcut would be asserting on a state the product can no
    longer reach, which is how a divergence test outlives the divergence it tested.

    One definition, shared by every test that needs the state, because the four that need it
    are in four files and a private copy of "how to break the archive" in each is how they
    stop agreeing about what broken means.
    """
    from core.docs import _point_id

    ix.register_request(repo)
    ix.add_files(repo, [path])
    ix.q.delete_points(ix.registry_name, [_point_id(f"registry:{repo}", 0)])


def make_emptied(ix, repo: str, path: str) -> None:
    """Leave an ENTRY CLAIMING CHUNKS OVER AN ARCHIVE THAT HAS NONE: the other divergence.

    It is what a `drop_repo` that fails at its registry step leaves — the recoverable
    remainder the chunks-first ordering exists to produce — so it is built that way here,
    by letting the registry delete fail, and not by deleting the chunks by hand. Same rule
    as `make_divergent`: a fixture that assembles a state through a path the product does
    not have is asserting about a state the product cannot reach.
    """
    ix.register_request(repo)
    ix.add_files(repo, [path])
    original = ix.q.delete_points

    def refuse(*a, **kw):
        raise OSError("the registry delete failed")

    ix.q.delete_points = refuse
    try:
        ix.drop_repo(repo)
    except Exception:                                   # noqa: BLE001
        pass
    finally:
        ix.q.delete_points = original
