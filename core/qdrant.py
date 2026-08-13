"""Qdrant adapter: implements the `ports.VectorStore` contract.

Why not the official SDK: this package is a dependency of hooks that run on EVERY
user prompt. A missing `pip install`, the wrong venv or a slow import turn a
dependency failure into a SILENT loss of functionality.

No business rule lives here — only translation between the contract's operations and
the HTTP API. That is what makes it possible to swap the vector store by writing
another adapter, without touching `memory`, `docs` or `retrieval`.
"""
from .errors import CoreError
from .http import HttpError, request_json


class QdrantError(CoreError):
    pass


class Qdrant:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method: str, path: str, body=None):
        headers = {"api-key": self.api_key} if self.api_key else {}
        try:
            return request_json(f"{self.base}{path}", method=method, body=body,
                                headers=headers, timeout=self.timeout)
        except HttpError as exc:
            # Keep the status on the object: telling a 404 apart by a substring of the
            # message breaks the day the message changes.
            error = QdrantError(str(exc))
            error.status = exc.status
            raise error from exc

    # ---- collections -------------------------------------------------------

    def list_collections(self) -> list[str]:
        res = self.request("GET", "/collections")

        return [c["name"] for c in res.get("result", {}).get("collections", [])]

    def collection_info(self, name: str) -> dict | None:
        """Returns {size, distance, points}, or None if it does not exist.

        It only recognizes a collection with a single unnamed vector; a collection with
        named vectors has a different search shape and is not compatible with this
        client.
        """
        try:
            res = self.request("GET", f"/collections/{name}")
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        result = res.get("result", {})
        vectors = result.get("config", {}).get("params", {}).get("vectors", {})
        if not isinstance(vectors, dict) or "size" not in vectors:
            return {"size": None, "distance": None, "points": result.get("points_count")}

        return {
            "size": vectors.get("size"),
            "distance": vectors.get("distance"),
            "points": result.get("points_count"),
        }

    def ensure_collection(self, name: str, size: int, distance: str = "Cosine") -> bool:
        """Creates it if absent. Returns True if it created it.

        If it exists with a DIFFERENT dimension, this raises instead of carrying on:
        writing a vector of the wrong dimension is refused by Qdrant point by point, but
        writing into an archive built by another embedding model goes through and degrades
        search silently.
        """
        info = self.collection_info(name)
        if info is not None:
            if info["size"] not in (None, size):
                raise QdrantError(
                    f"collection {name!r} has dimension {info['size']}, incompatible with "
                    f"the configured model ({size}). Pick another collection or another model."
                )

            return False
        self.request("PUT", f"/collections/{name}", {"vectors": {"size": size, "distance": distance}})

        return True

    def ensure_payload_index(self, name: str, field: str, schema: str) -> None:
        """A payload index is a filter optimization, never a requirement — a failure here
        must not bring down the operation the caller actually wanted."""
        try:
            self.request("PUT", f"/collections/{name}/index?wait=true",
                         {"field_name": field, "field_schema": schema})
        except QdrantError:
            pass

    def delete_collection(self, name: str) -> None:
        self.request("DELETE", f"/collections/{name}")

    # ---- points ------------------------------------------------------------

    def upsert(self, name: str, points: list[dict], batch: int = 256) -> int:
        for i in range(0, len(points), batch):
            self.request("PUT", f"/collections/{name}/points?wait=true",
                         {"points": points[i:i + batch]})

        return len(points)

    def search(self, name: str, vector: list[float], limit: int,
               filter_: dict | None = None, with_payload: bool = True) -> list[dict]:
        body = {"vector": vector, "limit": limit, "with_payload": with_payload}
        if filter_:
            body["filter"] = filter_
        res = self.request("POST", f"/collections/{name}/points/search", body)

        return res.get("result", []) or []

    def get_point(self, name: str, point_id) -> dict | None:
        try:
            res = self.request("GET", f"/collections/{name}/points/{point_id}")
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

        return res.get("result")

    def set_payload(self, name: str, point_id, payload: dict) -> None:
        self.request("POST", f"/collections/{name}/points/payload?wait=true",
                     {"payload": payload, "points": [point_id]})

    def delete_points(self, name: str, ids: list) -> None:
        self.request("POST", f"/collections/{name}/points/delete?wait=true", {"points": ids})

    def delete_by_filter(self, name: str, filter_: dict) -> None:
        self.request("POST", f"/collections/{name}/points/delete?wait=true", {"filter": filter_})

    def scroll(self, name: str, limit: int = 256, offset=None,
               with_vector: bool = False, filter_: dict | None = None) -> tuple[list[dict], object]:
        body = {"limit": limit, "with_payload": True, "with_vector": with_vector}
        if offset is not None:
            body["offset"] = offset
        if filter_:
            body["filter"] = filter_
        res = self.request("POST", f"/collections/{name}/points/scroll", body).get("result", {})

        return res.get("points", []), res.get("next_page_offset")

    def scroll_all(self, name: str, filter_: dict | None = None, with_vector: bool = False):
        """Iterates the whole collection, paging. A generator, so nothing is materialized."""
        offset = None
        while True:
            points, offset = self.scroll(name, offset=offset, with_vector=with_vector, filter_=filter_)
            for p in points:
                yield p
            if offset is None:
                break
