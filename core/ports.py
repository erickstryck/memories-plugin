"""Contracts for the external dependencies.

They are `Protocol`s and not abstract base classes on purpose: STRUCTURAL typing, no
inheritance, no registration, no runtime cost. A test fake does not have to inherit
from anything — having the methods is enough. An abstract base class with a single
real implementation would be ceremony without benefit.

What these contracts buy, concretely:

1. The business rules (`retrieval`, `memory`, `docs`) come to depend on these
   signatures rather than on a concrete `Qdrant`, `Embedder` or `Reranker`. Swapping
   Qdrant for another vector store, or the embedding endpoint for a local library, is
   writing an adapter — no rule file changes.

2. The retrieval pipeline, the most delicate logic in the package, becomes testable
   WITHOUT the network. Before, it could only be exercised in an integration test,
   which means nobody runs it while editing.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Turns text into a vector."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors in the SAME order as the texts. Raises if the response is short."""
        ...

    def embed_one(self, text: str) -> list[float]:
        ...


@runtime_checkable
class RerankModel(Protocol):
    """Judges how relevant documents are to a query, looking at both together."""

    def rank(self, query: str, documents: list[str]) -> tuple[list[tuple[int, float]], dict]:
        """Returns ((index, score 0..1) pairs sorted by score desc, info).

        `info` MUST carry at least `ok: bool`. The caller has to know whether the
        judgement happened: a pipeline that relaxes the first stage counting on the
        second ends up WORSE than the single stage when the second fails silently.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores and searches vectors with a payload, grouped into collections."""

    def ensure_collection(self, name: str, size: int, distance: str = ...) -> bool:
        ...

    def ensure_payload_index(self, name: str, field: str, schema: str) -> None:
        ...

    def collection_info(self, name: str) -> dict | None:
        ...

    def list_collections(self) -> list[str]:
        ...

    def delete_collection(self, name: str) -> None:
        ...

    def upsert(self, name: str, points: list[dict], batch: int = ...) -> int:
        ...

    def search(self, name: str, vector: list[float], limit: int,
               filter_: dict | None = ..., with_payload: bool = ...) -> list[dict]:
        ...

    def get_point(self, name: str, point_id) -> dict | None:
        ...

    def set_payload(self, name: str, point_id, payload: dict) -> None:
        """Replaces the payload WITHOUT touching the vector.

        It exists so that changing metadata does not require recomputing an
        embedding: beyond the waste, without it fixing a label was IMPOSSIBLE while
        the embedding endpoint was down — an operation that does not need it.
        """
        ...

    def delete_points(self, name: str, ids: list) -> None:
        ...

    def delete_by_filter(self, name: str, filter_: dict) -> None:
        ...

    def scroll(self, name: str, limit: int = ..., offset=...,
               with_vector: bool = ..., filter_: dict | None = ...) -> tuple[list[dict], object]:
        ...

    def scroll_all(self, name: str, filter_: dict | None = ..., with_vector: bool = ...):
        ...
