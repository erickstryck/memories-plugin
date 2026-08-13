"""Portable core: long-term semantic memory + ephemeral document index.

Nothing in here knows about the agent or the host calling it. Integrations (a
harness plugin, a server, a script) assemble their objects through this module and
stay thin shells — that is what lets the same core serve different hosts without
duplicating the embedding, re-ranking or Qdrant access logic.

The duplication this removes is not hypothetical: in an earlier version the hook
and the MCP server each had their own embedding code, and the re-rank scale
normalization existed in only one of them — the other would have inherited the bug
back the day it needed to rerank.
"""
from .config import Config, ConfigError, load, save, redacted
from .errors import CoreError
from .docs import DocIndex, DocsError, parse_ttl, doc_id_for
from .memory import MemoryStore, Recalled, search_collections
from .embedding import Embedder, EmbeddingError
from .reranking import Reranker, RerankError, normalize_scores, sigmoid
from .qdrant import Qdrant, QdrantError
from .retrieval import (CE, CE_WEAK, DENSE, Outcome, Policy, Scored,
                        fuse_by_id, needs_rerank, two_stage)

__all__ = [
    "CoreError",
    "Config", "ConfigError", "load", "save", "redacted",
    "DocIndex", "DocsError", "parse_ttl", "doc_id_for",
    "MemoryStore", "Recalled", "search_collections",
    "Embedder", "EmbeddingError", "Reranker", "RerankError",
    "normalize_scores", "sigmoid", "Policy", "two_stage", "fuse_by_id",
    "Qdrant", "QdrantError",
    "build_qdrant", "build_embedder", "build_reranker", "build_memory", "build_docs",
]


def build_qdrant(cfg: Config, timeout: float = 30.0) -> Qdrant:
    cfg.require_qdrant()

    return Qdrant(cfg.qdrant_url, cfg.qdrant_api_key, timeout=timeout)


def build_embedder(cfg: Config, timeout: float = 60.0) -> Embedder:
    return Embedder(cfg.resolved_embed_url(), cfg.embed_model, cfg.api_key, timeout=timeout)


def build_reranker(cfg: Config, timeout: float = 15.0, **kw) -> Reranker | None:
    """Returns None when no re-ranking is configured.

    None is a valid answer, not an error: re-ranking improves the ordering and is
    never a prerequisite. Callers have to cope with its absence — and coping well
    means going back to the strict cut, not relaxing the floor and ending up with no
    filter at all.
    """
    try:
        url = cfg.resolved_rerank_url()
    except ConfigError:
        return None
    if not cfg.rerank_model:
        return None

    return Reranker(url, cfg.rerank_model, cfg.api_key, timeout=timeout, **kw)


def build_memory(cfg: Config, *, timeouts: dict | None = None) -> MemoryStore:
    """Assembles memory access.

    `timeouts` exists because a caller inside a hook has a budget of its own: the
    host kills the hook after a deadline, and a dependency timeout LONGER than that
    deadline means the process dies before it can tell the model anything — the user
    pays the whole latency and nobody finds out. The defaults here are for
    interactive use, not for a hook.
    """
    t = timeouts or {}

    return MemoryStore(build_qdrant(cfg, timeout=t.get("qdrant", 30.0)),
                       build_embedder(cfg, timeout=t.get("embed", 60.0)),
                       build_reranker(cfg, timeout=t.get("rerank", 15.0)),
                       cfg.require_memory_collection(), cfg.vector_size)


def build_docs(cfg: Config) -> DocIndex:
    return DocIndex(build_qdrant(cfg), build_embedder(cfg), build_reranker(cfg),
                    cfg.require_docs_collection(), cfg.require_library_collection(),
                    cfg.vector_size)
