"""Núcleo portável: memória semântica de longo prazo + índice efêmero de documentos.

Nada aqui conhece o agente ou o host que está chamando. As integrações (plugin de
um harness, servidor, script) montam os objetos por este módulo e ficam sendo
cascas finas — é o que permite o mesmo núcleo servir hosts diferentes sem
duplicar a lógica de embedding, de re-rank ou de acesso ao Qdrant.

A duplicação que isto elimina não é hipotética: numa versão anterior o hook e o
servidor MCP tinham cada um o seu código de embedding, e a normalização de escala
do re-rank existia só num deles — o outro herdaria o bug de volta no dia em que
precisasse reranquear.
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
    """Devolve None quando não há re-rank configurado.

    None é resposta válida, não erro: o re-rank melhora a ordenação e nunca é
    pré-requisito. Quem consome precisa lidar com a ausência dele — e lidar bem
    significa voltar ao corte estrito, não relaxar o piso e ficar sem filtro.
    """
    try:
        url = cfg.resolved_rerank_url()
    except ConfigError:
        return None
    if not cfg.rerank_model:
        return None

    return Reranker(url, cfg.rerank_model, cfg.api_key, timeout=timeout, **kw)


def build_memory(cfg: Config, *, timeouts: dict | None = None) -> MemoryStore:
    """Monta o acesso à memória.

    `timeouts` existe porque quem chama de dentro de um hook tem orçamento próprio:
    o host mata o hook num prazo, e um timeout de dependência MAIOR que esse prazo
    significa que o processo morre antes de conseguir avisar o modelo — o usuário
    paga a latência inteira e ninguém fica sabendo. Os defaults aqui são de uso
    interativo, não de hook.
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
