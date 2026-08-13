"""Diagnóstico e sugestões para configurar o pacote.

Devolve DADOS, não texto: quem chama decide se renderiza para uma pessoa, para
JSON ou para um agente. É o que permite o mesmo diagnóstico servir um wizard de
terminal e uma ferramenta chamada por outro programa.

Cada verificação carrega a CORREÇÃO junto. Diagnóstico que diz "falhou" sem dizer
o que fazer obriga quem lê a ir procurar a documentação, e é aí que se desiste.
"""
from dataclasses import dataclass, asdict

from .config import Config, ConfigError
from .embedding import Embedder
from .errors import CoreError
from .reranking import Reranker

#: Prefixos de coleção gerada por outro sistema — não são candidatas úteis.
PREFIXOS_RUIDO = ("ws-",)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix_hint: str | None = None
    warning: bool = False  # falha que degrada, não impede


def _check_qdrant(cfg: Config) -> tuple[Check, object]:
    if not cfg.qdrant_url:
        return Check("Qdrant", False, "qdrant_url não configurado",
                     "export QCTX_QDRANT_URL=https://seu-qdrant"), None
    from .qdrant import Qdrant, QdrantError
    q = Qdrant(cfg.qdrant_url, cfg.qdrant_api_key, timeout=10.0)
    try:
        names = q.list_collections()
    except QdrantError as exc:
        pista = "confira QCTX_QDRANT_API_KEY" if "401" in str(exc) or "403" in str(exc) \
            else "confira a URL e se o serviço está no ar"
        return Check("Qdrant", False, f"não respondeu: {exc}", pista), None

    return Check("Qdrant", True, f"{len(names)} coleções em {cfg.qdrant_url}"), q


def _check_embed(cfg: Config) -> tuple[Check, int | None]:
    try:
        url = cfg.resolved_embed_url()
    except ConfigError as exc:
        return Check("Embedding", False, str(exc),
                     "export QCTX_EMBED_URL=… ou QCTX_API_BASE_URL=…"), None
    emb = Embedder(url, cfg.embed_model, cfg.api_key, timeout=30.0)
    try:
        dim = emb.detect_dimension()
    except CoreError as exc:
        return Check("Embedding", False, f"{url} não respondeu: {exc}",
                     f"confira se o modelo {cfg.embed_model!r} está servido nesse endereço"), None
    detail = f"{cfg.embed_model} devolve {dim} dimensões"
    if dim != cfg.vector_size:
        return Check("Embedding", False,
                     f"{detail}, mas vector_size está {cfg.vector_size}",
                     "qctx config detect"), dim

    return Check("Embedding", True, detail), dim


def _check_rerank(cfg: Config) -> Check:
    """O re-rank é opcional: sem ele a busca funciona, só ordena pior. Falha aqui é
    AVISO, nunca impedimento — tratar como erro faria alguém sem reranker concluir
    que o pacote não serve."""
    try:
        url = cfg.resolved_rerank_url()
    except ConfigError:
        return Check("Re-rank", False, "não configurado (opcional)",
                     "export QCTX_RERANK_URL=… para ganhar precisão na busca", warning=True)
    rr = Reranker(url, cfg.rerank_model, cfg.api_key, timeout=20.0)
    pairs, info = rr.rank(
        "qual a capital da França?",
        ["Paris é a capital da França.", "receita de bolo de cenoura com cobertura"],
    )
    if not info["ok"]:
        return Check("Re-rank", False, f"{url} falhou: {info['erro']}",
                     f"confira se o servidor subiu com suporte a rerank e se o modelo "
                     f"{cfg.rerank_model!r} está lá", warning=True)
    escala = "logit cru (normalizado para sigmoid)" if info["era_logit"] else "sigmoid 0..1"
    best = max(s for _, s in pairs)
    acertou = pairs[0][0] == 0
    detail = f"{cfg.rerank_model} responde em {escala}; melhor score {best:.3f}"
    if not acertou:
        return Check("Re-rank", False,
                     f"{detail} — mas ordenou a resposta ERRADA em primeiro",
                     "modelo pode não ser um cross-encoder de rerank", warning=True)

    return Check("Re-rank", True, detail)


def _check_collections(cfg: Config, q) -> list[Check]:
    checks = []
    roles = (
        ("memory_collection", cfg.memory_collection, "memory-collection", True),
        ("docs_collection", cfg.docs_collection, "docs-collection", False),
        ("library_collection", cfg.library_collection, "library-collection", False),
    )
    vistos: dict[str, str] = {}
    for field, value, cli_key, required in roles:
        if not value:
            checks.append(Check(field, not required,
                                "não configurada",
                                f"qctx config set {cli_key} <nome>",
                                warning=not required))
            continue
        if value in vistos:
            checks.append(Check(field, False,
                                f"{value!r} já é usada por {vistos[value]}",
                                f"qctx config set {cli_key} <outro-nome> — cada papel "
                                f"tem ciclo de vida diferente e precisa de coleção própria"))
            continue
        vistos[value] = field
        if q is None:
            checks.append(Check(field, True, f"{value!r} (Qdrant inacessível, não verificada)"))
            continue
        info = q.collection_info(value)
        if info is None:
            checks.append(Check(field, True, f"{value!r} será criada no primeiro uso"))
            continue
        dim = info.get("size")
        if dim not in (None, cfg.vector_size):
            checks.append(Check(field, False,
                                f"{value!r} tem dimensão {dim}, o modelo usa {cfg.vector_size}",
                                "escolha outra coleção ou outro modelo de embedding"))
            continue
        checks.append(Check(field, True, f"{value!r} — {info.get('points')} pontos"))

    return checks


def suggest_collections(q, vector_size: int, cutoff: int = 8) -> list[dict]:
    """Coleções existentes que servem como acervo de memória, melhores primeiro.

    Ordena por número de pontos porque acervo já povoado é o candidato óbvio, e
    esconde as geradas por outro sistema, que são ruído para esta escolha.
    """
    if q is None:
        return []
    output = []
    for name in q.list_collections():
        if name.startswith(PREFIXOS_RUIDO):
            continue
        info = q.collection_info(name) or {}
        if info.get("size") not in (None, vector_size):
            continue
        output.append({"collection": name, "points": info.get("points") or 0})
    output.sort(key=lambda c: -c["points"])

    return output[:cutoff]


def diagnose(cfg: Config) -> dict:
    """Roda todas as verificações e devolve o retrato completo."""
    check_q, q = _check_qdrant(cfg)
    check_emb, dim = _check_embed(cfg)
    checks = [check_q, check_emb, _check_rerank(cfg)]
    checks += _check_collections(cfg, q)

    blockers = [c for c in checks if not c.ok and not c.warning]
    warnings = [c for c in checks if not c.ok and c.warning]

    return {
        "pronto": not blockers,
        "checks": [asdict(c) for c in checks],
        "bloqueios": [asdict(c) for c in blockers],
        "avisos": [asdict(c) for c in warnings],
        "dim_detectada": dim,
        "sugestoes_memoria": suggest_collections(q, cfg.vector_size),
    }


def choose_by_index(options: list[str], entry: str) -> str | None:
    """Resolve a escolha do usuário: número da lista, ou nome digitado.

    Separado da leitura do terminal de propósito — é a única parte do wizard com
    lógica, então é a única que precisa de teste. Devolve None quando a entrada
    não seleciona nada (vazio = manter o atual).
    """
    entry = (entry or "").strip()
    if not entry:
        return None
    if entry.isdigit():
        ix = int(entry) - 1
        if 0 <= ix < len(options):
            return options[ix]

        return None

    return entry
