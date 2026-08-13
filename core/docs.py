"""Índice de documentos, em dois acervos com ciclos de vida diferentes.

Problema: responder uma pergunta sobre 40 linhas de um arquivo de 8.000 não deve
custar o arquivo inteiro em contexto. O arquivo é lido do disco por este processo,
fatiado, embedado e gravado; a busca devolve só os trechos que respondem — e,
quando o arquivo é relegível, devolve a LOCALIZAÇÃO para quem consome ler o
conteúdo atual em vez de uma foto.

DOIS ACERVOS, e a separação é estrutural, não convenção:

  - TEMPORÁRIO (`tmp`): documento aberto para uma tarefa. Cada trecho carrega
    `expires_at_ts` e toda operação varre o que venceu. O Qdrant não tem expiração
    nativa, então o TTL é isto: um delete por filtro em cada chamada, sem daemon.
    Este acervo é DESTRUTÍVEL por construção — existe comando que apaga a coleção.

  - BIBLIOTECA (`library`): documento que vale guardar para consulta. Sem TTL,
    nunca varrido, e fora do alcance de qualquer comando de limpeza. Vive em
    coleção própria justamente para que a destrutibilidade do temporário não possa
    alcançá-lo.

Nenhum dos dois é a coleção de MEMÓRIA: trecho de arquivo competindo com fato
curado numa busca de recall vence por volume e afunda o acervo que mais importa.
"""
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from . import retrieval
from .errors import CoreError
from .chunk import chunk_text, is_probably_binary, mode_for_suffix

DEFAULT_TTL_SECONDS = 24 * 3600
DENSE_TOP_K = 20
RERANK_MIN_SCORE = 0.10
#: Piso do primeiro estágio na busca de documentos. Mais generoso que na memória
#: porque aqui o segundo estágio não veta — o objetivo é não perder candidato.
DENSE_FLOOR = 0.30

# O limiar de detecção de colapso cross-lingual mora em `retrieval`, junto com
# a lógica que o usa — aqui só se escolhe a política.

SCOPES = ("all", "tmp", "library")


class DocsError(CoreError):
    pass


@dataclass
class Hit:
    score: float
    origin: str          # "CE" quando o cross-encoder julgou, "denso" quando não
    scope: str           # tmp | library
    path: str
    start_line: int
    end_line: int
    mode: str            # locator | snapshot
    text: str
    indexed_at: str
    stale: str | None    # motivo, quando o arquivo mudou desde a indexação


def parse_ttl(spec: str) -> float:
    """Aceita `30m`, `24h`, `7d` ou segundos puros."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", str(spec).strip().lower())
    if not m:
        raise DocsError(f"TTL inválido: {spec!r} (use 30m, 24h, 7d ou segundos)")
    mult = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]

    return float(m.group(1)) * mult


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def doc_id_for(path: str) -> str:
    """Id derivado do caminho absoluto, para que reindexar o mesmo arquivo
    SUBSTITUA o índice anterior em vez de acumular duas versões competindo."""
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]


def _point_id(doc_id: str, ix: int) -> int:
    """Id numérico estável e determinístico (o Qdrant aceita inteiro ou UUID)."""
    return int(hashlib.sha1(f"{doc_id}:{ix}".encode()).hexdigest()[:15], 16)


MTIME_TOLERANCE = 0.001


def source_changed(path: str, src_mtime, src_size) -> str | None:
    """Motivo da obsolescência, ou None se o arquivo está igual ao indexado.

    COMPARA COM TOLERÂNCIA, e isso não é frouxidão: `st_mtime` é float e o
    round-trip por JSON perde os últimos bits (medido: 1786646270.9956777 gravado
    contra 1786646270.9956775 no disco, 2.4e-7 de diferença). Igualdade exata dava
    falso positivo em TODA busca de TODO documento — o aviso viraria ruído
    ignorável e o `refresh` reindexaria o acervo inteiro a cada execução, pagando
    embedding por nada. Um milissegundo de tolerância mata o falso positivo e
    detecta qualquer edição real, que muda o mtime em segundos.
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return "arquivo não existe mais"
    if src_size is not None and st.st_size != src_size:
        return "tamanho do arquivo mudou desde a indexação"
    if src_mtime is not None and abs(st.st_mtime - float(src_mtime)) > MTIME_TOLERANCE:
        return "arquivo mudou desde a indexação"

    return None


def _read_source(path: str) -> tuple[str, os.stat_result, str]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise DocsError(f"não é um arquivo: {path}")
    st = os.stat(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    if is_probably_binary(content[:8192]):
        raise DocsError("arquivo parece binário — converta para texto antes de indexar")

    return path, st, content


class DocIndex:
    def __init__(self, qdrant, embedder, reranker, tmp_collection: str,
                 library_collection: str, vector_size: int):
        self.q = qdrant
        self.embedder = embedder
        self.reranker = reranker
        self.collections = {"tmp": tmp_collection, "library": library_collection}
        self.vector_size = vector_size

    # ---- manutenção --------------------------------------------------------

    def _collection(self, scope: str) -> str:
        if scope not in self.collections:
            raise DocsError(f"escopo inválido: {scope!r} (use tmp ou library)")

        return self.collections[scope]

    def ensure(self, scope: str) -> None:
        name = self._collection(scope)
        if self.q.ensure_collection(name, self.vector_size):
            self.q.ensure_payload_index(name, "doc_id", "keyword")
            if scope == "tmp":
                self.q.ensure_payload_index(name, "expires_at_ts", "float")

    def sweep(self) -> None:
        """Apaga o que venceu — só no temporário. A biblioteca nunca é varrida."""
        self.ensure("tmp")
        self.q.delete_by_filter(
            self._collection("tmp"),
            {"must": [{"key": "expires_at_ts", "range": {"lt": time.time()}}]},
        )

    # ---- indexação ---------------------------------------------------------

    def index_file(self, path: str, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                   doc_id: str | None = None) -> dict:
        """Indexa como TEMPORÁRIO, com TTL."""
        return self._write(path, "tmp", ttl_seconds, doc_id)

    def keep_file(self, path: str, doc_id: str | None = None) -> dict:
        """Guarda na BIBLIOTECA, sem expiração."""
        return self._write(path, "library", None, doc_id)

    def _write(self, path: str, scope: str, ttl_seconds: float | None,
               doc_id: str | None) -> dict:
        path, st, content = _read_source(path)
        chunks = chunk_text(content)
        if not chunks:
            raise DocsError("nada indexável (arquivo vazio ou só espaço em branco)")

        doc_id = doc_id or doc_id_for(path)
        mode = mode_for_suffix(os.path.splitext(path)[1])
        now_ts = time.time()
        name = self._collection(scope)

        self.ensure(scope)
        if scope == "tmp":
            self.sweep()
        # Reindexar substitui: sem isto, a versão antiga e a nova coexistem e a
        # busca mistura trechos de dois estados do mesmo arquivo.
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
                    "chunk_ix": ix,
                    "n_chunks": len(chunks),
                    "indexed_at": _iso(now_ts),
                    "src_mtime": round(st.st_mtime, 3),
                    "src_size": st.st_size,
                },
            }
            if ttl_seconds is not None:
                expires_at = now_ts + ttl_seconds
                payload["expires_at_ts"] = expires_at
                payload["metadata"]["expires_at"] = _iso(expires_at)
                # Guarda a DURAÇÃO, não só o instante: sem ela o `refresh` não tem
                # como saber que você pediu 1 hora e reindexava com o padrão de 24h,
                # esticando em silêncio o prazo que você mesmo escolheu.
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
        """Reindexa os documentos cujo arquivo mudou desde a indexação.

        Existe por causa do acervo permanente: um documento guardado em agosto cujo
        arquivo mudou em outubro devolve trecho que não existe mais. O aviso em
        cada hit alerta; isto conserta.
        """
        report = []
        for doc in self.list_docs(scope):
            path = doc["path"]
            reason = source_changed(path, doc.get("src_mtime"), doc.get("src_size"))
            if reason == "arquivo não existe mais":
                report.append({"doc_id": doc["doc_id"], "path": path, "acao": "ausente"})
                continue
            if reason is None:
                report.append({"doc_id": doc["doc_id"], "path": path, "acao": "ok"})
                continue
            if scope == "library":
                res = self.keep_file(path, doc["doc_id"])
            else:
                # Reusa a duração original. Cair no default aqui ignoraria o prazo
                # que o usuário pediu na indexação.
                res = self.index_file(path, doc.get("ttl_seconds") or DEFAULT_TTL_SECONDS,
                                      doc["doc_id"])
            report.append({"doc_id": doc["doc_id"], "path": path,
                              "acao": "reindexado", "chunks": res["chunks"]})

        return report

    # ---- busca -------------------------------------------------------------

    def search(self, query: str, scope: str = "all", doc_id: str | None = None,
               limit: int = 5, min_score: float = RERANK_MIN_SCORE
               ) -> tuple[list[Hit], retrieval.Outcome]:
        """Busca nos acervos pedidos e reranqueia a UNIÃO.

        Reranquear a união, e não cada acervo em separado, é o que torna os scores
        comparáveis: o cross-encoder julga todos os candidatos contra a mesma
        pergunta, então um trecho da biblioteca e um do temporário disputam a mesma
        vaga em pé de igualdade.

        A política aqui difere da memória em dois pontos, e os dois são deliberados:
        SEM VETO, porque quem pergunta já escolheu o documento e silêncio é pior que
        ordem imperfeita; e A ORDEM É O PRODUTO, porque o resultado é uma lista lida
        de cima para baixo — então vale reranquear mesmo quando tudo cabe.
        """
        if scope not in SCOPES:
            raise DocsError(f"escopo inválido: {scope!r} (use {', '.join(SCOPES)})")
        scopes = ("tmp", "library") if scope == "all" else (scope,)

        # Os dois pisos IGUAIS, de propósito: sem veto, "voltar ao corte estrito"
        # quando o julgamento não acontece tem de ser um no-op. Se fossem diferentes,
        # o colapso cross-lingual (denso na faixa de 0.46) devolveria silêncio —
        # exatamente o que esta política existe para evitar.
        policy = retrieval.Policy(
            dense_floor=DENSE_FLOOR, strict_floor=DENSE_FLOOR, min_score=min_score,
            max_results=limit, veto=False, detect_collapse=True, order_matters=True,
        )
        vector = self.embedder.embed_one(query)
        candidates: list[dict] = []
        for esc in scopes:
            for raw in self._search_scope(esc, vector, doc_id):
                candidates.append(raw)
        candidates.sort(key=lambda b: -(b.get("score") or 0.0))

        outcome = retrieval.two_stage(candidates, query, self.reranker, policy,
                                  text_of=lambda b: b["payload"]["document"])

        return [self._to_hit(s.item, s.score, s.origin) for s in outcome.scored], outcome

    def _search_scope(self, scope: str, vector: list[float],
                      doc_id: str | None) -> list[dict]:
        """Primeiro estágio num acervo. Só o temporário filtra por validade."""
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
        """Traduz o hit cru + o veredito do pipeline no formato de apresentação."""
        p = raw.get("payload", {})
        md = p.get("metadata", {})
        path = md.get("path", "?")
        reason = source_changed(path, md.get("src_mtime"), md.get("src_size"))
        stale = f"{reason} ({md.get('indexed_at')})" if reason else None

        return Hit(
            score=score, origin=origin, scope=raw.get("_scope", md.get("scope", "?")),
            path=path, start_line=md.get("start_line", 0), end_line=md.get("end_line", 0),
            mode=md.get("mode", "snapshot"), text=p.get("document", ""),
            indexed_at=md.get("indexed_at", "?"), stale=stale,
        )

    # ---- inventário e remoção ---------------------------------------------

    def list_docs(self, scope: str = "all") -> list[dict]:
        if scope not in SCOPES:
            raise DocsError(f"escopo inválido: {scope!r}")
        scopes = ("tmp", "library") if scope == "all" else (scope,)
        by_scope_doc: dict[tuple[str, str], dict] = {}
        for esc in scopes:
            self.ensure(esc)
            if esc == "tmp":
                self.sweep()
            for point in self.q.scroll_all(self._collection(esc)):
                p = point.get("payload", {})
                md = p.get("metadata", {})
                key = (esc, p.get("doc_id", "?"))
                d = by_scope_doc.setdefault(key, {
                    "doc_id": p.get("doc_id", "?"), "scope": esc, "chunks": 0,
                    "path": md.get("path", "?"), "mode": md.get("mode", "?"),
                    "indexed_at": md.get("indexed_at", "?"),
                    "expires_at_ts": p.get("expires_at_ts"),
                    "src_mtime": md.get("src_mtime"), "src_size": md.get("src_size"),
                    "ttl_seconds": md.get("ttl_seconds"),
                })
                d["chunks"] += 1

        return sorted(by_scope_doc.values(), key=lambda d: (d["scope"], d["path"]))

    def drop(self, doc_id: str, scope: str = "all") -> None:
        scopes = ("tmp", "library") if scope == "all" else (scope,)
        for esc in scopes:
            self.ensure(esc)
            self.q.delete_by_filter(self._collection(esc),
                                    {"must": [{"key": "doc_id", "match": {"value": doc_id}}]})

    def drop_all_tmp(self) -> str:
        """Apaga a coleção TEMPORÁRIA inteira.

        Só existe para o temporário. A biblioteca não tem equivalente de propósito:
        acervo permanente se remove documento por documento, com o id na mão.
        """
        name = self._collection("tmp")
        self.q.delete_collection(name)

        return name
