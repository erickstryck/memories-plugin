"""Memória semântica de longo prazo.

Substitui, sem migrar dado nenhum, um servidor MCP feito à mão: o FORMATO DO
PAYLOAD é deliberadamente idêntico ao que já está gravado —
`{document, metadata, created_at, updated_at}` — então um acervo existente
continua legível e gravável por este módulo sem conversão, sem reindexação e sem
janela de risco.

Duas direções, e a de leitura é a que costuma ser esquecida:
  - `recall` / `find`: buscar ANTES de afirmar, para não re-decidir o que já foi
    decidido nem contradizer o que já foi medido.
  - `store` / `update`: persistir fato durável, um por registro.

`recall` implementa o pipeline de DOIS PORTÕES: o denso relaxa o piso para ganhar
alcance e o cross-encoder aplica a precisão. A assimetria importa — se o segundo
portão não roda, o piso permissivo do primeiro TEM de voltar ao valor estrito,
senão o modo com re-rank fica pior que o modo sem, que é o oposto da intenção.
"""
import uuid
from dataclasses import dataclass

from . import retrieval
from .errors import CoreError
from datetime import datetime, timezone


class MemoryError_(CoreError):
    pass


@dataclass
class Recalled:
    id: str
    score: float
    origin: str            # "CE" ou "denso"
    dense_score: float
    document: str
    metadata: dict
    updated_at: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, qdrant, embedder, reranker, collection: str, vector_size: int):
        self.q = qdrant
        self.embedder = embedder
        self.reranker = reranker
        self.collection = collection
        self.vector_size = vector_size

    def ensure(self) -> None:
        """Garante a coleção. Só para caminho de ESCRITA."""
        self.q.ensure_collection(self.collection, self.vector_size)

    def require_existing(self) -> None:
        """Exige que a coleção exista. Para caminho de LEITURA.

        Ler não pode CRIAR: com um erro de digitação no nome, `ensure` criava uma
        coleção vazia, a busca devolvia zero hits e o consumidor concluía "não há
        precedente registrado sobre este assunto". Ou seja um typo de configuração
        virava a afirmação mais perigosa que este sistema pode fazer. Melhor falhar
        alto — e o hook transforma isso em aviso explícito de indisponibilidade.
        """
        if self.q.collection_info(self.collection) is None:
            raise MemoryError_(
                f"coleção de memória {self.collection!r} não existe. Confira o nome com "
                f"`collections list`; nada é criado por uma leitura."
            )

    # ---- escrita -----------------------------------------------------------

    def store(self, information: str, metadata: dict | None = None) -> dict:
        if not information or not information.strip():
            raise MemoryError_("memória vazia não é gravável")
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
        """Lote com UMA ida ao endpoint de embeddings, tudo-ou-nada.

        Os vetores são gerados ANTES de qualquer escrita: um timeout no meio do
        lote não deixa metade gravada, que é o pior estado possível para um acervo
        onde duplicata parcial é indistinguível de fato novo.
        """
        if not items:
            return {"status": "noop", "ids": [], "count": 0}
        texts = []
        for i, item in enumerate(items):
            info = (item or {}).get("information")
            if not isinstance(info, str) or not info.strip():
                raise MemoryError_(f"itens[{i}] precisa de 'information' (string não vazia)")
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
        new_doc = information if information is not None else previous.get("document", "")
        new_meta = metadata if metadata is not None else previous.get("metadata", {})
        payload = {"document": new_doc, "metadata": new_meta,
                   "created_at": previous.get("created_at", _now()), "updated_at": _now()}

        # Texto inalterado significa vetor inalterado: recalcular seria pagar uma ida
        # à rede para obter o mesmo número. Pior, tornava impossível corrigir uma
        # etiqueta enquanto o endpoint de embedding estivesse fora — uma operação que
        # não depende dele.
        if new_doc == previous.get("document"):
            self.q.set_payload(self.collection, mid, payload)

            return {"status": "updated", "id": mid, "reembedded": False}

        point_with_vector = {"id": mid, "vector": self.embedder.embed_one(new_doc),
                             "payload": payload}
        self.q.upsert(self.collection, [point_with_vector])

        return {"status": "updated", "id": mid, "reembedded": True}

    def delete(self, mid: str) -> dict:
        self.q.delete_points(self.collection, [mid])

        return {"status": "deleted", "id": mid}

    # ---- leitura -----------------------------------------------------------

    def find(self, query: str, limit: int = 5) -> list[dict]:
        """Busca densa pura, sem re-rank. Barata; use quando a ordem entre os
        relevantes não importa (por exemplo para deduplicar antes de gravar)."""
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

    def recall(self, queries: list[str], policy: retrieval.Policy,
               top_k: int) -> tuple[list[Recalled], retrieval.Outcome]:
        """Recupera memórias para vários ÂNGULOS da mesma pergunta.

        Esta classe cuida do PRIMEIRO estágio — embeddings, busca por vetor e fusão.
        O segundo estágio e a política de seleção vivem em `retrieval`, compartilhados
        com a busca de documentos: enquanto eram dois códigos, uma correção entrava
        num e não no outro.

        Os ângulos vão numa única chamada de embeddings, porque o endpoint aceita
        `input` como array. A fusão é por id pelo MAIOR score: um registro que aparece
        em dois ângulos não deve ser penalizado pelo pior deles.
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
                                   text_of=lambda h: h["document"])
        # `best_dense` do pipeline vê só os candidatos; para dizer "nada passou do
        # corte, o melhor foi X" o número útil é o melhor de TODOS os hits.
        if fused:
            outcome.best_dense = fused[0]["score"]

        return [self._to_recalled(s) for s in outcome.scored], outcome

    @staticmethod
    def _flatten_hit(hit: dict) -> dict | None:
        """Achata o hit do banco no formato que o pipeline consome.

        Devolve None para registro sem documento utilizável: vetor sem texto não tem
        como ser julgado pelo cross-encoder nem apresentado a ninguém.
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
        point = self.q.get_point(self.collection, mid)
        if point is None:
            return {"status": "not_found", "id": mid}
        p = point.get("payload", {})

        return {"id": point.get("id"), "document": p.get("document"),
                "metadata": p.get("metadata", {}), "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at")}

    def list_page(self, limit: int = 20, offset=None) -> dict:
        self.require_existing()
        points, proximo = self.q.scroll(self.collection, limit=limit, offset=offset)
        memories = [{
            "id": pt.get("id"),
            "document": pt.get("payload", {}).get("document"),
            "metadata": pt.get("payload", {}).get("metadata", {}),
            "updated_at": pt.get("payload", {}).get("updated_at"),
        } for pt in points]

        return {"count": len(memories), "memories": memories, "next_offset": proximo}

    def count(self) -> int | None:
        info = self.q.collection_info(self.collection)

        return info.get("points") if info else None


def search_collections(qdrant, embedder, query: str, collections: list[str] | None,
                       vector_size: int, limit: int = 5,
                       max_results: int = 25) -> dict:
    """Busca SOMENTE LEITURA em coleções arbitrárias.

    Serve para consultar acervos de outros sistemas que compartilham o mesmo
    modelo de embedding. Coleção de dimensão diferente é PULADA e reportada — ler
    silenciosamente de um acervo de outro modelo devolve vizinhos aleatórios com
    score plausível, que é pior que devolver nada.
    """
    vector = embedder.embed_one(query)
    targets = collections if collections else qdrant.list_collections()
    searched, skipped_cols, results = [], [], []
    for name in targets:
        info = qdrant.collection_info(name)
        if info is None or info.get("size") is None:
            skipped_cols.append({"collection": name, "motivo": "não encontrada / vetor nomeado"})
            continue
        if info["size"] != vector_size:
            skipped_cols.append({"collection": name, "motivo": f"dimensão {info['size']} ≠ {vector_size}"})
            continue
        try:
            hits = qdrant.search(name, vector, limit)
        except Exception as exc:
            skipped_cols.append({"collection": name, "motivo": f"erro: {type(exc).__name__}"})
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
