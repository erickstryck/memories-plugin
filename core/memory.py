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
    origem: str            # "CE" ou "denso"
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
        vetor = self.embedder.embed_one(information)
        agora = _now()
        self.q.upsert(self.collection, [{
            "id": mid,
            "vector": vetor,
            "payload": {"document": information, "metadata": metadata or {},
                        "created_at": agora, "updated_at": agora},
        }])

        return {"status": "created", "id": mid}

    def store_many(self, itens: list[dict]) -> dict:
        """Lote com UMA ida ao endpoint de embeddings, tudo-ou-nada.

        Os vetores são gerados ANTES de qualquer escrita: um timeout no meio do
        lote não deixa metade gravada, que é o pior estado possível para um acervo
        onde duplicata parcial é indistinguível de fato novo.
        """
        if not itens:
            return {"status": "noop", "ids": [], "count": 0}
        textos = []
        for i, item in enumerate(itens):
            info = (item or {}).get("information")
            if not isinstance(info, str) or not info.strip():
                raise MemoryError_(f"itens[{i}] precisa de 'information' (string não vazia)")
            textos.append(info)
        self.ensure()
        vetores = self.embedder.embed(textos)
        agora = _now()
        pontos, ids = [], []
        for item, texto, vetor in zip(itens, textos, vetores):
            mid = str(uuid.uuid4())
            ids.append(mid)
            pontos.append({
                "id": mid,
                "vector": vetor,
                "payload": {"document": texto, "metadata": (item or {}).get("metadata") or {},
                            "created_at": agora, "updated_at": agora},
            })
        self.q.upsert(self.collection, pontos)

        return {"status": "created", "ids": ids, "count": len(ids)}

    def update(self, mid: str, information: str | None = None,
               metadata: dict | None = None) -> dict:
        ponto = self.q.get_point(self.collection, mid)
        if ponto is None:
            return {"status": "not_found", "id": mid}
        antigo = ponto.get("payload", {})
        novo_doc = information if information is not None else antigo.get("document", "")
        novo_meta = metadata if metadata is not None else antigo.get("metadata", {})
        vetor = self.embedder.embed_one(novo_doc)
        self.q.upsert(self.collection, [{
            "id": mid,
            "vector": vetor,
            "payload": {"document": novo_doc, "metadata": novo_meta,
                        "created_at": antigo.get("created_at", _now()), "updated_at": _now()},
        }])

        return {"status": "updated", "id": mid}

    def delete(self, mid: str) -> dict:
        self.q.delete_points(self.collection, [mid])

        return {"status": "deleted", "id": mid}

    # ---- leitura -----------------------------------------------------------

    def find(self, query: str, limit: int = 5) -> list[dict]:
        """Busca densa pura, sem re-rank. Barata; use quando a ordem entre os
        relevantes não importa (por exemplo para deduplicar antes de gravar)."""
        self.require_existing()
        vetor = self.embedder.embed_one(query)
        saida = []
        for hit in self.q.search(self.collection, vetor, limit):
            p = hit.get("payload", {})
            saida.append({
                "id": hit.get("id"),
                "score": round(hit.get("score", 0.0), 4),
                "document": p.get("document"),
                "metadata": p.get("metadata", {}),
                "updated_at": p.get("updated_at"),
            })

        return saida

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
        vetores = self.embedder.embed(queries)
        lotes = [[self._normaliza(h) for h in self.q.search(self.collection, v, top_k)]
                 for v in vetores]
        lotes = [[h for h in lote if h is not None] for lote in lotes]
        fundidos = retrieval.fuse_by_id(lotes, id_de=lambda h: h["id"])

        piso = policy.floor_for(self.reranker is not None)
        candidatos = [h for h in fundidos if h["score"] >= piso]
        fora = retrieval.two_stage(candidatos, queries[0], self.reranker, policy,
                                   texto_de=lambda h: h["document"])
        # `best_dense` do pipeline vê só os candidatos; para dizer "nada passou do
        # corte, o melhor foi X" o número útil é o melhor de TODOS os hits.
        if fundidos:
            fora.best_dense = fundidos[0]["score"]

        return [self._to_recalled(s) for s in fora.scored], fora

    @staticmethod
    def _normaliza(hit: dict) -> dict | None:
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

        return Recalled(id=h["id"], score=s.score, origem=s.origin,
                        dense_score=h["score"], document=h["document"],
                        metadata=h["metadata"], updated_at=h.get("updated_at"))

    def get(self, mid: str) -> dict:
        ponto = self.q.get_point(self.collection, mid)
        if ponto is None:
            return {"status": "not_found", "id": mid}
        p = ponto.get("payload", {})

        return {"id": ponto.get("id"), "document": p.get("document"),
                "metadata": p.get("metadata", {}), "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at")}

    def list_page(self, limit: int = 20, offset=None) -> dict:
        self.require_existing()
        pontos, proximo = self.q.scroll(self.collection, limit=limit, offset=offset)
        memorias = [{
            "id": pt.get("id"),
            "document": pt.get("payload", {}).get("document"),
            "metadata": pt.get("payload", {}).get("metadata", {}),
            "updated_at": pt.get("payload", {}).get("updated_at"),
        } for pt in pontos]

        return {"count": len(memorias), "memories": memorias, "next_offset": proximo}

    def count(self) -> int | None:
        info = self.q.collection_info(self.collection)

        return info.get("points") if info else None


def search_collections(qdrant, embedder, query: str, colecoes: list[str] | None,
                       vector_size: int, limit: int = 5,
                       max_results: int = 25) -> dict:
    """Busca SOMENTE LEITURA em coleções arbitrárias.

    Serve para consultar acervos de outros sistemas que compartilham o mesmo
    modelo de embedding. Coleção de dimensão diferente é PULADA e reportada — ler
    silenciosamente de um acervo de outro modelo devolve vizinhos aleatórios com
    score plausível, que é pior que devolver nada.
    """
    vetor = embedder.embed_one(query)
    alvos = colecoes if colecoes else qdrant.list_collections()
    pesquisadas, puladas, resultados = [], [], []
    for nome in alvos:
        info = qdrant.collection_info(nome)
        if info is None or info.get("size") is None:
            puladas.append({"collection": nome, "motivo": "não encontrada / vetor nomeado"})
            continue
        if info["size"] != vector_size:
            puladas.append({"collection": nome, "motivo": f"dimensão {info['size']} ≠ {vector_size}"})
            continue
        try:
            hits = qdrant.search(nome, vetor, limit)
        except Exception as exc:
            puladas.append({"collection": nome, "motivo": f"erro: {type(exc).__name__}"})
            continue
        pesquisadas.append(nome)
        for hit in hits:
            p = hit.get("payload", {}) or {}
            doc = None
            for chave in ("document", "text", "content", "page_content", "chunk", "body"):
                if isinstance(p.get(chave), str) and p[chave]:
                    doc = p[chave]
                    break
            resultados.append({
                "collection": nome, "id": hit.get("id"),
                "score": round(hit.get("score", 0.0), 4), "document": doc,
                "metadata": p.get("metadata", {}),
                "payload": None if doc is not None else p,
            })
    resultados.sort(key=lambda h: -h["score"])

    return {"searched": sorted(pesquisadas), "skipped": puladas,
            "total_found": len(resultados), "results": resultados[:max_results]}
