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
from datetime import datetime, timezone


class MemoryError_(Exception):
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
        self.q.ensure_collection(self.collection, self.vector_size)

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
        self.ensure()
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

    def recall(self, queries: list[str], dense_floor: float, strict_floor: float,
               top_k: int, min_score: float, max_results: int) -> tuple[list[Recalled], dict]:
        """Pipeline de dois portões, com múltiplos ÂNGULOS da mesma pergunta.

        Ângulos diferentes do mesmo texto pescam registros diferentes, e vão numa
        única chamada de embeddings porque o endpoint aceita `input` como array.
        A fusão é por id pelo MAIOR score: um registro que aparece em dois ângulos
        não deve ser penalizado pelo pior deles.
        """
        self.ensure()
        vetores = self.embedder.embed(queries)
        fundidos: dict[str, dict] = {}
        for vetor in vetores:
            for hit in self.q.search(self.collection, vetor, top_k):
                p = hit.get("payload", {})
                doc = p.get("document")
                if not isinstance(doc, str) or not doc.strip():
                    continue
                mid = str(hit.get("id"))
                score = float(hit.get("score") or 0.0)
                anterior = fundidos.get(mid)
                if anterior is None or score > anterior["score"]:
                    fundidos[mid] = {
                        "id": mid, "score": score, "document": doc.strip(),
                        "metadata": p.get("metadata") or {}, "updated_at": p.get("updated_at"),
                    }

        ordenados = sorted(fundidos.values(), key=lambda h: -h["score"])
        candidatos = [h for h in ordenados if h["score"] >= dense_floor]
        info = {"hits": len(ordenados), "candidatos": len(candidatos), "rerank": None,
                "ce_ran": False, "melhor_denso": ordenados[0]["score"] if ordenados else 0.0}
        if not candidatos:
            return [], info

        # O cross-encoder tem dois papéis e só o segundo obriga a chamada:
        # ESCOLHER (mais candidatos que vagas) e FILTRAR (existe candidato na faixa
        # permissiva, que só entrou porque alguém ia julgá-lo). Sem nenhum dos dois,
        # reordenar não muda o que sai e a chamada é trabalho inútil.
        precisa_escolher = len(candidatos) > max_results
        precisa_filtrar = any(h["score"] < strict_floor for h in candidatos)
        if self.reranker is not None and (precisa_escolher or precisa_filtrar):
            pares, rinfo = self.reranker.rank(queries[0], [h["document"] for h in candidatos])
            info["rerank"] = rinfo
            info["ce_ran"] = rinfo["ok"]
            if rinfo["ok"]:
                selecionados = []
                for i, s in pares:
                    if s < min_score:
                        continue
                    h = dict(candidatos[i])
                    h["dense_score"] = h["score"]
                    h["score"] = s
                    h["origem"] = "CE"
                    selecionados.append(h)
            else:
                selecionados = self._strict(candidatos, strict_floor)
        else:
            selecionados = self._strict(candidatos, strict_floor)

        resultados = [
            Recalled(id=h["id"], score=h["score"], origem=h.get("origem", "denso"),
                     dense_score=h.get("dense_score", h["score"]), document=h["document"],
                     metadata=h["metadata"], updated_at=h.get("updated_at"))
            for h in selecionados
        ]

        return resultados, info

    @staticmethod
    def _strict(candidatos: list[dict], strict_floor: float) -> list[dict]:
        """Volta ao corte estrito. Necessário sempre que o cross-encoder não
        julgou: o piso permissivo só era seguro porque ele ia limpar depois."""
        saida = []
        for h in candidatos:
            if h["score"] < strict_floor:
                continue
            h = dict(h)
            h["origem"] = "denso"
            saida.append(h)

        return saida

    def get(self, mid: str) -> dict:
        ponto = self.q.get_point(self.collection, mid)
        if ponto is None:
            return {"status": "not_found", "id": mid}
        p = ponto.get("payload", {})

        return {"id": ponto.get("id"), "document": p.get("document"),
                "metadata": p.get("metadata", {}), "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at")}

    def list_page(self, limit: int = 20, offset=None) -> dict:
        self.ensure()
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
