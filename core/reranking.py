"""Cliente de re-rank (cross-encoder), com o contrato de rede como ESTRATÉGIA.

Duas coisas moram aqui, e nenhuma é o algoritmo de recuperação — esse está em
`retrieval`, compartilhado. Aqui é só a conversa com o servidor:

1. O CONTRATO DE REDE, que varia por implementação de servidor. Antes era um `if`
   dentro do método de ranquear, olhando o sufixo da URL. Agora cada forma é uma
   classe pequena: acrescentar um servidor novo é escrever uma estratégia e
   registrá-la, sem tocar no caminho que já funciona (OCP). O `if` também escondia
   que as duas formas diferem no CORPO e na LEITURA da resposta, não só no caminho.

2. A NORMALIZAÇÃO DE ESCALA, que é obrigatória e não opcional: o MESMO modelo
   devolve sigmoid (0..1) num servidor e logit cru noutro. Medido com
   bge-reranker-v2-m3 no mesmo documento irrelevante: 1.6e-05 num servidor e -11.04
   no outro, sendo -11.04 exatamente logit(1.6e-05). Um corte calibrado numa escala
   é INÓCUO na outra — 0.10 em escala de logit fica no centro da distribuição e
   aceita quase tudo. A falha é silenciosa: nenhum erro, só relevância pior.
"""
import math
from dataclasses import dataclass

from .errors import CoreError
from .http import HttpError, bearer, post_json


class RerankError(CoreError):
    pass


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)  # evita overflow quando x é muito negativo

    return e / (1.0 + e)


def normalize_scores(pairs: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], bool]:
    """Converte para a faixa 0..1 quando a resposta veio em logit.

    Detecta pela FAIXA e não por configuração: configuração erra em silêncio quando
    alguém troca o servidor e esquece de atualizar. Equivalência útil de cabeça:
    sigmoid 0.10 corresponde a logit -2.197.

    Devolve (pares normalizados, era_logit).
    """
    if not pairs:
        return [], False
    was_logit = any(s < 0.0 or s > 1.0 for _, s in pairs)
    if was_logit:
        pairs = [(i, sigmoid(s)) for i, s in pairs]

    return pairs, was_logit


# ---- estratégias de contrato de rede ---------------------------------------

@dataclass(frozen=True)
class WireContract:
    """Como montar o pedido e como ler a resposta de um servidor de re-rank."""
    name: str

    def body(self, model: str, query: str, documents: list[str]) -> dict:
        raise NotImplementedError

    def parse(self, response: dict) -> list[tuple[int, float]]:
        raise NotImplementedError


class JinaContract(WireContract):
    """`{model, query, documents}` -> `results[].relevance_score`.

    Forma exposta por servidores que seguem a API de rerank da JinaAI, incluindo o
    endpoint `/rerank` do llama.cpp e do vLLM.
    """

    def __init__(self):
        super().__init__(name="jina")

    def body(self, model: str, query: str, documents: list[str]) -> dict:
        return {"model": model, "query": query, "documents": documents}

    def parse(self, response: dict) -> list[tuple[int, float]]:
        return _pairs(response.get("results") or [], "relevance_score")


class ScoreContract(WireContract):
    """`{model, text_1, text_2}` -> `data[].score`.

    Forma do endpoint `/score`, que existe em paralelo ao `/rerank` em alguns
    servidores e é a única disponível em outros.
    """

    def __init__(self):
        super().__init__(name="score")

    def body(self, model: str, query: str, documents: list[str]) -> dict:
        return {"model": model, "text_1": query, "text_2": documents}

    def parse(self, response: dict) -> list[tuple[int, float]]:
        return _pairs(response.get("data") or [], "score")


def _pairs(lines: list, field: str) -> list[tuple[int, float]]:
    output = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        idx = line.get("index")
        value = line.get(field, line.get("relevance_score", line.get("score")))
        if not isinstance(idx, int) or value is None:
            continue
        output.append((idx, float(value)))

    return output


def contract_for(url: str) -> WireContract:
    """Escolhe a estratégia pelo endereço. Sufixo `/score` usa a forma de score;
    o resto usa a forma Jina, que é a mais comum."""
    return ScoreContract() if url.rstrip("/").endswith("score") else JinaContract()


# ---- cliente ----------------------------------------------------------------

class Reranker:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 15.0,
                 max_docs: int = 12, doc_chars: int = 8000, query_chars: int = 2000,
                 contract: WireContract | None = None):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_docs = max_docs
        self.doc_chars = doc_chars
        self.query_chars = query_chars
        self.contract = contract or contract_for(url)

    def rank(self, query: str, documents: list[str]) -> tuple[list[tuple[int, float]], dict]:
        """Reordena `documentos` para `query`. NUNCA levanta.

        Devolve (pares ordenados por score desc, info com `ok`). Falha é DEGRADAÇÃO,
        não exceção: o re-rank melhora a ordenação e nunca é pré-requisito dela, e
        quem chama precisa poder seguir com o primeiro estágio. Mas `ok` tem de ser
        checado — um pipeline que relaxa o primeiro corte contando com o segundo fica
        pior que o estágio único quando o segundo falha em silêncio.

        Um cross-encoder faz um forward por par (query, documento) e não tem vetor
        pré-computável: o custo é linear no total de tokens. Daí os tetos — o corte é
        do JULGAMENTO, não do que o chamador entrega adiante.
        """
        info = {"ok": False, "era_logit": False, "descartados": 0,
                "erro": None, "contrato": self.contract.name}
        if not documents:
            return [], info

        info["descartados"] = max(0, len(documents) - self.max_docs)
        candidates = [d[:self.doc_chars] for d in documents[:self.max_docs]]
        # A LEITURA da resposta fica DENTRO do try: um servidor que responde uma
        # lista, ou um score em string, fazia o parse explodir e quebrava a promessa
        # de "nunca levanta" — justamente com a resposta inesperada, que é quando a
        # promessa importa.
        try:
            response = post_json(self.url,
                                 self.contract.body(self.model, query[:self.query_chars], candidates),
                                 headers=bearer(self.api_key), timeout=self.timeout)
            pairs = [(i, s) for i, s in self.contract.parse(response)
                     if 0 <= i < len(candidates)]
        except HttpError as exc:
            info["erro"] = str(exc)

            return [], info
        except Exception as exc:  # resposta em forma inesperada, e o que mais vier
            info["erro"] = f"{type(exc).__name__}: {exc}"

            return [], info
        if not pairs:
            info["erro"] = f"resposta sem hits utilizáveis: {str(response)[:200]}"

            return [], info

        pairs, was_logit = normalize_scores(pairs)
        pairs.sort(key=lambda p: -p[1])
        info["ok"] = True
        info["era_logit"] = was_logit

        return pairs, info
