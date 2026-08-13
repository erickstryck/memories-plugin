"""Clientes de embedding e de re-rank, contra endpoints OpenAI-compatible.

Os dois modelos vivem atrás de HTTP e são intercambiáveis por configuração. O que
este módulo garante é que o RESTO do pacote não precise saber qual servidor está
do outro lado — e isso não é abstração gratuita: servidores diferentes devolvem o
score do re-rank em ESCALAS diferentes, e essa diferença já quebrou um corte
calibrado em silêncio (ver `normalize_scores`).
"""
import json
import math
import urllib.error
import urllib.request

EMBED_BATCH = 32


class ModelError(Exception):
    pass


def _post(url: str, api_key: str, body: dict, timeout: float):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        corpo = exc.read().decode()[:400]
        raise ModelError(f"HTTP {exc.code} em {url}: {corpo}") from exc
    except urllib.error.URLError as exc:
        raise ModelError(f"não alcancei {url}: {exc.reason}") from exc


class Embedder:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 60.0):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, textos: list[str]) -> list[list[float]]:
        """Embeda em lotes. O endpoint aceita `input` como array, então N textos
        custam N/EMBED_BATCH idas à rede em vez de N.

        Ordena por `index` mesmo que o contrato prometa ordem: já vi servidor
        devolver fora de ordem, e um vetor trocado de lugar produz busca errada
        sem nenhum erro visível.
        """
        if not textos:
            return []
        saida: list[list[float]] = []
        for i in range(0, len(textos), EMBED_BATCH):
            lote = textos[i:i + EMBED_BATCH]
            res = _post(self.url, self.api_key, {"model": self.model, "input": lote}, self.timeout)
            data = res.get("data")
            if not isinstance(data, list) or len(data) != len(lote):
                raise ModelError(
                    f"endpoint devolveu {len(data) if isinstance(data, list) else '?'} "
                    f"vetores para {len(lote)} textos — resposta incompleta, nada foi gravado"
                )
            for d in sorted(data, key=lambda x: x.get("index", 0)):
                saida.append(d["embedding"])

        return saida

    def embed_one(self, texto: str) -> list[float]:
        return self.embed([texto])[0]

    def detect_dimension(self) -> int:
        """Pergunta ao endpoint quantas dimensões o modelo devolve.

        Existe para `vector_size` não ser um número digitado à mão: trocar o modelo
        e esquecer de ajustar a dimensão faz a guarda de compatibilidade recusar
        coleções que estavam perfeitamente boas, com mensagem que aponta para o
        lugar errado. Uma sonda de um texto curto responde isso sem ambiguidade.
        """
        return len(self.embed_one("dimension probe"))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)  # evita overflow quando x é muito negativo

    return e / (1.0 + e)


def normalize_scores(pares: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], bool]:
    """Converte score de re-rank para a faixa 0..1, se necessário.

    O MESMO modelo devolve escalas diferentes conforme o servidor: um aplica
    sigmoid e devolve 0..1, outro devolve o LOGIT cru. Medido com bge-reranker-v2-m3
    num documento irrelevante: 1.6e-05 num servidor e -11.04 no outro, sendo
    -11.04 exatamente logit(1.6e-05) — mesma inferência, sem o squash final.

    Isso importa porque um corte calibrado numa escala é INÓCUO na outra: 0.10 em
    escala de logit fica praticamente no centro da distribuição e aceita quase
    tudo. A falha é silenciosa — nenhum erro, só relevância pior. Detectar pela
    faixa e normalizar mantém o número calibrado válido em qualquer servidor.

    Devolve (pares normalizados, era_logit).
    """
    if not pares:
        return [], False
    era_logit = any(s < 0.0 or s > 1.0 for _, s in pares)
    if era_logit:
        pares = [(i, sigmoid(s)) for i, s in pares]

    return pares, era_logit


class Reranker:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 15.0,
                 max_docs: int = 12, doc_chars: int = 8000, query_chars: int = 2000):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_docs = max_docs
        self.doc_chars = doc_chars
        self.query_chars = query_chars

    def rank(self, query: str, documentos: list[str]) -> tuple[list[tuple[int, float]], dict]:
        """Reordena `documentos` para `query`.

        Devolve (pares ordenados por score desc, info). `info` traz `ok`,
        `era_logit` e `descartados` — o chamador PRECISA saber se o re-rank rodou,
        porque um pipeline de dois estágios que relaxa o primeiro corte contando
        com o segundo fica PIOR que o estágio único quando o segundo falha.

        Um cross-encoder faz um forward por par (query, documento) e não tem
        vetor pré-computável: o custo é linear no total de tokens. Daí os tetos de
        número de pares e de tamanho — o corte é do JULGAMENTO, não do que o
        chamador entrega adiante.
        """
        info = {"ok": False, "era_logit": False, "descartados": 0, "erro": None}
        if not documentos:
            return [], info
        descartados = max(0, len(documentos) - self.max_docs)
        info["descartados"] = descartados
        candidatos = documentos[:self.max_docs]
        corpo = {
            "model": self.model,
            "query": query[:self.query_chars],
            "documents": [d[:self.doc_chars] for d in candidatos],
        }
        try:
            res = _post(self.url, self.api_key, corpo, self.timeout)
        except Exception as exc:
            info["erro"] = f"{type(exc).__name__}: {exc}"

            return [], info

        linhas = res.get("results") or res.get("data") or []
        pares: list[tuple[int, float]] = []
        for linha in linhas:
            if not isinstance(linha, dict):
                continue
            idx = linha.get("index")
            sc = linha.get("relevance_score", linha.get("score"))
            if not isinstance(idx, int) or sc is None or not 0 <= idx < len(candidatos):
                continue
            pares.append((idx, float(sc)))

        if not pares:
            info["erro"] = f"resposta sem hits utilizáveis: {str(res)[:200]}"

            return [], info

        pares, era_logit = normalize_scores(pares)
        pares.sort(key=lambda p: -p[1])
        info["ok"] = True
        info["era_logit"] = era_logit

        return pares, info
