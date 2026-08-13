"""Dublês em memória dos contratos de `core.ports`.

Existem porque a propriedade mais importante do pacote — o payload gravado ser
byte-compatível com o do servidor que ele substitui — estava verificada SÓ contra
infraestrutura viva. Teste que exige Qdrant e GPU não roda enquanto se edita, e o
que não roda não protege.

Nenhum dublê herda de nada: os contratos são `Protocol`, então basta ter os métodos.
`FakeVectorStore` guarda os pontos como estão, sem normalizar nada, justamente para
que um teste possa afirmar sobre as CHAVES do payload.
"""
import math
import zlib


class FakeVectorStore:
    """Banco vetorial em dicionário. Similaridade real (cosseno), não simulada."""

    def __init__(self):
        self.colecoes: dict[str, dict] = {}   # nome -> {"size", "pontos": {id: ponto}}
        self.chamadas: list[tuple] = []

    # ---- coleções ----
    def list_collections(self) -> list[str]:
        return list(self.colecoes)

    def collection_info(self, nome: str) -> dict | None:
        c = self.colecoes.get(nome)
        if c is None:
            return None

        return {"size": c["size"], "distance": "Cosine", "points": len(c["pontos"])}

    def ensure_collection(self, nome: str, size: int, distance: str = "Cosine") -> bool:
        if nome in self.colecoes:
            if self.colecoes[nome]["size"] != size:
                raise ValueError(f"dimensão {self.colecoes[nome]['size']} != {size}")

            return False
        self.colecoes[nome] = {"size": size, "pontos": {}}
        self.chamadas.append(("ensure_collection", nome))

        return True

    def ensure_payload_index(self, nome: str, campo: str, schema: str) -> None:
        self.chamadas.append(("ensure_payload_index", nome, campo))

    def delete_collection(self, nome: str) -> None:
        self.colecoes.pop(nome, None)

    # ---- pontos ----
    def upsert(self, nome: str, pontos: list[dict], batch: int = 256) -> int:
        self.ensure_collection(nome, self.colecoes.get(nome, {}).get("size", 4))
        for p in pontos:
            self.colecoes[nome]["pontos"][p["id"]] = p

        return len(pontos)

    def get_point(self, nome: str, ponto_id):
        return self.colecoes.get(nome, {}).get("pontos", {}).get(ponto_id)

    def set_payload(self, nome: str, ponto_id, payload: dict) -> None:
        ponto = self.colecoes.get(nome, {}).get("pontos", {}).get(ponto_id)
        if ponto is not None:
            ponto["payload"] = payload
        self.chamadas.append(("set_payload", nome, ponto_id))

    def delete_points(self, nome: str, ids: list) -> None:
        for i in ids:
            self.colecoes.get(nome, {}).get("pontos", {}).pop(i, None)

    def delete_by_filter(self, nome: str, filtro: dict) -> None:
        pontos = self.colecoes.get(nome, {}).get("pontos", {})
        for pid in [p for p, v in pontos.items() if _casa(v.get("payload", {}), filtro)]:
            pontos.pop(pid)

    def search(self, nome: str, vector: list[float], limit: int,
               filtro: dict | None = None, with_payload: bool = True) -> list[dict]:
        self.chamadas.append(("search", nome, limit))
        saida = []
        for pid, p in self.colecoes.get(nome, {}).get("pontos", {}).items():
            if filtro and not _casa(p.get("payload", {}), filtro):
                continue
            saida.append({"id": pid, "score": _cos(vector, p["vector"]),
                          "payload": p.get("payload", {})})
        saida.sort(key=lambda h: -h["score"])

        return saida[:limit]

    def scroll(self, nome: str, limit: int = 256, offset=None,
               with_vector: bool = False, filtro: dict | None = None):
        itens = [{"id": pid, "payload": p.get("payload", {})}
                 for pid, p in self.colecoes.get(nome, {}).get("pontos", {}).items()
                 if not filtro or _casa(p.get("payload", {}), filtro)]

        return itens[:limit], None

    def scroll_all(self, nome: str, filtro: dict | None = None, with_vector: bool = False):
        itens, _ = self.scroll(nome, limit=10_000, filtro=filtro)
        yield from itens


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0

    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _casa(payload: dict, filtro: dict) -> bool:
    """Suporta as duas formas que o pacote usa: `match.value` e `range`."""
    for cond in filtro.get("must", []):
        chave = cond.get("key")
        valor = payload.get(chave)
        if "match" in cond:
            if valor != cond["match"].get("value"):
                return False
        elif "range" in cond:
            faixa = cond["range"]
            if valor is None:
                return False
            if "gt" in faixa and not valor > faixa["gt"]:
                return False
            if "lt" in faixa and not valor < faixa["lt"]:
                return False

    return True


class FakeEmbedder:
    """Vetores determinísticos derivados do texto.

    Determinístico e não aleatório para que a similaridade seja PREVISÍVEL: textos
    que compartilham palavras ficam próximos, o que permite testar recuperação de
    verdade em vez de testar o dublê.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.chamadas: list[list[str]] = []

    def embed(self, textos: list[str]) -> list[list[float]]:
        self.chamadas.append(list(textos))

        return [self._vetor(t) for t in textos]

    def embed_one(self, texto: str) -> list[float]:
        return self.embed([texto])[0]

    def detect_dimension(self) -> int:
        return self.dim

    def _vetor(self, texto: str) -> list[float]:
        # `crc32` e NÃO `hash()`: o hash de string em Python é salgado por processo,
        # então o mesmo texto cai em posições diferentes a cada execução. Um dublê com
        # similaridade que muda entre execuções produz teste que passa hoje e falha
        # amanhã sem nada ter mudado — pior que não ter dublê.
        vec = [0.0] * self.dim
        for palavra in texto.lower().split():
            vec[zlib.crc32(palavra.encode()) % self.dim] += 1.0
        if not any(vec):
            vec[0] = 1.0

        return vec


class FakeEmbedderQueQuebra:
    """Levanta o erro do domínio. Para exercitar o caminho de degradação."""

    def __init__(self, erro):
        self.erro = erro

    def embed(self, textos):
        raise self.erro

    def embed_one(self, texto):
        raise self.erro


class FakeReranker:
    """Devolve os scores combinados, na ordem dos documentos recebidos."""

    def __init__(self, scores=None, ok=True, erro=None, era_logit=False):
        self.scores = scores
        self.ok = ok
        self.erro = erro
        self.era_logit = era_logit
        self.chamadas: list[tuple] = []

    def rank(self, query, documentos):
        self.chamadas.append((query, list(documentos)))
        if not self.ok:
            return [], {"ok": False, "erro": self.erro or "falhou", "era_logit": False}
        scores = self.scores if self.scores is not None else [1.0] * len(documentos)
        pares = sorted(enumerate(scores[:len(documentos)]), key=lambda p: -p[1])

        return pares, {"ok": True, "erro": None, "era_logit": self.era_logit}
