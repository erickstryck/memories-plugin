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
        self.collections: dict[str, dict] = {}   # nome -> {"size", "pontos": {id: ponto}}
        self.calls: list[tuple] = []

    # ---- coleções ----
    def list_collections(self) -> list[str]:
        return list(self.collections)

    def collection_info(self, name: str) -> dict | None:
        c = self.collections.get(name)
        if c is None:
            return None

        return {"size": c["size"], "distance": "Cosine", "points": len(c["pontos"])}

    def ensure_collection(self, name: str, size: int, distance: str = "Cosine") -> bool:
        if name in self.collections:
            if self.collections[name]["size"] != size:
                raise ValueError(f"dimensão {self.collections[name]['size']} != {size}")

            return False
        self.collections[name] = {"size": size, "pontos": {}}
        self.calls.append(("ensure_collection", name))

        return True

    def ensure_payload_index(self, name: str, field: str, schema: str) -> None:
        self.calls.append(("ensure_payload_index", name, field))

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    # ---- pontos ----
    def upsert(self, name: str, points: list[dict], batch: int = 256) -> int:
        self.ensure_collection(name, self.collections.get(name, {}).get("size", 4))
        for p in points:
            self.collections[name]["pontos"][p["id"]] = p

        return len(points)

    def get_point(self, name: str, point_id):
        return self.collections.get(name, {}).get("pontos", {}).get(point_id)

    def set_payload(self, name: str, point_id, payload: dict) -> None:
        point = self.collections.get(name, {}).get("pontos", {}).get(point_id)
        if point is not None:
            point["payload"] = payload
        self.calls.append(("set_payload", name, point_id))

    def delete_points(self, name: str, ids: list) -> None:
        for i in ids:
            self.collections.get(name, {}).get("pontos", {}).pop(i, None)

    def delete_by_filter(self, name: str, filter_: dict) -> None:
        points = self.collections.get(name, {}).get("pontos", {})
        for pid in [p for p, v in points.items() if _matches_filter(v.get("payload", {}), filter_)]:
            points.pop(pid)

    def search(self, name: str, vector: list[float], limit: int,
               filter_: dict | None = None, with_payload: bool = True) -> list[dict]:
        self.calls.append(("search", name, limit))
        output = []
        for pid, p in self.collections.get(name, {}).get("pontos", {}).items():
            if filter_ and not _matches_filter(p.get("payload", {}), filter_):
                continue
            output.append({"id": pid, "score": _cosine(vector, p["vector"]),
                          "payload": p.get("payload", {})})
        output.sort(key=lambda h: -h["score"])

        return output[:limit]

    def scroll(self, name: str, limit: int = 256, offset=None,
               with_vector: bool = False, filter_: dict | None = None):
        items = [{"id": pid, "payload": p.get("payload", {})}
                 for pid, p in self.collections.get(name, {}).get("pontos", {}).items()
                 if not filter_ or _matches_filter(p.get("payload", {}), filter_)]

        return items[:limit], None

    def scroll_all(self, name: str, filter_: dict | None = None, with_vector: bool = False):
        items, _ = self.scroll(name, limit=10_000, filter_=filter_)
        yield from items


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0

    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _matches_filter(payload: dict, filter_: dict) -> bool:
    """Suporta as duas formas que o pacote usa: `match.value` e `range`."""
    for cond in filter_.get("must", []):
        key = cond.get("key")
        value = payload.get(key)
        if "match" in cond:
            if value != cond["match"].get("value"):
                return False
        elif "range" in cond:
            range_ = cond["range"]
            if value is None:
                return False
            if "gt" in range_ and not value > range_["gt"]:
                return False
            if "lt" in range_ and not value < range_["lt"]:
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
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))

        return [self._vector_for(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def detect_dimension(self) -> int:
        return self.dim

    def _vector_for(self, text: str) -> list[float]:
        # `crc32` e NÃO `hash()`: o hash de string em Python é salgado por processo,
        # então o mesmo texto cai em posições diferentes a cada execução. Um dublê com
        # similaridade que muda entre execuções produz teste que passa hoje e falha
        # amanhã sem nada ter mudado — pior que não ter dublê.
        vec = [0.0] * self.dim
        for word in text.lower().split():
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        if not any(vec):
            vec[0] = 1.0

        return vec


class FailingFakeEmbedder:
    """Levanta o erro do domínio. Para exercitar o caminho de degradação."""

    def __init__(self, error):
        self.error = error

    def embed(self, texts):
        raise self.error

    def embed_one(self, text):
        raise self.error


class FakeReranker:
    """Devolve os scores combinados, na ordem dos documentos recebidos."""

    def __init__(self, scores=None, ok=True, error=None, was_logit=False):
        self.scores = scores
        self.ok = ok
        self.error = error
        self.was_logit = was_logit
        self.calls: list[tuple] = []

    def rank(self, query, documents):
        self.calls.append((query, list(documents)))
        if not self.ok:
            return [], {"ok": False, "error": self.error or "failed", "was_logit": False}
        scores = self.scores if self.scores is not None else [1.0] * len(documents)
        pairs = sorted(enumerate(scores[:len(documents)]), key=lambda p: -p[1])

        return pairs, {"ok": True, "error": None, "was_logit": self.was_logit}
