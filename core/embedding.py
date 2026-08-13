"""Cliente de embedding, contra endpoint OpenAI-compatible.

Uma responsabilidade: texto -> vetor. O que era um módulo `models.py` com duas
famílias de modelo virou dois, porque nada aqui muda quando o contrato do re-rank
muda, e vice-versa — juntá-los fazia toda alteração num tocar o outro.
"""
from .http import bearer, post_json

#: Lote por requisição. O endpoint aceita `input` como array, então N textos custam
#: N/EMBED_BATCH idas à rede em vez de N.
EMBED_BATCH = 32


class EmbeddingError(Exception):
    pass


class Embedder:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 60.0):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, textos: list[str]) -> list[list[float]]:
        """Vetores na mesma ordem dos textos.

        Ordena por `index` mesmo que o contrato prometa ordem: já vi servidor
        devolver fora de ordem, e um vetor trocado de lugar produz busca errada sem
        nenhum erro visível. Resposta incompleta LEVANTA em vez de devolver menos —
        gravar metade de um lote é o pior estado possível para um acervo.
        """
        if not textos:
            return []
        saida: list[list[float]] = []
        for i in range(0, len(textos), EMBED_BATCH):
            lote = textos[i:i + EMBED_BATCH]
            res = post_json(self.url, {"model": self.model, "input": lote},
                            headers=bearer(self.api_key), timeout=self.timeout)
            data = res.get("data")
            if not isinstance(data, list) or len(data) != len(lote):
                quantos = len(data) if isinstance(data, list) else "?"
                raise EmbeddingError(
                    f"endpoint devolveu {quantos} vetores para {len(lote)} textos — "
                    f"resposta incompleta, nada foi gravado"
                )
            for d in sorted(data, key=lambda x: x.get("index", 0)):
                saida.append(d["embedding"])

        return saida

    def embed_one(self, texto: str) -> list[float]:
        return self.embed([texto])[0]

    def detect_dimension(self) -> int:
        """Pergunta ao endpoint quantas dimensões o modelo devolve.

        Existe para `vector_size` não ser número digitado à mão: trocar o modelo e
        esquecer de ajustar faz a guarda de compatibilidade recusar coleções que
        estavam boas, culpando a coleção em vez da configuração.
        """
        return len(self.embed_one("dimension probe"))
