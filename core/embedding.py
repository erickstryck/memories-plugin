"""Cliente de embedding, contra endpoint OpenAI-compatible.

Uma responsabilidade: texto -> vetor. O que era um módulo `models.py` com duas
famílias de modelo virou dois, porque nada aqui muda quando o contrato do re-rank
muda, e vice-versa — juntá-los fazia toda alteração num tocar o outro.
"""
from .errors import CoreError
from .http import HttpError, bearer, post_json

#: Lote por requisição. O endpoint aceita `input` como array, então N textos custam
#: N/EMBED_BATCH idas à rede em vez de N.
EMBED_BATCH = 32


class EmbeddingError(CoreError):
    pass


class Embedder:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 60.0):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vetores na mesma ordem dos textos.

        Ordena por `index` mesmo que o contrato prometa ordem: já vi servidor
        devolver fora de ordem, e um vetor trocado de lugar produz busca errada sem
        nenhum erro visível. Resposta incompleta LEVANTA em vez de devolver menos —
        gravar metade de um lote é o pior estado possível para um acervo.
        """
        if not texts:
            return []
        output: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            try:
                res = post_json(self.url, {"model": self.model, "input": batch},
                                headers=bearer(self.api_key), timeout=self.timeout)
            except HttpError as exc:
                # Traduz para o erro do domínio: quem chama fala de embedding, não
                # de transporte, e capturar HttpError obrigaria todo consumidor a
                # conhecer a camada de baixo.
                raise EmbeddingError(str(exc)) from exc
            data = res.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                how_many = len(data) if isinstance(data, list) else "?"
                raise EmbeddingError(
                    f"endpoint devolveu {how_many} vetores para {len(batch)} textos — "
                    f"resposta incompleta, nada foi gravado"
                )
            for d in sorted(data, key=lambda x: x.get("index", 0)):
                output.append(d["embedding"])

        return output

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def detect_dimension(self) -> int:
        """Pergunta ao endpoint quantas dimensões o modelo devolve.

        Existe para `vector_size` não ser número digitado à mão: trocar o modelo e
        esquecer de ajustar faz a guarda de compatibilidade recusar coleções que
        estavam boas, culpando a coleção em vez da configuração.
        """
        return len(self.embed_one("dimension probe"))
