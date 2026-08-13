"""Embedding client, against an OpenAI-compatible endpoint.

One responsibility: text -> vector. What used to be a `models.py` module holding two
model families became two, because nothing here changes when the re-rank contract
changes, and vice versa — keeping them together made every change to one touch the
other.
"""
from .errors import CoreError
from .http import HttpError, bearer, post_json

#: Batch per request. The endpoint accepts `input` as an array, so N texts cost
#: N/EMBED_BATCH network round trips instead of N.
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
        """Vectors in the same order as the texts.

        It sorts by `index` even though the contract promises order: I have seen a
        server return them out of order, and a vector in the wrong slot produces a wrong
        search with no visible error. An incomplete response RAISES instead of returning
        fewer — storing half a batch is the worst possible state for an archive.
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
                # Translate into the domain error: the caller talks about embedding, not
                # about transport, and catching HttpError would force every consumer to
                # know about the layer below.
                raise EmbeddingError(str(exc)) from exc
            data = res.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                how_many = len(data) if isinstance(data, list) else "?"
                raise EmbeddingError(
                    f"endpoint returned {how_many} vectors for {len(batch)} texts — "
                    f"incomplete response, nothing was stored"
                )
            for d in sorted(data, key=lambda x: x.get("index", 0)):
                output.append(d["embedding"])

        return output

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def detect_dimension(self) -> int:
        """Asks the endpoint how many dimensions the model returns.

        It exists so `vector_size` is not a hand-typed number: swapping the model and
        forgetting to adjust it makes the compatibility guard refuse collections that
        were fine, blaming the collection instead of the configuration.
        """
        return len(self.embed_one("dimension probe"))
