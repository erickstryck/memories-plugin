"""Contratos das dependências externas.

São `Protocol` e não classe base abstrata de propósito: tipagem ESTRUTURAL, sem
herança, sem registro, sem custo em tempo de execução. Um dublê de teste não
precisa herdar de nada — basta ter os métodos. Classe base abstrata com uma única
implementação real seria cerimônia sem benefício.

O que estes contratos compram, concretamente:

1. As regras de negócio (`retrieval`, `memory`, `docs`) passam a depender destas
   assinaturas e não de `Qdrant`, `Embedder` ou `Reranker` concretos. Trocar Qdrant
   por outro banco vetorial, ou o endpoint de embedding por uma biblioteca local, é
   escrever um adaptador — nenhum arquivo de regra muda.

2. O pipeline de recuperação, que é a lógica mais delicada do pacote, fica
   testável SEM rede. Antes só dava para exercitá-lo em teste de integração, o que
   significa que ninguém o roda enquanto edita.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Transforma texto em vetor."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vetores na MESMA ordem dos textos. Erro se a resposta vier incompleta."""
        ...

    def embed_one(self, text: str) -> list[float]:
        ...


@runtime_checkable
class RerankModel(Protocol):
    """Julga a relevância de documentos para uma pergunta, olhando os dois juntos."""

    def rank(self, query: str, documents: list[str]) -> tuple[list[tuple[int, float]], dict]:
        """Devolve (pares (índice, score 0..1) ordenados por score desc, info).

        `info` PRECISA trazer pelo menos `ok: bool`. Quem chama tem de saber se o
        julgamento aconteceu: um pipeline que relaxa o primeiro estágio contando com
        o segundo fica PIOR que o estágio único quando o segundo falha em silêncio.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Armazena e busca vetores com payload, agrupados em coleções."""

    def ensure_collection(self, name: str, size: int, distance: str = ...) -> bool:
        ...

    def ensure_payload_index(self, name: str, field: str, schema: str) -> None:
        ...

    def collection_info(self, name: str) -> dict | None:
        ...

    def list_collections(self) -> list[str]:
        ...

    def delete_collection(self, name: str) -> None:
        ...

    def upsert(self, name: str, points: list[dict], batch: int = ...) -> int:
        ...

    def search(self, name: str, vector: list[float], limit: int,
               filter_: dict | None = ..., with_payload: bool = ...) -> list[dict]:
        ...

    def get_point(self, name: str, ponto_id) -> dict | None:
        ...

    def set_payload(self, name: str, ponto_id, payload: dict) -> None:
        """Substitui o payload SEM tocar no vetor.

        Existe para que alterar metadata não exija recalcular embedding: além do
        desperdício, sem isto uma correção de etiqueta ficava IMPOSSÍVEL enquanto o
        endpoint de embedding estivesse fora — uma operação que não precisa dele.
        """
        ...

    def delete_points(self, name: str, ids: list) -> None:
        ...

    def delete_by_filter(self, name: str, filter_: dict) -> None:
        ...

    def scroll(self, name: str, limit: int = ..., offset=...,
               with_vector: bool = ..., filter_: dict | None = ...) -> tuple[list[dict], object]:
        ...

    def scroll_all(self, name: str, filter_: dict | None = ..., with_vector: bool = ...):
        ...
