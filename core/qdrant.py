"""Adaptador do Qdrant: implementa o contrato `ports.VectorStore`.

Por que não o SDK oficial: este pacote é dependência de hooks que rodam a CADA
prompt do usuário. Um `pip install` faltando, um venv errado ou um import lento
transformam falha de dependência em perda SILENCIOSA de funcionalidade.

Nenhuma regra de negócio mora aqui — só tradução entre as operações do contrato e
a API HTTP. É o que permite trocar o banco vetorial escrevendo outro adaptador,
sem tocar em `memory`, `docs` ou `retrieval`.
"""
from .errors import CoreError
from .http import HttpError, request_json


class QdrantError(CoreError):
    pass


class Qdrant:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, method: str, path: str, body=None):
        headers = {"api-key": self.api_key} if self.api_key else {}
        try:
            return request_json(f"{self.base}{path}", method=method, body=body,
                                headers=headers, timeout=self.timeout)
        except HttpError as exc:
            # Preserva o status no objeto: distinguir 404 por substring da mensagem
            # quebra quando a mensagem muda.
            error = QdrantError(str(exc))
            error.status = exc.status
            raise error from exc

    # ---- coleções ----------------------------------------------------------

    def list_collections(self) -> list[str]:
        res = self.request("GET", "/collections")

        return [c["name"] for c in res.get("result", {}).get("collections", [])]

    def collection_info(self, name: str) -> dict | None:
        """Devolve {size, distance, points} ou None se não existe.

        Só reconhece coleção com um único vetor não-nomeado; coleção com vetores
        nomeados tem outra forma de busca e não é compatível com este cliente.
        """
        try:
            res = self.request("GET", f"/collections/{name}")
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        result = res.get("result", {})
        vectors = result.get("config", {}).get("params", {}).get("vectors", {})
        if not isinstance(vectors, dict) or "size" not in vectors:
            return {"size": None, "distance": None, "points": result.get("points_count")}

        return {
            "size": vectors.get("size"),
            "distance": vectors.get("distance"),
            "points": result.get("points_count"),
        }

    def ensure_collection(self, name: str, size: int, distance: str = "Cosine") -> bool:
        """Cria se não existir. Devolve True se criou.

        Se existir com dimensão DIFERENTE, levanta em vez de seguir: gravar vetor
        de dimensão errada é recusado pelo Qdrant ponto a ponto, mas gravar num
        acervo de outro modelo de embedding passa e degrada a busca em silêncio.
        """
        info = self.collection_info(name)
        if info is not None:
            if info["size"] not in (None, size):
                raise QdrantError(
                    f"coleção {name!r} tem dimensão {info['size']}, incompatível com "
                    f"o modelo configurado ({size}). Escolha outra coleção ou outro modelo."
                )

            return False
        self.request("PUT", f"/collections/{name}", {"vectors": {"size": size, "distance": distance}})

        return True

    def ensure_payload_index(self, name: str, field: str, schema: str) -> None:
        """Índice de payload é otimização de filtro, nunca requisito — falha aqui
        não pode derrubar a operação que o chamador queria fazer."""
        try:
            self.request("PUT", f"/collections/{name}/index?wait=true",
                         {"field_name": field, "field_schema": schema})
        except QdrantError:
            pass

    def delete_collection(self, name: str) -> None:
        self.request("DELETE", f"/collections/{name}")

    # ---- pontos ------------------------------------------------------------

    def upsert(self, name: str, points: list[dict], batch: int = 256) -> int:
        for i in range(0, len(points), batch):
            self.request("PUT", f"/collections/{name}/points?wait=true",
                         {"points": points[i:i + batch]})

        return len(points)

    def search(self, name: str, vector: list[float], limit: int,
               filter_: dict | None = None, with_payload: bool = True) -> list[dict]:
        body = {"vector": vector, "limit": limit, "with_payload": with_payload}
        if filter_:
            body["filter"] = filter_
        res = self.request("POST", f"/collections/{name}/points/search", body)

        return res.get("result", []) or []

    def get_point(self, name: str, point_id) -> dict | None:
        try:
            res = self.request("GET", f"/collections/{name}/points/{point_id}")
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

        return res.get("result")

    def set_payload(self, name: str, point_id, payload: dict) -> None:
        self.request("POST", f"/collections/{name}/points/payload?wait=true",
                     {"payload": payload, "points": [point_id]})

    def delete_points(self, name: str, ids: list) -> None:
        self.request("POST", f"/collections/{name}/points/delete?wait=true", {"points": ids})

    def delete_by_filter(self, name: str, filter_: dict) -> None:
        self.request("POST", f"/collections/{name}/points/delete?wait=true", {"filter": filter_})

    def scroll(self, name: str, limit: int = 256, offset=None,
               with_vector: bool = False, filter_: dict | None = None) -> tuple[list[dict], object]:
        body = {"limit": limit, "with_payload": True, "with_vector": with_vector}
        if offset is not None:
            body["offset"] = offset
        if filter_:
            body["filter"] = filter_
        res = self.request("POST", f"/collections/{name}/points/scroll", body).get("result", {})

        return res.get("points", []), res.get("next_page_offset")

    def scroll_all(self, name: str, filter_: dict | None = None, with_vector: bool = False):
        """Itera a coleção inteira, paginando. Gerador para não materializar tudo."""
        offset = None
        while True:
            points, offset = self.scroll(name, offset=offset, with_vector=with_vector, filter_=filter_)
            for p in points:
                yield p
            if offset is None:
                break
