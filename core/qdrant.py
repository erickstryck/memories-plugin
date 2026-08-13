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
            erro = QdrantError(str(exc))
            erro.status = exc.status
            raise erro from exc

    # ---- coleções ----------------------------------------------------------

    def list_collections(self) -> list[str]:
        res = self.request("GET", "/collections")

        return [c["name"] for c in res.get("result", {}).get("collections", [])]

    def collection_info(self, nome: str) -> dict | None:
        """Devolve {size, distance, points} ou None se não existe.

        Só reconhece coleção com um único vetor não-nomeado; coleção com vetores
        nomeados tem outra forma de busca e não é compatível com este cliente.
        """
        try:
            res = self.request("GET", f"/collections/{nome}")
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        result = res.get("result", {})
        vetores = result.get("config", {}).get("params", {}).get("vectors", {})
        if not isinstance(vetores, dict) or "size" not in vetores:
            return {"size": None, "distance": None, "points": result.get("points_count")}

        return {
            "size": vetores.get("size"),
            "distance": vetores.get("distance"),
            "points": result.get("points_count"),
        }

    def ensure_collection(self, nome: str, size: int, distance: str = "Cosine") -> bool:
        """Cria se não existir. Devolve True se criou.

        Se existir com dimensão DIFERENTE, levanta em vez de seguir: gravar vetor
        de dimensão errada é recusado pelo Qdrant ponto a ponto, mas gravar num
        acervo de outro modelo de embedding passa e degrada a busca em silêncio.
        """
        info = self.collection_info(nome)
        if info is not None:
            if info["size"] not in (None, size):
                raise QdrantError(
                    f"coleção {nome!r} tem dimensão {info['size']}, incompatível com "
                    f"o modelo configurado ({size}). Escolha outra coleção ou outro modelo."
                )

            return False
        self.request("PUT", f"/collections/{nome}", {"vectors": {"size": size, "distance": distance}})

        return True

    def ensure_payload_index(self, nome: str, campo: str, schema: str) -> None:
        """Índice de payload é otimização de filtro, nunca requisito — falha aqui
        não pode derrubar a operação que o chamador queria fazer."""
        try:
            self.request("PUT", f"/collections/{nome}/index?wait=true",
                         {"field_name": campo, "field_schema": schema})
        except QdrantError:
            pass

    def delete_collection(self, nome: str) -> None:
        self.request("DELETE", f"/collections/{nome}")

    # ---- pontos ------------------------------------------------------------

    def upsert(self, nome: str, pontos: list[dict], batch: int = 256) -> int:
        for i in range(0, len(pontos), batch):
            self.request("PUT", f"/collections/{nome}/points?wait=true",
                         {"points": pontos[i:i + batch]})

        return len(pontos)

    def search(self, nome: str, vector: list[float], limit: int,
               filtro: dict | None = None, with_payload: bool = True) -> list[dict]:
        body = {"vector": vector, "limit": limit, "with_payload": with_payload}
        if filtro:
            body["filter"] = filtro
        res = self.request("POST", f"/collections/{nome}/points/search", body)

        return res.get("result", []) or []

    def get_point(self, nome: str, ponto_id) -> dict | None:
        try:
            res = self.request("GET", f"/collections/{nome}/points/{ponto_id}")
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

        return res.get("result")

    def set_payload(self, nome: str, ponto_id, payload: dict) -> None:
        self.request("POST", f"/collections/{nome}/points/payload?wait=true",
                     {"payload": payload, "points": [ponto_id]})

    def delete_points(self, nome: str, ids: list) -> None:
        self.request("POST", f"/collections/{nome}/points/delete?wait=true", {"points": ids})

    def delete_by_filter(self, nome: str, filtro: dict) -> None:
        self.request("POST", f"/collections/{nome}/points/delete?wait=true", {"filter": filtro})

    def scroll(self, nome: str, limit: int = 256, offset=None,
               with_vector: bool = False, filtro: dict | None = None) -> tuple[list[dict], object]:
        body = {"limit": limit, "with_payload": True, "with_vector": with_vector}
        if offset is not None:
            body["offset"] = offset
        if filtro:
            body["filter"] = filtro
        res = self.request("POST", f"/collections/{nome}/points/scroll", body).get("result", {})

        return res.get("points", []), res.get("next_page_offset")

    def scroll_all(self, nome: str, filtro: dict | None = None, with_vector: bool = False):
        """Itera a coleção inteira, paginando. Gerador para não materializar tudo."""
        offset = None
        while True:
            pontos, offset = self.scroll(nome, offset=offset, with_vector=with_vector, filtro=filtro)
            for p in pontos:
                yield p
            if offset is None:
                break
