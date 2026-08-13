"""JSON sobre HTTP, em um só lugar.

Antes cada adaptador tinha o seu: o cliente do banco vetorial e o dos modelos
repetiam montagem de request, tratamento de erro e decodificação. Duplicação
dessas é onde a correção entra em metade dos lugares — foi exatamente o que
aconteceu com a normalização de escala do re-rank, que existiu num consumidor e
não no outro.

Stdlib de propósito: este código roda dentro de hook disparado a cada interação do
usuário, e uma dependência faltando transformaria erro de ambiente em perda
silenciosa de funcionalidade.
"""
import json
import urllib.error
import urllib.request


class HttpError(Exception):
    """Falha de transporte ou status de erro. Carrega o corpo, truncado.

    O corpo importa: sem ele, `HTTP 400` não distingue modelo inexistente de
    payload malformado de coleção com dimensão errada, e o diagnóstico vira
    tentativa e erro.
    """

    def __init__(self, mensagem: str, status: int | None = None, corpo: str = ""):
        super().__init__(mensagem)
        self.status = status
        self.corpo = corpo


def request_json(url: str, *, method: str = "GET", body=None, headers: dict | None = None,
                 timeout: float = 30.0):
    """Faz a requisição e decodifica JSON. Corpo vazio devolve `{}`."""
    dados = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=dados, method=method)
    for chave, valor in (headers or {}).items():
        req.add_header(chave, valor)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cru = resp.read()

            return json.loads(cru.decode()) if cru else {}
    except urllib.error.HTTPError as exc:
        corpo = exc.read().decode(errors="replace")[:400]
        raise HttpError(f"HTTP {exc.code} em {method} {url}: {corpo}",
                        status=exc.code, corpo=corpo) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"não alcancei {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise HttpError(f"{url} respondeu algo que não é JSON: {exc}") from exc


def post_json(url: str, body, *, headers: dict | None = None, timeout: float = 30.0):
    return request_json(url, method="POST", body=body, headers=headers, timeout=timeout)


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}
