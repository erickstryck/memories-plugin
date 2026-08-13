"""Resolução de configuração.

Precedência, do mais forte para o mais fraco: variável de ambiente > arquivo de
config > default. O arquivo existe para o que é escolha durável (qual coleção
usar), o ambiente para o que muda por máquina ou por deploy (endereços, chaves).

Os nomes canônicos são `QCTX_*`. Os nomes LEGADOS também são aceitos, porque este
pacote nasceu substituindo um servidor MCP feito à mão que já usava
`SERVER_BASE_URL` / `QDRANT_SERVICE_API_KEY` / `RECALL_*`; quebrar isso obrigaria
a reconfigurar um ambiente que já funciona, sem ganho nenhum.

Nada aqui conhece o host que está chamando — este módulo é a fronteira entre o
núcleo portável e o mundo.
"""
import json
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("QCTX_CONFIG")
    or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "memories-plugin" / "config.json"
)

# Cada campo lista os nomes de ambiente que o alimentam, em ordem de precedência.
# O primeiro é o canônico; os seguintes são legado aceito por compatibilidade.
ENV_ALIASES = {
    "qdrant_url": ("QCTX_QDRANT_URL", "QDRANT_URL"),
    "qdrant_api_key": ("QCTX_QDRANT_API_KEY", "QDRANT_SERVICE_API_KEY", "QDRANT_API_KEY"),
    "api_base_url": ("QCTX_API_BASE_URL", "SERVER_BASE_URL"),
    "api_key": ("QCTX_API_KEY", "SERVER_API_KEY"),
    "embed_url": ("QCTX_EMBED_URL", "RECALL_EMBED_URL"),
    "rerank_url": ("QCTX_RERANK_URL", "RECALL_RERANK_URL"),
    "embed_model": ("QCTX_EMBED_MODEL", "EMBEDDING_MODEL"),
    "rerank_model": ("QCTX_RERANK_MODEL", "RECALL_RERANK_MODEL"),
    "memory_collection": ("QCTX_MEMORY_COLLECTION", "COLLECTION_NAME"),
    "docs_collection": ("QCTX_DOCS_COLLECTION", "DOCS_COLLECTION"),
    "library_collection": ("QCTX_LIBRARY_COLLECTION", "LIBRARY_COLLECTION"),
    "vector_size": ("QCTX_VECTOR_SIZE", "VECTOR_SIZE"),
}

DEFAULTS = {
    "qdrant_url": "",
    "qdrant_api_key": "",
    "api_base_url": "",
    "api_key": "",
    "embed_url": "",
    "rerank_url": "",
    "embed_model": "bge-m3",
    "rerank_model": "bge-reranker-v2-m3",
    "memory_collection": "",
    "docs_collection": "memories_docs_tmp",
    "library_collection": "memories_docs_library",
    "vector_size": 1024,
}


class ConfigError(Exception):
    pass


@dataclass
class Config:
    qdrant_url: str
    qdrant_api_key: str
    api_base_url: str
    api_key: str
    embed_url: str
    rerank_url: str
    embed_model: str
    rerank_model: str
    memory_collection: str
    docs_collection: str
    library_collection: str
    vector_size: int

    def resolved_embed_url(self) -> str:
        """URL completa do /embeddings.

        Aceita as duas formas porque os dois consumidores históricos diferem: um
        guarda o caminho completo, o outro guarda a base e concatena.
        """
        if self.embed_url:
            return self.embed_url
        if self.api_base_url:
            return f"{self.api_base_url.rstrip('/')}/embeddings"

        raise ConfigError("nem embed_url nem api_base_url configurados")

    def resolved_rerank_url(self) -> str:
        if self.rerank_url:
            return self.rerank_url
        if self.api_base_url:
            return f"{self.api_base_url.rstrip('/')}/rerank"

        raise ConfigError("nem rerank_url nem api_base_url configurados")

    def require_qdrant(self) -> None:
        if not self.qdrant_url:
            raise ConfigError("qdrant_url não configurado (env QCTX_QDRANT_URL ou `config set qdrant-url`)")

    def require_memory_collection(self) -> str:
        if not self.memory_collection:
            raise ConfigError(
                "memory_collection não configurada. Veja as existentes com "
                "`collections list` e escolha com `config set memory-collection <nome>`"
            )

        return self.memory_collection

    def require_docs_collection(self) -> str:
        return self._require_doc_collection("docs_collection", self.docs_collection)

    def require_library_collection(self) -> str:
        return self._require_doc_collection("library_collection", self.library_collection)

    def _require_doc_collection(self, nome_campo: str, valor: str) -> str:
        """Garante que as TRÊS coleções sejam distintas.

        Cada colisão possível tem uma consequência concreta, e nenhuma delas dá
        erro na hora — todas degradam em silêncio:

        - documento na coleção de MEMÓRIA: um arquivo longo vira dezenas de
          trechos verbosos que competem com fato curado em toda busca e ganham por
          volume. É poluição permanente do acervo que mais importa.
        - biblioteca na coleção TEMPORÁRIA: a temporária é destruível por
          construção (`drop --all` apaga a coleção), então um comando de limpeza
          passaria a ser capaz de apagar acervo permanente.
        """
        if not valor:
            raise ConfigError(f"{nome_campo} não configurada")
        outras = {
            "memory_collection": self.memory_collection,
            "docs_collection": self.docs_collection,
            "library_collection": self.library_collection,
        }
        for outro_campo, outro_valor in outras.items():
            if outro_campo == nome_campo or not outro_valor:
                continue
            if outro_valor == valor:
                raise ConfigError(
                    f"{nome_campo} e {outro_campo} apontam para a mesma coleção "
                    f"({valor!r}). As três coleções têm ciclos de vida diferentes e "
                    f"precisam ser distintas — veja `collections list`."
                )

        return valor


def read_file(path: Path | None = None) -> dict:
    p = path or DEFAULT_CONFIG_PATH
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config inválido em {p}: {exc}") from exc


def load(path: Path | None = None, env: dict | None = None) -> Config:
    env = os.environ if env is None else env
    arquivo = read_file(path)
    valores = {}
    for campo, aliases in ENV_ALIASES.items():
        valor = None
        for nome in aliases:
            if env.get(nome):
                valor = env[nome]
                break
        if valor is None:
            valor = arquivo.get(campo, DEFAULTS[campo])
        valores[campo] = valor
    valores["vector_size"] = int(valores["vector_size"])

    return Config(**valores)


#: Campos que NUNCA vão para o arquivo de config. Segredo em arquivo de texto é
#: segredo vazado: entra em backup, em sincronização de dotfiles e em `cat` casual.
#: O ambiente já resolve, e é onde as chaves deste stack sempre viveram.
SECRET_FIELDS = frozenset({"qdrant_api_key", "api_key"})


def save(patch: dict, path: Path | None = None) -> Path:
    """Grava só o que mudou, preservando o resto do arquivo."""
    p = path or DEFAULT_CONFIG_PATH
    validos = {f.name for f in fields(Config)}
    desconhecidos = set(patch) - validos
    if desconhecidos:
        raise ConfigError(f"chave(s) desconhecida(s): {', '.join(sorted(desconhecidos))}")
    segredos = set(patch) & SECRET_FIELDS
    if segredos:
        nomes = ", ".join(sorted(segredos))
        canonicos = ", ".join(ENV_ALIASES[s][0] for s in sorted(segredos))
        raise ConfigError(
            f"{nomes} não vai para o arquivo de config — segredo em texto puro entra "
            f"em backup e em sincronização de dotfiles. Exporte no ambiente: {canonicos}"
        )
    atual = read_file(p)
    atual.update(patch)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(atual, indent=2, ensure_ascii=False) + "\n")

    return p


def redacted(cfg: Config) -> dict:
    """Config para exibição, sem vazar segredo em log ou terminal."""
    d = asdict(cfg)
    for chave in ("qdrant_api_key", "api_key"):
        if d[chave]:
            d[chave] = f"<{len(d[chave])} chars>"

    return d
