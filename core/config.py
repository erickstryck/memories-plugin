"""Configuration resolution.

Precedence, strongest to weakest: environment variable > config file > default. The
file exists for durable choices (which collection to use), the environment for what
changes per machine or per deploy (addresses, keys).

The canonical names are `QCTX_*`. The LEGACY names are accepted too, because this
package was born replacing a hand-made MCP server that already used
`SERVER_BASE_URL` / `QDRANT_SERVICE_API_KEY` / `RECALL_*`; breaking that would force
someone to reconfigure a working environment for no gain at all.

Nothing here knows about the host calling it — this module is the boundary between
the portable core and the world.
"""
import json
import os

from .errors import CoreError
from dataclasses import dataclass, asdict, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("QCTX_CONFIG")
    or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "memories-plugin" / "config.json"
)

# Each field lists the environment names that feed it, in order of precedence.
# The first is the canonical one; the rest are legacy, accepted for compatibility.
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
    "repos_collection": ("QCTX_REPOS_COLLECTION", "REPOS_COLLECTION"),
    "repos_registry_collection": ("QCTX_REPOS_REGISTRY_COLLECTION", "REPOS_REGISTRY_COLLECTION"),
    "vector_size": ("QCTX_VECTOR_SIZE", "VECTOR_SIZE"),
    "context_window": ("QCTX_CONTEXT_WINDOW",),
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
    "repos_collection": "memories_repos",
    "repos_registry_collection": "memories_repos_registry",
    "vector_size": 1024,
    "context_window": 0,
}


class ConfigError(CoreError):
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
    repos_collection: str
    repos_registry_collection: str
    vector_size: int
    context_window: int = 0

    def resolved_embed_url(self) -> str:
        """The full /embeddings URL.

        It accepts both forms because the two historical consumers differ: one stores
        the full path, the other stores the base and concatenates.
        """
        if self.embed_url:
            return self.embed_url
        if self.api_base_url:
            return f"{self.api_base_url.rstrip('/')}/embeddings"

        raise ConfigError("neither embed_url nor api_base_url is configured")

    def resolved_rerank_url(self) -> str:
        if self.rerank_url:
            return self.rerank_url
        if self.api_base_url:
            return f"{self.api_base_url.rstrip('/')}/rerank"

        raise ConfigError("neither rerank_url nor api_base_url is configured")

    def require_qdrant(self) -> None:
        if not self.qdrant_url:
            raise ConfigError("qdrant_url is not configured (env QCTX_QDRANT_URL or `config set qdrant-url`)")

    def require_memory_collection(self) -> str:
        """The memory collection, validated.

        The distinctness check runs HERE too, and not only on the document archives:
        before, only `build_docs` ran it, so a collision went unnoticed on every memory
        path — the recall hook, `store`, `find` — and only surfaced later, as an error in
        a document command, once the archive had already been polluted. A guard that only
        one path runs is not a guard.
        """
        if not self.memory_collection:
            raise ConfigError(
                "memory_collection is not configured. See the existing ones with "
                "`collections list` and pick one with `config set memory-collection <name>`"
            )

        return self._require_distinct("memory_collection", self.memory_collection)

    def require_docs_collection(self) -> str:
        return self._require_doc_collection("docs_collection", self.docs_collection)

    def require_library_collection(self) -> str:
        return self._require_doc_collection("library_collection", self.library_collection)

    def require_repos_collection(self) -> str:
        return self._require_doc_collection("repos_collection", self.repos_collection)

    def require_repos_registry_collection(self) -> str:
        return self._require_doc_collection("repos_registry_collection",
                                            self.repos_registry_collection)

    def _require_doc_collection(self, field_name: str, value: str) -> str:
        if not value:
            raise ConfigError(f"{field_name} is not configured")

        return self._require_distinct(field_name, value)

    def _require_distinct(self, field_name: str, value: str) -> str:
        """Ensures the FIVE collections are distinct.

        Every possible collision has a concrete consequence, and none of them raises at
        the time — they all degrade silently:

        - a document in the MEMORY collection: one long file becomes dozens of verbose
          chunks that compete with curated facts in every search and win on volume. It is
          permanent pollution of the archive that matters most.
        - the library in the TEMPORARY collection: the temporary one is destroyable by
          construction (`drop --all` deletes the collection), so a cleanup command would
          become able to erase a permanent archive.
        - a repo archive on top of the LIBRARY: tens of thousands of automatic code chunks
          drown the hand-picked documents, which is the same volume argument that keeps
          documents out of the memory collection, one level down.
        - the REGISTRY sharing the chunk collection: they are apart so that no search has to
          filter registry rows out, and a filter forgotten once turns a registry row into a
          search hit.
        """
        others = {
            "memory_collection": self.memory_collection,
            "docs_collection": self.docs_collection,
            "library_collection": self.library_collection,
            "repos_collection": self.repos_collection,
            "repos_registry_collection": self.repos_registry_collection,
        }
        for other_field, other_value in others.items():
            if other_field == field_name or not other_value:
                continue
            if other_value == value:
                raise ConfigError(
                    f"{field_name} and {other_field} point at the same collection "
                    f"({value!r}). The five collections have different lifecycles and "
                    f"have to be distinct — see `collections list`."
                )

        return value


def read_file(path: Path | None = None) -> dict:
    p = path or DEFAULT_CONFIG_PATH
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid config at {p}: {exc}") from exc


def load(path: Path | None = None, env: dict | None = None) -> Config:
    env = os.environ if env is None else env
    from_file = read_file(path)
    values = {}
    for field, aliases in ENV_ALIASES.items():
        value = None
        for name in aliases:
            if env.get(name):
                value = env[name]
                break
        if value is None:
            value = from_file.get(field, DEFAULTS[field])
        values[field] = value
    values["vector_size"] = int(values["vector_size"])
    values["context_window"] = int(values["context_window"])

    return Config(**values)


#: Fields that NEVER go into the config file. A secret in a text file is a leaked
#: secret: it ends up in backups, in dotfile sync and in a casual `cat`.
#: The environment already solves this, and it is where this stack's keys have always
#: lived.
SECRET_FIELDS = frozenset({"qdrant_api_key", "api_key"})


def save(patch: dict, path: Path | None = None) -> Path:
    """Writes only what changed, preserving the rest of the file."""
    p = path or DEFAULT_CONFIG_PATH
    valid_keys = {f.name for f in fields(Config)}
    unknown_keys = set(patch) - valid_keys
    if unknown_keys:
        raise ConfigError(f"unknown key(s): {', '.join(sorted(unknown_keys))}")
    secret_keys = set(patch) & SECRET_FIELDS
    if secret_keys:
        names = ", ".join(sorted(secret_keys))
        canonical = ", ".join(ENV_ALIASES[s][0] for s in sorted(secret_keys))
        raise ConfigError(
            f"{names} does not go into the config file — a plaintext secret ends up in "
            f"backups and in dotfile sync. Export it in the environment: {canonical}"
        )
    current = read_file(p)
    current.update(patch)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")

    return p


def redacted(cfg: Config) -> dict:
    """Config for display, without leaking a secret into a log or a terminal."""
    d = asdict(cfg)
    for key in ("qdrant_api_key", "api_key"):
        if d[key]:
            d[key] = f"<{len(d[key])} chars>"

    return d
