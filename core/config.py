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

from . import statefile
from .bigfile import FLOOR_PCT, SHARE_PCT
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
    "checkpoint_interval": ("QCTX_CHECKPOINT_INTERVAL", "REMEMBER_INTERVAL"),
    "bigfile_floor_pct": ("QCTX_BIGFILE_FLOOR_PCT", "BIGFILE_FLOOR_PCT"),
    "bigfile_share_pct": ("QCTX_BIGFILE_SHARE_PCT", "BIGFILE_SHARE_PCT"),
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
    "checkpoint_interval": 5,
    # The guard's own constants, so the numbers have one owner and `decide` called without
    # a config still means what the config means by default.
    "bigfile_floor_pct": FLOOR_PCT,
    "bigfile_share_pct": SHARE_PCT,
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
    #: Turns between the write procedure being handed to the model; 0 turns it off. Both
    #: hosts read it from here, so `config set` reaches both and the file is the one place
    #: it can be set, under the environment's override like every other field.
    checkpoint_interval: int = 5
    #: The big-file guard's two thresholds, as fractions. A read is refused when the context
    #: left after it would be less than `bigfile_floor_pct` of the window, or when the read
    #: alone would take more than `bigfile_share_pct` of what is free. Both hosts read them
    #: from here; the rule that applies them is `core.bigfile._blocks`.
    bigfile_floor_pct: float = FLOOR_PCT
    bigfile_share_pct: float = SHARE_PCT

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


#: The fields that must hold a number, read off the dataclass instead of listed by hand so a
#: numeric field added later inherits the tolerant coercion in `load` without anyone having to
#: remember. A hand-kept copy of this list is exactly the shape that once left one knob using
#: bare `int()` six lines below the tolerant helper its nine siblings used.
_NUMERIC_FIELDS = tuple(f.name for f in fields(Config) if f.type in (int, "int"))


def numeric_fields() -> tuple:
    """The config fields that must hold a whole number.

    Public because the CLI has to REFUSE a non-number at the moment it is typed: `load` falls
    back to the default so a bad file cannot take the plugin down, and a `config set` that
    wrote the value anyway would leave the user reading back a setting the loader ignores.
    Two places needing the same list is exactly how they drift, so there is one.
    """
    return _NUMERIC_FIELDS


#: The fields that hold a fraction, from 0 to 1, read off the dataclass for the same reason
#: `_NUMERIC_FIELDS` is: a fraction field added later inherits the tolerance.
_FRACTION_FIELDS = tuple(f.name for f in fields(Config) if f.type in (float, "float"))


def fraction_fields() -> tuple:
    """The config fields that must hold a fraction from 0 to 1. Public for the reason
    `numeric_fields` is: the CLI and the wizard refuse at the door what `load` tolerates."""
    return _FRACTION_FIELDS


def as_fraction(value) -> float:
    """`value` as a fraction from 0 to 1, or ValueError (TypeError for a non-scalar).

    ONE rule for the three doors a fraction comes through: the loader, which falls back,
    and `config set` and the wizard, which refuse. NaN fails the range test like any value
    outside it, and so does infinity; a percentage typed as `20` is out of range too, which
    is the typo this exists for.
    """
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{value!r} is not between 0 and 1")

    return number


def read_file(path: Path | None = None) -> dict:
    p = path or DEFAULT_CONFIG_PATH
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid config at {p}: {exc}") from exc


#: Pass as `path` to resolve WITHOUT the file: the environment and the defaults only. For a
#: caller that must still answer when the file is unreadable (the checkpoint hook).
NO_FILE = False


def load(path: Path | None = None, env: dict | None = None, note=None) -> Config:
    """The resolved configuration: environment, then file, then default.

    `note` is the CALLER's channel for a value that could not be used, the way it is in
    `core/knobs.py`: the decision (fall back to the default) lives here, while where the
    operator reads about it (a hook's stderr, a prefixed line from hermes) belongs to the
    host. Without a `note` the fallback is silent, as it always was.
    """
    env = os.environ if env is None else env
    from_file = {} if path is NO_FILE else read_file(path)
    values = {}
    source = {}
    for field, aliases in ENV_ALIASES.items():
        value = None
        for name in aliases:
            # BLANK IS UNSET, as `core.knobs.env` reads every other knob and as 1.1.0 read the
            # checkpoint interval: a blank canonical name falls through to the legacy one.
            # Measured by review, `QCTX_CHECKPOINT_INTERVAL="  "` beside REMEMBER_INTERVAL=4
            # fired every 5 turns with a note on every prompt, where 1.1.0 fired every 4.
            if (env.get(name) or "").strip():
                value, source[field] = env[name], (name, "")
                break
        if value is None:
            value = from_file.get(field, DEFAULTS[field])
            source[field] = (field, f" in {path or DEFAULT_CONFIG_PATH}")
        values[field] = value
    # TOLERANT, AND DERIVED FROM THE DATACLASS. These are values a person types, in a file or
    # in the environment, so a typo is ordinary — and bare `int()` here made one typo fatal to
    # everything: every command raised on load, including the one that repairs the file, and
    # the hermes loader swallowed the ValueError (not a CoreError) so the memory provider
    # disappeared with a single debug line. Reading the field list off `Config` rather than
    # naming the two by hand is the same lesson this project already paid for once, when nine
    # knobs read through a tolerant helper and one did not: the odd one out is invisible.
    for numeric in _NUMERIC_FIELDS:
        try:
            values[numeric] = int(values[numeric])
        except (TypeError, ValueError):
            if note:
                name, where = source[numeric]
                note(f"{name}={values[numeric]!r}{where} is not a number, "
                     f"using {DEFAULTS[numeric]}")
            values[numeric] = DEFAULTS[numeric]
    for fraction in _FRACTION_FIELDS:
        try:
            values[fraction] = as_fraction(values[fraction])
        except (TypeError, ValueError):
            if note:
                name, where = source[fraction]
                note(f"{name}={values[fraction]!r}{where} is not a fraction between 0 and 1, "
                     f"using {DEFAULTS[fraction]}")
            values[fraction] = DEFAULTS[fraction]

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
    statefile.ensure_dir(p.parent)
    statefile.write_text(p, json.dumps(current, indent=2, ensure_ascii=False) + "\n")

    return p


def redacted(cfg: Config) -> dict:
    """Config for display, without leaking a secret into a log or a terminal."""
    d = asdict(cfg)
    for key in ("qdrant_api_key", "api_key"):
        if d[key]:
            d[key] = f"<{len(d[key])} chars>"

    return d
