"""The operations the MODEL may invoke, as hermes tool schemas.

Fifteen of the CLI's twenty. The five left out are `setup` (interactive, wants a TTY) and
`config show/set/detect` plus `collections` — configuration belongs to the operator, and a
`config set` tool would let the model point the archive somewhere else mid-conversation.
The CLI still carries all twenty in both hosts; this is only about what the model reaches
on its own.

Every handler returns a JSON STRING, and no handler raises. hermes surfaces a raise as a
crashed turn; a JSON error is something the model can read and react to. The string part is
not a detail either: the ABC says "Must return a JSON string" (agent/memory_provider.py,
:182 in v0.20.0 and :232 in the v0.20.1 now installed — same sentence, moved), and a dict
there fails at the host boundary, past every test of a handler's own logic.

Each handler routes to the SAME core function the CLI's matching `cmd_*` handler calls, so
the two hosts cannot drift on what an operation does. Where a `cmd_*` handler carried logic
rather than rendering, that logic now lives in the core and both call it:
`core.metadata_from`, `core.outcome_payload`, `DocIndex.drop_request`.
"""
import json
import os
import sys

#: The SAME bootstrap `hosts/hermes/__init__.py` does, and it has to be here too — this
#: module can be the first file of the package to run.
#:
#: Measured against the loader that actually loads memory providers
#: (`plugins/memory/__init__.py::_load_provider_from_dir`, lines 282-297 in v0.20.0 and
#: 486-501 in the v0.20.1 now installed — unchanged code, moved down the file; NOT
#: `hermes_cli/plugins.py::_load_local_module`, which skips `kind: exclusive` plugins):
#: before exec'ing `__init__.py` it pre-execs every sibling `*.py`, registering each in
#: `sys.modules` FIRST and swallowing any failure at `logger.debug`. Without these three
#: lines, `import core` here raised ModuleNotFoundError during that pre-exec — `core` only
#: becomes importable once `__init__.py` inserts the repo root — and the broken shell STAYED
#: in `sys.modules`, so the package's own `from . import tools` then succeeded and handed
#: back a module with nothing in it. The provider failed to load entirely, taking recall and
#: the checkpoint cadence with it, and the only symptom was one debug line.
#:
#: `realpath` and THREE `dirname` levels, for the reason the package docstring gives: the
#: plugin is installed as a symlink, and `abspath` resolves to the symlink's directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core  # noqa: E402
import core.docs  # noqa: E402

#: Filled by `hosts/hermes/__init__.py` at import time — see `bind_tuning`.
_TUNING = None


def bind_tuning(provider_class) -> None:
    """Hand this module the provider class whose constants are the recall tuning.

    `memory_recall` has to run with THIS host's floors, not a second copy of the numbers.
    The `QCTX_RECALL_*` reads cannot move here — tests/test_host_equivalence.py extracts
    those names from `hosts/hermes/__init__.py` and from `hooks/checkpoint.py`/`recall.py`
    and requires the two sets to match, so a knob that left the adapter's own file would
    read as a setting the two hosts no longer share. And this module cannot import back
    into the package while the package is still executing. So the provider hands its class
    over, once, on the line that wires the two together.
    """
    global _TUNING
    _TUNING = provider_class


def _tuning():
    if _TUNING is not None:
        return _TUNING
    try:                       # someone imported this module without the package wiring
        from . import MemoriesProvider

        return MemoriesProvider
    except ImportError as exc:  # pragma: no cover — defensive; the wiring is tested
        raise ToolArgError(f"the memories adapter is not fully loaded: {exc}")


def _ok(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _err(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


class ToolArgError(Exception):
    """A bad or missing argument, reported TO THE MODEL rather than raised at the host.

    Separate from `core.CoreError` on purpose: a core error describes the archive, this one
    describes the call. The model can only fix the second kind, and it can only fix it if
    the message names the argument.
    """


# ---- argument coercion -----------------------------------------------------
#
# The model emits arguments from a schema, not from a type checker, and it WILL get one
# wrong: an integer as "5", a boolean as "true", an object as its JSON text. Accepting what
# is unambiguous and naming what is not costs one turn less than refusing everything.


def _require(args: dict, name: str):
    value = args.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ToolArgError(f"missing required argument: {name!r}")

    return value


def _text(args: dict, name: str, *, blank_is_absent: bool = True):
    """An optional string argument, absent when not given.

    TYPE-CHECKED like every other coercer here. It was the only one without a check, and it
    showed: `memory_update(id, information=42)` came back as
    `{"error": "AttributeError: 'int' object has no attribute 'strip'"}` — the interpreter's
    words about the core's internals, which names neither the argument nor the fix, so the
    model cannot act on it. Every other coercer answers "'x' must be an integer, got …".

    `blank_is_absent` is the difference between an OPTIONAL FILTER and a REPLACEMENT VALUE.
    For `doc_id`, `""` reasonably means "no filter given". For `memory_update`'s
    `information` it cannot: mapping `""` to None made the tool answer
    `{"status": "updated", "reembedded": false}` with the text untouched, while the schema
    says "It cannot be empty or blank" (so the schema was false) and the CLI raised on the
    same input (so the two hosts disagreed). The blanking refusal in `core` never saw it —
    None never reaches the guard, by design, because None means "keep the current text".
    """
    value = args.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolArgError(f"{name!r} must be a string, got {type(value).__name__} "
                           f"({value!r}) — send the text itself")
    if not value.strip():
        if blank_is_absent:
            return None
        raise ToolArgError(f"{name!r} cannot be empty or blank: omit it to keep the "
                           f"current text, or delete the record")

    return value


def _int(args: dict, name: str, default: int) -> int:
    raw = args.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ToolArgError(f"{name!r} must be an integer, got {raw!r}")


TRUE = {"true", "1", "yes", "on"}
FALSE = {"false", "0", "no", "off", ""}


def _bool(args: dict, name: str, default: bool = False) -> bool:
    raw = args.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in TRUE:
        return True
    if text in FALSE:
        return False
    raise ToolArgError(f"{name!r} must be true or false, got {raw!r}")


def _choice(args: dict, name: str, allowed, default: str) -> str:
    """A closed set of values, mirroring the CLI's argparse `choices`.

    Checked here and not only in the core because one of them is load-bearing:
    `DocIndex.refresh` BRANCHES on the scope, so `all` would reindex a library document as
    a temporary one and silently give a permanent document an expiry.
    """
    value = args.get(name)
    if value in (None, ""):
        return default
    if value not in allowed:
        raise ToolArgError(f"{name!r} must be one of {', '.join(allowed)}, got {value!r}")

    return value


def _object(args: dict, name: str, required: bool = False):
    """A JSON value that must be a list or dict, accepted as its JSON text too."""
    raw = args.get(name)
    if raw in (None, "") and required:
        raise ToolArgError(f"missing required argument: {name!r}")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise ToolArgError(f"{name!r} is not valid JSON: {exc}")

    return raw


def _metadata(args: dict):
    """The metadata rule, from the core: base object plus the shortcuts, shortcuts win."""
    return core.metadata_from(args.get("metadata"),
                              **{field: args.get(field) for field in core.METADATA_FIELDS})


# ---- the archives ----------------------------------------------------------


def _configured(cfg):
    if cfg is None:
        raise ToolArgError(
            "long-term memory is not configured on this host — the operator has to run "
            "`qctx setup`. Nothing was read or written.")

    return cfg


def _memory(cfg):
    return core.build_memory(_configured(cfg))


def _docs(cfg):
    return core.build_docs(_configured(cfg))


# ---- memory ----------------------------------------------------------------


def _memory_store(args: dict, cfg) -> str:
    information = _require(args, "information")
    metadata = _metadata(args)

    return _ok(_memory(cfg).store(information, metadata))


def _memory_store_many(args: dict, cfg) -> str:
    items = _object(args, "items", required=True)
    if not isinstance(items, list):
        raise ToolArgError("'items' must be an array of {information, metadata?}, "
                           f"got {type(items).__name__}")
    store = _memory(cfg)

    return _ok(store.store_many(items))


def _memory_find(args: dict, cfg) -> str:
    query = _require(args, "query")
    limit = _int(args, "limit", 5)          # the CLI's own default

    return _ok(_memory(cfg).find(query, limit))


def _memory_recall(args: dict, cfg) -> str:
    query = _require(args, "query")
    tuning = _tuning()
    limit = _int(args, "limit", tuning.MAX_MEMORIES)
    store = _memory(cfg)
    policy = core.Policy(dense_floor=tuning.DENSE_FLOOR, strict_floor=tuning.STRICT_FLOOR,
                         min_score=tuning.MIN_SCORE, max_results=limit,
                         veto=True, order_matters=False)
    hits, outcome = store.recall([query], policy, tuning.TOP_K)

    return _ok({"info": core.outcome_payload(outcome), "hits": [h.__dict__ for h in hits]})


def _memory_get(args: dict, cfg) -> str:
    # The argument is validated BEFORE the archive is built, in every handler here, so an
    # unconfigured host does not answer a malformed call by complaining about the
    # configuration: the model can fix its own argument and cannot fix the deployment.
    mid = _require(args, "id")

    return _ok(_memory(cfg).get(mid))


def _memory_update(args: dict, cfg) -> str:
    mid = _require(args, "id")
    # `or None` is the difference between "leave the labels alone" and "clear them": an
    # assembled `{}` REPLACES the metadata, so a call that only fixes the text would strip
    # every label. Same rule the CLI's `cmd_memory_update` follows.
    metadata = _metadata(args) or None

    # `blank_is_absent=False`: here an empty string is a REPLACEMENT, not an omission, and
    # the schema promises it is refused. See `_text`.
    information = _text(args, "information", blank_is_absent=False)

    return _ok(_memory(cfg).update(mid, information, metadata))


def _memory_delete(args: dict, cfg) -> str:
    mid = _require(args, "id")

    return _ok(_memory(cfg).delete(mid))


def _memory_list(args: dict, cfg) -> str:
    return _ok(_memory(cfg).list_page(_int(args, "limit", 20)))


def _memory_search_collections(args: dict, cfg) -> str:
    query = _require(args, "query")
    limit = _int(args, "limit", 5)
    collections = _object(args, "collections")
    if collections is not None and not isinstance(collections, list):
        raise ToolArgError("'collections' must be an array of collection names")
    cfg = _configured(cfg)
    result = core.search_collections(core.build_qdrant(cfg), core.build_embedder(cfg),
                                     query, collections or None, cfg.vector_size,
                                     limit=limit)

    return _ok(result)


# ---- documents -------------------------------------------------------------


def _docs_index(args: dict, cfg) -> str:
    path = _require(args, "path")
    ttl = core.parse_ttl(args.get("ttl") or "24h")     # the CLI's own default

    return _ok(_docs(cfg).index_file(path, ttl, _text(args, "doc_id")))


def _docs_keep(args: dict, cfg) -> str:
    path = _require(args, "path")

    return _ok(_docs(cfg).keep_file(path, _text(args, "doc_id")))


def _docs_search(args: dict, cfg) -> str:
    query = _require(args, "query")
    scope = _choice(args, "scope", core.docs.SCOPES, "all")
    doc_id, limit = _text(args, "doc_id"), _int(args, "limit", 5)
    hits, outcome = _docs(cfg).search(query, scope, doc_id, limit)

    return _ok({"info": core.outcome_payload(outcome), "hits": [h.__dict__ for h in hits]})


def _docs_list(args: dict, cfg) -> str:
    return _ok(_docs(cfg).list_docs(_choice(args, "scope", core.docs.SCOPES, "all")))


def _docs_refresh(args: dict, cfg) -> str:
    scope = _choice(args, "scope", ("library", "tmp"), "library")

    return _ok(_docs(cfg).refresh(scope))


def _docs_drop(args: dict, cfg) -> str:
    return _ok(_docs(cfg).drop_request(_text(args, "doc_id"),
                                       _choice(args, "scope", core.docs.SCOPES, "all"),
                                       purge_tmp=_bool(args, "purge_tmp"),
                                       expired=_bool(args, "expired")))


# ---- schemas ---------------------------------------------------------------
#
# The `description` is what the model reads to decide, so each one says WHEN to reach for
# the tool and not only what it does. Automatic recall already ran for the user's prompt
# before the turn started; the memory read tools are for a facet the prompt did not name.

_META_PROPERTIES = {
    "metadata": {"type": "object",
                 "description": "Free-form labels stored with the record, e.g. "
                                "{\"type\": \"decision\", \"project\": \"x\"}."},
    "type": {"type": "string",
             "description": "Shortcut for metadata.type: decision, reference, "
                            "preference, bug, measurement…"},
    "project": {"type": "string", "description": "Shortcut for metadata.project."},
    "area": {"type": "string", "description": "Shortcut for metadata.area."},
}

#: NOT the shared `_META_PROPERTIES`, for the same reason `docs_drop` does not share `_SCOPE`:
#: on an UPDATE the labels already exist, and passing any of them REPLACES the whole set —
#: `core.MemoryStore.update` writes the metadata it is given, it does not merge. That is the
#: existing, intended contract of this system and it is not changing; what must not happen is
#: a model discovering it by "fixing one label" and losing the other two. Measured, through
#: the real dispatcher: a record labelled {type, project, area} updated with `type` alone ends
#: up with `{"type": ...}` and nothing else. So every property that can trigger the
#: replacement says so, because a model fills in one argument by reading that argument.
_REPLACES_THE_LABELS = ("Passing this REPLACES the record's whole label set — any label not "
                        "sent here or alongside it is REMOVED. Include every label you want "
                        "to keep (memory_get shows the current ones), or send no labels at "
                        "all to leave them untouched.")

#: Measured, on both surfaces: `metadata={}` here and `--json-meta '{}'` in the CLI leave the
#: labels untouched and report `"updated"`. Both assemble the metadata and then apply `or
#: None`, which is what makes "no labels sent" mean "leave them alone" — and an empty object
#: is indistinguishable from that. Saying "the record's labels after the update" without this
#: made the schema false for the one call a model would try in order to clear them.
_CANNOT_BE_CLEARED = ("An EMPTY object does not clear the labels: it reads as \"no labels "
                      "sent\", so they are left as they are. There is no way to remove every "
                      "label — replace them with the ones you want instead.")

_UPDATE_META_PROPERTIES = {
    "metadata": {"type": "object",
                 "description": "The labels the record should END UP with, e.g. "
                                "{\"type\": \"decision\", \"project\": \"x\"}. "
                                + _REPLACES_THE_LABELS + " " + _CANNOT_BE_CLEARED},
    "type": {"type": "string",
             "description": "Shortcut for metadata.type: decision, reference, preference, "
                            "bug, measurement… " + _REPLACES_THE_LABELS},
    "project": {"type": "string",
                "description": "Shortcut for metadata.project. " + _REPLACES_THE_LABELS},
    "area": {"type": "string",
             "description": "Shortcut for metadata.area. " + _REPLACES_THE_LABELS},
}

_SCOPE = {"type": "string", "enum": list(core.docs.SCOPES),
          "description": "Which archive: tmp (expiring), library (permanent) or all."}

SCHEMAS = [
    # -- memory, reading --
    {
        "name": "memory_recall",
        "description": ("Search long-term memory with the two-stage pipeline (dense plus "
                        "cross-encoder). Use before asserting a fact about this codebase, "
                        "an SDK or a past decision, and before proposing a design in an "
                        "area with history. Automatic recall already ran for the user's "
                        "prompt; use this for a facet the prompt did not name."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Topic in natural language, not a symbol name."},
                "limit": {"type": "integer",
                          "description": "Max memories to return (default 6)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_find",
        "description": ("Cheap dense-only search of long-term memory, with no re-ranking. "
                        "Use it when the ORDER among the results does not matter — above "
                        "all to check whether a fact is already stored before writing it, "
                        "so the archive does not accumulate near-duplicates. For deciding "
                        "what is true, use memory_recall instead."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic in natural language."},
                "limit": {"type": "integer", "description": "Max hits (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_get",
        "description": ("Fetch one memory by id, whole. Use it when a recalled memory was "
                        "delivered as a pointer (id plus a summary line) and you need its "
                        "full text, or to read a record before updating it."),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string",
                       "description": "The memory id, as shown by recall, find or list."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "memory_list",
        "description": ("Page through the archive without a query, newest page first. Use "
                        "it to inspect or audit what is stored — never to answer a "
                        "question, which is what memory_recall is for."),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer",
                          "description": "Records in this page (default 20)."},
            },
            "required": [],
        },
    },
    {
        "name": "memory_search_collections",
        "description": ("Read-only search across OTHER systems' Qdrant collections that "
                        "share this embedding model. Use it only when the answer might "
                        "live in an archive this plugin does not own; collections built "
                        "with a different model are skipped and reported rather than read."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic in natural language."},
                "collections": {"type": "array", "items": {"type": "string"},
                                "description": "Collection names; omit to try them all."},
                "limit": {"type": "integer",
                          "description": "Max hits per collection (default 5)."},
            },
            "required": ["query"],
        },
    },
    # -- memory, writing --
    {
        "name": "memory_store",
        "description": ("Write ONE durable, atomic fact to long-term memory. Use it for a "
                        "decision and its reason, a measured number, a user preference, a "
                        "non-obvious constraint — anything that would have to be "
                        "rediscovered in a later session. Search first (memory_find) so a "
                        "near-duplicate becomes an update instead of a second record. Do "
                        "not store conversation, plans in progress or anything the code "
                        "already says."),
        "parameters": {
            "type": "object",
            "properties": {
                "information": {"type": "string",
                                "description": "The fact, self-contained: it will be read "
                                               "with no conversation around it."},
                **_META_PROPERTIES,
            },
            "required": ["information"],
        },
    },
    {
        "name": "memory_store_many",
        "description": ("Write several facts in ONE all-or-nothing batch. Use it at a "
                        "checkpoint, when a session produced a handful of durable facts at "
                        "once: it costs one embedding round trip instead of N, and a "
                        "failure leaves nothing half-written."),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {"type": "array",
                          "items": {"type": "object",
                                    "properties": {
                                        "information": {"type": "string",
                                                        "description": "The fact."},
                                        "metadata": {"type": "object",
                                                     "description": "Labels for it."},
                                    },
                                    "required": ["information"]},
                          "description": "The facts, each atomic and self-contained."},
            },
            "required": ["items"],
        },
    },
    {
        "name": "memory_update",
        "description": ("Correct an existing memory in place, keeping its id. Use it when "
                        "a stored fact turned out wrong or incomplete, instead of writing "
                        "a second record that contradicts the first. Omitting `information` "
                        "keeps the text, and it cannot be blanked — a memory with no text is "
                        "refused. The labels are REPLACED WHOLESALE, not merged: send every "
                        "label the record should end up with, or send none at all to leave "
                        "them as they are."),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The memory id to correct."},
                "information": {"type": "string",
                                "description": "Replacement text; omit to keep the current "
                                               "one. It cannot be empty or blank."},
                **_UPDATE_META_PROPERTIES,
            },
            "required": ["id"],
        },
    },
    {
        "name": "memory_delete",
        "description": ("Remove a memory permanently. Use it only for something that is "
                        "wrong and has no corrected form — a fact that is merely outdated "
                        "should be updated, so the reasoning behind it is not lost. There "
                        "is no undo."),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The memory id to remove."},
            },
            "required": ["id"],
        },
    },
    # -- documents --
    {
        "name": "docs_index",
        "description": ("Index a long document TEMPORARILY (it expires) so you can search "
                        "the parts that answer instead of reading the whole file into "
                        "context. Use it BEFORE opening a large log, dump, transcript or "
                        "report when the question is about one part of it. Then search with "
                        "docs_search."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Path to a text file on this machine."},
                "ttl": {"type": "string",
                        "description": "How long it lives: 30m, 24h, 7d (default 24h)."},
                "doc_id": {"type": "string",
                           "description": "Optional stable id; reindexing the same id "
                                          "replaces the previous version."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "docs_keep",
        "description": ("Index a document into the PERMANENT library, with no expiry. Use "
                        "it for reference material worth searching in later sessions — a "
                        "specification, an SDK's documentation, a runbook. For a document "
                        "needed only for the task at hand, use docs_index."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Path to a text file on this machine."},
                "doc_id": {"type": "string",
                           "description": "Optional stable id; reindexing it replaces the "
                                          "previous version."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "docs_search",
        "description": ("Search the indexed documents and get back only the relevant "
                        "chunks. Use it after docs_index or docs_keep, and whenever the "
                        "answer may already be in the library. A result in locator mode "
                        "gives file and line range so you read the CURRENT content."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question, in prose."},
                "scope": _SCOPE,
                "doc_id": {"type": "string",
                           "description": "Restrict to one document — the full pass over "
                                          "it, with no candidate ceiling."},
                "limit": {"type": "integer", "description": "Max chunks (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "docs_list",
        "description": ("List the indexed documents, with their chunk counts and expiry. "
                        "Use it to find out whether a document is already indexed (and "
                        "under which doc_id) before indexing it again, or when a search "
                        "returns nothing and the index may have expired."),
        "parameters": {
            "type": "object",
            "properties": {"scope": _SCOPE},
            "required": [],
        },
    },
    {
        "name": "docs_refresh",
        "description": ("Reindex the documents whose file changed on disk since indexing. "
                        "Use it when a search result carried a stale warning, or before "
                        "trusting the library on a file that is edited often — a chunk from "
                        "an old version answers with text that no longer exists."),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["library", "tmp"],
                          "description": "Which archive to check (default library)."},
            },
            "required": [],
        },
    },
    {
        "name": "docs_drop",
        "description": ("Remove indexed documents: one by doc_id, everything temporary "
                        "(purge_tmp), or just what has expired (expired). Use it when a "
                        "task's documents are no longer needed, or to take something out "
                        "of the library. The library is never touched by purge_tmp."),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string",
                           "description": "The document to remove, as shown by docs_list."},
                # NOT the shared _SCOPE: the default matters more here than anywhere else,
                # because this is the one tool where omitting an argument REMOVES something.
                # The default is `all` to match the CLI — but a model has to be told that,
                # or the permanent library gets touched by an argument nobody supplied.
                "scope": {"type": "string", "enum": list(core.docs.SCOPES),
                          "description": "Which archive to remove from. DEFAULTS TO all, "
                                         "which means a doc_id drop reaches the PERMANENT "
                                         "library as well as the temporary archive. Pass "
                                         "tmp explicitly to leave the library alone."},
                "purge_tmp": {"type": "boolean",
                              "description": "Delete the whole TEMPORARY collection; the "
                                             "permanent library is left alone."},
                "expired": {"type": "boolean",
                            "description": "Remove only entries whose TTL has passed."},
            },
            "required": [],
        },
    },
]

ROUTES = {
    "memory_store": _memory_store,
    "memory_store_many": _memory_store_many,
    "memory_find": _memory_find,
    "memory_recall": _memory_recall,
    "memory_get": _memory_get,
    "memory_update": _memory_update,
    "memory_delete": _memory_delete,
    "memory_list": _memory_list,
    "memory_search_collections": _memory_search_collections,
    "docs_index": _docs_index,
    "docs_keep": _docs_keep,
    "docs_search": _docs_search,
    "docs_list": _docs_list,
    "docs_refresh": _docs_refresh,
    "docs_drop": _docs_drop,
}


def dispatch(name: str, args, *, cfg) -> str:
    """Run a tool. Returns a JSON string, always; raises never.

    `BaseException` is deliberately NOT caught, unlike in `prefetch`: a KeyboardInterrupt
    or a SystemExit turned into a tool result the model then reasons about would be worse
    than the crash it hid. Everything the model can cause is an `Exception`.
    """
    handler = ROUTES.get(name)
    if handler is None:
        return _err(f"unknown tool: {name}")
    try:
        if isinstance(args, str):
            # Measured shape of the problem: some models emit the arguments object as its
            # JSON text. Refusing it costs a turn and teaches the model nothing.
            #
            # Its own try, so unparseable text gets the message below rather than a raw
            # JSONDecodeError from the catch-all at the bottom — "Expecting value: line 1
            # column 1" says nothing about what the argument should have been.
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return _err("arguments must be an object; got text that is not JSON: "
                            f"{args[:80]!r}")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return _err(f"arguments must be an object, got {type(args).__name__}")
        out = handler(args, cfg)

        # The string contract enforced in ONE place, so no handler can break it alone.
        return out if isinstance(out, str) else _ok(out)
    except ToolArgError as exc:
        return _err(str(exc))
    except core.CoreError as exc:
        return _err(f"{type(exc).__name__}: {exc}")
    except KeyError as exc:
        return _err(f"missing required argument: {exc}")
    except Exception as exc:  # noqa: BLE001 — a raise becomes a crashed turn
        return _err(f"{type(exc).__name__}: {exc}")
