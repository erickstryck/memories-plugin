#!/usr/bin/env python3
"""qctx — the core's command-line interface.

This is how an agent with no MCP support (or any host at all) uses long-term memory
and the ephemeral document index: one process call, JSON or text on the output. All
the logic lives in `core/`; this file only translates arguments.

    qctx collections list
    qctx config show | set <key> <value>
    qctx memory store <text> [--type T] [--project P] [--json META]
    qctx memory find <question> [--limit N]
    qctx memory recall <question> [--limit N]
    qctx memory get|delete <id>
    qctx memory list [--limit N]
    qctx memory update <id> [--text T] [--json META]
    qctx docs index <path> [--ttl 24h]           temporary, expires
    qctx docs keep <path>                        library, permanent
    qctx docs search <question> [--scope all|tmp|library] [--doc-id ID]
    qctx docs list [--scope ...]
    qctx docs refresh [--scope library|tmp]      reindexes what changed on disk
    qctx docs drop <doc-id> [--scope ...] | --purge-tmp | --expired

Three archives, three lifecycles: MEMORY holds curated facts and never expires;
LIBRARY holds documents for reference and never expires; TEMPORARY holds a task's
documents and expires. Distinct collections, by configuration.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
import core.docs  # noqa: E402
import core.setup  # noqa: E402
from core.config import ConfigError  # noqa: E402


def as_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def output(obj, want_json: bool) -> None:
    """Prints the JSON form, or NOTHING when JSON was not asked for.

    The silent branch is deliberate but it is a landmine, so it is named here rather
    than discovered: every caller has to print the human form itself. A handler written
    as `output(x, args.json)` and nothing else prints nothing at all in the default
    mode.
    """
    if want_json:
        print(as_json(obj))


# ---- collections / config --------------------------------------------------

# One definition, in the core, imported here. It used to exist twice under two names
# (`NOISE` here, `NOISE_PREFIXES` in core.setup) with identical values — the kind of
# duplication where the two drift and only one of the two listings changes.
from core.setup import NOISE_PREFIXES as NOISE  # noqa: E402


def cmd_collections(args, cfg):
    q = core.build_qdrant(cfg)
    names = q.list_collections()
    configured = {cfg.memory_collection, cfg.docs_collection, cfg.library_collection}
    if not args.all:
        visible = [n for n in names
                    if n in configured or not n.startswith(NOISE)]
    else:
        visible = list(names)
    hidden = len(names) - len(visible)
    lines = []
    for name in visible:
        info = q.collection_info(name) or {}
        size = info.get("size")
        lines.append({
            "collection": name,
            "points": info.get("points"),
            "dim": size,
            "compatible": size == cfg.vector_size,
            "role": ("memory" if name == cfg.memory_collection
                    else "temporary" if name == cfg.docs_collection
                    else "library" if name == cfg.library_collection else ""),
        })
    lines.sort(key=lambda l: (not l["role"], -(l["points"] or 0)))
    if args.json:
        output({"vector_size": cfg.vector_size, "hidden": hidden,
               "collections": lines}, True)

        return
    print(f"model {cfg.embed_model} uses dimension {cfg.vector_size}\n")
    print(f"{'collection':34} {'points':>8} {'dim':>6}  {'':10} role")
    for l in lines:
        mark = "ok" if l["compatible"] else "INCOMPAT."
        print(f"{l['collection']:34} {str(l['points']):>8} {str(l['dim']):>6}  {mark:10} {l['role']}")
    if hidden:
        print(f"\n({hidden} collection(s) from another system hidden — `--all` shows them)")
    print("\nto choose: qctx config set memory-collection|docs-collection|"
          "library-collection <name>")


def cmd_config_show(args, cfg):
    data = core.redacted(cfg)
    if args.json:
        output(data, True)

        return
    for k, v in data.items():
        print(f"  {k:20} {v}")


def cmd_config_set(args, cfg):
    key = args.key.replace("-", "_")
    value = int(args.value) if key == "vector_size" else args.value
    path = core.save({key: value})
    if args.json:
        output({"key": key, "value": value, "path": str(path)}, True)
    else:
        print(f"{key} = {value}  (written to {path})")
    # A warning, not an error: the collection may be created later. But an incompatible
    # dimension is a silent trap, so it is worth shouting at the moment of choice.
    if key in ("memory_collection", "docs_collection", "library_collection"):
        try:
            q = core.build_qdrant(core.load())
            info = q.collection_info(value)
            if info is None:
                print(f"  (collection {value!r} does not exist yet — it will be created on first use)")
            elif info.get("size") not in (None, cfg.vector_size):
                print(f"  CAUTION: {value!r} has dimension {info['size']}, "
                      f"incompatible with the model ({cfg.vector_size})")
        except Exception:
            pass


def cmd_config_detect(args, cfg):
    """Finds the model's real dimension instead of trusting the number that was typed."""
    dim = core.build_embedder(cfg).detect_dimension()
    if dim == cfg.vector_size:
        print(f"{cfg.embed_model} returns {dim} dimensions — the config is already correct")

        return
    path = core.save({"vector_size": dim})
    print(f"{cfg.embed_model} returns {dim} dimensions (the config said {cfg.vector_size})")
    print(f"  vector_size updated in {path}")
    print("  check with `qctx collections list` which collections are still compatible")


def _render_check(c: dict) -> None:
    mark = "ok  " if c["ok"] else ("warn" if c["warning"] else "FAIL")
    print(f"  [{mark:5}] {c['name']:20} {c['detail']}")
    if not c["ok"] and c["fix_hint"]:
        print(f"            -> {c['fix_hint']}")


def cmd_setup(args, cfg):
    """Guided diagnostics. It does NOT block on stdin when there is no terminal.

    That is not a detail: this command exists to be called by an agent or a script too,
    and an `input()` waiting for an answer that never comes would hang the call until the
    timeout. With no TTY, the command diagnoses, prints the exact commands that are
    missing, and exits.
    """
    rel = core.setup.diagnose(cfg)
    if args.json:
        output(rel, True)

        return

    print("diagnostics:\n")
    for c in rel["checks"]:
        _render_check(c)

    if rel["ready"]:
        print("\nready to use.")
    else:
        print(f"\n{len(rel['blockers'])} item(s) block use — the commands above fix them.")

    interactive = sys.stdin.isatty() and not args.check
    if not interactive:
        if rel["memory_suggestions"] and not cfg.memory_collection:
            print("\ncandidate collections for memory (most populated first):")
            for i, s_ in enumerate(rel["memory_suggestions"], 1):
                print(f"  {i}. {s_['collection']:34} {s_['points']:>8} points")
            print("\nchoose with: qctx config set memory-collection <name>")
        if not sys.stdin.isatty():
            print("\n(no interactive terminal — nothing was changed)")

        return

    print("\n--- configure (Enter keeps the current value) ---")
    options = [s_["collection"] for s_ in rel["memory_suggestions"]]
    for i, s_ in enumerate(options, 1):
        print(f"  {i}. {s_}")
    choice = core.setup.choose_by_index(
        options, input(f"memory collection [{cfg.memory_collection or 'none'}]: "))
    if choice:
        core.save({"memory_collection": choice})
        print(f"  memory_collection = {choice}")
    for key, current in (("docs_collection", cfg.docs_collection),
                         ("library_collection", cfg.library_collection)):
        resp = input(f"{key} [{current}]: ").strip()
        if resp:
            core.save({key: resp})
            print(f"  {key} = {resp}")
    if rel["detected_dim"] and rel["detected_dim"] != cfg.vector_size:
        core.save({"vector_size": rel["detected_dim"]})
        print(f"  vector_size = {rel['detected_dim']} (detected from the endpoint)")
    print("\nrunning the diagnostics again:\n")
    for c in core.setup.diagnose(core.load())["checks"]:
        _render_check(c)


# ---- memory ----------------------------------------------------------------

def _metadata_from_args(args) -> dict:
    meta = {}
    if getattr(args, "json_meta", None):
        meta.update(json.loads(args.json_meta))
    for field in ("type", "project", "area"):
        value = getattr(args, field, None)
        if value:
            meta[field] = value

    return meta


def cmd_memory_store(args, cfg):
    result = core.build_memory(cfg).store(args.text, _metadata_from_args(args))
    print(json.dumps(result, ensure_ascii=False) if args.json else f"stored id={result['id']}")


def cmd_memory_find(args, cfg):
    hits = core.build_memory(cfg).find(args.query, args.limit)
    if args.json:
        output(hits, True)

        return
    if not hits:
        print("no memory found")

        return
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h['score']:.3f}  {h['id']}  {json.dumps(h['metadata'], ensure_ascii=False)}")
        print(f"   {h['document'][:400]}\n")


def cmd_memory_recall(args, cfg):
    store = core.build_memory(cfg)
    policy = core.Policy(dense_floor=args.dense_floor, strict_floor=args.strict_floor,
                           min_score=args.min_score, max_results=args.limit,
                           veto=True, order_matters=False)
    hits, outcome = store.recall([args.query], policy, args.top_k)
    if args.json:
        output({"info": outcome.__dict__ | {"scored": None}, "hits": [h.__dict__ for h in hits]}, True)

        return
    if outcome.scale_converted:
        print("(logit scale detected and normalized to sigmoid)")
    if not hits:
        print(f"nothing above the cut (best dense {outcome.best_dense:.3f})")

        return
    for i, h in enumerate(hits[:args.limit], 1):
        print(f"{i}. {h.origin} {h.score:.3f} (dense {h.dense_score:.3f})  {h.id}")
        print(f"   {json.dumps(h.metadata, ensure_ascii=False)}")
        print(f"   {h.document[:600]}\n")


def cmd_memory_get(args, cfg):
    output(core.build_memory(cfg).get(args.id), True)


def cmd_memory_delete(args, cfg):
    output(core.build_memory(cfg).delete(args.id), True)


def cmd_memory_update(args, cfg):
    meta = _metadata_from_args(args) or None
    res = core.build_memory(cfg).update(args.id, args.text, meta)
    print(json.dumps(res, ensure_ascii=False) if args.json else f"{res['status']} id={res['id']}")


def cmd_memory_store_many(args, cfg):
    """A batch with ONE trip to the embeddings endpoint and all-or-nothing semantics.

    It existed in the core with no surface: without this, a checkpoint with N facts cost
    N processes and N embedding calls, and lost the atomicity the method was written to
    provide.
    """
    raw = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    items = json.loads(raw)
    if not isinstance(items, list):
        print("error: expected a JSON array of {information, metadata?}", file=sys.stderr)
        raise SystemExit(2)
    res = core.build_memory(cfg).store_many(items)
    print(json.dumps(res, ensure_ascii=False) if args.json
          else f"stored {res['count']}: {' '.join(res['ids'])}")


def cmd_memory_search_collections(args, cfg):
    """READ-ONLY search across other systems' collections."""
    res = core.search_collections(core.build_qdrant(cfg), core.build_embedder(cfg),
                                  args.query, args.collections or None,
                                  cfg.vector_size, limit=args.limit)
    if args.json:
        output(res, True)

        return
    if res["skipped"]:
        for s_ in res["skipped"]:
            print(f"  (skipped {s_['collection']}: {s_['reason']})")
    for i, h in enumerate(res["results"], 1):
        print(f"{i}. [{h['collection']}] {h['score']:.3f}  {h['id']}")
        print(f"   {(h['document'] or str(h['payload']))[:300]}\n")


def cmd_memory_list(args, cfg):
    output(core.build_memory(cfg).list_page(args.limit), True)


# ---- docs ------------------------------------------------------------------

def _report_write(res: dict, as_json: bool) -> None:
    if as_json:
        output(res, True)

        return
    label = "kept in the library" if res["scope"] == "library" else "indexed (temporary)"
    print(f"{label}: {os.path.basename(res['path'])} -> doc_id={res['doc_id']}")
    print(f"  {res['lines']} lines, {res['chars']} chars -> {res['chunks']} chunks "
          f"(mode {res['mode']}, collection {res['collection']})")
    if res["expires_at"]:
        print(f"  expires at {res['expires_at']}")
    else:
        print("  no expiry — remove it with `qctx docs drop <doc-id> --scope library`")
    print(f"  search: qctx docs search \"<question>\" --doc-id {res['doc_id']}")


def cmd_docs_index(args, cfg):
    res = core.build_docs(cfg).index_file(args.path, core.parse_ttl(args.ttl), args.doc_id)
    _report_write(res, args.json)


def cmd_docs_keep(args, cfg):
    res = core.build_docs(cfg).keep_file(args.path, args.doc_id)
    _report_write(res, args.json)


def cmd_docs_refresh(args, cfg):
    report = core.build_docs(cfg).refresh(args.scope)
    if args.json:
        output(report, True)

        return
    if not report:
        print("nothing to check")

        return
    for r in report:
        mark = {"ok": "  ", "reindexed": "->", "missing": "!!"}.get(r["action"], "  ")
        print(f"{mark} {r['action']:11} {r['doc_id']}  {r['path']}")


def cmd_docs_search(args, cfg):
    hits, outcome = core.build_docs(cfg).search(args.query, args.scope, args.doc_id, args.limit)
    if args.json:
        output({"info": outcome.__dict__ | {"scored": None}, "hits": [h.__dict__ for h in hits]}, True)

        return
    if not hits:
        print("no relevant chunk (or the index expired — see `qctx docs list`)")

        return
    if outcome.dropped_above_floor:
        # The same obligation recall.py honours: a candidate cut by the pair ceiling was
        # never judged, and one of them could have ranked first. Saying the list is not
        # exhaustive is the difference between a ranking and a verdict.
        print(f"({outcome.dropped_above_floor} candidate(s) above the dense floor went "
              f"unjudged — the pair ceiling cut them; narrow with --doc-id for a full pass)\n")
    if outcome.collapsed:
        print(f"(re-rank collapsed — best CE {outcome.best_rerank:.4f}, typical of a question "
              f"and a document in different languages; using DENSE order, which is "
              f"language-agnostic)\n")
    elif not outcome.reranked:
        print(f"(warning: the re-rank did not run — {outcome.rerank_error}; DENSE order, "
              f"not a verdict)\n")
    for i, h in enumerate(hits, 1):
        warning = f"  ⚠ {h.stale}" if h.stale else ""
        label = "library" if h.scope == "library" else "temporary"
        if h.mode == "locator":
            print(f"{i}. [{label}] {h.path}:{h.start_line}-{h.end_line}  "
                  f"({h.origin} {h.score:.3f}){warning}")
            print(f"   {' '.join(h.text.split())[:300]}…")
            print(f"   -> read lines {h.start_line}-{h.end_line} of the file for the current content")
        else:
            print(f"{i}. [{label}] {os.path.basename(h.path)}  "
                  f"({h.origin} {h.score:.3f}){warning}")
            print(f"   [SNAPSHOT from {h.indexed_at} — the source cannot be re-read by region]")
            print("   " + h.text.replace("\n", "\n   "))
        print()


def cmd_docs_list(args, cfg):
    docs = core.build_docs(cfg).list_docs(args.scope)
    if args.json:
        output(docs, True)

        return
    if not docs:
        print("nothing indexed")

        return
    import time
    print(f"{len(docs)} document(s):")
    for d in docs:
        if d["expires_at_ts"]:
            expiry = f"expires in {(d['expires_at_ts'] - time.time()) / 3600:5.1f}h"
        else:
            expiry = "permanent      "
        print(f"  [{d['scope']:7}] {d['doc_id']}  {d['chunks']:>4} chunks  {expiry}  "
              f"{d['mode']:9} {d['path']}")


def cmd_docs_drop(args, cfg):
    idx = core.build_docs(cfg)
    if args.purge_tmp:
        name = idx.drop_all_tmp()
        print(f"temporary collection {name} removed (recreated on next use); "
              f"library untouched")

        return
    if args.expired:
        idx.sweep()
        print("expired entries removed from the temporary archive")

        return
    if not args.doc_id:
        print("give a doc-id, --purge-tmp or --expired", file=sys.stderr)
        raise SystemExit(2)
    idx.drop(args.doc_id, args.scope)
    if args.json:
        output({"doc_id": args.doc_id, "scope": args.scope, "status": "removed"}, True)
    else:
        print(f"doc_id {args.doc_id} removed from {args.scope}")


# ---- parser ----------------------------------------------------------------

def _propagate_json(parser: argparse.ArgumentParser) -> None:
    """Adds `--json` to EVERY subcommand, recursively.

    People type the flag at the end (`qctx memory find x --json`) and the documentation
    promised it worked, but it existed only on the top-level parser — the natural call
    failed with "unrecognized arguments". Walking the subparsers after they are built
    solves it in one place; repeating the definition across twenty `add_parser` calls
    would be the same duplication this project spent an afternoon removing.
    """
    for action in parser._subparsers._group_actions if parser._subparsers else []:
        for sub in getattr(action, "choices", {}).values():
            if not any(o == "--json" for a in sub._actions for o in a.option_strings):
                sub.add_argument("--json", action="store_true", help="JSON output")
            _propagate_json(sub)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qctx", description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="JSON output")
    sub = ap.add_subparsers(dest="group", required=True)

    p = sub.add_parser("setup", help="guided configuration diagnostics")
    p.add_argument("--check", action="store_true",
                   help="diagnose only, never ask and never change anything")
    p.set_defaults(fn=cmd_setup)

    col = sub.add_parser("collections", help="inspect Qdrant collections")
    colsub = col.add_subparsers(dest="action", required=True)
    p = colsub.add_parser("list")
    p.add_argument("--all", action="store_true",
                   help="include collections from other systems (ws-*)")
    p.set_defaults(fn=cmd_collections)

    cfgp = sub.add_parser("config", help="view or change configuration")
    cfgsub = cfgp.add_subparsers(dest="action", required=True)
    cfgsub.add_parser("show").set_defaults(fn=cmd_config_show)
    cfgsub.add_parser("detect", help="detect the model dimension and store it").set_defaults(
        fn=cmd_config_detect)
    p = cfgsub.add_parser("set")
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(fn=cmd_config_set)

    mem = sub.add_parser("memory", help="long-term semantic memory")
    memsub = mem.add_subparsers(dest="action", required=True)

    p = memsub.add_parser("store")
    p.add_argument("text")
    p.add_argument("--type")
    p.add_argument("--project")
    p.add_argument("--area")
    p.add_argument("--json-meta", dest="json_meta")
    p.set_defaults(fn=cmd_memory_store)

    p = memsub.add_parser("find", help="dense search (cheap, no re-rank)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_memory_find)

    p = memsub.add_parser("recall", help="search with re-rank (two gates)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--dense-floor", type=float, default=0.45)
    p.add_argument("--strict-floor", type=float, default=0.58)
    p.add_argument("--min-score", type=float, default=0.10)
    p.add_argument("--top-k", type=int, default=20)
    p.set_defaults(fn=cmd_memory_recall)

    p = memsub.add_parser("get")
    p.add_argument("id")
    p.set_defaults(fn=cmd_memory_get)

    p = memsub.add_parser("delete")
    p.add_argument("id")
    p.set_defaults(fn=cmd_memory_delete)

    p = memsub.add_parser("update")
    p.add_argument("id")
    p.add_argument("--text")
    p.add_argument("--type")
    p.add_argument("--project")
    p.add_argument("--area")
    p.add_argument("--json-meta", dest="json_meta")
    p.set_defaults(fn=cmd_memory_update)

    p = memsub.add_parser("list")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_memory_list)

    p = memsub.add_parser("store-many", help="a batch of facts, all-or-nothing")
    p.add_argument("file", nargs="?", default="-",
                   help="JSON file with [{information, metadata?}], or - for stdin")
    p.set_defaults(fn=cmd_memory_store_many)

    p = memsub.add_parser("search-collections", help="read-only search in other archives")
    p.add_argument("query")
    p.add_argument("--collections", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_memory_search_collections)

    docs = sub.add_parser("docs", help="ephemeral index for long documents")
    docsub = docs.add_subparsers(dest="action", required=True)

    p = docsub.add_parser("index", help="index as TEMPORARY (with a TTL)")
    p.add_argument("path")
    p.add_argument("--ttl", default="24h", help="30m, 24h, 7d (default 24h)")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.set_defaults(fn=cmd_docs_index)

    p = docsub.add_parser("keep", help="keep in the LIBRARY, with no expiry")
    p.add_argument("path")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.set_defaults(fn=cmd_docs_keep)

    p = docsub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--scope", choices=core.docs.SCOPES, default="all")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_docs_search)

    p = docsub.add_parser("list")
    p.add_argument("--scope", choices=core.docs.SCOPES, default="all")
    p.set_defaults(fn=cmd_docs_list)

    p = docsub.add_parser("refresh", help="reindex what changed on disk")
    p.add_argument("--scope", choices=("library", "tmp"), default="library")
    p.set_defaults(fn=cmd_docs_refresh)

    p = docsub.add_parser("drop")
    p.add_argument("doc_id", nargs="?", default=None)
    p.add_argument("--scope", choices=core.docs.SCOPES, default="all")
    p.add_argument("--purge-tmp", dest="purge_tmp", action="store_true",
                   help="delete the entire temporary collection (library untouched)")
    p.add_argument("--expired", action="store_true")
    p.set_defaults(fn=cmd_docs_drop)

    _propagate_json(ap)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.fn(args, core.load())
    except core.CoreError as exc:
        # The root of the hierarchy: a new core error is caught here the day it is born.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
