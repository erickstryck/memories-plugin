#!/usr/bin/env python3
"""qctx — interface de linha de comando do núcleo.

É por aqui que um agente sem suporte a MCP (ou um host qualquer) usa a memória de
longo prazo e o índice efêmero de documentos: uma chamada de processo, JSON ou
texto na saída. Toda a lógica vive em `core/`; este arquivo só traduz argumentos.

    qctx collections list
    qctx config show | set <chave> <valor>
    qctx memory store <texto> [--type T] [--project P] [--json META]
    qctx memory find <pergunta> [--limit N]
    qctx memory recall <pergunta> [--limit N]
    qctx memory get|delete <id>
    qctx memory list [--limit N]
    qctx memory update <id> [--text T] [--json META]
    qctx docs index <caminho> [--ttl 24h]        temporário, expira
    qctx docs keep <caminho>                     biblioteca, permanente
    qctx docs search <pergunta> [--scope all|tmp|library] [--doc-id ID]
    qctx docs list [--scope ...]
    qctx docs refresh [--scope library|tmp]      reindexa o que mudou no disco
    qctx docs drop <doc-id> [--scope ...] | --purge-tmp | --expired

Três acervos, três ciclos de vida: MEMÓRIA guarda fato curado e não expira;
BIBLIOTECA guarda documento para consulta e não expira; TEMPORÁRIO guarda
documento de uma tarefa e expira. Coleções distintas, por configuração.
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


def output(obj, como_json: bool) -> None:
    if como_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# ---- collections / config --------------------------------------------------

# Prefixos de coleção gerada por outro sistema. Com 84 coleções no Qdrant, uma
# listagem crua afoga o que interessa — e escolher acervo é justamente a operação
# em que a pessoa precisa VER as opções. Escondidas por padrão, `--all` mostra.
NOISE = ("ws-",)


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
            "compativel": size == cfg.vector_size,
            "uso": ("memória" if name == cfg.memory_collection
                    else "temporário" if name == cfg.docs_collection
                    else "biblioteca" if name == cfg.library_collection else ""),
        })
    lines.sort(key=lambda l: (not l["uso"], -(l["points"] or 0)))
    if args.json:
        output({"vector_size": cfg.vector_size, "hidden": hidden,
               "collections": lines}, True)

        return
    print(f"modelo {cfg.embed_model} usa dimensão {cfg.vector_size}\n")
    print(f"{'coleção':34} {'pontos':>8} {'dim':>6}  {'':10} uso")
    for l in lines:
        mark = "ok" if l["compativel"] else "INCOMPAT."
        print(f"{l['collection']:34} {str(l['points']):>8} {str(l['dim']):>6}  {mark:10} {l['uso']}")
    if hidden:
        print(f"\n({hidden} coleção(ões) de outro sistema escondidas — `--all` mostra)")
    print("\nescolher: qctx config set memory-collection|docs-collection|"
          "library-collection <nome>")


def cmd_config_show(args, cfg):
    dados = core.redacted(cfg)
    if args.json:
        output(dados, True)

        return
    for k, v in dados.items():
        print(f"  {k:20} {v}")


def cmd_config_set(args, cfg):
    key = args.key.replace("-", "_")
    value = int(args.value) if key == "vector_size" else args.value
    path = core.save({key: value})
    print(f"{key} = {value}  (gravado em {path})")
    # Aviso, não erro: a coleção pode ser criada depois. Mas dimensão incompatível
    # é armadilha silenciosa, então vale gritar na hora da escolha.
    if key in ("memory_collection", "docs_collection", "library_collection"):
        try:
            q = core.build_qdrant(core.load())
            info = q.collection_info(value)
            if info is None:
                print(f"  (a coleção {value!r} ainda não existe — será criada no primeiro uso)")
            elif info.get("size") not in (None, cfg.vector_size):
                print(f"  ATENÇÃO: {value!r} tem dimensão {info['size']}, "
                      f"incompatível com o modelo ({cfg.vector_size})")
        except Exception:
            pass


def cmd_config_detect(args, cfg):
    """Descobre a dimensão real do modelo em vez de confiar no número digitado."""
    dim = core.build_embedder(cfg).detect_dimension()
    if dim == cfg.vector_size:
        print(f"{cfg.embed_model} devolve {dim} dimensões — config já está correta")

        return
    path = core.save({"vector_size": dim})
    print(f"{cfg.embed_model} devolve {dim} dimensões (config dizia {cfg.vector_size})")
    print(f"  vector_size atualizado em {path}")
    print("  confira com `qctx collections list` quais coleções seguem compatíveis")


def _render_check(c: dict) -> None:
    mark = "ok  " if c["ok"] else ("aviso" if c["aviso"] else "FALHA")
    print(f"  [{mark:5}] {c['nome']:20} {c['detalhe']}")
    if not c["ok"] and c["correcao"]:
        print(f"            -> {c['correcao']}")


def cmd_setup(args, cfg):
    """Diagnóstico guiado. NÃO bloqueia em stdin quando não há terminal.

    Isso não é detalhe: este comando existe para ser chamado também por um agente
    ou por um script, e um `input()` esperando resposta que nunca vem penduraria a
    chamada até o timeout. Sem TTY, o comando diagnostica, imprime os comandos
    exatos que faltam e sai.
    """
    rel = core.setup.diagnose(cfg)
    if args.json:
        output(rel, True)

        return

    print("diagnóstico:\n")
    for c in rel["checks"]:
        _render_check(c)

    if rel["pronto"]:
        print("\npronto para usar.")
    else:
        print(f"\n{len(rel['bloqueios'])} item(ns) impedem o uso — os comandos acima resolvem.")

    interactive = sys.stdin.isatty() and not args.check
    if not interactive:
        if rel["sugestoes_memoria"] and not cfg.memory_collection:
            print("\ncoleções candidatas para memória (mais povoadas primeiro):")
            for i, s_ in enumerate(rel["sugestoes_memoria"], 1):
                print(f"  {i}. {s_['collection']:34} {s_['points']:>8} pontos")
            print("\nescolha com: qctx config set memory-collection <nome>")
        if not sys.stdin.isatty():
            print("\n(sem terminal interativo — nada foi alterado)")

        return

    print("\n--- configurar (Enter mantém o valor atual) ---")
    options = [s_["collection"] for s_ in rel["sugestoes_memoria"]]
    for i, s_ in enumerate(options, 1):
        print(f"  {i}. {s_}")
    choice = core.setup.choose_by_index(
        options, input(f"coleção de memória [{cfg.memory_collection or 'nenhuma'}]: "))
    if choice:
        core.save({"memory_collection": choice})
        print(f"  memory_collection = {choice}")
    for key, current in (("docs_collection", cfg.docs_collection),
                         ("library_collection", cfg.library_collection)):
        resp = input(f"{key} [{current}]: ").strip()
        if resp:
            core.save({key: resp})
            print(f"  {key} = {resp}")
    if rel["dim_detectada"] and rel["dim_detectada"] != cfg.vector_size:
        core.save({"vector_size": rel["dim_detectada"]})
        print(f"  vector_size = {rel['dim_detectada']} (detectado do endpoint)")
    print("\nrodando o diagnóstico de novo:\n")
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
    print(json.dumps(result, ensure_ascii=False) if args.json else f"gravado id={result['id']}")


def cmd_memory_find(args, cfg):
    hits = core.build_memory(cfg).find(args.query, args.limit)
    if args.json:
        output(hits, True)

        return
    if not hits:
        print("nenhuma memória encontrada")

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
        print("(escala logit detectada e normalizada para sigmoid)")
    if not hits:
        print(f"nada acima do corte (melhor denso {outcome.best_dense:.3f})")

        return
    for i, h in enumerate(hits[:args.limit], 1):
        print(f"{i}. {h.origin} {h.score:.3f} (denso {h.dense_score:.3f})  {h.id}")
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
    """Lote com UMA ida ao endpoint de embeddings e semântica tudo-ou-nada.

    Existia no núcleo e não tinha superfície: sem isto, um checkpoint com N fatos
    custava N processos e N chamadas de embedding, e perdia a atomicidade que o
    método foi escrito para dar.
    """
    raw = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    items = json.loads(raw)
    if not isinstance(items, list):
        print("erro: esperado um array JSON de {information, metadata?}", file=sys.stderr)
        raise SystemExit(2)
    res = core.build_memory(cfg).store_many(items)
    print(json.dumps(res, ensure_ascii=False) if args.json
          else f"gravados {res['count']}: {' '.join(res['ids'])}")


def cmd_memory_search_collections(args, cfg):
    """Busca SOMENTE LEITURA em coleções de outros sistemas."""
    res = core.search_collections(core.build_qdrant(cfg), core.build_embedder(cfg),
                                  args.query, args.collections or None,
                                  cfg.vector_size, limit=args.limit)
    if args.json:
        output(res, True)

        return
    if res["skipped"]:
        for s_ in res["skipped"]:
            print(f"  (pulada {s_['collection']}: {s_['motivo']})")
    for i, h in enumerate(res["results"], 1):
        print(f"{i}. [{h['collection']}] {h['score']:.3f}  {h['id']}")
        print(f"   {(h['document'] or str(h['payload']))[:300]}\n")


def cmd_memory_list(args, cfg):
    output(core.build_memory(cfg).list_page(args.limit), True)


# ---- docs ------------------------------------------------------------------

def _report_write(res: dict, como_json: bool) -> None:
    if como_json:
        output(res, True)

        return
    label = "guardado na biblioteca" if res["scope"] == "library" else "indexado (temporário)"
    print(f"{label}: {os.path.basename(res['path'])} -> doc_id={res['doc_id']}")
    print(f"  {res['lines']} linhas, {res['chars']} chars -> {res['chunks']} trechos "
          f"(modo {res['mode']}, coleção {res['collection']})")
    if res["expires_at"]:
        print(f"  expira em {res['expires_at']}")
    else:
        print("  sem expiração — remova com `qctx docs drop <doc-id> --scope library`")
    print(f"  buscar: qctx docs search \"<pergunta>\" --doc-id {res['doc_id']}")


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
        print("nada para verificar")

        return
    for r in report:
        mark = {"ok": "  ", "reindexado": "->", "ausente": "!!"}.get(r["acao"], "  ")
        print(f"{mark} {r['acao']:11} {r['doc_id']}  {r['path']}")


def cmd_docs_search(args, cfg):
    hits, outcome = core.build_docs(cfg).search(args.query, args.scope, args.doc_id, args.limit)
    if args.json:
        output({"info": outcome.__dict__ | {"scored": None}, "hits": [h.__dict__ for h in hits]}, True)

        return
    if not hits:
        print("nenhum trecho relevante (ou o índice expirou — veja `qctx docs list`)")

        return
    if outcome.collapsed:
        print(f"(re-rank colapsou — melhor CE {outcome.best_rerank:.4f}, típico de pergunta "
              f"e documento em línguas diferentes; usando ordem DENSA, que é indiferente "
              f"à língua)\n")
    elif not outcome.reranked:
        print(f"(aviso: re-rank não rodou — {outcome.rerank_error}; ordem DENSA, "
              f"não é veredito)\n")
    for i, h in enumerate(hits, 1):
        warning = f"  ⚠ {h.stale}" if h.stale else ""
        label = "biblioteca" if h.scope == "library" else "temporário"
        if h.mode == "locator":
            print(f"{i}. [{label}] {h.path}:{h.start_line}-{h.end_line}  "
                  f"({h.origin} {h.score:.3f}){warning}")
            print(f"   {' '.join(h.text.split())[:300]}…")
            print(f"   -> ler linhas {h.start_line}-{h.end_line} do arquivo para o conteúdo atual")
        else:
            print(f"{i}. [{label}] {os.path.basename(h.path)}  "
                  f"({h.origin} {h.score:.3f}){warning}")
            print(f"   [FOTO de {h.indexed_at} — origem não relegível por região]")
            print("   " + h.text.replace("\n", "\n   "))
        print()


def cmd_docs_list(args, cfg):
    docs = core.build_docs(cfg).list_docs(args.scope)
    if args.json:
        output(docs, True)

        return
    if not docs:
        print("nada indexado")

        return
    import time
    print(f"{len(docs)} documento(s):")
    for d in docs:
        if d["expires_at_ts"]:
            expiry = f"expira em {(d['expires_at_ts'] - time.time()) / 3600:5.1f}h"
        else:
            expiry = "permanente     "
        print(f"  [{d['scope']:7}] {d['doc_id']}  {d['chunks']:>4} trechos  {expiry}  "
              f"{d['mode']:9} {d['path']}")


def cmd_docs_drop(args, cfg):
    idx = core.build_docs(cfg)
    if args.purge_tmp:
        name = idx.drop_all_tmp()
        print(f"coleção temporária {name} removida (recriada no próximo uso); "
              f"biblioteca intacta")

        return
    if args.expired:
        idx.sweep()
        print("expirados removidos do temporário")

        return
    if not args.doc_id:
        print("informe um doc-id, --purge-tmp ou --expired", file=sys.stderr)
        raise SystemExit(2)
    idx.drop(args.doc_id, args.scope)
    print(f"doc_id {args.doc_id} removido de {args.scope}")


# ---- parser ----------------------------------------------------------------

def _propagate_json(parser: argparse.ArgumentParser) -> None:
    """Acrescenta `--json` a TODO subcomando, recursivamente.

    Quem digita põe a flag no fim (`qctx memory find x --json`) e a documentação
    prometia que funcionava, mas ela existia só no parser de topo — a chamada
    natural falhava com "unrecognized arguments". Percorrer os subparsers depois de
    montados resolve num lugar só; repetir a definição em vinte `add_parser` seria a
    mesma duplicação que este projeto passou a tarde eliminando.
    """
    for action in parser._subparsers._group_actions if parser._subparsers else []:
        for sub in getattr(action, "choices", {}).values():
            if not any(o == "--json" for a in sub._actions for o in a.option_strings):
                sub.add_argument("--json", action="store_true", help="saída em JSON")
            _propagate_json(sub)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qctx", description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="saída em JSON")
    sub = ap.add_subparsers(dest="grupo", required=True)

    p = sub.add_parser("setup", help="diagnóstico guiado da configuração")
    p.add_argument("--check", action="store_true",
                   help="só diagnostica, nunca pergunta nem altera")
    p.set_defaults(fn=cmd_setup)

    col = sub.add_parser("collections", help="inspecionar coleções do Qdrant")
    colsub = col.add_subparsers(dest="acao", required=True)
    p = colsub.add_parser("list")
    p.add_argument("--all", action="store_true",
                   help="inclui coleções de outros sistemas (ws-*)")
    p.set_defaults(fn=cmd_collections)

    cfgp = sub.add_parser("config", help="ver ou alterar configuração")
    cfgsub = cfgp.add_subparsers(dest="acao", required=True)
    cfgsub.add_parser("show").set_defaults(fn=cmd_config_show)
    cfgsub.add_parser("detect", help="detecta a dimensão do modelo e grava").set_defaults(
        fn=cmd_config_detect)
    p = cfgsub.add_parser("set")
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(fn=cmd_config_set)

    mem = sub.add_parser("memory", help="memória semântica de longo prazo")
    memsub = mem.add_subparsers(dest="acao", required=True)

    p = memsub.add_parser("store")
    p.add_argument("text")
    p.add_argument("--type")
    p.add_argument("--project")
    p.add_argument("--area")
    p.add_argument("--json-meta", dest="json_meta")
    p.set_defaults(fn=cmd_memory_store)

    p = memsub.add_parser("find", help="busca densa (barata, sem re-rank)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_memory_find)

    p = memsub.add_parser("recall", help="busca com re-rank (dois portões)")
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

    p = memsub.add_parser("store-many", help="lote de fatos, tudo-ou-nada")
    p.add_argument("file", nargs="?", default="-",
                   help="arquivo JSON com [{information, metadata?}], ou - para stdin")
    p.set_defaults(fn=cmd_memory_store_many)

    p = memsub.add_parser("search-collections", help="busca read-only em outros acervos")
    p.add_argument("query")
    p.add_argument("--collections", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_memory_search_collections)

    docs = sub.add_parser("docs", help="índice efêmero de documentos longos")
    docsub = docs.add_subparsers(dest="acao", required=True)

    p = docsub.add_parser("index", help="indexa como TEMPORÁRIO (com TTL)")
    p.add_argument("path")
    p.add_argument("--ttl", default="24h", help="30m, 24h, 7d (default 24h)")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.set_defaults(fn=cmd_docs_index)

    p = docsub.add_parser("keep", help="guarda na BIBLIOTECA, sem expiração")
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

    p = docsub.add_parser("refresh", help="reindexa o que mudou no disco")
    p.add_argument("--scope", choices=("library", "tmp"), default="library")
    p.set_defaults(fn=cmd_docs_refresh)

    p = docsub.add_parser("drop")
    p.add_argument("doc_id", nargs="?", default=None)
    p.add_argument("--scope", choices=core.docs.SCOPES, default="all")
    p.add_argument("--purge-tmp", dest="purge_tmp", action="store_true",
                   help="apaga a coleção temporária inteira (biblioteca intacta)")
    p.add_argument("--expired", action="store_true")
    p.set_defaults(fn=cmd_docs_drop)

    _propagate_json(ap)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.fn(args, core.load())
    except core.CoreError as exc:
        # Raiz da hierarquia: erro novo no núcleo já nasce capturado aqui.
        print(f"erro: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
