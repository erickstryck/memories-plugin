#!/usr/bin/env python3
"""Hook de RECALL AUTOMÁTICO: busca na memória antes de o prompt chegar ao modelo.

Adaptador de host. Toda a recuperação vive em `core`; aqui ficam só as três coisas
que são do host: ler o payload do hook, decidir o que injetar dentro de um
orçamento de contexto, e formatar o bloco.

O ponto do hook é que a memória chegue SEM depender de o modelo decidir buscar.
Sem ele, a leitura fica na disciplina do agente — e a leitura é justamente a
direção que se esquece, porque nada falha visivelmente quando ela não acontece.

FALHA EM SILÊNCIO PARA O USUÁRIO, NUNCA PARA O MODELO. Se a busca não roda, o
prompt segue normalmente (o usuário não é penalizado), mas o modelo recebe um
aviso EXPLÍCITO de indisponibilidade. Sem esse aviso, ausência de resultado é
indistinguível de "não há precedente", e aí o modelo afirma que algo é inédito
quando ninguém consultou o acervo — o exato modo de falha que este hook existe
para impedir.

Configuração (todas opcionais; nomes QCTX_* canônicos, RECALL_* aceitos por
compatibilidade com a versão anterior):
    QCTX_RECALL_DISABLED       "1" desliga
    QCTX_RECALL_STRICT_FLOOR   corte denso quando sozinho        (0.58)
    QCTX_RECALL_DENSE_FLOOR    piso denso quando há cross-encoder (0.45)
    QCTX_RECALL_MIN_SCORE      corte do cross-encoder             (0.10)
    QCTX_RECALL_TOP_K          hits por ângulo                    (8 / 20 com CE)
    QCTX_RECALL_MAX_MEMORIES   memórias injetadas por vez         (6)
    QCTX_RECALL_MAX_CHARS      orçamento total                    (14000)
    QCTX_RECALL_MAX_PER_MEM    teto por memória                   (4500)
    QCTX_RECALL_BREAKER        espera do disjuntor em segundos    (300)
    QCTX_STATE_DIR             onde guardar estado e log
"""
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
from core import query  # noqa: E402
from core.breaker import Breaker  # noqa: E402


def env(nome: str, legado: str, default: str) -> str:
    return os.environ.get(nome) or os.environ.get(legado) or default


STATE_DIR = Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))
LOG = STATE_DIR / "recall.log"
LOG_MAX_BYTES = 256 * 1024

STRICT_FLOOR = float(env("QCTX_RECALL_STRICT_FLOOR", "RECALL_MIN_SCORE", "0.58"))
DENSE_FLOOR = float(env("QCTX_RECALL_DENSE_FLOOR", "RECALL_DENSE_FLOOR", "0.45"))
MIN_SCORE = float(env("QCTX_RECALL_MIN_SCORE", "RECALL_RERANK_MIN_SCORE", "0.10"))
MAX_MEMORIES = int(env("QCTX_RECALL_MAX_MEMORIES", "RECALL_MAX_MEMORIES", "6"))
MAX_CHARS = int(env("QCTX_RECALL_MAX_CHARS", "RECALL_MAX_CHARS", "14000"))
MAX_PER_MEM = int(env("QCTX_RECALL_MAX_PER_MEM", "RECALL_MAX_PER_MEM", "4500"))
BREAKER_SECONDS = float(env("QCTX_RECALL_BREAKER", "RECALL_RERANK_BREAKER", "300"))

#: Rounds antes de reinjetar uma memória por inteiro em vez do ponteiro de uma
#: linha. O contexto pode ter sido compactado nesse meio-tempo.
REINJECT_AFTER = 8

INSTRUCOES = """Como usar, sem exceção:
- Precedente ou veto do usuário PREVALECE. Não re-derive, não re-proponha o que foi \
vetado; se achar que deve mudar, diga explicitamente que é uma reversão.
- Memória que cita arquivo, linha, flag ou versão: VERIFIQUE na árvore atual antes \
de agir. Ela reflete o que era verdade quando foi escrita.
- Memória que contradiz o que você acabou de medir: a medição ganha — e então \
CORRIJA a memória, não deixe as duas conviverem.
- Faceta do assunto não coberta abaixo: faça uma busca explícita com outro ângulo."""


def log(msg: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if LOG.exists() and LOG.stat().st_size > LOG_MAX_BYTES:
            LOG.write_text(LOG.read_text(errors="replace")[-LOG_MAX_BYTES // 2:])
        with LOG.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def extrai_prompt(data: dict) -> str:
    """O nome do campo varia por versão de host; aceita os candidatos conhecidos."""
    for chave in ("prompt", "user_prompt", "userPrompt", "message", "current_prompt", "text"):
        v = data.get(chave)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def emitir(contexto: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": contexto,
        }
    }))


def bloco_indisponivel(estagio: str, erro: str) -> str:
    return (
        "[recall automático — INDISPONÍVEL neste prompt]\n"
        f"A busca na memória de longo prazo NÃO foi executada: {estagio} falhou ({erro}). "
        "Isto NÃO significa que não há precedente — significa que o acervo não foi "
        "consultado. Não afirme que algo é inédito ou sem histórico apoiado neste turno. "
        "Se o assunto puder ter precedente, tente uma busca explícita; se ela também "
        "falhar, diga ao usuário que está sem memória em vez de responder como se o "
        "acervo estivesse vazio."
    )


def _nota_degradacao(fora) -> str:
    """Uma linha dizendo que o julgamento foi PARCIAL, quando foi.

    Sem isto o bloco afirmava "não há precedente registrado sobre este assunto" mesmo
    quando o segundo estágio havia falhado — ou seja apresentava resultado de um
    pipeline degradado com a confiança de um pipeline completo.
    """
    partes = []
    if fora.rerank_error:
        partes.append(f"o re-rank NÃO rodou ({fora.rerank_error[:80]}), então a ordem é "
                      "densa e o corte estrito foi reaplicado")
    elif fora.collapsed:
        partes.append(f"o re-rank colapsou (melhor {fora.best_rerank:.4f}), típico de "
                      "pergunta e memória em línguas diferentes; ordem densa")
    # `dropped` só é notícia quando as vagas NÃO foram preenchidas: os descartados
    # são a cauda de menor score denso, cortada por desenho, e avisar sobre eles em
    # todo prompt é gritar lobo — o aviso perde o valor justamente quando importa.
    if fora.dropped and len(fora.scored) < MAX_MEMORIES:
        partes.append(f"{fora.dropped} candidato(s) não foram julgados por teto de pares, "
                      f"e as vagas não foram preenchidas — pode haver memória relevante fora")
    if not partes:
        return ""

    return "ATENÇÃO, julgamento parcial: " + "; ".join(partes) + ".\n"


def bloco_vazio(fora, n_angulos: int) -> str:
    if fora.rerank_error or fora.collapsed:
        # Com o julgamento degradado, "não há precedente" seria uma afirmação que
        # os dados não sustentam.
        conclusao = ("O acervo foi consultado mas o julgamento foi PARCIAL, então isto "
                     "não é evidência de ausência de precedente — se o assunto puder ter "
                     "histórico, faça uma busca dirigida.")
    else:
        conclusao = ("Não há precedente registrado sobre este assunto — não repita esta "
                     "busca genérica. Se o trabalho abrir um sub-assunto específico, aí "
                     "sim vale uma busca dirigida.")

    return (
        "[recall automático — memória de longo prazo]\n"
        f"Busca executada a partir do seu prompt ({n_angulos} ângulos semânticos): nenhuma "
        f"memória acima do corte de relevância (melhor score {fora.best_dense:.3f}).\n"
        + _nota_degradacao(fora) + conclusao +
        " E considere se a resposta que você vai produzir merece ser salva no fim."
    )


def linha_meta(meta: dict) -> str:
    campos = [meta.get(k) for k in ("type", "project", "connector", "area", "date")]

    return " · ".join(str(c) for c in campos if c)


def monta_bloco(cheias: list, ponteiros: list, n_angulos: int, fora) -> str:
    partes = [
        "[recall automático — memória de longo prazo]",
        f"Esta busca foi EXECUTADA pelo harness a partir do seu prompt ({n_angulos} ângulos "
        "semânticos, fundidos pelo maior score). O que segue é conhecimento de sessões "
        "anteriores — leia ANTES de responder, investigar ou propor design.",
    ]
    nota = _nota_degradacao(fora)
    if nota:
        partes.append(nota.rstrip())
    partes += ["", INSTRUCOES, ""]
    for i, h in enumerate(cheias, 1):
        doc = h.document
        corte = ""
        if len(doc) > MAX_PER_MEM:
            doc = doc[:MAX_PER_MEM]
            corte = (f"\n[… truncado em {MAX_PER_MEM} chars — recupere o restante pelo "
                     f"id {h.id} se o assunto for central]")
        cabecalho = f"── {i}. {h.origem} {h.score:.3f}"
        meta = linha_meta(h.metadata)
        if meta:
            cabecalho += f" · {meta}"
        cabecalho += f" · id {h.id}"
        partes += [cabecalho, doc + corte, ""]

    if ponteiros:
        partes.append("Também relevantes, não incluídas por inteiro (já injetadas nesta "
                      "sessão, ou fora do orçamento de contexto deste turno — recupere "
                      "pelo id se precisar do texto):")
        for p in ponteiros:
            resumo = re.sub(r"\s+", " ", p.document)[:110]
            partes.append(f"- {p.id} (score {p.score:.3f}) — {resumo}…")
        partes.append("")

    return "\n".join(partes).rstrip()


def carrega_estado(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"round": 0, "seen": {}}


def salva_estado(path: Path, estado: dict) -> None:
    try:
        path.write_text(json.dumps(estado))
    except Exception:
        pass


def main() -> None:
    """Ponto de entrada blindado.

    O aviso de indisponibilidade ao modelo é propriedade do frame MAIS EXTERNO, não
    de uma lista de tipos a capturar. Lista de tipos é frágil por construção: precisa
    ser atualizada em todo consumidor quando um erro novo aparece, e o esquecimento
    não dá erro de compilação — dá traceback para o usuário e silêncio para o modelo,
    que é o inverso exato do contrato. Isto já aconteceu: `HttpError` não estava na
    lista, e é a falha mais comum.
    """
    try:
        _executa()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — ver docstring
        try:
            log(f"falha inesperada ({type(exc).__name__}: {exc})")
            emitir(bloco_indisponivel("o hook", type(exc).__name__))
        except Exception:
            pass  # se nem isso funcionar, silêncio é o único caminho restante


def _executa() -> None:
    if os.environ.get("QCTX_RECALL_DISABLED") == "1" or os.environ.get("RECALL_DISABLED") == "1":
        return

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    prompt = extrai_prompt(data)
    if not prompt:
        log(f"sem prompt no payload; chaves={sorted(data.keys())}")
        return

    motivo = query.motivo_para_pular(prompt)
    if motivo:
        log(f"skip ({motivo}): {prompt[:60]!r}")
        return

    try:
        cfg = core.load()
        # Orçamento CABE dentro do timeout do hook (20s no hooks.json): 8+5+6 = 19s
        # no pior caso, deixando margem para emitir o aviso. Sem isto, o host matava
        # o processo antes de o aviso sair.
        store = core.build_memory(cfg, timeouts={"embed": 8.0, "qdrant": 5.0,
                                                 "rerank": 6.0})
    except core.ConfigError as exc:
        log(f"config incompleta ({exc}) — hook inerte")
        return

    # O disjuntor decide se o cross-encoder entra nesta invocação. Desligá-lo aqui,
    # em vez de dentro do núcleo, é o que faz o `recall` aplicar sozinho o corte
    # estrito: sem o segundo portão, o piso permissivo não tem quem o limpe.
    breaker = Breaker(STATE_DIR / "rerank-breaker", BREAKER_SECONDS)
    ocioso = breaker.aberto()
    if ocioso is not None:
        store.reranker = None
        log(f"re-rank em disjuntor: falhou há {ocioso:.0f}s — corte denso estrito")

    top_k = int(env("QCTX_RECALL_TOP_K", "RECALL_TOP_K", "20" if store.reranker else "8"))
    # Política da MEMÓRIA: o cross-encoder VETA (falso positivo polui o contexto do
    # agente) e a ordem entre os aprovados é cosmética, porque todos são injetados.
    politica = core.Policy(dense_floor=DENSE_FLOOR, strict_floor=STRICT_FLOOR,
                           min_score=MIN_SCORE, max_results=MAX_MEMORIES,
                           veto=True, order_matters=False)
    angulos = query.angulos(prompt)
    t0 = time.monotonic()
    try:
        hits, fora = store.recall(angulos, politica, top_k)
    except core.EmbeddingError as exc:
        log(f"embeddings falhou ({exc}) — sem recall neste prompt")
        emitir(bloco_indisponivel("embeddings", type(exc).__name__))
        return
    except core.QdrantError as exc:
        log(f"Qdrant falhou ({exc}) — sem recall neste prompt")
        emitir(bloco_indisponivel("Qdrant", type(exc).__name__))
        return
    decorrido = time.monotonic() - t0

    if fora.rerank_error:
        breaker.armar()
        log(f"re-rank falhou ({fora.rerank_error}) — disjuntor armado por {BREAKER_SECONDS:.0f}s")
    elif fora.by_rerank:
        breaker.limpar()

    sessao = "".join(c if c.isalnum() or c in "-_" else "_"
                     for c in str(data.get("session_id") or "default"))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    caminho_estado = STATE_DIR / f"recall-{sessao}.json"
    estado = carrega_estado(caminho_estado)
    estado["round"] = int(estado.get("round", 0)) + 1
    vistas = estado.setdefault("seen", {})
    rodada = estado["round"]

    if not hits:
        log(f"round {rodada}: 0 acima do corte (melhor {fora.best_dense:.3f}) "
            f"em {decorrido:.1f}s | {len(angulos)} ângulos | {prompt[:60]!r}")
        salva_estado(caminho_estado, estado)
        emitir(bloco_vazio(fora, len(angulos)))
        return

    # Orça contexto. Memória já injetada há pouco volta como ponteiro de uma linha:
    # repetir o documento inteiro a cada prompt do mesmo assunto infla o contexto
    # sem acrescentar nada, e a vaga liberada revela MAIS do acervo.
    cheias, ponteiros = [], []
    orcamento = MAX_CHARS
    for h in hits:
        ultima = vistas.get(h.id)
        recente = isinstance(ultima, int) and (rodada - ultima) < REINJECT_AFTER
        custo = min(len(h.document), MAX_PER_MEM)
        if recente or len(cheias) >= MAX_MEMORIES or custo > orcamento:
            ponteiros.append(h)
            continue
        cheias.append(h)
        vistas[h.id] = rodada
        orcamento -= custo

    salva_estado(caminho_estado, estado)
    escala = " (escala convertida)" if fora.scale_converted else ""
    log(f"round {rodada}: {len(cheias)} injetadas + {len(ponteiros)} ponteiros "
        f"(de {len(hits)} relevantes / {fora.candidates} candidatos) em {decorrido:.1f}s | "
        f"{len(angulos)} ângulos | CE={fora.by_rerank}{escala} | {prompt[:60]!r}")

    emitir(monta_bloco(cheias, ponteiros, len(angulos), fora))


if __name__ == "__main__":
    main()
