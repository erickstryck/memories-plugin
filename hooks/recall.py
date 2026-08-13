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


def env(name: str, legado: str, default: str) -> str:
    return os.environ.get(name) or os.environ.get(legado) or default


def env_num(name: str, legado: str, default: str, kind=float):
    """Lê número do ambiente SEM derrubar o processo se estiver mal escrito.

    Isto é lido no carregamento do módulo, ou seja ANTES do catch-all do `main` — um
    `QCTX_RECALL_MAX_CHARS=14k` explodia antes de qualquer código nosso rodar, e o
    usuário perdia o recall recebendo um traceback em vez do aviso de
    indisponibilidade. Valor inválido cai no default e fica registrado no log.
    """
    raw = env(name, legado, default)
    try:
        return kind(raw)
    except (TypeError, ValueError):
        _pending_notes.append(f"{name}={raw!r} não é número — usando {default}")

        return kind(default)


#: Avisos coletados antes de o log existir (o log depende de STATE_DIR, que depende
#: de env). Despejados na primeira escrita de log.
_pending_notes: list[str] = []


STATE_DIR = Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))
LOG = STATE_DIR / "recall.log"
LOG_MAX_BYTES = 256 * 1024

STRICT_FLOOR = env_num("QCTX_RECALL_STRICT_FLOOR", "RECALL_MIN_SCORE", "0.58")
DENSE_FLOOR = env_num("QCTX_RECALL_DENSE_FLOOR", "RECALL_DENSE_FLOOR", "0.45")
MIN_SCORE = env_num("QCTX_RECALL_MIN_SCORE", "RECALL_RERANK_MIN_SCORE", "0.10")
MAX_MEMORIES = env_num("QCTX_RECALL_MAX_MEMORIES", "RECALL_MAX_MEMORIES", "6", int)
MAX_CHARS = env_num("QCTX_RECALL_MAX_CHARS", "RECALL_MAX_CHARS", "14000", int)
MAX_PER_MEM = env_num("QCTX_RECALL_MAX_PER_MEM", "RECALL_MAX_PER_MEM", "4500", int)
BREAKER_SECONDS = env_num("QCTX_RECALL_BREAKER", "RECALL_RERANK_BREAKER", "300")

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
        while _pending_notes:
            _write_log(f"config: {_pending_notes.pop(0)}")
        if LOG.exists() and LOG.stat().st_size > LOG_MAX_BYTES:
            LOG.write_text(LOG.read_text(errors="replace")[-LOG_MAX_BYTES // 2:])
        _write_log(msg)
    except Exception:
        pass


def _write_log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def extract_prompt(data: dict) -> str:
    """O nome do campo varia por versão de host; aceita os candidatos conhecidos."""
    for key in ("prompt", "user_prompt", "userPrompt", "message", "current_prompt", "text"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def emit(context: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }))


def unavailable_block(estagio: str, error: str) -> str:
    return (
        "[recall automático — INDISPONÍVEL neste prompt]\n"
        f"A busca na memória de longo prazo NÃO foi executada: {estagio} falhou ({error}). "
        "Isto NÃO significa que não há precedente — significa que o acervo não foi "
        "consultado. Não afirme que algo é inédito ou sem histórico apoiado neste turno. "
        "Se o assunto puder ter precedente, tente uma busca explícita; se ela também "
        "falhar, diga ao usuário que está sem memória em vez de responder como se o "
        "acervo estivesse vazio."
    )


def _degradation_note(outcome) -> str:
    """Uma linha dizendo que o julgamento foi PARCIAL, quando foi.

    Sem isto o bloco afirmava "não há precedente registrado sobre este assunto" mesmo
    quando o segundo estágio havia falhado — ou seja apresentava resultado de um
    pipeline degradado com a confiança de um pipeline completo.
    """
    parts = []
    if outcome.rerank_error:
        parts.append(f"o re-rank NÃO rodou ({outcome.rerank_error[:80]}), então a ordem é "
                      "densa e o corte estrito foi reaplicado")
    elif outcome.collapsed:
        parts.append(f"o re-rank colapsou (melhor {outcome.best_rerank:.4f}), típico de "
                      "pergunta e memória em línguas diferentes; ordem densa")
    # `dropped` só é notícia quando as vagas NÃO foram preenchidas: os descartados
    # são a cauda de menor score denso, cortada por desenho, e avisar sobre eles em
    # todo prompt é gritar lobo — o aviso perde o valor justamente quando importa.
    if outcome.dropped and len(outcome.scored) < MAX_MEMORIES:
        parts.append(f"{outcome.dropped} candidato(s) não foram julgados por teto de pares, "
                      f"e as vagas não foram preenchidas — pode haver memória relevante fora")
    if not parts:
        return ""

    return "ATENÇÃO, julgamento parcial: " + "; ".join(parts) + ".\n"


def empty_block(outcome, n_angles: int) -> str:
    if outcome.rerank_error or outcome.collapsed:
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
        f"Busca executada a partir do seu prompt ({n_angles} ângulos semânticos): nenhuma "
        f"memória acima do corte de relevância (melhor score {outcome.best_dense:.3f}).\n"
        + _degradation_note(outcome) + conclusao +
        " E considere se a resposta que você vai produzir merece ser salva no fim."
    )


def meta_line(meta: dict) -> str:
    fields = [meta.get(k) for k in ("type", "project", "connector", "area", "date")]

    return " · ".join(str(c) for c in fields if c)


def build_block(full_hits: list, pointers: list, n_angles: int, outcome) -> str:
    parts = [
        "[recall automático — memória de longo prazo]",
        f"Esta busca foi EXECUTADA pelo harness a partir do seu prompt ({n_angles} ângulos "
        "semânticos, fundidos pelo maior score). O que segue é conhecimento de sessões "
        "anteriores — leia ANTES de responder, investigar ou propor design.",
    ]
    nota = _degradation_note(outcome)
    if nota:
        parts.append(nota.rstrip())
    parts += ["", INSTRUCOES, ""]
    for i, h in enumerate(full_hits, 1):
        doc = h.document
        truncated = ""
        if len(doc) > MAX_PER_MEM:
            doc = doc[:MAX_PER_MEM]
            truncated = (f"\n[… truncado em {MAX_PER_MEM} chars — recupere o restante pelo "
                     f"id {h.id} se o assunto for central]")
        header = f"── {i}. {h.origin} {h.score:.3f}"
        meta = meta_line(h.metadata)
        if meta:
            header += f" · {meta}"
        header += f" · id {h.id}"
        parts += [header, doc + truncated, ""]

    if pointers:
        parts.append("Também relevantes, não incluídas por inteiro (já injetadas nesta "
                      "sessão, ou fora do orçamento de contexto deste turno — recupere "
                      "pelo id se precisar do texto):")
        for p in pointers:
            summary = re.sub(r"\s+", " ", p.document)[:110]
            parts.append(f"- {p.id} (score {p.score:.3f}) — {summary}…")
        parts.append("")

    return "\n".join(parts).rstrip()


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"round": 0, "seen": {}}


def prune_state(state: dict) -> int:
    """Descarta de `seen` o que já não muda decisão nenhuma.

    Uma entrada só importa enquanto `round - visto < REINJECT_AFTER`: passado isso a
    memória volta INTEIRA de qualquer forma, então guardá-la é só ocupar espaço. Sem
    poda, uma sessão longa acumula uma entrada por memória por rodada para sempre.
    """
    round_no = int(state.get("round", 0))
    seen_map = state.get("seen", {})
    velhas = [mid for mid, r in seen_map.items()
              if not isinstance(r, int) or (round_no - r) >= REINJECT_AFTER]
    for mid in velhas:
        seen_map.pop(mid, None)

    return len(velhas)


def purge_dead_sessions(days: float = 7.0) -> int:
    """Apaga estado de sessões que não são tocadas há dias.

    Cada sessão cria um arquivo e nada os removia: o diretório crescia para sempre.
    Sessão parada há uma semana não vai voltar, e se voltar o custo é começar com
    `seen` vazio — o pior efeito é uma memória reinjetada uma vez.
    """
    cutoff = time.time() - days * 86400
    deleted = 0
    try:
        for file_path in STATE_DIR.glob("recall-*.json"):
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                deleted += 1
    except Exception:
        pass

    return deleted


def save_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state))
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
        _run()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — ver docstring
        try:
            log(f"falha inesperada ({type(exc).__name__}: {exc})")
            emit(unavailable_block("o hook", type(exc).__name__))
        except Exception:
            pass  # se nem isso funcionar, silêncio é o único caminho restante


def _run() -> None:
    if os.environ.get("QCTX_RECALL_DISABLED") == "1" or os.environ.get("RECALL_DISABLED") == "1":
        return

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    prompt = extract_prompt(data)
    if not prompt:
        log(f"sem prompt no payload; chaves={sorted(data.keys())}")
        return

    reason = query.skip_reason(prompt)
    if reason:
        log(f"skip ({reason}): {prompt[:60]!r}")
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
    ocioso = breaker.is_open()
    if ocioso is not None:
        store.reranker = None
        log(f"re-rank em disjuntor: falhou há {ocioso:.0f}s — corte denso estrito")

    top_k = int(env("QCTX_RECALL_TOP_K", "RECALL_TOP_K", "20" if store.reranker else "8"))
    # Política da MEMÓRIA: o cross-encoder VETA (falso positivo polui o contexto do
    # agente) e a ordem entre os aprovados é cosmética, porque todos são injetados.
    policy = core.Policy(dense_floor=DENSE_FLOOR, strict_floor=STRICT_FLOOR,
                           min_score=MIN_SCORE, max_results=MAX_MEMORIES,
                           veto=True, order_matters=False)
    angles = query.angles(prompt)
    t0 = time.monotonic()
    try:
        hits, outcome = store.recall(angles, policy, top_k)
    except core.EmbeddingError as exc:
        log(f"embeddings falhou ({exc}) — sem recall neste prompt")
        emit(unavailable_block("embeddings", type(exc).__name__))
        return
    except core.QdrantError as exc:
        log(f"Qdrant falhou ({exc}) — sem recall neste prompt")
        emit(unavailable_block("Qdrant", type(exc).__name__))
        return
    elapsed = time.monotonic() - t0

    if outcome.rerank_error:
        breaker.arm()
        log(f"re-rank falhou ({outcome.rerank_error}) — disjuntor armado por {BREAKER_SECONDS:.0f}s")
    elif outcome.by_rerank:
        breaker.clear()

    session = "".join(c if c.isalnum() or c in "-_" else "_"
                     for c in str(data.get("session_id") or "default"))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / f"recall-{session}.json"
    state = load_state(state_path)
    state["round"] = int(state.get("round", 0)) + 1
    seen_map = state.setdefault("seen", {})
    round_no = state["round"]

    if not hits:
        prune_state(state)
        log(f"round {round_no}: 0 acima do corte (melhor {outcome.best_dense:.3f}) "
            f"em {elapsed:.1f}s | {len(angles)} ângulos | {prompt[:60]!r}")
        save_state(state_path, state)
        emit(empty_block(outcome, len(angles)))
        return

    # Orça contexto. Memória já injetada há pouco volta como ponteiro de uma linha:
    # repetir o documento inteiro a cada prompt do mesmo assunto infla o contexto
    # sem acrescentar nada, e a vaga liberada revela MAIS do acervo.
    full_hits, pointers = [], []
    budget = MAX_CHARS
    for h in hits:
        ultima = seen_map.get(h.id)
        recent = isinstance(ultima, int) and (round_no - ultima) < REINJECT_AFTER
        cost = min(len(h.document), MAX_PER_MEM)
        if recent or len(full_hits) >= MAX_MEMORIES or cost > budget:
            pointers.append(h)
            continue
        full_hits.append(h)
        seen_map[h.id] = round_no
        budget -= cost

    pruned = prune_state(state)
    save_state(state_path, state)
    if round_no % 20 == 0:
        # Varredura barata e ocasional: uma vez a cada 20 rodadas basta para o
        # diretório não crescer, e não paga `glob` em todo prompt.
        mortas = purge_dead_sessions()
        if mortas:
            log(f"limpeza: {mortas} estado(s) de sessão morta removido(s)")
    escala = " (escala convertida)" if outcome.scale_converted else ""
    log(f"round {round_no}: {len(full_hits)} injetadas + {len(pointers)} ponteiros "
        f"(de {len(hits)} relevantes / {outcome.candidates} candidatos) em {elapsed:.1f}s | "
        f"{len(angles)} ângulos | CE={outcome.by_rerank}{escala} | {prompt[:60]!r}")

    emit(build_block(full_hits, pointers, len(angles), outcome))


if __name__ == "__main__":
    main()
