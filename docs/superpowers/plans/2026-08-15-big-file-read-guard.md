# Guarda de leitura de arquivo grande — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bloquear a leitura de um arquivo para dentro do contexto quando ela custaria caro demais em relação ao que RESTA da janela, com mensagem que diz o que fazer no lugar.

**Architecture:** Decisão pura em `core/`, protocolo em adaptador por host. `core/windows.py` resolve o tamanho da janela; `core/bigfile.py` decide; `hooks/bigfile.py` colhe o orçamento do transcript do claude-code; `hosts/hermes/bigfile.py` colhe do `state.db` do hermes. Os dois traduzem o mesmo `Verdict` para o protocolo de bloqueio do seu host.

**Tech Stack:** Python 3, stdlib apenas (sem dependência externa, por decisão de projeto — este código roda dentro de hook a cada interação).

**Spec:** `docs/superpowers/specs/2026-08-15-big-file-read-guard-design.md`

## Global Constraints

- Código, comentários, docstrings, mensagens ao usuário e commits em **inglês**. Este plano e a spec ficam em português.
- **Só stdlib.** Nada de `watchdog`, `requests` ou qualquer pacote externo.
- **Falha ABRE.** Qualquer erro na guarda libera a leitura. Nunca bloquear por falha própria.
- **Nunca escrever fora do protocolo.** No claude-code o stdout do hook é o canal JSON; texto solto ali corrompe o protocolo.
- **A guarda decide, não age.** Não indexa nada.
- Suíte atual: **565 testes, `OK (skipped=17)`** via `python3 -m unittest discover -s tests`. A contagem nunca cai.
- `claude_memory`, `memories_docs_library`, `memories_docs_tmp` são PRODUÇÃO: somente leitura em teste. Escrita só em coleção descartável que o próprio teste apaga.
- Nunca escrever em `~/.claude/`, `~/.claude.json`, `~/.hermes/`, `~/.config/memories-plugin/config.json` ou `~/.memories-plugin/state/`.
- Limpar bytecode antes de medir depois de qualquer edição: `find . -name __pycache__ -type d -prune -exec rm -rf {} +`.
- Ao provar que um teste morde, **verificar que a mutação PEGOU** (contar ocorrências antes de substituir). Este repo já embarcou seis testes que passavam pelo motivo errado.

---

### Task 1: A janela de contexto — tabela com config sobrescrevendo

> **EMENDA de 2026-08-16 — leia antes do código abaixo.** Esta task foi implementada como
> escrita e está commitada (`7cf1746`), mas a T4 provou por execução que a semântica da tabela
> estava errada: `MODEL_WINDOWS` guarda **tetos**, não janelas nominais, e `used >= window`
> significa palpite refutado (⇒ janela desconhecida ⇒ libera). O bloco de código abaixo é o
> registro histórico do que foi executado, não o alvo atual. Ver a emenda no spec e a ruling no
> ledger. A correção vive na Task 4-fix.

**Files:**
- Create: `core/windows.py`
- Modify: `core/config.py` (ENV_ALIASES, DEFAULTS, dataclass `Config`)
- Test: `tests/test_windows.py`

**Interfaces:**
- Produces: `core.windows.window_for(model: str, cfg) -> int` — devolve `0` quando desconhecida. `core.windows.MODEL_WINDOWS: dict[str, int]`.
- Consumes: `cfg.context_window` (campo novo, default `0`).

**Por que `0` e não um default plausível:** a spec exige que janela desconhecida LIBERE a leitura. Um default plausível faria a guarda bloquear com base em palpite, que é o oposto do pedido.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_windows.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import windows
from tests.test_hermes_tools import a_config


class TestWindowFor(unittest.TestCase):
    def test_a_known_model_resolves_from_the_table(self):
        self.assertGreater(windows.window_for("claude-opus-5", a_config()), 0)

    def test_an_unknown_model_is_zero_not_a_guess(self):
        """Zero means "unknown", and the caller must then ALLOW the read. A plausible
        default here would make the guard block on a guess, which is what the design
        forbids: the window is not derivable from disk on either host."""
        self.assertEqual(windows.window_for("some-model-nobody-shipped-yet", a_config()), 0)

    def test_the_config_beats_the_table(self):
        """Measured trap that motivated this: the session where this was designed ran a
        1M variant of claude-opus-5, whose bare name would otherwise map to 200k — a 5x
        silent error. The operator has to be able to state the truth."""
        cfg = a_config(context_window=1_000_000)
        self.assertEqual(windows.window_for("claude-opus-5", cfg), 1_000_000)

    def test_the_config_beats_the_table_for_an_unknown_model_too(self):
        cfg = a_config(context_window=333_000)
        self.assertEqual(windows.window_for("whatever", cfg), 333_000)

    def test_a_nonsense_config_value_falls_back_to_the_table(self):
        """The config is read tolerantly everywhere else in this plugin; a typo must not
        turn the guard into a blocker calibrated on garbage."""
        self.assertGreater(windows.window_for("claude-opus-5", a_config(context_window=-5)), 0)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_windows -v 2>&1 | tail -5`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.windows'`

- [ ] **Step 3: Acrescente o campo na config**

Em `core/config.py`, acrescente às três estruturas, mantendo a ordem alfabética do resto não é exigida — acrescente ao FIM de cada uma:

```python
# em ENV_ALIASES
    "context_window": ("QCTX_CONTEXT_WINDOW",),

# em DEFAULTS
    "context_window": 0,

# no dataclass Config, depois de vector_size
    context_window: int = 0
```

O campo tem default no dataclass porque `Config` é construído por posição em vários testes; sem o default, todos quebram.

- [ ] **Step 4: Implemente `core/windows.py`**

```python
"""How large is the context window? Neither host can tell us from disk.

Measured on 2026-08-15: the claude-code transcript records the MODEL (`claude-opus-5`)
and not the window; hermes has `sessions.model`, a `model_config` whose `max_tokens` is
the OUTPUT cap, and a `context_length` that lives only on the in-process compressor.

So the table below is a guess and the operator's config is the truth. The guess exists so
the guard works out of the box; the override exists because the guess WILL be wrong — the
session this was designed in ran a 1M variant of a model whose bare name maps to 200k
here, a 5x error that would have been silent.
"""

#: Best-effort, by bare model name. Wrong for any variant that changes the window without
#: changing the name — which is exactly why `context_window` in the config wins.
MODEL_WINDOWS = {
    "claude-opus-5": 200_000,
    "claude-sonnet-5": 200_000,
    "claude-haiku-4-5": 200_000,
}


def window_for(model: str, cfg) -> int:
    """Tokens the context window holds, or 0 when we do not know.

    0 is load-bearing: the caller must ALLOW the read when the window is unknown. Blocking
    on a guessed window is the one failure this whole guard must not produce.
    """
    declared = getattr(cfg, "context_window", 0)
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        declared = 0
    if declared > 0:
        return declared

    return int(MODEL_WINDOWS.get((model or "").strip(), 0))
```

- [ ] **Step 5: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_windows -v 2>&1 | tail -4`
Expected: `Ran 5 tests`, `OK`

- [ ] **Step 6: Rode a suíte inteira**

Run: `python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK (skipped=17)`, contagem ≥ 570

- [ ] **Step 7: Commit**

```bash
git add core/windows.py core/config.py tests/test_windows.py
git commit -F - <<'MSG'
feat: resolve the context window from a table the operator can override

Neither host exposes the window on disk — measured. The table is a guess so the guard
works unconfigured; `context_window` in the config is the truth because the guess will be
wrong. The session this was designed in ran a 1M variant of a model whose bare name maps
to 200k here: a 5x error, and silent.

Unknown resolves to 0, and 0 means the caller ALLOWS the read. Blocking on a guessed
window is the one failure this guard must never produce.
MSG
```

---

### Task 2: O núcleo da decisão

**Files:**
- Create: `core/bigfile.py`
- Test: `tests/test_bigfile.py`

**Interfaces:**
- Consumes: nada de `core/` além de stdlib. `Budget` é dado a ele; não o obtém.
- Produces:
  - `Budget(window: int, used: int, exact: bool)` — frozen dataclass
  - `Verdict(block: bool, reason: str, cost: int, free: int)` — frozen dataclass
  - `cost_of(path: str) -> int`
  - `decide(path: str, budget: Budget, *, floor_pct: float = 0.20, share_pct: float = 0.40) -> Verdict`
  - `FLOOR_PCT = 0.20`, `SHARE_PCT = 0.40`, `CHARS_PER_TOKEN = 4`

A mensagem e os casos especiais (binário, já indexado) são a Task 3; aqui `reason` carrega só o texto aritmético.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_bigfile.py
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bigfile
from core.bigfile import Budget


def a_file(size_bytes: int, suffix: str = ".txt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write("x" * size_bytes)

    return path


class TestCostOf(unittest.TestCase):
    def test_cost_is_derived_from_size_without_reading_the_file(self):
        """The whole point is not paying for the file. Reading it to measure it would
        defeat the guard on the very path it protects."""
        path = a_file(4000)
        self.assertEqual(bigfile.cost_of(path), 1000)   # 4000 / CHARS_PER_TOKEN

    def test_a_missing_file_costs_zero(self):
        """Fail open: the read will fail on its own, with a better message than ours."""
        self.assertEqual(bigfile.cost_of("/nonexistent/nope.txt"), 0)


class TestTheTwoCriteria(unittest.TestCase):
    def test_a_small_file_in_a_fresh_window_is_allowed(self):
        path = a_file(4000)                                   # 1k tokens
        v = bigfile.decide(path, Budget(window=1_000_000, used=10_000, exact=True))
        self.assertFalse(v.block)

    def test_the_final_remainder_floor_blocks(self):
        """After reading, less than 20% of the window would remain."""
        path = a_file(4000 * 100)                             # 100k tokens
        v = bigfile.decide(path, Budget(window=200_000, used=90_000, exact=True))
        self.assertTrue(v.block)          # 90k + 100k = 190k of 200k -> 5% left

    def test_the_share_of_free_blocks(self):
        """The measured case that motivated the guard: 604,023 used of 1M, a file worth
        ~171k. The floor does NOT fire (775k of 1M leaves 22%), the share does
        (171/396 = 43% > 40%)."""
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        self.assertTrue(v.block)
        self.assertIn("43%", v.reason)

    def test_neither_criterion_fires_just_below_both(self):
        path = a_file(4 * 100_000)                            # 100k tokens
        v = bigfile.decide(path, Budget(window=1_000_000, used=100_000, exact=True))
        self.assertFalse(v.block)         # 200k of 1M left 80%; 100/900 = 11%

    def test_an_unknown_window_allows(self):
        """window=0 means we could not learn it. Blocking on a guessed window is the one
        failure this guard must not produce."""
        path = a_file(4 * 900_000)
        v = bigfile.decide(path, Budget(window=0, used=0, exact=False))
        self.assertFalse(v.block)

    def test_used_above_window_does_not_divide_by_a_negative(self):
        """Defensive: the hermes estimate can drift above a mis-declared window."""
        path = a_file(4000)
        v = bigfile.decide(path, Budget(window=100, used=500, exact=False))
        self.assertTrue(v.block)
        self.assertGreaterEqual(v.free, 0)


class TestTheNumbersInTheMessage(unittest.TestCase):
    def test_an_estimated_budget_marks_the_number_as_approximate(self):
        """hermes cannot measure the context; it sums message bodies. A number that looks
        exact and is a guess is worse than an admitted guess."""
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=False))
        self.assertIn("≈", v.reason)

    def test_an_exact_budget_does_not(self):
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        self.assertNotIn("≈", v.reason)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_bigfile -v 2>&1 | tail -5`
Expected: FAIL — `No module named 'core.bigfile'`

- [ ] **Step 3: Implemente `core/bigfile.py`**

```python
"""Would reading this file cost more context than it is worth?

The criterion is RELATIVE to what remains, never the file's size alone: 171k tokens is
irrelevant in a freshly opened 1M window and fatal with 100k left. Both halves of the rule
come from the same measured incident — a 586 KB JSON read straight into context when a
search over 258 indexed chunks would have answered in ~6k tokens.

This module decides and nothing else. It does not learn the budget (the host adapters do),
it does not index (the model does, after reading the message), and it touches the network
never. That is what makes it testable with a fabricated Budget and no infrastructure.
"""
import os
from dataclasses import dataclass

#: Fraction of the window that must REMAIN after the read.
FLOOR_PCT = 0.20
#: Fraction of the free space a single file may take.
SHARE_PCT = 0.40
#: Same ratio hermes uses in `agent/context_breakdown.py::_chars_to_tokens`. Bytes are not
#: characters under UTF-8, but the error is far below the precision a threshold needs.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Budget:
    """What the host could tell us about the context right now.

    `exact` is not decoration: claude-code reports measured token usage, hermes can only
    sum message bodies. The message says which one it is, because a number that looks
    precise and is a guess is worse than a guess that admits it.
    """
    window: int
    used: int
    exact: bool


@dataclass(frozen=True)
class Verdict:
    block: bool
    reason: str
    cost: int
    free: int


def cost_of(path: str) -> int:
    """Tokens the file would cost, from its SIZE — the file is never read.

    Reading it to measure it would spend exactly what the guard exists to save.
    """
    try:
        return int(os.path.getsize(path) // CHARS_PER_TOKEN)
    except OSError:
        return 0


def decide(path: str, budget: Budget, *,
           floor_pct: float = FLOOR_PCT, share_pct: float = SHARE_PCT) -> Verdict:
    """Block when either criterion fires, whichever comes first."""
    cost = cost_of(path)
    free = max(0, budget.window - budget.used)
    after = budget.used + cost

    if budget.window <= 0:
        # We could not learn the window. Allowing is mandatory: blocking here would mean
        # blocking on a guess, and the window is not derivable from disk on either host.
        return Verdict(False, "", cost, free)

    floor_hit = after > budget.window * (1 - floor_pct)
    share_hit = free > 0 and cost > free * share_pct
    if not (floor_hit or share_hit):
        return Verdict(False, "", cost, free)

    about = "≈" if not budget.exact else ""
    pct = int(round(cost / free * 100)) if free else 100
    reason = (f"reading this file would cost {about}{cost:,} tokens, "
              f"{pct}% of the {about}{free:,} you have left")

    return Verdict(True, reason, cost, free)
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_bigfile -v 2>&1 | tail -4`
Expected: `Ran 10 tests`, `OK`

- [ ] **Step 5: Prove que os dois critérios mordem SEPARADAMENTE**

```bash
cp core/bigfile.py /tmp/bf.bak
python3 - <<'PY'
p = "core/bigfile.py"
s = open(p).read()
old = "    floor_hit = after > budget.window * (1 - floor_pct)\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "    floor_hit = False\n"))
print("mutation landed: floor criterion disabled")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_bigfile 2>&1 | tail -3
cp /tmp/bf.bak core/bigfile.py; rm /tmp/bf.bak
```
Expected: `FAILED` nomeando `test_the_final_remainder_floor_blocks`. Repita trocando `share_hit` por `False` e espere `test_the_share_of_free_blocks`.

- [ ] **Step 6: Commit**

```bash
git add core/bigfile.py tests/test_bigfile.py
git commit -F - <<'MSG'
feat: decide whether reading a file costs more context than it is worth

Two criteria, whichever fires first: less than 20% of the window would remain after the
read, or the file alone takes more than 40% of what is free. Relative to what remains and
never to size alone — 171k tokens is nothing in a fresh 1M window and fatal with 100k left.

Pure: it is handed a Budget, never learns one; never reads the file it prices; never
touches the network. That is what makes it testable without infrastructure, and what lets
a second host reuse it with a different way of measuring.

An unknown window (0) ALLOWS. Blocking on a guessed window is the one failure this guard
must not produce, because the window is not derivable from disk on either host.
MSG
```

---

### Task 3: A mensagem — binário, já indexado, e a ordem que protege o caminho comum

**Files:**
- Modify: `core/bigfile.py`
- Test: `tests/test_bigfile.py`

**Interfaces:**
- Consumes: `core.docs.doc_id_for(path) -> str` (existe, `core/docs.py:78`).
- Produces: `bigfile.is_indexable(path) -> bool`; `decide(..., indexed_ids: set[str] | None = None)`.

**A ordem é requisito, não estilo.** Saber se o arquivo já está indexado exige ida ao Qdrant (`DocIndex.list_docs`). Isso NÃO pode acontecer antes de toda leitura de arquivo. O adaptador só busca os ids indexados DEPOIS de `decide` dizer que bloquearia — que é raro. `decide` recebe `indexed_ids` já pronto e nunca faz I/O de rede.

- [ ] **Step 1: Escreva o teste que falha**

```python
# acrescente a tests/test_bigfile.py
from core import docs as core_docs


class TestSpecialCases(unittest.TestCase):
    def test_a_binary_file_is_allowed_because_indexing_it_is_not_an_option(self):
        """`docs_index` slices TEXT. Telling the model to index a binary is wrong advice,
        and blocking without an alternative is just a wall."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"\x00\x01\x02" * 400_000)
        v = bigfile.decide(path, Budget(window=200_000, used=190_000, exact=True))
        self.assertFalse(v.block)

    def test_a_text_file_is_indexable(self):
        self.assertTrue(bigfile.is_indexable(a_file(100)))

    def test_an_already_indexed_file_is_told_to_SEARCH_not_to_reindex(self):
        """Reindexing 258 chunks the archive already holds is waste, and the model would
        do it because the message told it to."""
        path = a_file(4 * 171_000)
        known = {core_docs.doc_id_for(path)}
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True),
                           indexed_ids=known)
        self.assertTrue(v.block)
        self.assertIn("already indexed", v.reason)
        self.assertIn(core_docs.doc_id_for(path), v.reason)

    def test_a_file_not_yet_indexed_is_told_to_INDEX(self):
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True),
                           indexed_ids=set())
        self.assertTrue(v.block)
        self.assertIn("docs_index", v.reason)

    def test_the_message_names_the_escape(self):
        """A block with no way out is a cage. The message has to carry its own key."""
        path = a_file(4 * 171_000)
        v = bigfile.decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        self.assertIn("--full", v.reason)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_bigfile.TestSpecialCases -v 2>&1 | tail -5`
Expected: FAIL — `module 'core.bigfile' has no attribute 'is_indexable'`

- [ ] **Step 3: Implemente**

Acrescente a `core/bigfile.py`:

```python
#: Enough bytes to catch a NUL without paying for the file.
_SNIFF_BYTES = 8192

#: The literal the user types to force the read through. A marker and not a natural phrase
#: on purpose: "read it whole" false-positives easily ("read the whole paragraph"), and
#: `--full` is only ever typed deliberately.
ESCAPE_MARKER = "--full"


def is_indexable(path: str) -> bool:
    """Whether `docs_index` could do anything with this file.

    A NUL byte in the first few KB is the classic text/binary test, and it is enough: the
    question is not "is this valid UTF-8" but "would slicing it into chunks produce
    something searchable".
    """
    try:
        with open(path, "rb") as fh:
            return b"\x00" not in fh.read(_SNIFF_BYTES)
    except OSError:
        return False
```

E substitua o bloco final de `decide` (a partir de `about = ...`) por:

```python
    if not is_indexable(path):
        # Nothing to offer instead, so blocking would only take away an option.
        return Verdict(False, "", cost, free)

    about = "≈" if not budget.exact else ""
    pct = int(round(cost / free * 100)) if free else 100
    head = (f"reading this file would cost {about}{cost:,} tokens, "
            f"{pct}% of the {about}{free:,} you have left")

    from core.docs import doc_id_for          # local: keeps the pure path import-light
    doc_id = doc_id_for(path)
    if indexed_ids and doc_id in indexed_ids:
        what = f"it is already indexed as {doc_id} — search it with docs_search"
    else:
        what = "index it with docs_index and search the parts that answer"

    reason = f"{head}. Instead, {what}. To read it anyway, put {ESCAPE_MARKER} in your request."

    return Verdict(True, reason, cost, free)
```

E acrescente o parâmetro à assinatura:

```python
def decide(path: str, budget: Budget, *, indexed_ids: set | None = None,
           floor_pct: float = FLOOR_PCT, share_pct: float = SHARE_PCT) -> Verdict:
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_bigfile 2>&1 | tail -3`
Expected: `Ran 15 tests`, `OK`

- [ ] **Step 5: Prove que a guarda de binário morde**

```bash
cp core/bigfile.py /tmp/bf.bak
python3 - <<'PY'
p = "core/bigfile.py"; s = open(p).read()
old = "    if not is_indexable(path):\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "    if False:\n"))
print("mutation landed")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_bigfile 2>&1 | tail -3
cp /tmp/bf.bak core/bigfile.py; rm /tmp/bf.bak
```
Expected: `FAILED` nomeando `test_a_binary_file_is_allowed_because_indexing_it_is_not_an_option`

- [ ] **Step 6: Commit**

```bash
git add core/bigfile.py tests/test_bigfile.py
git commit -F - <<'MSG'
feat: say what to do instead, and know when there is nothing to say

Three cases the arithmetic alone gets wrong. A binary file is ALLOWED, because docs_index
slices text and telling the model to index a binary is wrong advice — blocking without an
alternative is just a wall. An already-indexed file is told to SEARCH, not to reindex 258
chunks the archive already holds. And every block carries the escape marker, because a
block with no way out is a cage.

`indexed_ids` is passed IN rather than looked up here, and that ordering is a requirement:
knowing it costs a Qdrant round trip, and this runs before every file read. The adapter
fetches it only once `decide` has already said it would block — which is rare.
MSG
```

---

### Task 4: Adaptador claude-code

**Files:**
- Create: `hooks/bigfile.py`
- Modify: `hooks/hooks.json`
- Test: `tests/test_bigfile_claude.py`

**Interfaces:**
- Consumes: `core.bigfile.{Budget, decide}`, `core.windows.window_for`, `core.load`.
- Produces: nada para tasks seguintes além do arquivo em si.

**Primeiro passo é uma MEDIÇÃO, não código.** O payload de `PreToolUse` do claude-code precisa ser observado, não suposto: não está verificado se ele traz `transcript_path`. O adaptador usa `transcript_path` quando presente e deriva de `session_id` quando não.

- [ ] **Step 1: Meça o payload real antes de escrever o adaptador**

Registre um hook temporário que só despeja o payload, rode UM prompt, leia, e REMOVA o hook. Não adivinhe.

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".claude" / "settings.json"
print("payload keys observados vão para /tmp/pretooluse-payload.json")
print("registre manualmente um PreToolUse com:")
print('  python3 -c "import sys,json;open(\'/tmp/pretooluse-payload.json\',\'w\').write(sys.stdin.read())"')
PY
```
Anote as chaves observadas num comentário no topo de `hooks/bigfile.py`. Se `transcript_path` existir, use-o; se não, derive: `~/.claude/projects/<slug>/<session_id>.jsonl`, onde `<slug>` é o `cwd` com `/` trocado por `-`.

- [ ] **Step 2: Escreva o teste que falha**

```python
# tests/test_bigfile_claude.py
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

import bigfile as adapter


def a_transcript(lines) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")

    return path


ASSISTANT = {"message": {"role": "assistant", "model": "claude-opus-5",
                         "usage": {"input_tokens": 2,
                                   "cache_creation_input_tokens": 5_906,
                                   "cache_read_input_tokens": 598_115}}}
REAL_TURN = {"userType": "external", "entrypoint": "cli",
             "message": {"role": "user", "content": "analise o arquivo"}}
SKILL_NOISE = {"userType": "external", "entrypoint": "cli",
               "message": {"role": "user", "content": "Base directory for this skill: /x"}}
TOOL_RESULT = {"toolUseResult": {"ok": 1}, "userType": "external", "entrypoint": "cli",
               "message": {"role": "user", "content": "--full"}}


class TestReadingTheBudget(unittest.TestCase):
    def test_used_is_the_sum_of_the_three_usage_fields(self):
        """Measured on a real session: 2 + 5,906 + 598,115 = 604,023."""
        b = adapter.budget_from(a_transcript([ASSISTANT]), lambda m: 1_000_000)
        self.assertEqual(b.used, 604_023)
        self.assertTrue(b.exact, "claude-code MEASURES this; it is not an estimate")

    def test_an_unreadable_transcript_yields_an_unusable_budget(self):
        b = adapter.budget_from("/nonexistent/none.jsonl", lambda m: 1_000_000)
        self.assertEqual(b.window, 0, "window 0 makes decide() allow — fail open")


class TestTheEscapeMarker(unittest.TestCase):
    def test_it_is_found_in_a_real_user_turn(self):
        t = a_transcript([REAL_TURN, ASSISTANT,
                          dict(REAL_TURN, message={"role": "user", "content": "leia --full"})])
        self.assertTrue(adapter.escape_requested(t))

    def test_a_tool_result_is_not_a_user_turn(self):
        """Tool results carry role=user. Counting them would let any tool output that
        happens to contain the marker unlock the guard."""
        t = a_transcript([REAL_TURN, TOOL_RESULT])
        self.assertFalse(adapter.escape_requested(t))

    def test_the_marker_only_counts_in_the_LAST_turn(self):
        """The escape is scoped to one turn by construction: it evaporates on the next
        prompt, so the guard can never be left off by forgetting."""
        t = a_transcript([dict(REAL_TURN, message={"role": "user", "content": "--full"}),
                          ASSISTANT, REAL_TURN])
        self.assertFalse(adapter.escape_requested(t))


class TestFailOpen(unittest.TestCase):
    def test_a_broken_transcript_line_does_not_raise(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            fh.write("{not json\n")
        self.assertEqual(adapter.budget_from(path, lambda m: 1_000_000).window, 0)
```

- [ ] **Step 3: Rode e veja falhar**

Run: `python3 -m unittest tests.test_bigfile_claude -v 2>&1 | tail -5`
Expected: FAIL — `No module named 'bigfile'`

- [ ] **Step 4: Implemente `hooks/bigfile.py`**

```python
#!/usr/bin/env python3
"""PreToolUse guard for claude-code: refuse a file read that would cost too much context.

WHY THE TAIL AND NOT THE FILE. The transcript of the session this was written in was
15.8 MB. This hook runs before EVERY file read, so reading the whole transcript would cost
more than the guard saves. Only the last few KB are needed: the most recent `usage` block
and the most recent real user turn.

WHAT COUNTS AS A REAL USER TURN. `role=user` alone is not it — tool results and injected
skill text carry that role too. Measured on a live transcript, the discriminator is
`userType=external` AND `entrypoint=cli` AND no `toolUseResult`. Without the filter, the
"last user message" is skill boilerplate, and any tool output containing the escape marker
would unlock the guard.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bigfile, windows  # noqa: E402

#: Enough for the last usage block and the last user turn, never the whole file.
TAIL_BYTES = 256 * 1024


def _tail_objects(path: str) -> list:
    """Parsed JSON objects from the tail, oldest first. Never raises."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - TAIL_BYTES))
            raw = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for line in raw.splitlines()[1:]:       # first line may be truncated mid-object
        try:
            out.append(json.loads(line))
        except ValueError:
            continue

    return out


def budget_from(path: str, window_of) -> bigfile.Budget:
    """The context budget, measured. window=0 when anything was unavailable — fail open."""
    used, model = 0, ""
    for obj in _tail_objects(path):
        m = obj.get("message") or {}
        u = m.get("usage")
        if u:
            used = (int(u.get("input_tokens") or 0)
                    + int(u.get("cache_creation_input_tokens") or 0)
                    + int(u.get("cache_read_input_tokens") or 0))
            model = m.get("model") or model
    if not used:
        return bigfile.Budget(window=0, used=0, exact=True)

    return bigfile.Budget(window=int(window_of(model) or 0), used=used, exact=True)


def escape_requested(path: str) -> bool:
    """Did the LAST real user turn carry the marker?

    Scoped to one turn on purpose: nothing to clear, and no way to leave the guard off by
    forgetting — which is what ruled out an environment variable.
    """
    for obj in reversed(_tail_objects(path)):
        if obj.get("toolUseResult") is not None:
            continue
        if obj.get("userType") != "external" or obj.get("entrypoint") != "cli":
            continue
        m = obj.get("message") or {}
        if m.get("role") != "user":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else " ".join(
            b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else ""

        return bigfile.ESCAPE_MARKER in text

    return False
```

Acrescente `main()` que lê o payload de stdin, resolve o transcript, chama `decide`, e em bloqueio emite o JSON de bloqueio do contrato observado no Step 1 — envolvido em `try/except BaseException` que, em qualquer erro, **não emite nada e sai 0**.

- [ ] **Step 5: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_bigfile_claude 2>&1 | tail -3`
Expected: `Ran 6 tests`, `OK`

- [ ] **Step 6: Registre o hook**

Em `hooks/hooks.json`, acrescente ao lado do bloco `UserPromptSubmit`:

```json
    "PreToolUse": [
      {
        "matcher": "Read",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/bigfile.py\"",
            "shell": "bash",
            "timeout": 5
          }
        ]
      }
    ]
```

`timeout: 5` e não 20: isto roda antes de cada leitura, e um hook lento é pior que a guarda é boa.

- [ ] **Step 7: Commit**

```bash
git add hooks/bigfile.py hooks/hooks.json tests/test_bigfile_claude.py
git commit -F - <<'MSG'
feat: claude-code adapter — measure the context from the transcript tail

Reads the TAIL, never the file: the transcript of the session this was written in was
15.8 MB, and this runs before every file read.

`role=user` alone does not identify a user turn — tool results and injected skill text
carry it too. Measured on a live transcript, the discriminator is userType=external AND
entrypoint=cli AND no toolUseResult. Without it, the "last user message" is skill
boilerplate, and any tool output containing --full would unlock the guard.

Anything unavailable yields window=0, which makes decide() allow. Fail open is the rule:
a guard that breaks must get out of the way, not become a cage.
MSG
```

---

### Task 5: Adaptador hermes

**Files:**
- Create: `hosts/hermes/bigfile.py`
- Test: `tests/test_bigfile_hermes.py`

**Interfaces:**
- Consumes: `core.bigfile.{Budget, decide}`, `core.windows.window_for`.
- Produces: nada além do arquivo.

**Bootstrap obrigatório.** Este arquivo é irmão de `hosts/hermes/__init__.py`, e o loader do hermes PRE-EXECUTA cada irmão ANTES do `__init__.py`, engolindo a falha em `logger.debug`. Sem bootstrar o `sys.path` por conta própria, `import core` falha e o provedor inteiro deixa de carregar — foi o Critical do porte. Copie o padrão de `hosts/hermes/tools.py:40-44`: `realpath` e **três** `dirname`.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_bigfile_hermes.py
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hosts.hermes import bigfile as adapter


def a_state_db(messages) -> str:
    path = os.path.join(tempfile.mkdtemp(), "state.db")
    c = sqlite3.connect(path)
    c.execute("create table messages (session_id text, role text, content text, "
              "active int, timestamp real)")
    c.execute("create table sessions (session_id text, model text)")
    c.execute("insert into sessions values ('s1', 'MiniMax-M2.7')")
    for i, (role, content, active) in enumerate(messages):
        c.execute("insert into messages values ('s1', ?, ?, ?, ?)", (role, content, active, i))
    c.commit()
    c.close()

    return path


class TestReadingTheBudget(unittest.TestCase):
    def test_used_is_estimated_from_active_message_bodies(self):
        """hermes leaves messages.token_count NULL on every row — measured. The ratio is
        hermes' own `_chars_to_tokens`: (len+3)//4."""
        db = a_state_db([("user", "x" * 400, 1), ("assistant", "y" * 400, 1)])
        b = adapter.budget_from(db, "s1", lambda m: 200_000)
        self.assertEqual(b.used, 200)
        self.assertFalse(b.exact, "hermes cannot measure it; the message must say so")

    def test_compacted_messages_do_not_count(self):
        db = a_state_db([("user", "x" * 400, 1), ("assistant", "y" * 4000, 0)])
        self.assertEqual(adapter.budget_from(db, "s1", lambda m: 200_000).used, 100)

    def test_a_missing_db_fails_open(self):
        b = adapter.budget_from("/nonexistent/state.db", "s1", lambda m: 200_000)
        self.assertEqual(b.window, 0)


class TestTheEscapeMarker(unittest.TestCase):
    def test_the_marker_is_read_from_the_last_user_message(self):
        db = a_state_db([("user", "primeiro", 1), ("assistant", "r", 1),
                         ("user", "leia --full", 1)])
        self.assertTrue(adapter.escape_requested(db, "s1"))

    def test_an_older_marker_does_not_leak_forward(self):
        db = a_state_db([("user", "--full", 1), ("assistant", "r", 1), ("user", "agora nao", 1)])
        self.assertFalse(adapter.escape_requested(db, "s1"))


class TestItNeverWrites(unittest.TestCase):
    def test_the_database_is_opened_read_only(self):
        """hermes writes to state.db live. A write lock from a hook that fires before every
        file read would be a self-inflicted outage."""
        db = a_state_db([("user", "x", 1)])
        before = os.path.getmtime(db)
        adapter.budget_from(db, "s1", lambda m: 200_000)
        self.assertEqual(os.path.getmtime(db), before)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_bigfile_hermes -v 2>&1 | tail -5`
Expected: FAIL — `cannot import name 'bigfile'`

- [ ] **Step 3: Implemente `hosts/hermes/bigfile.py`**

```python
#!/usr/bin/env python3
"""pre_tool_call guard for hermes: refuse a file read that would cost too much context.

WHY THE sys.path BOOTSTRAP. hermes' loader pre-execs every sibling `*.py` in the provider
directory BEFORE `__init__.py`, and swallows the failure at logger.debug. Without finding
`core` on its own, this module raises during that pre-exec, leaves a broken shell in
sys.modules, and the whole provider then fails to load — no recall, no checkpoint, no
tools, one debug line. That is exactly how this plugin once shipped broken.

WHY AN ESTIMATE. `messages.token_count` is NULL on every row — measured — and
`session_model_usage` is cumulative across API calls, not the current context. So the used
figure is summed from message bodies with hermes' own ratio. It UNDERSTATES, because
`messages` holds no system prompt, tool definitions or skills index: the guard fires later
here than on claude-code, and the message marks the number approximate.
"""
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import bigfile  # noqa: E402

#: Short on purpose: hermes writes to this database live, and this runs before every read.
SQLITE_TIMEOUT_S = 0.5


def _rows(db_path: str, sql: str, args=()) -> list:
    """Read-only query that never raises and never waits long."""
    try:
        uri = f"file:{db_path}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)
        try:
            return list(con.execute(sql, args))
        finally:
            con.close()
    except Exception:      # noqa: BLE001 — fail open, always
        return []


def budget_from(db_path: str, session_id: str, window_of) -> bigfile.Budget:
    rows = _rows(db_path,
                 "select content from messages where session_id=? and active=1", (session_id,))
    if not rows:
        return bigfile.Budget(window=0, used=0, exact=False)
    used = sum((len(r[0] or "") + 3) // 4 for r in rows)
    model = ""
    got = _rows(db_path, "select model from sessions where session_id=? limit 1", (session_id,))
    if got:
        model = got[0][0] or ""

    return bigfile.Budget(window=int(window_of(model) or 0), used=used, exact=False)


def escape_requested(db_path: str, session_id: str) -> bool:
    rows = _rows(db_path,
                 "select content from messages where session_id=? and role='user' and "
                 "active=1 order by timestamp desc limit 1", (session_id,))

    return bool(rows) and bigfile.ESCAPE_MARKER in (rows[0][0] or "")
```

Acrescente `main()` lendo o payload JSON do stdin (`tool_name`, `args`, `session_id`), e em bloqueio imprimindo `{"decision": "block", "reason": <reason>}` e saindo com **2**. Qualquer erro: não imprime nada, sai **0**.

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_bigfile_hermes 2>&1 | tail -3`
Expected: `Ran 6 tests`, `OK`

- [ ] **Step 5: Prove que o bootstrap é necessário**

```bash
cp hosts/hermes/bigfile.py /tmp/hb.bak
python3 - <<'PY'
p = "hosts/hermes/bigfile.py"; s = open(p).read()
old = "REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))\nif REPO_ROOT not in sys.path:\n    sys.path.insert(0, REPO_ROOT)\n\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, ""))
print("mutation landed: bootstrap removed")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
cd /tmp && python3 -c "import sys; sys.path.insert(0,'$OLDPWD'); import hosts.hermes.bigfile" 2>&1 | tail -2
cd - >/dev/null; cp /tmp/hb.bak hosts/hermes/bigfile.py; rm /tmp/hb.bak
```
Expected: `ModuleNotFoundError: No module named 'core'` com o bootstrap removido

- [ ] **Step 6: Commit**

```bash
git add hosts/hermes/bigfile.py tests/test_bigfile_hermes.py
git commit -F - <<'MSG'
feat: hermes adapter — estimate the context from state.db, read-only

Estimates, and says so. messages.token_count is NULL on every row (measured) and
session_model_usage is cumulative across API calls, not current context. So the figure is
summed from message bodies with hermes' own (len+3)//4 ratio, and it UNDERSTATES: messages
holds no system prompt, tool definitions or skills index. The guard fires later here than
on claude-code, and the message marks the number approximate rather than pretending.

Opened read-only with a short timeout because hermes writes to this database live, and
this runs before every file read. A write lock here would be a self-inflicted outage.

Bootstraps sys.path itself: the loader pre-execs siblings before __init__.py and swallows
the failure at debug level, so a module that cannot find `core` on its own takes the whole
provider down silently. That is how this plugin once shipped broken.
MSG
```

---

### Task 6: Equivalência entre os hosts, e os knobs sob a varredura existente

**Files:**
- Modify: `tests/test_host_equivalence.py`
- Modify: `tests/test_hermes_provider.py` (`KNOB_SOURCES`, linha 704)
- Test: os dois acima

**Interfaces:**
- Consumes: os dois adaptadores das Tasks 4 e 5.

- [ ] **Step 1: Escreva o teste de equivalência**

```python
# acrescente a tests/test_host_equivalence.py
class TestBothHostsGuardTheSameWay(unittest.TestCase):
    """The same file and the same Budget must produce the same Verdict on both hosts.

    The hosts MEASURE differently — claude-code exactly, hermes by estimate — and that
    asymmetry is deliberate and documented. What must not differ is the DECISION taken on
    identical inputs, because that is the equivalence the plugin promises.
    """

    def test_identical_budgets_give_identical_verdicts(self):
        import tempfile
        from core.bigfile import Budget, decide

        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as fh:
            fh.write("x" * (4 * 171_000))

        for used, window in ((604_023, 1_000_000), (10_000, 1_000_000), (190_000, 200_000)):
            exact = decide(path, Budget(window=window, used=used, exact=True))
            approx = decide(path, Budget(window=window, used=used, exact=False))
            with self.subTest(used=used):
                self.assertEqual(exact.block, approx.block,
                                 "the decision must not depend on how the number was obtained")
                self.assertEqual(exact.cost, approx.cost)

    def test_only_the_wording_marks_the_estimate(self):
        import tempfile
        from core.bigfile import Budget, decide

        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as fh:
            fh.write("x" * (4 * 171_000))
        exact = decide(path, Budget(window=1_000_000, used=604_023, exact=True))
        approx = decide(path, Budget(window=1_000_000, used=604_023, exact=False))
        self.assertNotIn("≈", exact.reason)
        self.assertIn("≈", approx.reason)
```

- [ ] **Step 2: Ponha os dois adaptadores sob a varredura de knobs**

Em `tests/test_hermes_provider.py:704`, acrescente ao dict `KNOB_SOURCES`:

```python
    "hooks/bigfile.py": "import bigfile as M\n",
    "hosts/hermes/bigfile.py": "from hosts.hermes import bigfile as M\n",
```

Isto faz os testes de leitura tolerante e de paridade cobrirem os arquivos novos **sem congelar uma lista** — se alguém acrescentar um knob numérico com `int()` cru num deles, o teste que já existe pega.

- [ ] **Step 3: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK (skipped=17)`

- [ ] **Step 4: Prove que a equivalência morde**

```bash
cp core/bigfile.py /tmp/bf.bak
python3 - <<'PY'
p = "core/bigfile.py"; s = open(p).read()
old = "    share_hit = free > 0 and cost > free * share_pct\n"
assert s.count(old) == 1
new = "    share_hit = free > 0 and cost > free * (share_pct if budget.exact else 0.9)\n"
open(p, "w").write(s.replace(old, new))
print("mutation landed: hosts now decide differently")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_host_equivalence 2>&1 | tail -3
cp /tmp/bf.bak core/bigfile.py; rm /tmp/bf.bak
```
Expected: `FAILED` nomeando `test_identical_budgets_give_identical_verdicts`

- [ ] **Step 5: Commit**

```bash
git add tests/test_host_equivalence.py tests/test_hermes_provider.py
git commit -F - <<'MSG'
test: the two hosts must decide alike, and their knobs stay under the existing scan

The hosts measure differently on purpose — exact on claude-code, estimated on hermes — but
the DECISION on identical inputs must not differ. Only the wording may, and it must: the
estimate is marked approximate rather than presented as measurement.

The two adapters join KNOB_SOURCES rather than getting a frozen list of their own, so the
tolerant-read and parity tests that already exist cover them. A future knob added with a
bare int() fails a test nobody had to remember to write.
MSG
```

---

### Task 7: Instalação nos dois hosts e documentação

**Files:**
- Modify: `scripts/hermes_cutover.sh`
- Modify: `README.md`
- Test: `tests/test_hermes_cutover.py`

- [ ] **Step 1: Escreva o teste que falha**

```python
# acrescente a tests/test_hermes_cutover.py, na classe de checagens
    def test_it_reports_whether_the_big_file_guard_is_registered(self):
        """The guard is useless if hermes never calls it, and a cutover script that stays
        silent about that is the defect this script already paid for once."""
        out = self.run_script()
        self.assertLine(out, "big-file guard")
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_hermes_cutover -v 2>&1 | tail -4`
Expected: FAIL — a linha não aparece

- [ ] **Step 3: Acrescente a checagem e a escrita do bloco `hooks:`**

No `hermes_cutover.sh`, na seção de checagens, reporte se o `$HERMES_HOME/config.yaml` já tem entrada `pre_tool_call` apontando para `hosts/hermes/bigfile.py`; e no `--apply`, escreva-a, com o mesmo backup datado e a mesma releitura de verificação que o `memory.provider` já usa.

```yaml
hooks:
  - event: pre_tool_call
    matcher: read_file
    command: python3 $REPO_ROOT/hosts/hermes/bigfile.py
```

> **EMENDA de 2026-08-16, na implementação da T7 — o YAML acima está ERRADO e não registra
> nada.** Medido no hermes instalado (v0.20.1, `agent/shell_hooks.py::_parse_hooks_block`):
> a primeira linha é `if not isinstance(hooks_cfg, dict): return []`, e em seguida ele itera
> `hooks_cfg.items()` como **nome do evento -> LISTA de entradas**. Um `hooks:` escrito como
> sequência de `{event, command}` resolve para ZERO hooks e **não loga nada** nesse caminho:
> a guarda fica instalada, reportada como instalada, e nunca roda. Nem `event` é campo de
> entrada — o evento é a CHAVE. A forma que o host lê é esta, e é a que o script escreve:
>
> ```yaml
> hooks:
>   pre_tool_call:
>     - matcher: read_file
>       command: python3 "<raiz do checkout>/hosts/hermes/bigfile.py"
>       timeout: 5
> ```
>
> `timeout: 5` porque `DEFAULT_TIMEOUT_SECONDS` do host é **60**, e sessenta segundos na
> frente de cada leitura de arquivo é travamento, não guarda (é o mesmo 5 que o
> `hooks/hooks.json` dá ao irmão). `fail_closed` fica no default `false` de propósito: esta
> guarda falha ABERTA. O caminho é DERIVADO de `$ROOT`, como o alvo do symlink já era.

- [ ] **Step 4: Rode e veja passar; depois a suíte**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK (skipped=17)`

- [ ] **Step 5: Documente nos dois lados**

No `README.md`, na seção `## Hosts`, acrescente a guarda: o que ela faz, os dois critérios com os defaults, `--full`, e a frase que evita a confusão mais provável — **ela não indexa nada; ela avisa e o modelo indexa**.

Em `skills/doc-index/SKILL.md`, uma linha dizendo que a guarda existe e pode bloquear a leitura, para o modelo não interpretar o bloqueio como erro de ferramenta.

- [ ] **Step 6: Commit**

```bash
git add scripts/hermes_cutover.sh README.md skills/doc-index/SKILL.md tests/test_hermes_cutover.py
git commit -F - <<'MSG'
feat: install the guard on both hosts, and say so in the cutover report

The guard does nothing if the host never calls it, and a cutover script that stays quiet
about that is the exact defect this script already paid for: it once printed ok for a state
it had not verified.

The skill gets a line too, so a blocked read reads as a deliberate guard rather than a
broken tool — otherwise the model retries or reports a failure that never happened.
MSG
```

---

## Auto-revisão do plano

**Cobertura da spec:** os dois critérios (T2), binário e já-indexado (T3), tabela de janela com override (T1), cauda do transcript (T4), `state.db` somente-leitura (T5), `--full` no último turno real (T4, T5), falha-abre nas quatro falhas nomeadas (T1 janela desconhecida, T2 window=0, T4 transcript ilegível, T5 db ausente/travado), equivalência (T6), instalação e documentação (T7). Sem lacuna.

**Placeholders:** nenhum "TBD"/"a definir". O único passo sem código pronto é o T4 Step 1, e por desenho: o payload de `PreToolUse` **precisa ser medido**, não suposto — supor foi o que produziu o pior defeito do porte anterior.

**Consistência de valor** (o check que este projeto aprendeu a fazer depois de a spec dizer três `dirname` e o plano escrever dois): `FLOOR_PCT = 0.20` e `SHARE_PCT = 0.40` aparecem idênticos na spec e aqui; `CHARS_PER_TOKEN = 4` bate com `_chars_to_tokens` do hermes; `604.023` e `171k` são os números medidos e batem nos dois documentos; `--full` é o mesmo literal em T3, T4 e T5; três `dirname` em T5 batem com `hosts/hermes/tools.py`.
