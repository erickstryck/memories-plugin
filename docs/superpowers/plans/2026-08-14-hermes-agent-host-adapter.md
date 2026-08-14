# memories-plugin no hermes-agent — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tornar o `memories-plugin` instalável no hermes-agent com as mesmas funções e a mesma configuração que ele tem no claude-code.

**Architecture:** O que é DECISÃO move de `hooks/` (adaptador do claude-code) para `core/`; os adaptadores ficam só com protocolo de host. `hosts/hermes/` implementa o contrato `MemoryProvider` do hermes chamando o mesmo `core/`. A equivalência entre os dois hosts é provada por teste, não afirmada.

**Tech Stack:** Python 3.12, só stdlib no núcleo. `unittest`. Qdrant por HTTP. hermes-agent v0.20.0.

**Spec:** `docs/superpowers/specs/2026-08-14-hermes-agent-host-adapter-design.md`

## Global Constraints

- **Baseline medido: 258 testes** (`python3 -m unittest discover -s tests` → `OK (skipped=17)`). Toda task termina com esse número ou maior, nunca menor.
- **Zero dependência externa** no núcleo — só stdlib. Uma dependência faltando viraria perda silenciosa de função num hook que roda a cada prompt.
- **Código, comentários, identificadores e mensagens em INGLÊS.** Specs e planos em português.
- **Nunca traduzir** o conteúdo de `TRIVIAL_WORDS` e `STOPWORDS` em `core/query.py` — são dados casados contra o que o usuário digita.
- **Todo teste novo exige VERMELHO PROVADO** antes do verde: quebre a fonte, veja falhar, desfaça.
- **Limpe o bytecode entre probes:** `find . -name __pycache__ -type d -prune -exec rm -rf {} +`. A validação de `.pyc` compara `(mtime em segundos, tamanho)`, então uma edição que preserva os dois deixa bytecode velho no lugar.
- **hermes-agent v0.20.0 instalado em `~/.hermes/hermes-agent`** NÃO tem `RecallStatus`, `recall_status()` nem `unavailable_reason()`. Abstratos: `name`, `is_available`, `initialize`, `get_tool_schemas`.
- **Orçamento do `prefetch` externo: 8.0s** (`_EXTERNAL_PREFETCH_TIMEOUT_S`). O recall mede 0,5–1,7s, então bloqueante cabe.
- **`realpath`, nunca `abspath`**, para achar a raiz do repo em `hosts/hermes/` — o plugin é instalado por symlink.
- Comandos assumem `cd ~/dev/memories-plugin`. Integração exige `set -a; . ~/.bashrc; . ~/.secrets; set +a` e `QCTX_INTEGRATION=1`.

## Estrutura de arquivos

| arquivo | responsabilidade |
|---|---|
| `core/prompts.py` (criar) | os dois textos injetados: `INSTRUCTIONS`, `CHECKPOINT_PROCEDURE` |
| `core/blocks.py` (criar) | montar os 4 estados de bloco + orçamento de contexto |
| `core/session_state.py` (criar) | rodada, `seen`, poda, purga, cadência |
| `hosts/hermes/__init__.py` (criar) | o provedor: contrato do hermes → `core/` |
| `hosts/hermes/tools.py` (criar) | os 15 schemas de ferramenta + roteamento |
| `hosts/hermes/plugin.yaml` (criar) | manifesto do plugin |
| `hooks/recall.py` (modificar) | casca fina: stdin → `core.blocks` → stdout |
| `hooks/checkpoint.py` (modificar) | casca fina: cadência de `core.session_state` |
| `scripts/hermes_cutover.sh` (criar) | ensaio por padrão, `--apply` para valer |
| `tests/test_blocks.py` (criar) | os 4 estados e o orçamento, direto no núcleo |
| `tests/test_session_state.py` (criar) | estado e cadência, direto no núcleo |
| `tests/test_hermes_provider.py` (criar) | contrato, disponibilidade, prefetch, cadência |
| `tests/test_hermes_tools.py` (criar) | schemas e roteamento |
| `tests/test_host_equivalence.py` (criar) | **os dois adaptadores produzem bloco idêntico** |

---

### Task 1: `core/prompts.py` — os textos injetados

**Files:**
- Create: `core/prompts.py`
- Modify: `hooks/recall.py` (remove `INSTRUCTIONS`, importa de `core.prompts`)
- Modify: `hooks/checkpoint.py` (remove `PROCEDURE`, importa de `core.prompts`)
- Test: `tests/test_blocks.py`

**Interfaces:**
- Consumes: nada.
- Produces: `core.prompts.INSTRUCTIONS: str`, `core.prompts.CHECKPOINT_PROCEDURE: str`.
  `CHECKPOINT_PROCEDURE` tem os placeholders `{count}` e `{interval}` e chaves DOBRADAS no
  exemplo JSON (`{{"type": ...}}`), porque é consumido por `.format()`.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_blocks.py
"""Prose and block assembly live in core, so both hosts render the same text.

They used to live in hooks/, which is the claude-code adapter. A second host would have
had to copy them, and copies drift on the first fix — this repo already paid that bill
with three copies of the two-stage pipeline.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import prompts


class TestPrompts(unittest.TestCase):
    def test_instructions_carry_the_four_rules(self):
        for fragment in ("PREVAILS", "VERIFY it against the current tree",
                         "the measurement wins", "another angle"):
            self.assertIn(fragment, prompts.INSTRUCTIONS)

    def test_checkpoint_procedure_formats_without_leftover_braces(self):
        rendered = prompts.CHECKPOINT_PROCEDURE.format(count=5, interval=5)
        self.assertIn("Interaction 5 of this conversation (every 5)", rendered)
        head = rendered.split("Mandatory metadata")[0]
        self.assertNotIn("{", head, "an unfilled placeholder means the format keys drifted")

    def test_the_metadata_example_survives_formatting(self):
        rendered = prompts.CHECKPOINT_PROCEDURE.format(count=1, interval=1)
        self.assertIn('{"type": "user|feedback|project|reference"', rendered)

    def test_the_five_steps_are_all_there(self):
        for step in ("1. SWEEP", "2. DEDUPE", "3. FIX", "4. WRITE", "5. CONFIRM"):
            self.assertIn(step, prompts.CHECKPOINT_PROCEDURE)
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_blocks -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.prompts'`

- [ ] **Step 3: Crie `core/prompts.py` movendo os dois textos**

Mova o texto VERBATIM. Não reescreva, não resuma: cada frase desses blocos foi calibrada
contra um modo de falha observado, e reescrever perde o porquê.

```python
"""The text this package injects into a host's context.

It lives in the core, not in an adapter, because every host injects the SAME words. When
these lived in hooks/ — the claude-code adapter — a second host could only get them by
copying, and a copy drifts on the first fix. This repo already paid that bill: three
copies of the two-stage pipeline, with the re-rank scale normalization present in one and
missing from the other.

Neither string is decoration. INSTRUCTIONS exists because a memory delivered without its
rules of use gets applied out of context — a stale `file:line` acted on as if current, a
vetoed design re-proposed. CHECKPOINT_PROCEDURE is deliberately self-sufficient: a
one-line "save what matters" reminder produces vague, duplicated, metadata-less records,
and the cost only appears months later when a search returns three contradictory versions
of the same fact.
"""

#: Rules that travel WITH the recalled memories, every time.
INSTRUCTIONS = """How to use this, without exception:
- A precedent or a veto from the user PREVAILS. Do not re-derive, do not re-propose \
what was vetoed; if you think it should change, say explicitly that it is a reversal.
- A memory that cites a file, a line, a flag or a version: VERIFY it against the \
current tree before acting. It reflects what was true when it was written.
- A memory that contradicts what you just measured: the measurement wins — and then \
FIX the memory, do not let the two coexist.
- A facet of the subject not covered below: run an explicit search from another angle."""
```

Em seguida cole `CHECKPOINT_PROCEDURE` com o corpo EXATO de `PROCEDURE` que está hoje em
`hooks/checkpoint.py` (do `[memory checkpoint —` até `the essence of the procedure is
here."""`), mantendo `{count}`, `{interval}` e as chaves dobradas do exemplo de metadata.

- [ ] **Step 4: Aponte os dois hooks para o núcleo**

Em `hooks/recall.py`: apague o bloco `INSTRUCTIONS = """..."""` e acrescente ao grupo de
imports `from core.prompts import INSTRUCTIONS  # noqa: E402`.

Em `hooks/checkpoint.py`: apague `PROCEDURE = """..."""`, acrescente
`from core.prompts import CHECKPOINT_PROCEDURE as PROCEDURE`, e — porque esse arquivo hoje
não importa nada do pacote — insira antes do import a mesma linha de path que o `recall.py`
usa:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 5: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 262 tests` (258 + 4), `OK (skipped=17)`

- [ ] **Step 6: Prove que o texto não mudou na mudança de casa**

O risco de um "move" é reescrever sem perceber. Compare com a versão anterior:

```bash
python3 - <<'EOF'
import subprocess, re, pathlib, sys
sys.path.insert(0, ".")
from core.prompts import INSTRUCTIONS, CHECKPOINT_PROCEDURE
old_recall = subprocess.run(["git","show","HEAD:hooks/recall.py"],capture_output=True,text=True).stdout
old_ckpt   = subprocess.run(["git","show","HEAD:hooks/checkpoint.py"],capture_output=True,text=True).stdout
old_i = re.search(r'INSTRUCTIONS = """(.*?)"""', old_recall, re.S).group(1)
old_p = re.search(r'PROCEDURE = """(.*?)"""', old_ckpt, re.S).group(1)
assert INSTRUCTIONS == old_i, "INSTRUCTIONS mudou no move"
assert CHECKPOINT_PROCEDURE == old_p, "CHECKPOINT_PROCEDURE mudou no move"
print("  os dois textos são byte-identicos ao que estava em hooks/")
EOF
```
Expected: `os dois textos são byte-identicos ao que estava em hooks/`

- [ ] **Step 7: Confirme que o checkpoint ainda renderiza como processo real**

Run:
```bash
echo '{"session_id":"t1"}' | QCTX_STATE_DIR=$(mktemp -d) QCTX_CHECKPOINT_INTERVAL=1 python3 hooks/checkpoint.py | head -c 120
```
Expected: JSON com `[memory checkpoint — writing to the long-term archive]` e
`Interaction 1 of this conversation (every 1)`

- [ ] **Step 8: Commit**

```bash
git add core/prompts.py hooks/recall.py hooks/checkpoint.py tests/test_blocks.py
git commit -m "Move the injected prose into core, where both hosts can share it

The two texts lived in hooks/, which is the claude-code adapter, so a second host could
only get them by copying — and a copy drifts on the first fix. This repo already paid that
bill with three copies of the two-stage pipeline.

Verified byte-identical to the previous version rather than assumed: a move that quietly
rewrites is the failure mode, and every sentence in these blocks was calibrated against an
observed failure."
```

---

### Task 2: `core/blocks.py` — montar os 4 estados e orçar o contexto

**Files:**
- Create: `core/blocks.py`
- Modify: `hooks/recall.py` (apaga as funções de bloco e o laço de orçamento; passa a chamar o núcleo)
- Test: `tests/test_blocks.py` (acrescenta)

**Interfaces:**
- Consumes: `core.prompts.INSTRUCTIONS`; `core.retrieval.Outcome`; `core.memory.Recalled`.
- Produces:
  - `core.blocks.Budget(max_memories: int, max_chars: int, max_per_mem: int, reinject_after: int)` — dataclass frozen.
  - `core.blocks.split_by_budget(hits: list, seen: dict, round_no: int, budget: Budget) -> tuple[list, list]` — devolve `(full_hits, pointers)` e MUTA `seen` marcando o que entrou inteiro.
  - `core.blocks.degradation_note(outcome, max_memories: int) -> str`
  - `core.blocks.recall_block(full_hits, pointers, n_angles, outcome, budget) -> str`
  - `core.blocks.empty_block(outcome, n_angles) -> str`
  - `core.blocks.unavailable_block(stage: str, error: str) -> str`
  - `core.blocks.meta_line(meta: dict) -> str`

- [ ] **Step 1: Escreva os testes que falham**

Acrescente a `tests/test_blocks.py`:

```python
from dataclasses import dataclass

from core import blocks
from core.retrieval import CE, DENSE, Outcome

FLAT_CLAIM = "There is no recorded precedent"
HEDGE = "not evidence that no precedent exists"
BUDGET = blocks.Budget(max_memories=6, max_chars=14000, max_per_mem=4500, reinject_after=8)


@dataclass
class FakeHit:
    """Structural stand-in for core.memory.Recalled — blocks only reads these fields."""
    id: str = "abc123"
    document: str = "a durable fact"
    origin: str = CE
    score: float = 0.9
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"type": "reference", "date": "2026-08-14"}


class TestTheFourStates(unittest.TestCase):
    def test_populated_block_carries_the_rules_and_the_id(self):
        out = blocks.recall_block([FakeHit()], [], 2, Outcome(candidates=1, reranked=True), BUDGET)
        self.assertIn(prompts.INSTRUCTIONS, out)
        self.assertIn("abc123", out)
        self.assertIn("a durable fact", out)

    def test_a_complete_empty_search_may_claim_no_precedent(self):
        out = blocks.empty_block(Outcome(candidates=4, best_dense=0.31, reranked=True), 3)
        self.assertIn(FLAT_CLAIM, out)
        self.assertIn("0.310", out)

    def test_a_partial_judgement_withdraws_the_claim(self):
        for outcome in (
            Outcome(candidates=5, best_dense=0.5, rerank_error="timeout"),
            Outcome(candidates=5, best_dense=0.46, reranked=True, collapsed=True),
            Outcome(candidates=27, best_dense=0.54, suppressed="circuit breaker: 12s ago"),
            Outcome(candidates=26, best_dense=0.5, reranked=True, dropped=14, dropped_above_floor=3),
        ):
            out = blocks.empty_block(outcome, 2)
            self.assertNotIn(FLAT_CLAIM, out, repr(outcome))
            self.assertIn(HEDGE, out)

    def test_the_note_and_the_conclusion_cannot_disagree(self):
        cases = [
            Outcome(candidates=3, best_dense=0.2, reranked=True),
            Outcome(candidates=26, best_dense=0.5, reranked=True, dropped=14),
            Outcome(candidates=5, best_dense=0.5, rerank_error="boom"),
            Outcome(candidates=27, best_dense=0.54, suppressed="breaker"),
        ]
        for outcome in cases:
            note = blocks.degradation_note(outcome, BUDGET.max_memories)
            out = blocks.empty_block(outcome, 2)
            if note:
                self.assertNotIn(FLAT_CLAIM, out, f"note present but claim flat: {note!r}")
            else:
                self.assertIn(FLAT_CLAIM, out)

    def test_unavailable_forbids_concluding_absence(self):
        out = blocks.unavailable_block("embeddings", "HttpError")
        self.assertIn("UNAVAILABLE", out)
        self.assertIn("was not consulted", out)
        self.assertNotIn(FLAT_CLAIM, out)
        self.assertIn("embeddings", out)


class TestBudget(unittest.TestCase):
    def test_a_recently_seen_memory_becomes_a_pointer(self):
        hits = [FakeHit(id="old"), FakeHit(id="new")]
        seen = {"old": 10}
        full, pointers = blocks.split_by_budget(hits, seen, round_no=12, budget=BUDGET)
        self.assertEqual([h.id for h in full], ["new"])
        self.assertEqual([h.id for h in pointers], ["old"])

    def test_an_old_enough_memory_comes_back_in_full(self):
        seen = {"old": 1}
        full, _ = blocks.split_by_budget([FakeHit(id="old")], seen, round_no=12, budget=BUDGET)
        self.assertEqual([h.id for h in full], ["old"], "12-1 >= reinject_after")

    def test_what_goes_in_full_is_marked_seen_at_this_round(self):
        seen = {}
        blocks.split_by_budget([FakeHit(id="a")], seen, round_no=7, budget=BUDGET)
        self.assertEqual(seen, {"a": 7})

    def test_the_char_budget_stops_before_the_slot_ceiling(self):
        tight = blocks.Budget(max_memories=6, max_chars=100, max_per_mem=4500, reinject_after=8)
        hits = [FakeHit(id=str(i), document="x" * 80) for i in range(3)]
        full, pointers = blocks.split_by_budget(hits, {}, round_no=1, budget=tight)
        self.assertEqual(len(full), 1)
        self.assertEqual(len(pointers), 2)

    def test_the_slot_ceiling_stops_before_the_char_budget(self):
        two = blocks.Budget(max_memories=2, max_chars=99999, max_per_mem=4500, reinject_after=8)
        hits = [FakeHit(id=str(i)) for i in range(5)]
        full, pointers = blocks.split_by_budget(hits, {}, round_no=1, budget=two)
        self.assertEqual(len(full), 2)
        self.assertEqual(len(pointers), 3)

    def test_an_oversized_memory_is_truncated_and_names_its_id(self):
        big = FakeHit(id="big1", document="y" * 9000)
        out = blocks.recall_block([big], [], 2, Outcome(candidates=1, reranked=True), BUDGET)
        self.assertIn("truncated at", out)
        self.assertIn("big1", out)
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_blocks -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name 'blocks' from 'core'`

- [ ] **Step 3: Crie `core/blocks.py`**

Mova o corpo das funções `unavailable_block`, `_degradation_note`, `empty_block`,
`meta_line` e `build_block` de `hooks/recall.py`, e o laço de orçamento que hoje vive
dentro de `_run`. As mudanças de forma são só três: as constantes de módulo viram o
dataclass `Budget`; `_degradation_note` recebe `max_memories` como argumento em vez de ler
global; `build_block` vira `recall_block` e recebe `budget`.

```python
"""Assembling the text this package injects, and deciding how much of it fits.

In `core` and not in an adapter, because every host injects the same four states and needs
the same budget discipline. When this lived in hooks/ it was reachable only by the
claude-code adapter, and a second host could only copy it.

THE FOUR STATES, and the contract that makes them worth distinguishing:

  populated    memories were found and are included, with the rules for using them
  empty        the archive was consulted and nothing cleared the cut
  partial      it was consulted, but the judgement was incomplete — NOT evidence of absence
  unavailable  the search did not run at all

The third is the one that took a real defect to get right. The conclusion is derived from
whether a degradation note EXISTS, never from re-testing which degradation happened: the
two were separate conditions once, they drifted, and a block went out saying both "there
may be relevant memory outside" and "there is no recorded precedent". A model that reads
the second goes on to call something unprecedented when nothing was exhaustively searched,
which is the exact failure this package exists to prevent.
"""
from dataclasses import dataclass

from .prompts import INSTRUCTIONS


@dataclass(frozen=True)
class Budget:
    """How much context the injected block may spend.

    A host parameter, not a core one: what fits depends on the window the host gives us,
    so the adapter reads its own environment and hands the numbers in.
    """
    max_memories: int
    max_chars: int
    max_per_mem: int
    reinject_after: int
```

Depois as funções, com estes corpos (transcritos de `hooks/recall.py`, com as três
mudanças acima):

```python
def split_by_budget(hits, seen, round_no, budget):
    """Split hits into (full, pointers) and MARK what went in full as seen.

    A memory injected recently comes back as a one-line pointer: repeating the whole
    document on every prompt about the same subject inflates the context without adding
    anything, and the slot it frees reveals MORE of the archive.

    Mutates `seen` on purpose — the caller owns persistence, and threading a returned copy
    through every adapter would be ceremony for the same effect.
    """
    full_hits, pointers = [], []
    remaining = budget.max_chars
    for h in hits:
        last_seen = seen.get(h.id)
        recent = isinstance(last_seen, int) and (round_no - last_seen) < budget.reinject_after
        cost = min(len(h.document), budget.max_per_mem)
        if recent or len(full_hits) >= budget.max_memories or cost > remaining:
            pointers.append(h)
            continue
        full_hits.append(h)
        seen[h.id] = round_no
        remaining -= cost

    return full_hits, pointers
```

`degradation_note(outcome, max_memories)`, `empty_block(outcome, n_angles)`,
`unavailable_block(stage, error)`, `meta_line(meta)` e
`recall_block(full_hits, pointers, n_angles, outcome, budget)` recebem os corpos atuais
sem mudança de comportamento — inclusive os comentários que explicam por que cada um é
como é, que viajam junto.

- [ ] **Step 4: Faça `hooks/recall.py` consumir o núcleo**

Apague de `hooks/recall.py`: `INSTRUCTIONS` (já saiu na Task 1), `unavailable_block`,
`_degradation_note`, `empty_block`, `meta_line`, `build_block`, e o laço de orçamento
dentro de `_run`. No lugar:

```python
from core.blocks import (Budget, empty_block, recall_block,  # noqa: E402
                         unavailable_block)

BUDGET = Budget(max_memories=MAX_MEMORIES, max_chars=MAX_CHARS,
                max_per_mem=MAX_PER_MEM, reinject_after=REINJECT_AFTER)
```

e dentro de `_run`, no lugar do laço:

```python
    full_hits, pointers = split_by_budget(hits, seen_map, round_no, BUDGET)
```

- [ ] **Step 5: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 275 tests`, `OK (skipped=17)`

`tests/test_recall_block.py` continua verde porque chama pelo hook, que agora delega —
é justamente essa a prova de que o move preservou comportamento.

- [ ] **Step 6: Prove que cada função nova morde**

```bash
for probe in \
  's/if note:/if False:/' \
  's/if recent or len(full_hits) >= budget.max_memories/if recent or False/' ; do
  cp core/blocks.py /tmp/b.bak; sed -i "$probe" core/blocks.py
  find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  echo "  $probe -> $(python3 -m unittest discover -s tests 2>&1 | tail -1)"
  cp /tmp/b.bak core/blocks.py; rm /tmp/b.bak
done
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: as duas linhas mostram `FAILED`

- [ ] **Step 7: Confirme o hook real contra a infra**

Run:
```bash
set -a; . ~/.bashrc >/dev/null 2>&1; . ~/.secrets >/dev/null 2>&1; set +a
echo '{"session_id":"t2","prompt":"o que foi decidido sobre o colapso cross-lingual?"}' \
 | QCTX_STATE_DIR=$(mktemp -d) python3 hooks/recall.py \
 | python3 -c "import json,sys; c=json.load(sys.stdin)['hookSpecificOutput']['additionalContext']; print(c.splitlines()[0]); print('memórias:', c.count('── '))"
```
Expected: `[automatic recall — long-term memory]` e pelo menos uma memória

- [ ] **Step 8: Commit**

```bash
git add core/blocks.py hooks/recall.py tests/test_blocks.py
git commit -m "Move block assembly and context budgeting into core

The four injected states and the budget discipline are the same in every host, so they
belong in core rather than in the claude-code adapter. hooks/recall.py now delegates,
which is why the existing recall-block tests stay green without being touched — that is
the evidence the move preserved behaviour.

Two shape changes, both to remove reliance on module globals: the ceilings become a frozen
Budget dataclass the adapter fills from its own environment, and degradation_note takes
max_memories as an argument."
```

---

### Task 3: `core/session_state.py` — estado de sessão e cadência

**Files:**
- Create: `core/session_state.py`
- Modify: `hooks/recall.py` (apaga `load_state`/`save_state`/`prune_state`/`purge_dead_sessions`)
- Modify: `hooks/checkpoint.py` (usa a cadência do núcleo)
- Test: `tests/test_session_state.py`

**Interfaces:**
- Consumes: nada do pacote.
- Produces:
  - `core.session_state.REINJECT_AFTER: int` (= 8)
  - `core.session_state.load(path) -> dict` — nunca levanta; devolve `{"round": 0, "seen": {}}`
  - `core.session_state.save(path, state) -> None` — nunca levanta; `path=None` é no-op
  - `core.session_state.prune(state, reinject_after=REINJECT_AFTER) -> int`
  - `core.session_state.purge_dead(state_dir, days=7.0, pattern="recall-*.json") -> int`
  - `core.session_state.next_round(state) -> int` — incrementa e devolve
  - `core.session_state.due(turn: int, interval: int) -> bool` — cadência do checkpoint

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_session_state.py
"""Session state and cadence, in core because both hosts keep the same kind of state.

The claude-code hook counts turns in a file because the host does not tell it; hermes
passes turn_number to on_turn_start. Same decision, two sources — so the DECISION lives
here and each adapter supplies the number.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import session_state as st


class TestLoadAndSave(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "recall-s1.json"

    def test_a_missing_file_is_a_fresh_session_not_an_error(self):
        self.assertEqual(st.load(self.path), {"round": 0, "seen": {}})

    def test_corrupted_content_is_a_fresh_session_not_an_error(self):
        self.path.write_text("{not json")
        self.assertEqual(st.load(self.path), {"round": 0, "seen": {}})

    def test_a_round_trip_preserves_the_state(self):
        st.save(self.path, {"round": 3, "seen": {"a": 2}})
        self.assertEqual(st.load(self.path), {"round": 3, "seen": {"a": 2}})

    def test_saving_to_None_is_a_no_op_not_a_crash(self):
        st.save(None, {"round": 1, "seen": {}})

    def test_an_unwritable_path_does_not_raise(self):
        st.save(Path("/proc/impossible/state.json"), {"round": 1, "seen": {}})


class TestPrune(unittest.TestCase):
    def test_it_drops_what_can_no_longer_change_a_decision(self):
        state = {"round": 20, "seen": {"recent": 19, "at_the_edge": 12, "old": 5}}
        self.assertEqual(st.prune(state), 2)
        self.assertEqual(set(state["seen"]), {"recent"})

    def test_it_keeps_what_still_prevents_a_reinjection(self):
        state = {"round": 10, "seen": {"a": 9, "b": 4}}
        st.prune(state)
        self.assertEqual(set(state["seen"]), {"a", "b"})

    def test_a_corrupted_value_is_discarded(self):
        state = {"round": 5, "seen": {"ok": 4, "junk": "not a number", "null": None}}
        st.prune(state)
        self.assertEqual(set(state["seen"]), {"ok"})

    def test_empty_state_does_not_break(self):
        self.assertEqual(st.prune({}), 0)


class TestPurgeDead(unittest.TestCase):
    def test_it_removes_the_old_and_keeps_the_recent(self):
        d = Path(tempfile.mkdtemp())
        old, new = d / "recall-dead.json", d / "recall-alive.json"
        for f in (old, new):
            f.write_text(json.dumps({"round": 1, "seen": {}}))
        stamp = time.time() - 10 * 86400
        os.utime(old, (stamp, stamp))
        self.assertEqual(st.purge_dead(d, days=7.0), 1)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_it_does_not_touch_a_log(self):
        d = Path(tempfile.mkdtemp())
        log = d / "recall.log"
        log.write_text("a line")
        stamp = time.time() - 30 * 86400
        os.utime(log, (stamp, stamp))
        st.purge_dead(d, days=1.0)
        self.assertTrue(log.exists(), "the log is not session state")

    def test_a_missing_directory_does_not_raise(self):
        self.assertEqual(st.purge_dead(Path("/proc/impossible"), days=1.0), 0)


class TestCadence(unittest.TestCase):
    def test_it_is_due_on_the_multiple_and_only_there(self):
        self.assertEqual([t for t in range(1, 13) if st.due(t, 5)], [5, 10])

    def test_an_interval_of_zero_or_less_never_fires(self):
        for interval in (0, -1):
            self.assertEqual([t for t in range(1, 20) if st.due(t, interval)], [])

    def test_an_interval_of_one_fires_every_turn(self):
        self.assertEqual([t for t in range(1, 5) if st.due(t, 1)], [1, 2, 3, 4])


class TestNextRound(unittest.TestCase):
    def test_it_increments_and_returns(self):
        state = {"round": 4, "seen": {}}
        self.assertEqual(st.next_round(state), 5)
        self.assertEqual(state["round"], 5)

    def test_a_corrupted_round_restarts_from_one(self):
        state = {"round": "wat", "seen": {}}
        self.assertEqual(st.next_round(state), 1)
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_session_state -v 2>&1 | tail -4`
Expected: FAIL — `ImportError: cannot import name 'session_state' from 'core'`

- [ ] **Step 3: Crie `core/session_state.py`**

```python
"""Per-session state, and when the checkpoint is due.

Both hosts keep the same state and want the same cadence; only the SOURCE of the turn
number differs — the claude-code hook counts in a file because the host does not tell it,
hermes hands `turn_number` to `on_turn_start`. So the decision lives here and each adapter
supplies the number.

Nothing in this module raises. State is a convenience — it decides pointer-versus-full
reinjection and nothing more — so losing it must never cost a search that already
succeeded. An unwritable state directory once made the recall hook discard results it was
already holding and tell the model the search had not run: a safe direction with the wrong
message.
"""
import json
import time
from pathlib import Path

#: Rounds before a memory is reinjected in full instead of as a one-line pointer.
#: The context may have been compacted in between, so a pointer eventually stops being
#: enough to recover the content.
REINJECT_AFTER = 8


def load(path) -> dict:
    """Read the state, or a fresh one. Any failure is a fresh session, never an error."""
    try:
        state = json.loads(Path(path).read_text())
    except Exception:
        return {"round": 0, "seen": {}}
    if not isinstance(state, dict):
        return {"round": 0, "seen": {}}
    state.setdefault("round", 0)
    state.setdefault("seen", {})

    return state


def save(path, state: dict) -> None:
    """Persist the state. `None` means state was unavailable this round, which is not an error."""
    if path is None:
        return
    try:
        Path(path).write_text(json.dumps(state))
    except Exception:
        pass


def next_round(state: dict) -> int:
    """Advance the round counter and return it. A corrupted counter restarts at 1."""
    try:
        current = int(state.get("round", 0))
    except (TypeError, ValueError):
        current = 0
    state["round"] = current + 1

    return state["round"]


def prune(state: dict, reinject_after: int = REINJECT_AFTER) -> int:
    """Drop `seen` entries that can no longer change a decision.

    An entry only matters while `round - seen < reinject_after`: past that the memory comes
    back in full anyway, so keeping it just occupies space. Without pruning, a long session
    accumulates one entry per memory per round forever.
    """
    round_no = int(state.get("round", 0) or 0)
    seen = state.get("seen", {})
    stale = [mid for mid, r in seen.items()
             if not isinstance(r, int) or (round_no - r) >= reinject_after]
    for mid in stale:
        seen.pop(mid, None)

    return len(stale)


def purge_dead(state_dir, days: float = 7.0, pattern: str = "recall-*.json") -> int:
    """Delete state files untouched for `days`.

    Each session creates a file and nothing removed them: the directory grew forever. A
    session idle for a week is not coming back, and if it does the cost is starting with an
    empty `seen` — the worst effect is one memory reinjected once. The pattern is narrow on
    purpose: the log is not session state.
    """
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        for path in Path(state_dir).glob(pattern):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
    except Exception:
        pass

    return removed


def due(turn: int, interval: int) -> bool:
    """Whether the checkpoint is due on this turn. A non-positive interval disables it."""
    if interval <= 0:
        return False

    return turn % interval == 0
```

- [ ] **Step 4: Aponte os dois hooks para o núcleo**

Em `hooks/recall.py`: apague `load_state`, `save_state`, `prune_state`,
`purge_dead_sessions` e a constante `REINJECT_AFTER`; importe
`from core import session_state as st` e troque as chamadas por `st.load`, `st.save`,
`st.prune`, `st.purge_dead`, e o incremento de rodada por `st.next_round(state)`.

Em `hooks/checkpoint.py`: troque `if INTERVAL <= 0 or n % INTERVAL != 0: return` por
`if not st.due(n, INTERVAL): return`.

`tests/test_hygiene_fixes.py` chama `recall.prune_state` e `recall.purge_dead_sessions`.
Reaponte esses testes para `core.session_state`, mantendo os nomes de teste e a prosa que
explica o porquê — o defeito que eles pinam não mudou de lugar.

- [ ] **Step 5: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 296 tests`, `OK (skipped=17)`

- [ ] **Step 6: Prove que a cadência morde**

```bash
cp core/session_state.py /tmp/s.bak
sed -i 's/return turn % interval == 0/return True/' core/session_state.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  cadência sempre verdadeira -> $(python3 -m unittest discover -s tests 2>&1 | tail -1)"
cp /tmp/s.bak core/session_state.py; rm /tmp/s.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: `FAILED`

- [ ] **Step 7: Confirme os dois hooks como processos reais**

Run:
```bash
D=$(mktemp -d)
for i in 1 2 3 4 5; do echo "{\"session_id\":\"c1\"}" | QCTX_STATE_DIR=$D QCTX_CHECKPOINT_INTERVAL=5 python3 hooks/checkpoint.py; done | grep -c "memory checkpoint"
```
Expected: `1` — dispara no turno 5 e só nele

- [ ] **Step 8: Commit**

```bash
git add core/session_state.py hooks/recall.py hooks/checkpoint.py tests/test_session_state.py tests/test_hygiene_fixes.py
git commit -m "Move session state and checkpoint cadence into core

Both hosts keep the same state and want the same cadence; only the source of the turn
number differs, so the decision belongs in core and each adapter supplies the number.

Nothing in the module raises: state decides pointer-versus-full reinjection and nothing
more, so losing it must never cost a search that already succeeded."
```

---

### Task 4: `hosts/hermes/` — o provedor carrega e sabe se pode funcionar

**Files:**
- Create: `hosts/hermes/__init__.py`
- Create: `hosts/hermes/plugin.yaml`
- Test: `tests/test_hermes_provider.py`

**Interfaces:**
- Consumes: `core.load`, `core.ConfigError`, `core.build_memory`.
- Produces:
  - `hosts.hermes.MemoriesProvider` — classe com `name`, `is_available()`,
    `unavailable_reason()`, `initialize(session_id, **kwargs)`, `get_tool_schemas()`,
    `system_prompt_block()`, `shutdown()`.
  - `hosts.hermes.register(ctx)` — chama `ctx.register_memory_provider(MemoriesProvider())`.
  - `hosts.hermes.REPO_ROOT: str` — a raiz do repo, resolvida por `realpath`.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_hermes_provider.py
"""The hermes-agent host adapter.

It is verified WITHOUT hermes importable, because this repo's suite has to run offline and
hermes lives in its own venv. What the adapter promises is therefore checked structurally:
the method set hermes v0.20.0 actually calls, measured from the install rather than assumed
from the published source — the two differ, and the installed one is what runs.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hosts.hermes import MemoriesProvider, register

#: Where hermes is installed on this machine. The ABC IS importable from there (measured);
#: only `RecallStatus` is absent in v0.20.0. Used by the test below, skipped elsewhere so
#: the suite stays portable.
HERMES_INSTALL = Path.home() / ".hermes" / "hermes-agent"

#: What the adapter implements EXPLICITLY — no reliance on the ABC's defaults, because the
#: suite runs with `_Base = object` and because a default silently absorbs a method hermes
#: renames. Everything here must exist whether or not hermes is importable.
IMPLEMENTED = {
    "name", "is_available", "unavailable_reason", "initialize", "shutdown",
    "prefetch", "queue_prefetch", "recall_status", "system_prompt_block",
    "on_turn_start", "on_session_end", "sync_turn",
    "get_tool_schemas", "handle_tool_call", "get_config_schema", "save_config",
}
HERMES_ABSTRACTS = {"get_tool_schemas", "initialize", "is_available", "name"}


class TestTheContract(unittest.TestCase):
    def test_it_implements_every_abstract(self):
        p = MemoriesProvider()
        for method in HERMES_ABSTRACTS:
            self.assertTrue(hasattr(p, method), f"missing abstract {method}")

    def test_it_defines_its_surface_without_leaning_on_inherited_defaults(self):
        """Explicit, not inherited. Two reasons: this suite runs without hermes, so the
        defaults are not there to lean on; and a default silently absorbs a method a hermes
        upgrade renames, turning a rename into a feature that quietly stops working."""
        p = MemoriesProvider()
        missing = {m for m in IMPLEMENTED if m not in vars(type(p))
                   and not any(m in vars(k) for k in type(p).__mro__[:-1])}
        self.assertEqual(missing, set(), f"declared as implemented but absent: {missing}")

    @unittest.skipUnless(HERMES_INSTALL.exists(), "hermes-agent not installed here")
    def test_it_answers_every_method_the_REAL_installed_abc_declares(self):
        """The honest version of the contract check: read the surface off the install
        instead of trusting a list I typed. Skipped where hermes is absent, so a machine
        without it still runs the rest."""
        import subprocess
        script = (
            "import sys, json\n"
            "sys.path.insert(0, %r)\n"
            "from agent.memory_provider import MemoryProvider as M\n"
            "print(json.dumps([m for m in dir(M) if not m.startswith('_')]))\n"
        ) % str(HERMES_INSTALL)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        surface = set(json.loads(out.stdout))
        p = MemoriesProvider()
        unanswered = {m for m in surface if not hasattr(p, m)}
        self.assertEqual(unanswered, set(),
                         f"hermes calls these and the provider has no answer: {unanswered}")

    def test_the_name_is_the_install_directory_name(self):
        self.assertEqual(MemoriesProvider().name, "memories")

    def test_register_hands_the_provider_to_the_collector(self):
        class Collector:
            provider = None

            def register_memory_provider(self, provider):
                self.provider = provider

        c = Collector()
        register(c)
        self.assertIsInstance(c.provider, MemoriesProvider)


class TestAvailability(unittest.TestCase):
    """is_available gates initialization, so a provider that reports unavailable is never
    initialized — any diagnostic it would log from initialize() is unreachable. The reason
    has to be actionable and it has to come from here."""

    def _clean_env(self):
        keep = {"PATH", "HOME", "LANG", "PYTHONPATH"}

        return {k: v for k, v in os.environ.items() if k in keep}

    def test_unconfigured_is_unavailable_with_an_actionable_reason(self):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "p = MemoriesProvider()\n"
            "print(int(p.is_available())); print(p.unavailable_reason())\n"
        ) % str(REPO)
        env = self._clean_env()
        env["QCTX_CONFIG"] = "/nonexistent/config.json"
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        available, reason = out.stdout.strip().splitlines()[:2]
        self.assertEqual(available, "0")
        self.assertIn("QCTX_", reason, "the reason must say WHICH variable to set")

    def test_is_available_makes_no_network_call(self):
        """Contract: 'Should not make network calls — just check config and installed deps.'
        Pointed at an unroutable address it must still answer fast."""
        import time
        env = dict(os.environ, QCTX_QDRANT_URL="http://10.255.255.1:9/qdrant")
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from hosts.hermes import MemoriesProvider\n"
            "MemoriesProvider().is_available()\n"
        ) % str(REPO)
        t0 = time.monotonic()
        subprocess.run([sys.executable, "-c", script], capture_output=True, env=env)
        self.assertLess(time.monotonic() - t0, 3.0, "is_available reached the network")


class TestSymlinkInstall(unittest.TestCase):
    def test_it_finds_the_repo_when_installed_through_a_symlink(self):
        """Measured trap: through a symlink, abspath(__file__) resolves to the SYMLINK's
        directory ($HERMES_HOME/plugins), not the repo. The path pattern hooks/recall.py
        uses would fail here; only realpath finds core/."""
        home = Path(tempfile.mkdtemp()) / "plugins"
        home.mkdir(parents=True)
        link = home / "memories"
        link.symlink_to(REPO / "hosts" / "hermes")

        script = (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('u.memories', %r,\n"
            "    submodule_search_locations=[%r])\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(m.REPO_ROOT)\n"
            "print(m.MemoriesProvider().name)\n"
        ) % (str(link / "__init__.py"), str(link))
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        root, name = out.stdout.strip().splitlines()[:2]
        self.assertEqual(Path(root).resolve(), REPO.resolve())
        self.assertEqual(name, "memories")


class TestManifest(unittest.TestCase):
    def test_the_yaml_declares_what_the_loader_reads(self):
        text = (REPO / "hosts" / "hermes" / "plugin.yaml").read_text()
        for key in ("name: memories", "category: memory", "kind: exclusive"):
            self.assertIn(key, text)

    def test_the_init_contains_the_string_discovery_greps_for(self):
        """_is_memory_provider_dir is a cheap text scan of the first 8192 bytes for
        'register_memory_provider' or 'MemoryProvider'. Without one of those the directory
        is not even considered a provider."""
        head = (REPO / "hosts" / "hermes" / "__init__.py").read_text()[:8192]
        self.assertTrue("register_memory_provider" in head or "MemoryProvider" in head)
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_hermes_provider -v 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'hosts'`

- [ ] **Step 3: Crie `hosts/hermes/plugin.yaml`**

```yaml
name: memories
version: 0.3.0
description: "Long-term semantic memory with automatic recall, plus an index for long documents (permanent library and a temporary archive with TTL), on top of Qdrant. Shares its archives with the claude-code plugin."
category: memory
kind: exclusive
license: MIT
homepage: https://github.com/erickstryck/memories-plugin
```

Sem `dependencies:` — o núcleo é stdlib puro, e é isso que faz o plugin não poder falhar
por dependência faltando num caminho disparado a cada prompt.

- [ ] **Step 4: Crie `hosts/hermes/__init__.py`**

```python
"""hermes-agent host adapter.

A thin shell over `core`, like `hooks/` is for claude-code. Everything that decides
anything — which queries to build, the two-stage retrieval, the four block states, the
context budget, the cadence — lives in `core` and is shared. What lives here is the shape
hermes expects: a provider object with methods it calls, returning strings it injects.

WHY register(ctx) AND NOT INHERITANCE. The loader tries `register(ctx)` first and only
falls back to scanning for a `MemoryProvider` subclass, and the register path does no
issubclass check. That matters because this repo's test suite runs WITHOUT hermes on the
path — it lives in its own venv — so a hard `from agent.memory_provider import
MemoryProvider` at module level would make the adapter untestable here. The ABC is
imported when available, purely to inherit its defaults, and `object` otherwise.

WHY realpath AND NOT abspath. The plugin is installed as a symlink into
`$HERMES_HOME/plugins/memories`. Measured: through that symlink `abspath(__file__)` is
`$HERMES_HOME/plugins/memories/__init__.py`, so walking up gives `$HERMES_HOME/plugins`
and `core` is never found. Only `realpath` resolves to the repo.

VERSION REALITY. Written against hermes v0.20.0 as INSTALLED, not as published: the
install has no `RecallStatus`, no `recall_status()` and no `unavailable_reason()`. Both
methods are implemented anyway — inert where nothing calls them, working if hermes is
upgraded — and `RecallStatus` is imported with a local fallback.
"""
import os
import sys

#: `realpath` first — see the module docstring. This is the one line that must not be
#: "simplified" to abspath.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import core  # noqa: E402

try:  # hermes is present in production and absent in this repo's test run
    from agent.memory_provider import MemoryProvider as _Base
except ImportError:
    _Base = object


class MemoriesProvider(_Base):
    """memories-plugin as a hermes memory provider."""

    #: Must equal the install directory name — that is what `memory.provider` selects.
    name = "memories"

    def __init__(self):
        self._cfg = None
        self._store = None
        self._reason = ""
        self._session_id = ""

    # -- availability ---------------------------------------------------------

    def is_available(self) -> bool:
        """Whether the plugin is configured. NO network calls — the contract forbids them,
        and `is_available` gates initialization, so a slow probe here delays every start."""
        try:
            cfg = core.load()
            cfg.require_qdrant()
            cfg.resolved_embed_url()
            cfg.require_memory_collection()
        except core.ConfigError as exc:
            self._reason = str(exc)

            return False
        self._cfg = cfg

        return True

    def unavailable_reason(self) -> str:
        """Actionable reason, since `initialize` is never reached when unavailable.

        Not called by v0.20.0; kept so an upgraded hermes surfaces it for free.
        """
        return self._reason

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or "default"

    def shutdown(self) -> None:
        self._store = None

    def get_tool_schemas(self) -> list:
        return []          # Task 8 fills this

    def system_prompt_block(self) -> str:
        return ""          # Task 5 fills this

    def prefetch(self, query_text: str, *, session_id: str = "") -> str:
        return ""          # Task 5 fills this

    def recall_status(self):
        return None        # Task 5 fills this

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass               # Task 7 fills this

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return "{}"        # Task 8 fills this

    def get_config_schema(self) -> list:
        return []          # Task 9 fills this

    def save_config(self, values: dict, hermes_home: str) -> None:
        pass               # Task 9 fills this

    # -- explicit no-ops, with the reason each one is a no-op -----------------
    #
    # Defined rather than inherited. The ABC would supply these as defaults in production,
    # but this suite runs without hermes on the path, and a default also silently absorbs a
    # method a hermes upgrade renames — a rename would become a feature that quietly stops
    # working instead of a test that fails.

    def queue_prefetch(self, query_text: str, *, session_id: str = "") -> None:
        """No background prefetch: the recall measures 0.5-1.7s against an 8s ceiling, so
        blocking in `prefetch` is simpler and costs nothing worth reclaiming."""

    def on_session_end(self, messages: list) -> None:
        """No end-of-session extraction, deliberately. claude-code has no session-end hook,
        and implementing extraction only here would make the two hosts behave differently —
        the asymmetry this whole adapter exists to avoid. See the spec, §3.4."""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages=None) -> None:
        """Writing is the checkpoint procedure's job, and the model performs it through the
        memory tools. Capturing turns here would store raw conversation, which the
        procedure explicitly discards as filler."""


def register(ctx) -> None:
    """Entry point the loader prefers. Also the string discovery greps for."""
    ctx.register_memory_provider(MemoriesProvider())
```

- [ ] **Step 5: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 305 tests`, `OK (skipped=17)`

- [ ] **Step 6: Prove que a armadilha do symlink é real**

```bash
cp hosts/hermes/__init__.py /tmp/h.bak
sed -i 's/os.path.realpath(__file__)/os.path.abspath(__file__)/' hosts/hermes/__init__.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  com abspath -> $(python3 -m unittest tests.test_hermes_provider 2>&1 | tail -1)"
cp /tmp/h.bak hosts/hermes/__init__.py; rm /tmp/h.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  com realpath -> $(python3 -m unittest tests.test_hermes_provider 2>&1 | tail -1)"
```
Expected: `FAILED` com abspath, `OK` com realpath

- [ ] **Step 7: Confirme que o hermes REAL descobre o provedor**

```bash
H=~/.hermes/hermes-agent
LINK=$(mktemp -d)/plugins; mkdir -p "$LINK"
ln -s ~/dev/memories-plugin/hosts/hermes "$LINK/memories"
HERMES_HOME=$(dirname "$LINK") python3 -c "
import sys; sys.path.insert(0,'$H')
from plugins.memory import discover_memory_providers
found = {n: (d, ok) for n, d, ok in discover_memory_providers()}
print('  memories descoberto?', 'memories' in found)
print('  ', found.get('memories'))
"
```
Expected: `memories descoberto? True`

- [ ] **Step 8: Commit**

```bash
git add hosts/ tests/test_hermes_provider.py
git commit -m "Add the hermes host adapter skeleton: discovery, availability, symlink load

Verified against the INSTALLED hermes v0.20.0 rather than the published source — the two
differ, and the installed one is what runs. The method surface it calls is frozen in the
test so a hermes upgrade that adds one shows up as a failing test instead of an
AttributeError in production.

register(ctx) rather than inheritance because the loader prefers it and does no issubclass
check there, which is what lets this repo's suite verify the adapter without hermes on the
path. The ABC is imported when available purely for its defaults.

realpath and not abspath: through the install symlink, abspath walks up into
\$HERMES_HOME/plugins and never finds core. Proved by reverting the one line and watching
the test go red."
```

---

### Task 5: `prefetch` — o recall no hermes, com os 4 estados

**Files:**
- Modify: `hosts/hermes/__init__.py`
- Test: `tests/test_hermes_provider.py` (acrescenta)

**Interfaces:**
- Consumes: `core.blocks.{Budget, recall_block, empty_block, unavailable_block, split_by_budget}`, `core.session_state`, `core.query.{skip_reason, angles}`, `core.build_memory`, `core.Policy`, `core.breaker.Breaker`, `core.prompts.INSTRUCTIONS`.
- Produces: `MemoriesProvider.prefetch(query, *, session_id="") -> str`;
  `MemoriesProvider.system_prompt_block() -> str`;
  `MemoriesProvider.recall_status()`; `MemoriesProvider.HERMES_PREFETCH_BUDGET_S = 8.0`.

- [ ] **Step 1: Escreva o teste que falha**

```python
class TestPrefetch(unittest.TestCase):
    """The read direction, through the hermes entry point.

    The provider is driven with fakes injected into its store slot, so these run offline.
    """

    def _provider(self, hits, outcome):
        from core.retrieval import Outcome  # noqa: F401  (documents the shape below)
        p = MemoriesProvider()
        p._cfg = object()

        class FakeStore:
            def recall(self, queries, policy, top_k, suppressed=None):
                return hits, outcome

        p._store = FakeStore()
        p._state_dir = Path(tempfile.mkdtemp())

        return p

    def test_a_populated_result_carries_the_rules_and_the_memory(self):
        from core.retrieval import CE, Outcome
        from tests.test_blocks import FakeHit
        p = self._provider([FakeHit(id="m1", document="a durable fact", origin=CE)],
                           Outcome(candidates=1, reranked=True))
        out = p.prefetch("how does the poll paginate?")
        self.assertIn("a durable fact", out)
        self.assertIn("m1", out)
        self.assertIn("PREVAILS", out, "the rules of use travel with the memory")

    def test_an_empty_complete_result_says_there_is_no_precedent(self):
        from core.retrieval import Outcome
        p = self._provider([], Outcome(candidates=4, best_dense=0.31, reranked=True))
        self.assertIn("There is no recorded precedent", p.prefetch("an absent subject"))

    def test_a_partial_result_withdraws_the_claim(self):
        from core.retrieval import Outcome
        p = self._provider([], Outcome(candidates=27, best_dense=0.54,
                                       suppressed="circuit breaker: 12s ago"))
        out = p.prefetch("a subject that might have history")
        self.assertNotIn("There is no recorded precedent", out)
        self.assertIn("not evidence that no precedent exists", out)

    def test_a_failure_reaches_the_MODEL_and_not_only_the_log(self):
        """The central contract: silent to the user, never to the model."""
        p = MemoriesProvider()
        p._cfg = object()

        class Broken:
            def recall(self, *a, **kw):
                raise core.EmbeddingError("endpoint down")

        p._store = Broken()
        p._state_dir = Path(tempfile.mkdtemp())
        out = p.prefetch("a real question about the archive")
        self.assertIn("UNAVAILABLE", out)
        self.assertIn("was not consulted", out)

    def test_an_unexpected_exception_also_reaches_the_model(self):
        p = MemoriesProvider()
        p._cfg = object()

        class Exploding:
            def recall(self, *a, **kw):
                raise RuntimeError("something nobody predicted")

        p._store = Exploding()
        p._state_dir = Path(tempfile.mkdtemp())
        self.assertIn("UNAVAILABLE", p.prefetch("a real question about the archive"))

    def test_a_trivial_prompt_costs_nothing(self):
        """hermes has its own is_trivial_prompt, but it is ENGLISH ONLY — measured: it does
        not match "ok, pode continuar". The plugin's composed Portuguese filter runs too."""
        p = MemoriesProvider()
        p._cfg = object()

        class Counting:
            calls = 0

            def recall(self, *a, **kw):
                Counting.calls += 1

                return [], None

        p._store = Counting()
        p._state_dir = Path(tempfile.mkdtemp())
        for prompt in ("ok", "sim, pode ser", "ok, pode continuar", "/status"):
            self.assertEqual(p.prefetch(prompt), "", prompt)
        self.assertEqual(Counting.calls, 0, "a trivial prompt reached the network")

    def test_the_system_prompt_block_is_static_and_not_the_recall(self):
        block = MemoriesProvider().system_prompt_block()
        self.assertIn("qctx memory", block)
        self.assertNotIn("── ", block, "recalled memories go through prefetch, not here")
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_hermes_provider -v 2>&1 | tail -5`
Expected: FAIL — `AttributeError: 'MemoriesProvider' object has no attribute 'prefetch'`
(o `prefetch` herdado devolve `""`, então a primeira asserção de conteúdo falha)

- [ ] **Step 3: Implemente `prefetch`**

Acrescente a `hosts/hermes/__init__.py`:

```python
from core import blocks, query, session_state  # noqa: E402
from core.breaker import Breaker  # noqa: E402
from core.prompts import INSTRUCTIONS  # noqa: E402

#: hermes runs an external provider's prefetch in a thread with this ceiling
#: (`_EXTERNAL_PREFETCH_TIMEOUT_S` in agent/memory_manager.py, measured 8.0). The recall
#: measures 0.5-1.7s against the real archive, so blocking is fine and a background queue
#: would be complexity without a reason. The dependency timeouts below are derived from
#: this the same way the claude-code hook derives its own: divided among the calls that
#: will actually be made, never repeated per call.
HERMES_PREFETCH_BUDGET_S = 8.0
```

e os métodos:

```python
    def prefetch(self, query_text: str, *, session_id: str = "") -> str:
        """Recall for the upcoming turn. NEVER raises, ALWAYS tells the model the truth.

        Failure is degradation, not an exception: the turn must proceed. But an absent
        result is indistinguishable from "there is no precedent" unless we say so, and a
        model that reads silence goes on to call something unprecedented when nobody
        looked. So every failure path returns the unavailability block rather than "".
        """
        try:
            return self._prefetch(query_text, session_id or self._session_id)
        except core.CoreError as exc:
            return blocks.unavailable_block(type(exc).__name__, str(exc)[:200])
        except BaseException as exc:  # noqa: BLE001 — see docstring
            return blocks.unavailable_block("the memory provider",
                                            f"{type(exc).__name__}: {exc}"[:200])

    def _prefetch(self, query_text: str, session_id: str) -> str:
        skip = query.skip_reason(query_text)
        if skip:
            return ""

        angles = query.angles(query_text)
        store = self._ensure_store(len(angles))
        breaker = Breaker(self._state_path("rerank-breaker"), self.BREAKER_SECONDS)
        idle = breaker.is_open()
        suppressed = None
        if idle is not None:
            store.reranker = None
            suppressed = f"circuit breaker: the re-rank failed {idle:.0f}s ago"

        policy = core.Policy(dense_floor=self.DENSE_FLOOR, strict_floor=self.STRICT_FLOOR,
                             min_score=self.MIN_SCORE, max_results=self.BUDGET.max_memories,
                             veto=True, order_matters=False)
        hits, outcome = store.recall(angles, policy, self.TOP_K, suppressed=suppressed)

        if outcome is not None and outcome.rerank_error:
            breaker.arm()
        elif outcome is not None and outcome.by_rerank:
            breaker.clear()

        path = self._state_path(f"recall-{_safe(session_id)}.json")
        state = session_state.load(path)
        round_no = session_state.next_round(state)
        seen = state.setdefault("seen", {})

        if not hits:
            session_state.prune(state)
            session_state.save(path, state)
            self._last_count = 0

            return blocks.empty_block(outcome, len(angles))

        full, pointers = blocks.split_by_budget(hits, seen, round_no, self.BUDGET)
        session_state.prune(state)
        session_state.save(path, state)
        self._last_count = len(full)

        return blocks.recall_block(full, pointers, len(angles), outcome, self.BUDGET)

    def recall_status(self):
        """Deterministic "recalled N" indicator. Absent from v0.20.0; free if upgraded."""
        if not getattr(self, "_last_count", 0):
            return None
        try:
            from agent.memory_provider import RecallStatus
        except ImportError:
            return None

        return RecallStatus(provider_label="memories", count=self._last_count)

    def system_prompt_block(self) -> str:
        """STATIC provider info. Recall goes through prefetch, never here."""
        return (
            "Long-term memory is available and searched automatically before each turn.\n"
            "To search or write it yourself, use the memory tools, or the CLI:\n"
            "  qctx memory recall \"<topic>\"   ·   qctx memory store \"<atomic fact>\"\n"
            + INSTRUCTIONS
        )
```

Acrescente os ajudantes `_ensure_store`, `_state_path`, `_safe`, e as constantes de
sintonia (`STRICT_FLOOR`, `DENSE_FLOOR`, `MIN_SCORE`, `TOP_K`, `BREAKER_SECONDS`,
`BUDGET`) lidas do ambiente com os mesmos nomes `QCTX_*` do hook, para que a configuração
seja equivalente. `_ensure_store(n_angles)` monta o `MemoryStore` com
`timeouts={"embed": 3.0, "qdrant": 2.0 / max(1, n_angles + 1), "rerank": 2.5}` — derivado
de `HERMES_PREFETCH_BUDGET_S`, não repetido por chamada.

- [ ] **Step 4: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 312 tests`, `OK (skipped=17)`

- [ ] **Step 5: Prove que o contrato de silêncio morde**

```bash
cp hosts/hermes/__init__.py /tmp/h.bak
python3 - <<'EOF'
import pathlib
f=pathlib.Path("hosts/hermes/__init__.py"); s=f.read_text()
s=s.replace('return blocks.unavailable_block(type(exc).__name__, str(exc)[:200])','return ""')
f.write_text(s)
EOF
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  falha devolvendo vazio -> $(python3 -m unittest tests.test_hermes_provider 2>&1 | tail -1)"
cp /tmp/h.bak hosts/hermes/__init__.py; rm /tmp/h.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: `FAILED`

- [ ] **Step 6: Confirme o prefetch contra o acervo REAL**

```bash
set -a; . ~/.bashrc >/dev/null 2>&1; . ~/.secrets >/dev/null 2>&1; set +a
python3 -c "
import sys, time; sys.path.insert(0,'.')
from hosts.hermes import MemoriesProvider
p = MemoriesProvider()
assert p.is_available(), p.unavailable_reason()
p.initialize('probe-hermes')
t0=time.monotonic()
out = p.prefetch('o que foi decidido sobre o colapso cross-lingual do cross-encoder?')
print(f'  {time.monotonic()-t0:.2f}s, {len(out)} chars, memórias: {out.count(chr(9472)*2)}')
print(' ', out.splitlines()[0])
"
```
Expected: menos de 8s, e o cabeçalho `[automatic recall — long-term memory]`

- [ ] **Step 7: Commit**

```bash
git add hosts/hermes/__init__.py tests/test_hermes_provider.py
git commit -m "Implement prefetch: the recall path for hermes, same four states

Every failure path returns the unavailability block, never the empty string. An absent
result is indistinguishable from 'there is no precedent' unless we say so, and a model
that reads silence goes on to call something unprecedented when nobody looked.

The dependency timeouts are derived from the 8s external-prefetch ceiling rather than set
per call — the same discipline the claude-code hook needed after its budget turned out to
multiply by the number of query angles."
```

---

### Task 6: O teste de equivalência entre os dois hosts

**Files:**
- Create: `tests/test_host_equivalence.py`

**Interfaces:**
- Consumes: `hooks/recall.py` (por subprocesso), `hosts.hermes.MemoriesProvider`, `core.blocks`.
- Produces: nada de produção — é o contrato que impede divergência.

- [ ] **Step 1: Escreva o teste**

```python
# tests/test_host_equivalence.py
"""The two hosts must inject the SAME text. Tested, not asserted.

This file is the reason core/prompts.py, core/blocks.py and core/session_state.py exist.
Equivalence claimed in a README is a promise; equivalence with a test that bites is a
contract. The failure it prevents is specific and this repo has already lived it: the
two-stage pipeline once existed in three copies, and the re-rank scale normalization was
present in one and missing from another.

The two adapters are driven differently on purpose — claude-code as a real subprocess with
JSON on stdin, hermes as an in-process method call — because if the test drove them
through a shared helper it would prove the helper consistent rather than the hosts
equivalent.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import blocks
from core.retrieval import CE, DENSE, Outcome
from tests.test_blocks import BUDGET, FakeHit


class TestBothHostsRenderTheSameBlock(unittest.TestCase):
    """Same Outcome and same hits through both adapters -> byte-identical text."""

    CASES = {
        "populated": ([FakeHit(id="m1", document="a durable fact about pagination")],
                      Outcome(candidates=3, reranked=True)),
        "populated_dense": ([FakeHit(id="m2", document="a fact judged by proximity",
                                     origin=DENSE, score=0.61)],
                            Outcome(candidates=1)),
        "empty_complete": ([], Outcome(candidates=4, best_dense=0.31, reranked=True)),
        "empty_partial_error": ([], Outcome(candidates=5, best_dense=0.5,
                                            rerank_error="timeout")),
        "empty_partial_breaker": ([], Outcome(candidates=27, best_dense=0.54,
                                              suppressed="circuit breaker: 12s ago")),
        "empty_partial_dropped": ([], Outcome(candidates=26, best_dense=0.5, reranked=True,
                                              dropped=14, dropped_above_floor=3)),
    }

    def _claude_side(self, hits, outcome):
        """Render through the claude-code hook's own module, in a subprocess."""
        script = (
            "import sys, json, os, tempfile\n"
            "sys.path.insert(0, %r)\n"
            "os.environ['QCTX_STATE_DIR'] = tempfile.mkdtemp()\n"
            "sys.path.insert(0, os.path.join(%r, 'hooks'))\n"
            "import recall\n"
            "from core.blocks import recall_block, empty_block\n"
            "from tests.test_blocks import FakeHit, BUDGET\n"
            "from core.retrieval import Outcome\n"
            "payload = json.loads(sys.stdin.read())\n"
            "hits = [FakeHit(**h) for h in payload['hits']]\n"
            "outcome = Outcome(**payload['outcome'])\n"
            "if hits:\n"
            "    print(recall_block(hits, [], payload['n_angles'], outcome, recall.BUDGET), end='')\n"
            "else:\n"
            "    print(empty_block(outcome, payload['n_angles']), end='')\n"
        ) % (str(REPO), str(REPO))
        payload = json.dumps({
            "hits": [dict(id=h.id, document=h.document, origin=h.origin, score=h.score,
                          metadata=h.metadata) for h in hits],
            "outcome": {k: v for k, v in vars(outcome).items() if k != "scored"},
            "n_angles": 2,
        })
        out = subprocess.run([sys.executable, "-c", script], input=payload,
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

        return out.stdout

    def _hermes_side(self, hits, outcome):
        """Render through the hermes provider, in process."""
        from hosts.hermes import MemoriesProvider
        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = Path(tempfile.mkdtemp())

        class FakeStore:
            def recall(self, queries, policy, top_k, suppressed=None):
                return hits, outcome

        p._store = FakeStore()

        return p.prefetch("how does the connector poll paginate its results?")

    def test_every_state_renders_identically_in_both_hosts(self):
        for label, (hits, outcome) in self.CASES.items():
            with self.subTest(state=label):
                claude = self._claude_side(hits, outcome)
                hermes = self._hermes_side(hits, outcome)
                self.assertEqual(claude.strip(), hermes.strip(),
                                 f"the two hosts diverged on the {label} state")

    def test_the_unavailable_block_is_identical(self):
        from hosts.hermes import MemoriesProvider
        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = Path(tempfile.mkdtemp())

        class Broken:
            def recall(self, *a, **kw):
                raise __import__("core").EmbeddingError("endpoint down")

        p._store = Broken()
        hermes = p.prefetch("a real question about the archive")
        expected = blocks.unavailable_block("EmbeddingError", "endpoint down")
        self.assertEqual(hermes.strip(), expected.strip())


class TestBothHostsShareOneConfiguration(unittest.TestCase):
    def test_they_resolve_the_same_collections_from_the_same_file(self):
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        cfg_path.write_text(json.dumps({
            "qdrant_url": "http://example.invalid/qdrant",
            "api_base_url": "http://example.invalid/v1",
            "memory_collection": "shared_memory",
            "docs_collection": "shared_tmp",
            "library_collection": "shared_library",
        }))
        env = dict(os.environ, QCTX_CONFIG=str(cfg_path))
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import core\n"
            "cfg = core.load()\n"
            "print(cfg.require_memory_collection())\n"
            "print(cfg.require_docs_collection())\n"
            "print(cfg.require_library_collection())\n"
        ) % str(REPO)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(),
                         ["shared_memory", "shared_tmp", "shared_library"])

    def test_the_tuning_knobs_have_the_same_names_in_both_hosts(self):
        """Equivalent CONFIGURATION means the same env var moves the same number in both.
        Read from each adapter's source rather than restated, so a rename in one shows up
        here instead of surfacing as a host that ignores a setting."""
        import re
        pattern = re.compile(r'QCTX_RECALL_[A-Z_]+')
        hook = set(pattern.findall((REPO / "hooks" / "recall.py").read_text()))
        host = set(pattern.findall((REPO / "hosts" / "hermes" / "__init__.py").read_text()))
        self.assertTrue(hook, "no QCTX_RECALL_* names found in the hook")
        self.assertEqual(hook, host,
                         f"only in claude-code: {hook - host}; only in hermes: {host - hook}")
```

- [ ] **Step 2: Rode e veja falhar ou passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_host_equivalence -v 2>&1 | tail -8`
Expected: pode passar de primeira se as Tasks 1-5 ficaram certas. Se falhar, a mensagem
nomeia o estado divergente — corrija o ADAPTADOR, nunca o teste.

- [ ] **Step 3: Prove que o teste morde — divirja um host de propósito**

```bash
cp hosts/hermes/__init__.py /tmp/h.bak
python3 - <<'EOF'
import pathlib
f=pathlib.Path("hosts/hermes/__init__.py"); s=f.read_text()
s=s.replace("return blocks.recall_block(full, pointers, len(angles), outcome, self.BUDGET)",
            "return 'Memories:\\n' + blocks.recall_block(full, pointers, len(angles), outcome, self.BUDGET)")
f.write_text(s)
EOF
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  hermes com prefixo próprio -> $(python3 -m unittest tests.test_host_equivalence 2>&1 | tail -1)"
cp /tmp/h.bak hosts/hermes/__init__.py; rm /tmp/h.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  restaurado -> $(python3 -m unittest tests.test_host_equivalence 2>&1 | tail -1)"
```
Expected: `FAILED` com o prefixo, `OK` depois de restaurar

- [ ] **Step 4: Prove que a divergência de CONFIG também morde**

```bash
cp hosts/hermes/__init__.py /tmp/h.bak
sed -i 's/QCTX_RECALL_MAX_MEMORIES/QCTX_HERMES_MAX_MEMORIES/' hosts/hermes/__init__.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  nome de env divergente -> $(python3 -m unittest tests.test_host_equivalence 2>&1 | tail -1)"
cp /tmp/h.bak hosts/hermes/__init__.py; rm /tmp/h.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: `FAILED`

- [ ] **Step 5: Rode a suíte inteira e commit**

Run: `python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 320 tests`, `OK (skipped=17)`

```bash
git add tests/test_host_equivalence.py
git commit -m "Pin host equivalence with a test that bites

This is why core/prompts.py, core/blocks.py and core/session_state.py exist. Equivalence
claimed in a README is a promise; equivalence with a test is a contract.

The two adapters are driven differently on purpose — claude-code as a subprocess with JSON
on stdin, hermes as a method call — because driving both through one helper would prove the
helper consistent rather than the hosts equivalent. Configuration equivalence is checked by
extracting the QCTX_RECALL_* names from each adapter's source rather than restating them,
so a rename in one surfaces here instead of as a host that silently ignores a setting.

Proved by diverging each side in turn and watching it go red."
```

---

### Task 7: Cadência do checkpoint no hermes

**Files:**
- Modify: `hosts/hermes/__init__.py`
- Test: `tests/test_hermes_provider.py` (acrescenta)

**Interfaces:**
- Consumes: `core.session_state.due`, `core.prompts.CHECKPOINT_PROCEDURE`.
- Produces: `MemoriesProvider.on_turn_start(turn_number, message, **kwargs)`;
  `MemoriesProvider.CHECKPOINT_INTERVAL: int`.

- [ ] **Step 1: Escreva o teste que falha**

```python
class TestCheckpointCadence(unittest.TestCase):
    """The write nudge. It rides along in prefetch, and the reason is structural:
    on_turn_start runs every turn and CARRIES the number, but returns None — so it cannot
    inject. The only injection points hermes offers are prefetch and system_prompt_block.
    """

    def _provider(self, interval):
        from core.retrieval import Outcome
        p = MemoriesProvider()
        p._cfg = object()
        p._state_dir = Path(tempfile.mkdtemp())
        p.CHECKPOINT_INTERVAL = interval

        class FakeStore:
            def recall(self, queries, policy, top_k, suppressed=None):
                return [], Outcome(candidates=1, best_dense=0.2, reranked=True)

        p._store = FakeStore()

        return p

    def test_the_procedure_appears_on_the_interval_and_only_there(self):
        p = self._provider(3)
        seen = []
        for turn in range(1, 8):
            p.on_turn_start(turn, "a real question about the archive")
            out = p.prefetch("a real question about the archive")
            if "memory checkpoint" in out:
                seen.append(turn)
        self.assertEqual(seen, [3, 6])

    def test_the_procedure_renders_with_the_turn_number(self):
        p = self._provider(1)
        p.on_turn_start(1, "a real question about the archive")
        out = p.prefetch("a real question about the archive")
        self.assertIn("Interaction 1 of this conversation (every 1)", out)
        self.assertNotIn("{count}", out)

    def test_an_interval_of_zero_disables_it(self):
        p = self._provider(0)
        for turn in range(1, 6):
            p.on_turn_start(turn, "a real question about the archive")
            self.assertNotIn("memory checkpoint", p.prefetch("a real question about the archive"))

    def test_the_recall_block_still_comes_first(self):
        """Order matters: the memories and their rules, then the write nudge."""
        from core.retrieval import CE, Outcome
        from tests.test_blocks import FakeHit
        p = self._provider(1)
        p._store = type("S", (), {"recall": lambda self, *a, **kw: (
            [FakeHit(id="m1", document="a durable fact", origin=CE)],
            Outcome(candidates=1, reranked=True))})()
        p.on_turn_start(1, "a real question about the archive")
        out = p.prefetch("a real question about the archive")
        self.assertLess(out.index("a durable fact"), out.index("memory checkpoint"))
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_hermes_provider.TestCheckpointCadence -v 2>&1 | tail -4`
Expected: FAIL — a lista de turnos vem `[]` porque nada injeta o procedimento

- [ ] **Step 3: Implemente**

```python
    #: Turns between checkpoint nudges. Same env var as claude-code, so the setting moves
    #: both hosts.
    CHECKPOINT_INTERVAL = int(os.environ.get("QCTX_CHECKPOINT_INTERVAL")
                              or os.environ.get("REMEMBER_INTERVAL") or "5")

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Record the turn. It cannot inject — it returns None — so `prefetch` carries the
        checkpoint when this number says it is due."""
        self._turn = int(turn_number or 0)
```

e no fim de `_prefetch`, antes de cada `return`, componha:

```python
        return self._with_checkpoint(block)

    def _with_checkpoint(self, block: str) -> str:
        """Append the write procedure when the cadence says so.

        The recall comes first and the nudge second: the memories and their rules of use are
        what the turn needs to answer, the nudge is what the session needs to remember.
        """
        if not session_state.due(getattr(self, "_turn", 0), self.CHECKPOINT_INTERVAL):
            return block
        nudge = CHECKPOINT_PROCEDURE.format(count=self._turn,
                                            interval=self.CHECKPOINT_INTERVAL)

        return f"{block}\n\n{nudge}" if block else nudge
```

- [ ] **Step 4: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 324 tests`, `OK (skipped=17)`

- [ ] **Step 5: Prove que morde**

```bash
cp hosts/hermes/__init__.py /tmp/h.bak
sed -i 's/if not session_state.due(getattr(self, "_turn", 0), self.CHECKPOINT_INTERVAL):/if False:/' hosts/hermes/__init__.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  cadência desligada -> $(python3 -m unittest tests.test_hermes_provider 2>&1 | tail -1)"
cp /tmp/h.bak hosts/hermes/__init__.py; rm /tmp/h.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: `FAILED`

- [ ] **Step 6: Commit**

```bash
git add hosts/hermes/__init__.py tests/test_hermes_provider.py
git commit -m "Carry the checkpoint cadence in prefetch, because on_turn_start cannot inject

on_turn_start runs every turn and carries the number, but returns None. The only injection
points hermes offers are prefetch and system_prompt_block, so the write nudge rides along
in prefetch on the turn the cadence names — recall first, nudge second."
```

---

### Task 8: As 15 ferramentas

**Files:**
- Create: `hosts/hermes/tools.py`
- Modify: `hosts/hermes/__init__.py` (`get_tool_schemas`, `handle_tool_call`)
- Test: `tests/test_hermes_tools.py`

**Interfaces:**
- Consumes: `core.build_memory`, `core.build_docs`, `core.search_collections`, `core.parse_ttl`.
- Produces:
  - `hosts.hermes.tools.SCHEMAS: list[dict]` — 15 entradas OpenAI function-calling.
  - `hosts.hermes.tools.dispatch(name, args, *, cfg) -> str` — devolve string JSON.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_hermes_tools.py
"""The 15 tools the model may call, and the 5 it may not.

Configuration is the operator's, not the model's: exposing `config set` as a tool would let
the model point the archive somewhere else mid-conversation. `setup` is interactive and
wants a TTY. The CLI keeps all 20 in both hosts — the restriction is only about what the
MODEL can reach on its own.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hosts.hermes import MemoriesProvider, tools

EXPECTED = {
    "memory_store", "memory_store_many", "memory_find", "memory_recall", "memory_get",
    "memory_update", "memory_delete", "memory_list", "memory_search_collections",
    "docs_index", "docs_keep", "docs_search", "docs_list", "docs_refresh", "docs_drop",
}
FORBIDDEN = {"setup", "config_set", "config_detect", "config_show", "collections"}


class TestSchemas(unittest.TestCase):
    def test_exactly_the_fifteen_are_exposed(self):
        self.assertEqual({s["name"] for s in tools.SCHEMAS}, EXPECTED)

    def test_no_configuration_tool_is_reachable_by_the_model(self):
        names = {s["name"] for s in tools.SCHEMAS}
        self.assertEqual(names & FORBIDDEN, set(),
                         "config is the operator's; a tool here could redirect the archive")

    def test_every_schema_has_the_three_required_keys(self):
        for s in tools.SCHEMAS:
            self.assertEqual({"name", "description", "parameters"} - set(s), set(), s["name"])
            self.assertEqual(s["parameters"]["type"], "object", s["name"])
            self.assertIn("properties", s["parameters"], s["name"])

    def test_every_description_says_when_to_use_it(self):
        for s in tools.SCHEMAS:
            self.assertGreater(len(s["description"]), 40,
                               f"{s['name']}: a description the model cannot act on is a stub")

    def test_the_provider_exposes_them(self):
        self.assertEqual({s["name"] for s in MemoriesProvider().get_tool_schemas()}, EXPECTED)

    def test_required_parameters_are_declared(self):
        by_name = {s["name"]: s for s in tools.SCHEMAS}
        self.assertIn("information", by_name["memory_store"]["parameters"]["required"])
        self.assertIn("query", by_name["memory_recall"]["parameters"]["required"])
        self.assertIn("path", by_name["docs_index"]["parameters"]["required"])
        self.assertIn("id", by_name["memory_get"]["parameters"]["required"])


class TestDispatch(unittest.TestCase):
    def test_every_declared_tool_is_routed(self):
        """A schema the dispatcher does not know is a tool the model calls and gets an
        error for — worse than not offering it."""
        for name in EXPECTED:
            self.assertIn(name, tools.ROUTES, f"{name} is declared but not routed")

    def test_an_unknown_tool_returns_json_not_an_exception(self):
        out = tools.dispatch("memory_teleport", {}, cfg=None)
        self.assertEqual(json.loads(out)["error"], "unknown tool: memory_teleport")

    def test_a_failure_comes_back_as_json_not_an_exception(self):
        """handle_tool_call must return a JSON string. A raise would surface to the user as
        a crashed turn instead of a result the model can react to."""
        import core
        cfg = core.Config(qdrant_url="", qdrant_api_key="", api_base_url="", api_key="",
                          embed_url="", rerank_url="", embed_model="m", rerank_model="r",
                          memory_collection="", docs_collection="d", library_collection="l",
                          vector_size=1024)
        out = tools.dispatch("memory_recall", {"query": "anything"}, cfg=cfg)
        payload = json.loads(out)
        self.assertIn("error", payload)

    def test_a_missing_required_argument_is_reported_not_raised(self):
        out = tools.dispatch("memory_store", {}, cfg=None)
        self.assertIn("error", json.loads(out))

    def test_the_provider_routes_through_dispatch(self):
        p = MemoriesProvider()
        out = p.handle_tool_call("memory_teleport", {})
        self.assertIn("error", json.loads(out))
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_hermes_tools -v 2>&1 | tail -4`
Expected: FAIL — `ImportError: cannot import name 'tools' from 'hosts.hermes'`

- [ ] **Step 3: Crie `hosts/hermes/tools.py`**

```python
"""The operations the MODEL may invoke, as hermes tool schemas.

Fifteen of the CLI's twenty. The five left out are `setup` (interactive, wants a TTY) and
`config show/set/detect` plus `collections` — configuration belongs to the operator, and a
`config set` tool would let the model point the archive somewhere else mid-conversation.
The CLI still carries all twenty in both hosts; this is only about what the model reaches
on its own.

Every handler returns a JSON STRING, and no handler raises. hermes surfaces a raise as a
crashed turn; a JSON error is something the model can read and react to.
"""
import json

import core


def _ok(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _err(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)
```

Depois `SCHEMAS` com as 15 entradas. Cada `description` diz QUANDO usar, não só o que faz —
é o que o modelo lê para decidir. Por exemplo:

```python
SCHEMAS = [
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
                "query": {"type": "string", "description": "Topic in natural language, not a symbol name."},
                "limit": {"type": "integer", "description": "Max memories to return (default 6)."},
            },
            "required": ["query"],
        },
    },
    ...
]
```

e o roteamento:

```python
ROUTES = {
    "memory_store": _memory_store,
    ...
}


def dispatch(name: str, args: dict, *, cfg) -> str:
    """Run a tool. Returns a JSON string, always; raises never."""
    handler = ROUTES.get(name)
    if handler is None:
        return _err(f"unknown tool: {name}")
    try:
        return handler(args or {}, cfg)
    except core.CoreError as exc:
        return _err(f"{type(exc).__name__}: {exc}")
    except KeyError as exc:
        return _err(f"missing required argument: {exc}")
    except Exception as exc:  # noqa: BLE001 — a raise becomes a crashed turn
        return _err(f"{type(exc).__name__}: {exc}")
```

Em `hosts/hermes/__init__.py`:

```python
    def get_tool_schemas(self) -> list:
        return list(tools.SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return tools.dispatch(tool_name, args, cfg=self._cfg or _load_cfg_quietly())
```

- [ ] **Step 4: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 337 tests`, `OK (skipped=17)`

- [ ] **Step 5: Prove que o roteamento morde**

```bash
cp hosts/hermes/tools.py /tmp/t.bak
python3 - <<'EOF'
import pathlib
f=pathlib.Path("hosts/hermes/tools.py"); s=f.read_text()
s=s.replace('    "docs_drop": _docs_drop,\n','')
f.write_text(s)
EOF
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  ferramenta declarada e não roteada -> $(python3 -m unittest tests.test_hermes_tools 2>&1 | tail -1)"
cp /tmp/t.bak hosts/hermes/tools.py; rm /tmp/t.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: `FAILED`

- [ ] **Step 6: Exercite as ferramentas contra a infra real, em coleção descartável**

```bash
set -a; . ~/.bashrc >/dev/null 2>&1; . ~/.secrets >/dev/null 2>&1; set +a
python3 -c "
import json, sys, uuid; sys.path.insert(0,'.')
import core
from hosts.hermes import tools
base = core.load()
fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
throwaway = f'hermes_tools_probe_{uuid.uuid4().hex[:8]}'
fields['memory_collection'] = throwaway
cfg = core.Config(**fields)
try:
    r = json.loads(tools.dispatch('memory_store', {'information': 'probe fact from the hermes tools'}, cfg=cfg))
    print('  store ->', r.get('status'), r.get('id','')[:8])
    f = json.loads(tools.dispatch('memory_find', {'query': 'probe fact hermes'}, cfg=cfg))
    print('  find  ->', len(f), 'hits')
    print('  delete->', json.loads(tools.dispatch('memory_delete', {'id': r['id']}, cfg=cfg)).get('status'))
finally:
    core.build_qdrant(cfg).delete_collection(throwaway)
    print('  coleção descartável removida')
"
```
Expected: `store -> created`, `find -> 1 hits`, `delete -> deleted`

- [ ] **Step 7: Commit**

```bash
git add hosts/hermes/tools.py hosts/hermes/__init__.py tests/test_hermes_tools.py
git commit -m "Expose the 15 model-callable operations as hermes tools

Fifteen of the CLI's twenty. setup wants a TTY; config show/set/detect and collections stay
out because configuration belongs to the operator — a config-set tool would let the model
point the archive somewhere else mid-conversation. The CLI still carries all twenty in both
hosts.

No handler raises: hermes surfaces a raise as a crashed turn, while a JSON error is
something the model can read and act on. A schema that is declared but not routed is worse
than one not offered, so a test walks the declared names against the route table."
```

---

### Task 9: Configuração equivalente pelo wizard do hermes

**Files:**
- Modify: `hosts/hermes/__init__.py` (`get_config_schema`, `save_config`)
- Test: `tests/test_hermes_provider.py` (acrescenta)

**Interfaces:**
- Consumes: `core.config.{ENV_ALIASES, SECRET_FIELDS, DEFAULTS, save}`.
- Produces: `MemoriesProvider.get_config_schema() -> list[dict]`;
  `MemoriesProvider.save_config(values, hermes_home) -> None`.

- [ ] **Step 1: Escreva o teste que falha**

```python
class TestConfigSchema(unittest.TestCase):
    """`hermes memory setup` walks this, and it must write where claude-code reads."""

    def test_the_two_api_keys_are_declared_secret(self):
        by_key = {f["key"]: f for f in MemoriesProvider().get_config_schema()}
        for key in ("qdrant_api_key", "api_key"):
            self.assertTrue(by_key[key]["secret"],
                            f"{key} in a config file is a leaked secret")
            self.assertTrue(by_key[key]["env_var"].startswith("QCTX_"),
                            "a secret needs the env var name it will be read from")

    def test_no_non_secret_field_is_marked_secret(self):
        by_key = {f["key"]: f for f in MemoriesProvider().get_config_schema()}
        self.assertFalse(by_key["memory_collection"]["secret"])
        self.assertFalse(by_key["qdrant_url"]["secret"])

    def test_the_schema_covers_every_configurable_field(self):
        from core.config import Config
        from dataclasses import fields as dc_fields
        declared = {f["key"] for f in MemoriesProvider().get_config_schema()}
        self.assertEqual({f.name for f in dc_fields(Config)}, declared,
                         "a field missing here is a setting the hermes wizard cannot set")

    def test_save_config_writes_where_claude_code_reads(self):
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        os.environ["QCTX_CONFIG"] = str(cfg_path)
        try:
            MemoriesProvider().save_config({"memory_collection": "shared_memory"},
                                           str(cfg_dir))
            self.assertEqual(json.loads(cfg_path.read_text())["memory_collection"],
                             "shared_memory")
        finally:
            del os.environ["QCTX_CONFIG"]

    def test_save_config_refuses_a_secret_even_if_hermes_passes_one(self):
        """core.save already refuses; this asserts the adapter does not route around it."""
        cfg_dir = Path(tempfile.mkdtemp())
        cfg_path = cfg_dir / "config.json"
        os.environ["QCTX_CONFIG"] = str(cfg_path)
        try:
            MemoriesProvider().save_config({"api_key": "SUPERSECRET"}, str(cfg_dir))
            written = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            self.assertNotIn("api_key", written)
            self.assertNotIn("SUPERSECRET", json.dumps(written))
        finally:
            del os.environ["QCTX_CONFIG"]
```

- [ ] **Step 2: Rode e confirme que falha**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_hermes_provider.TestConfigSchema -v 2>&1 | tail -4`
Expected: FAIL — `get_config_schema` herdado devolve `[]`

- [ ] **Step 3: Implemente**

```python
    def get_config_schema(self) -> list:
        """Fields `hermes memory setup` walks. The same settings `qctx setup` collects.

        Derived from core.config rather than restated, so a field added there cannot become
        a setting the hermes wizard silently cannot reach.
        """
        from dataclasses import fields as dc_fields

        from core.config import DEFAULTS, ENV_ALIASES, SECRET_FIELDS, Config

        described = {
            "qdrant_url": "Qdrant base URL, e.g. https://host/qdrant",
            "qdrant_api_key": "Qdrant API key",
            "api_base_url": "OpenAI-compatible base URL serving the models",
            "api_key": "API key for that endpoint",
            "embed_url": "Full /embeddings URL (optional if api_base_url is set)",
            "rerank_url": "Full /rerank URL (optional; without it search still works)",
            "embed_model": "Embedding model name",
            "rerank_model": "Cross-encoder model name",
            "memory_collection": "Collection holding curated facts. Point it at the one "
                                 "claude-code uses to share the archive.",
            "docs_collection": "Collection for temporary document chunks (TTL)",
            "library_collection": "Collection for permanently kept documents",
            "vector_size": "Embedding dimension; `qctx config detect` measures it",
        }
        out = []
        for f in dc_fields(Config):
            secret = f.name in SECRET_FIELDS
            field = {
                "key": f.name,
                "description": described[f.name],
                "secret": secret,
                "required": f.name in ("qdrant_url", "memory_collection"),
                "type": "integer" if f.name == "vector_size" else "text",
            }
            default = DEFAULTS.get(f.name)
            if default not in ("", None):
                field["default"] = default
            if secret:
                field["env_var"] = ENV_ALIASES[f.name][0]
            out.append(field)

        return out

    def save_config(self, values: dict, hermes_home: str) -> None:
        """Write to the plugin's native location — the SAME file claude-code reads.

        That is what makes the two hosts share one configuration instead of two that drift.
        Secrets are dropped here, not merely unsaved: `core.save` refuses them, and routing
        around that refusal would put a key in a text file that ends up in backups and in
        dotfile sync.
        """
        from core.config import SECRET_FIELDS

        patch = {k: v for k, v in (values or {}).items()
                 if k not in SECRET_FIELDS and v not in ("", None)}
        if patch:
            core.save(patch)
```

- [ ] **Step 4: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `Ran 342 tests`, `OK (skipped=17)`

- [ ] **Step 5: Prove que a guarda de segredo morde**

```bash
cp hosts/hermes/__init__.py /tmp/h.bak
sed -i 's/if k not in SECRET_FIELDS and v not in ("", None)}/if v not in ("", None)}/' hosts/hermes/__init__.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  sem filtrar segredo -> $(python3 -m unittest tests.test_hermes_provider 2>&1 | tail -1)"
cp /tmp/h.bak hosts/hermes/__init__.py; rm /tmp/h.bak
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
```
Expected: `FAILED`

- [ ] **Step 6: Commit**

```bash
git add hosts/hermes/__init__.py tests/test_hermes_provider.py
git commit -m "Share one configuration between the hosts through the hermes wizard

get_config_schema is derived from core.config rather than restated, so a field added there
cannot become a setting the hermes wizard silently cannot reach. save_config writes the
plugin's native file — the same one claude-code reads — which is what makes the two share a
configuration instead of drifting apart with two.

The two API keys are declared secret so hermes routes them to .env, matching core.save's
existing refusal to put a key in a text file that ends up in backups and dotfile sync. The
adapter drops them rather than routing around that refusal."
```

---

### Task 10: Cutover local e documentação

**Files:**
- Create: `scripts/hermes_cutover.sh`
- Modify: `README.md`
- Modify: `skills/memory/SKILL.md`, `skills/doc-index/SKILL.md`
- Test: manual, com ensaio

**Interfaces:**
- Consumes: nada de código.
- Produces: `scripts/hermes_cutover.sh` — ensaio por padrão, `--apply` para valer, `exit` diferente de zero quando um passo falha.

- [ ] **Step 1: Escreva o script**

```bash
#!/usr/bin/env bash
# Install memories-plugin as the hermes-agent memory provider, in ONE atomic pass.
#
# hermes allows exactly ONE external memory provider, so this REPLACES whatever is active
# — here, a third-party `qdrant` provider holding 1423 points in its own collection. That
# provider is disabled by configuration, never deleted, and its points are left untouched:
# they stay reachable read-only with
#     qctx memory search-collections "<topic>" --collections hermes_memory
#
# With no argument this is a DRY RUN and writes nothing. With --apply it applies after
# taking dated backups.
#
# Close other hermes sessions first: the provider list is read at session start, so a
# running session keeps the old one either way.
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
CONFIG="$HERMES_HOME/config.yaml"
LINK="$HERMES_HOME/plugins/memories"
APPLY="${1:-}"
STAMP="$(date +%Y%m%d-%H%M%S)"

failed=0
say()  { printf '%s\n' "$*"; }
ok()   { say "  ok    $*"; }
fail() { say "  FAIL  $*"; failed=1; }
note() { say "  ..    $*"; }

say "=== checks ==="
[ -d "$HERMES_HOME" ] && ok "HERMES_HOME at $HERMES_HOME" || fail "no HERMES_HOME at $HERMES_HOME"
[ -f "$CONFIG" ] && ok "config.yaml found" || fail "no config.yaml at $CONFIG"
command -v python3 >/dev/null && ok "python3 available" || fail "python3 is required"

if python3 -m unittest discover -s "$ROOT/tests" >/dev/null 2>&1; then
  ok "offline suite passes"
else
  fail "the offline suite does not pass — do not flip the switch like this"
fi

if python3 "$ROOT/cli/qctx.py" setup --check </dev/null >/dev/null 2>&1; then
  ok "qctx answers"
else
  note "qctx setup --check reported something pending — fix that before applying"
fi

current="$(grep -oP '^\s*provider:\s*\K\S+' "$CONFIG" 2>/dev/null | head -1 || true)"
note "active memory provider: ${current:-<none>}"
[ -e "$LINK" ] && note "$LINK already exists" || note "$LINK will be created"

if [ "$failed" -ne 0 ]; then
  say ""; say "checks failed — nothing was changed."; exit 1
fi

say ""
say "=== what changes ==="
say "  $LINK -> $ROOT/hosts/hermes   (symlink; one source of truth)"
say "  $CONFIG: memory.provider: ${current:-<none>} -> memories"
say ""
say "  UNCHANGED: every Qdrant collection, the 1423 points in hermes_memory, the qdrant"
say "  provider directory (disabled by config, not deleted), ~/.secrets, the .bashrc URLs."

if [ "$APPLY" != "--apply" ]; then
  say ""; say "DRY RUN. To apply: $0 --apply"; exit 0
fi

say ""
say "=== applying ==="
cp "$CONFIG" "$CONFIG.bak-$STAMP" && ok "backup $CONFIG.bak-$STAMP"
mkdir -p "$(dirname "$LINK")"
if [ -L "$LINK" ] || [ ! -e "$LINK" ]; then
  ln -sfn "$ROOT/hosts/hermes" "$LINK" && ok "symlink in place"
else
  fail "$LINK exists and is not a symlink — move it aside by hand"
fi

tmp="$(mktemp)"
if python3 - "$CONFIG" > "$tmp" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
out, done = [], False
in_memory = False
for line in src.splitlines(keepends=True):
    if re.match(r'^memory:\s*$', line):
        in_memory = True
        out.append(line); continue
    if in_memory and re.match(r'^\S', line):
        in_memory = False
    if in_memory and re.match(r'^(\s+)provider:\s*\S+', line):
        indent = re.match(r'^(\s+)', line).group(1)
        out.append(f"{indent}provider: memories\n"); done = True; continue
    out.append(line)
if not done:
    sys.exit("no memory.provider line found — set it by hand")
sys.stdout.write("".join(out))
PY
then
  mv "$tmp" "$CONFIG"; ok "memory.provider = memories"
else
  rm -f "$tmp"; fail "could not rewrite config.yaml — set memory.provider by hand"
fi

if [ "$failed" -ne 0 ]; then
  say ""
  say "one or more steps FAILED — see above. The .bak-$STAMP backup restores the previous"
  say "state, and the old provider is still installed."
  exit 1
fi

say ""
say "=== now ==="
say "  1. Close every hermes session and start a NEW one from a fresh terminal."
say "  2. Confirm the provider is the new one:"
say "       hermes memory setup   # 'memories' should be listed and selected"
say "  3. The old points remain reachable read-only:"
say "       qctx memory search-collections \"<topic>\" --collections hermes_memory"
say "  4. To go back: restore $CONFIG.bak-$STAMP and remove $LINK."

exit "$failed"
```

- [ ] **Step 2: Rode o ensaio e confirme que nada mudou**

```bash
chmod +x scripts/hermes_cutover.sh
bash -n scripts/hermes_cutover.sh && echo "  sintaxe ok"
cp ~/.hermes/config.yaml /tmp/cfg.before
./scripts/hermes_cutover.sh
diff -q /tmp/cfg.before ~/.hermes/config.yaml && echo "  o ensaio não tocou o config.yaml"
```
Expected: relatório de checks, `DRY RUN`, e o `diff` silencioso

- [ ] **Step 3: Prove que o `--apply` sai diferente de zero quando um passo falha**

```bash
FAKE=$(mktemp -d); mkdir -p "$FAKE/plugins"
printf 'memory:\n  provider: qdrant\n' > "$FAKE/config.yaml"
mkdir -p "$FAKE/plugins/memories"          # não-symlink no lugar: força o FAIL
HERMES_HOME="$FAKE" ./scripts/hermes_cutover.sh --apply >/tmp/co.out 2>&1
echo "  exit: $?"
grep -E "FAIL|one or more steps" /tmp/co.out | head -2
rm -rf "$FAKE" /tmp/co.out
```
Expected: `exit: 1` e a linha de `FAIL`

- [ ] **Step 4: Prove a idempotência num HOME falso**

```bash
FAKE=$(mktemp -d); mkdir -p "$FAKE/plugins"
printf 'memory:\n  memory_enabled: true\n  provider: qdrant\n  nudge_interval: 10\nother:\n  keep: yes\n' > "$FAKE/config.yaml"
for i in 1 2; do HERMES_HOME="$FAKE" ./scripts/hermes_cutover.sh --apply >/dev/null 2>&1; echo "  run $i exit: $?"; done
grep -E "provider|keep|nudge" "$FAKE/config.yaml"
rm -rf "$FAKE"
```
Expected: os dois com `exit: 0`, `provider: memories`, e `nudge_interval`/`keep` preservados

- [ ] **Step 5: Atualize README e as duas skills**

No `README.md`, acrescente uma seção `## Hosts` logo depois de `## Installation`:

```markdown
## Hosts

The same core serves two hosts, with the same operations and the same configuration.

| | claude-code | hermes-agent |
|---|---|---|
| adapter | `hooks/` | `hosts/hermes/` |
| install | `claude plugin marketplace add .` | symlink into `$HERMES_HOME/plugins/memories` |
| recall | `UserPromptSubmit` hook | `prefetch()` |
| checkpoint | second `UserPromptSubmit` hook | rides along in `prefetch()` on the Nth turn |
| operations | `qctx` CLI + skills | 15 tools + the same CLI |
| configuration | `~/.config/memories-plugin/config.json` | the same file |

Equivalence is not a claim in this table — `tests/test_host_equivalence.py` renders every
block state through both adapters and requires byte-identical output.

    ./scripts/hermes_cutover.sh            # dry run
    ./scripts/hermes_cutover.sh --apply    # install as the hermes memory provider
```

Nas duas skills, troque cada afirmação de host único por linguagem neutra: a skill é lida
pelos dois agentes. Onde `skills/memory/SKILL.md` diz `A UserPromptSubmit hook runs a
search`, troque por `The host runs a search before you see the text`.

- [ ] **Step 6: Rode a suíte inteira uma última vez, com integração**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest discover -s tests 2>&1 | tail -3
set -a; . ~/.bashrc >/dev/null 2>&1; . ~/.secrets >/dev/null 2>&1; set +a
QCTX_INTEGRATION=1 python3 -m unittest tests.test_integration 2>&1 | tail -3
```
Expected: `Ran 342 tests OK (skipped=17)` e `Ran 17 tests OK`

- [ ] **Step 7: Commit**

```bash
git add scripts/hermes_cutover.sh README.md skills/
git commit -m "Add the hermes cutover script and document both hosts

Dry run by default, non-zero exit when any apply step fails — the lesson from the
claude-code cutover, where the checks ran before the apply block and the script printed
success after a failure.

It REPLACES the active provider because hermes allows exactly one external memory
provider. The third-party qdrant provider is disabled by configuration, never deleted, and
its 1423 points stay reachable read-only through search-collections.

The skills stop naming a single host: they are read by both agents now."
```

---

## Auto-revisão

**Cobertura da spec.** §1 objetivo → Tasks 4-9. §2 D1/D2/D3 → Task 4. D4/D5 → Task 10.
D6 → documentado no script da Task 10 (nada a implementar). D7 → Tasks 1-3. D8 → Task 5
(`HERMES_PREFETCH_BUDGET_S`). D9 → Task 9. §3 mapeamento → Tasks 4,5,7,8,9. §3.1 as 15
ferramentas → Task 8. §3.2/§3.3 versão instalada e forward-compat → Task 4 (o teste lê a superfície da ABC
INSTALADA, com `skipUnless`) e Task 5 (`recall_status`). §3.4 `on_session_end` no-op →
Task 4, definido explicitamente com o motivo, não herdado. §4 o que move →
Tasks 1-3. §5 os 4 estados → Tasks 2,5,6. §6 filtro trivial → Task 5. §7 config → Task 9.
§8 symlink → Task 4. §9 testes → um por task, mais Task 6. §10 cutover → Task 10.
§11 fora de escopo → nenhuma task, correto.

**Placeholders.** Nenhum "TBD"/"TODO". Onde o corpo é longo e transcrito (o
`CHECKPOINT_PROCEDURE` na Task 1, as 15 `SCHEMAS` na Task 8, os corpos movidos na Task 2),
o passo nomeia o arquivo e o intervalo exatos da fonte em vez de repetir 60 linhas —
é transcrição verificável, não instrução vaga.

**Consistência de tipos.** `Budget(max_memories, max_chars, max_per_mem, reinject_after)`
igual nas Tasks 2,5,6. `split_by_budget(hits, seen, round_no, budget)` igual nas Tasks 2 e 5.
`recall_block(full_hits, pointers, n_angles, outcome, budget)` igual nas Tasks 2,5,6.
`degradation_note(outcome, max_memories)` igual nas Tasks 2 e 6. `session_state.due(turn,
interval)` igual nas Tasks 3 e 7. `tools.dispatch(name, args, *, cfg)` igual nas Tasks 8 e
`handle_tool_call`. `REPO_ROOT` definido na Task 4 e usado no teste de symlink da mesma task.
`FakeHit` e `BUDGET` definidos na Task 2 e importados de `tests.test_blocks` nas Tasks 5,6,7.

**Contagem de testes.** 258 → 262 (T1) → 275 (T2) → 296 (T3) → 305 (T4) → 312 (T5) →
320 (T6) → 324 (T7) → 337 (T8) → 342 (T9). São estimativas de piso; o que a task exige é
"nunca menor que o passo anterior".
