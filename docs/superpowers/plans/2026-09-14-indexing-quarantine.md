# Quarentena de indexação — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parar o loop de reindexação que satura o servidor de embeddings e derruba o recall
automático, retendo em quarentena os arquivos que não podem ser indexados.

**Architecture:** Um módulo novo `core/quarantine.py` — JSON por repo em
`state_dir/quarantine/`, no mesmo idioma de `core/jobs.py` e `core/lease.py` — registra o
`skipped` que `add_files` **já devolve** e que o `indexer` hoje descarta. O watcher subtrai o
que está retido dos candidatos, fechando o loop. Três correções menores acompanham: o chunker
passa a honrar o próprio `HARD_MAX_CHARS`, a guarda de minificado passa a olhar o arquivo todo,
e `refresh` para de quebrar com `AttributeError` em arquivo pulado.

**Tech Stack:** Python 3, **só stdlib** (restrição do projeto). Testes com `unittest`, offline.

**Spec:** `docs/superpowers/specs/2026-09-14-indexing-quarantine-design.md`

## Global Constraints

- **Só stdlib.** Nenhuma dependência nova, em nenhuma tarefa. Restrição documentada no README.
- **KISS + S.O.L.I.D.**, com o qualificador: simples não pode custar completo. Ver a seção
  "Restrições de engenharia" da spec — vale para todas as tarefas.
- **`quarantine.py` na ordem de grandeza de `core/jobs.py` (237 linhas) e `core/lease.py` (160).**
  Passar muito disso é sinal de responsabilidade que não é dele.
- **O motivo da falha é dado opaco** — string de quem falhou. A quarentena nunca interpreta,
  nunca enumera motivos.
- **Direção da dependência: `indexer → quarantine`.** `quarantine` importa só `knobs.state_dir`
  e `names.safe`. Nunca importa `repos` nem `core`.
- **Falha tolerada não pode virar mentira.** `OSError`/JSON corrompido degrada para "nada
  retido" (o comportamento de hoje), nunca para "está tudo indexado".
- **Comentar o porquê com o número que o justifica**, seguindo o padrão da base de código.
- **Rodar a suíte inteira antes de cada commit:** `python3 -m unittest discover -s tests`
- **Um commit por tarefa.** O dono revisa e faz push; não fazer push.

---

### Task 1: `refresh` para de quebrar em arquivo pulado (defeito D)

Vem primeiro porque a Task 4 depende de ler esse campo, e hoje ele levanta exceção.

`add_files` devolve `skipped` como **tupla** `(path, motivo)` — `core/repos.py:191`, e o CLI lê
assim corretamente em `cli/qctx.py:1318`. Mas `refresh` lê como **dict**, em `core/repos.py:418`:
`out["skipped"][0].get("reason", "unreadable")` → `'tuple' object has no attribute 'get'`. A
exceção sobe até `daemon._run_one`, que marca o job inteiro como FAILED, e os arquivos restantes
daquele refresh nunca são reindexados.

**Files:**
- Modify: `core/repos.py:416-419`
- Test: `tests/test_repos_refresh.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces: `refresh()` passa a devolver de forma confiável entradas
  `{"path": str, "action": "skipped", "reason": str}`. A Task 4 lê exatamente esse shape.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_repos_refresh.py`, no fim do arquivo:

```python
class TestASkippedFileIsReportedNotRaised(unittest.TestCase):
    """`add_files` reports a skip as the TUPLE `(path, reason)` — `cli/qctx.py` reads it that
    way. `refresh` read it as a dict, so the first unindexable file raised AttributeError,
    `daemon._run_one` marked the whole job FAILED, and every remaining changed file in that
    refresh was never reindexed."""

    def test_a_file_that_became_unindexable_is_reported_as_skipped(self):
        ix = an_index("alpha")
        path = a_file("real content here\n")
        ix.add_files("alpha", [path])
        # Emptying it makes `_write_one` raise "nothing indexable", which is the exact
        # condition 20 of the 22 looping files on the user's machine are in.
        rewrite(path, "")
        report = ix.refresh("alpha")
        entry = next(r for r in report if r["path"] == path)
        self.assertEqual(entry["action"], "skipped")
        self.assertIn("nothing indexable", entry["reason"])
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

```bash
python3 -m unittest tests.test_repos_refresh.TestASkippedFileIsReportedNotRaised -v
```

Esperado: FAIL com `AttributeError: 'tuple' object has no attribute 'get'`.

- [ ] **Step 3: Corrigir a leitura**

Em `core/repos.py`, trocar o bloco das linhas 416-419 por:

```python
            if out.get("skipped"):
                # `add_files` reports a skip as the TUPLE `(path, reason)` — the shape
                # `cli/qctx.py` unpacks. Reading it as a dict raised AttributeError here, and
                # `daemon._run_one` turns any raise into a FAILED job: one unindexable file
                # cost every remaining changed file in the same refresh.
                _, why = out["skipped"][0]
                report.append({"path": path, "action": "skipped", "reason": why})
                continue
```

- [ ] **Step 4: Rodar o teste e confirmar que passa**

```bash
python3 -m unittest tests.test_repos_refresh -v
python3 -m unittest discover -s tests
```

Esperado: PASS, suíte inteira verde.

- [ ] **Step 5: Commit**

```bash
git add core/repos.py tests/test_repos_refresh.py
git commit -m "fix: report a skipped file in refresh instead of raising on its shape

add_files reports a skip as the tuple (path, reason) and cli/qctx.py
unpacks it that way; refresh read it as a dict. AttributeError on the
first unindexable file, which daemon._run_one turns into a FAILED job —
so every remaining changed file in that refresh went unindexed."
```

---

### Task 2: O chunker honra o próprio `HARD_MAX_CHARS` (defeito B)

`HARD_MAX_CHARS = 6000` está documentado em `core/chunk.py:16` como invariante: *"o par (query,
chunk) tem que caber no reranker com folga"*. Mas `_window()` (`core/chunk.py:100`) acumula **por
linha** e nunca corta dentro de uma linha. Medido no `api.json` real: linha de 107.648 chars →
chunk de 108.141 chars, 18× o teto, e HTTP 500 no servidor de embeddings
(`input (83086 tokens) is too large to process`).

**Files:**
- Modify: `core/chunk.py:100-115` (`_window`)
- Test: `tests/test_chunk.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces: `chunk_text(content)` passa a garantir `len(c.text) <= hard_max` para todo chunk.
  Nenhuma mudança de assinatura.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_chunk.py`, no fim do arquivo:

```python
class TestTheHardMaxIsActuallyHonoured(unittest.TestCase):
    """HARD_MAX_CHARS is documented as the ceiling that keeps a (query, chunk) pair inside the
    reranker. `_window` accumulated whole LINES and never cut inside one, so a single long line
    became a chunk 18x the ceiling: measured on a generated OpenAPI client, a 107,648-char line
    produced a 108,141-char chunk, and the embedding server answered HTTP 500 —
    "input (83086 tokens) is too large to process"."""

    def test_a_single_enormous_line_is_split_to_respect_the_ceiling(self):
        content = "x" * 107_648 + "\n"
        chunks = chunk_text(content)
        self.assertTrue(chunks, "an enormous line produced no chunk at all")
        longest = max(len(c.text) for c in chunks)
        self.assertLessEqual(longest, HARD_MAX_CHARS,
                             f"a chunk of {longest} chars exceeds the documented ceiling")

    def test_nothing_of_the_long_line_is_lost_in_the_split(self):
        content = "y" * 20_000 + "\n"
        rebuilt = "".join(c.text for c in chunk_text(content))
        self.assertEqual(rebuilt.count("y"), 20_000, "the split dropped or duplicated content")

    def test_a_long_line_among_normal_ones_keeps_the_normal_ones_intact(self):
        content = "def a():\n    return 1\n" + "z" * 50_000 + "\ndef b():\n    return 2\n"
        chunks = chunk_text(content)
        self.assertLessEqual(max(len(c.text) for c in chunks), HARD_MAX_CHARS)
        self.assertTrue(any("def a()" in c.text for c in chunks))
        self.assertTrue(any("def b()" in c.text for c in chunks))
```

Acrescentar `HARD_MAX_CHARS` ao import no topo do arquivo:

```python
from core.chunk import (Chunk, HARD_MAX_CHARS, chunk_text, is_probably_binary,
                        mode_for_suffix, pack_chunks, split_blocks)
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

```bash
python3 -m unittest tests.test_chunk.TestTheHardMaxIsActuallyHonoured -v
```

Esperado: FAIL — `108141 exceeds the documented ceiling` no primeiro teste.

- [ ] **Step 3: Fatiar a linha longa dentro de `_window`**

Em `core/chunk.py`, substituir `_window` inteira (linhas 100-115) por:

```python
def _slice_long_line(line: str, target: int) -> list[str]:
    """A single line longer than `target`, cut into pieces of at most `target` chars.

    WHY THIS IS NEEDED AT ALL. `_window` accumulates whole lines, so a line longer than the
    ceiling produced a chunk as long as the line — `HARD_MAX_CHARS` was a promise the code
    broke. Measured on a generated OpenAPI client: a 107,648-char line became a 108,141-char
    chunk, 18x the ceiling, and the embedding endpoint refused it with HTTP 500
    ("input (83086 tokens) is too large to process"). Generated JSON, minified bundles and
    wide CSV rows all reach this shape.
    """
    return [line[i:i + target] for i in range(0, len(line), target)]


def _window(lines: list[str], start: int, end: int, target: int) -> list[tuple[int, int]]:
    """Fixed window with overlap, for a block that overflows the ceiling on its own."""
    windows = []
    step = start
    while step < end:
        accumulated = 0
        cursor = step
        while cursor < end and accumulated < target:
            accumulated += len(lines[cursor])
            cursor += 1
        windows.append((step, cursor))
        if cursor >= end:
            break
        step = max(step + 1, cursor - OVERLAP_LINES)

    return windows
```

E em `pack_chunks`, dentro do ramo `if block > hard_max:`, tratar o caso da linha única longa
antes de janelar. O bloco passa a ser:

```python
        if block > hard_max:
            if open_start is not None:
                emit(open_start, bi)
                open_start, size = None, 0
            # A block of ONE line longer than the ceiling cannot be split by any window over
            # lines — the cut has to happen inside the line itself. Emitted directly, with the
            # line's own number on every piece: `mode_for_suffix` marks these suffixes
            # `locator`, and reading that whole line back is the correct behaviour there.
            if bf - bi == 1 and len(lines[bi]) > hard_max:
                for piece in _slice_long_line(lines[bi].strip("\n"), target):
                    if piece.strip():
                        chunks.append(Chunk(bi + 1, bf, piece))
                continue
            for ji, jf in _window(lines, bi, bf, target):
                emit(ji, jf)
            continue
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

```bash
python3 -m unittest tests.test_chunk -v
python3 -m unittest discover -s tests
```

Esperado: PASS, suíte inteira verde.

- [ ] **Step 5: Verificar contra o arquivo real que motivou isto**

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from core.chunk import chunk_text, HARD_MAX_CHARS
c = open('/home/me/projects/awesome-cv3/connectors/ibm-tririga-source/src/clients/ibm-tririga-source/api.json').read()
ch = chunk_text(c)
print('chunks:', len(ch), '| maior:', max(len(x.text) for x in ch), '| teto:', HARD_MAX_CHARS)
assert max(len(x.text) for x in ch) <= HARD_MAX_CHARS
print('OK')
"
```

Esperado: o maior chunk ≤ 6000, `OK` impresso.

- [ ] **Step 6: Commit**

```bash
git add core/chunk.py tests/test_chunk.py
git commit -m "fix: cut inside a long line, so HARD_MAX_CHARS is a ceiling and not a wish

_window accumulated whole lines and never cut inside one, so a single
long line became a chunk as long as the line. Measured on a generated
OpenAPI client: a 107,648-char line produced a 108,141-char chunk, 18x
the documented ceiling, which the embedding endpoint refused with
HTTP 500 (input 83086 tokens is too large to process)."
```

---

### Task 3: A guarda de minificado olha o arquivo todo (defeito C)

`scan._sniff()` decide "minificado" lendo só os primeiros 8192 bytes (`core/scan.py:34`). No
`api.json` a primeira linha acima de 2.000 chars começa no **byte 16.687** — passa batido pela
guarda que existe exatamente para barrá-la. Maior linha nos primeiros 8 KB: 722 chars.

**Files:**
- Modify: `core/scan.py:104-127` (`_sniff`)
- Test: `tests/test_scan.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces: `scan.eligible()` passa a classificar como `minified` arquivos cuja linha longa
  esteja em qualquer posição. Nenhuma mudança de assinatura.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_scan.py`, no fim do arquivo:

```python
class TestMinifiedDetectionSeesPastTheFirstBytes(unittest.TestCase):
    """The sniff read only the first 8192 bytes, so a long line further in was invisible.
    Measured on a generated OpenAPI client: the longest line in the first 8 KB was 722 chars,
    while the real one — 107,648 chars — started at byte 16,687. It passed the very guard
    written to stop it, and then failed at the embedding endpoint instead."""

    def test_a_long_line_starting_after_the_first_8kb_is_detected(self):
        padding = "".join(f"line {i} with ordinary content here\n" for i in range(600))
        self.assertGreater(len(padding), 8192, "the padding must push the long line past 8 KB")
        root = a_repo(**{"generated.json": padding + "x" * 107_648 + "\n"})
        found = scan.eligible(root)
        self.assertEqual(found["eligible"], [],
                         "a file whose long line sits past the sniff window was let through")
        self.assertEqual(found["skipped"]["minified"], 1)

    def test_an_ordinary_file_of_the_same_size_is_still_eligible(self):
        root = a_repo(**{"normal.py": "".join(f"x = {i}\n" for i in range(4000))})
        self.assertEqual(len(scan.eligible(root)["eligible"]), 1,
                         "scanning the whole file must not start refusing ordinary ones")
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

```bash
python3 -m unittest tests.test_scan.TestMinifiedDetectionSeesPastTheFirstBytes -v
```

Esperado: FAIL no primeiro teste — o arquivo aparece em `eligible`.

- [ ] **Step 3: Varrer o arquivo inteiro**

Em `core/scan.py`, substituir `_sniff` (linhas 104-127) por:

```python
def _sniff(path: str) -> str | None:
    """`"binary"`, `"minified"`, `"unreadable"` or None.

    A NUL byte is the same test `_read_source` uses, applied earlier so an image is not read in
    full only to be refused — decided from the HEAD alone, because a binary file announces
    itself immediately.

    THE LINE LENGTH IS MEASURED OVER THE WHOLE FILE, and the head is not enough. Measured on a
    generated OpenAPI client: the longest line in the first 8 KB was 722 chars while the real
    one — 107,648 chars — started at byte 16,687, so the file passed this very guard and failed
    at the embedding endpoint instead (HTTP 500, "input is too large to process"). The read is
    bounded by `MAX_FILE_BYTES` (1 MB), already applied by the caller before this runs, and it
    happens on the eligibility pass — never on the watcher's cycle, which compares mtime/size.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SNIFF_BYTES)
            if b"\0" in head:
                return "binary"
            longest = carry = 0
            buffer = head
            while buffer:
                segments = buffer.split(b"\n")
                # The last segment may continue into the next read, so it is carried rather
                # than measured — every other one is a complete line.
                for segment in segments[:-1]:
                    longest = max(longest, carry + len(segment))
                    carry = 0
                carry += len(segments[-1])
                longest = max(longest, carry)
                buffer = fh.read(_SNIFF_BYTES)
    except OSError:
        return "unreadable"
    if longest > MINIFIED_LINE_CHARS:
        return "minified"

    return None
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

```bash
python3 -m unittest tests.test_scan -v
python3 -m unittest discover -s tests
```

Esperado: PASS, suíte inteira verde.

- [ ] **Step 5: Verificar contra os dois arquivos reais**

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from core.scan import _sniff
for p in ['/home/me/projects/awesome-cv3/connectors/ibm-tririga-source/src/clients/ibm-tririga-source/api.json',
          '/home/me/projects/awesome-cv3/connectors/utsw-moveit-occupancy-source/src/clients/utsw-moveit-occupancy-source/api.json']:
    print(_sniff(p), p.split('/')[-3])
"
```

Esperado: `minified` para os dois.

- [ ] **Step 6: Commit**

```bash
git add core/scan.py tests/test_scan.py
git commit -m "fix: measure the longest line over the whole file, not the first 8 KB

The sniff read 8192 bytes, so a long line further in was invisible to
the guard written to catch it. Measured on a generated OpenAPI client:
722 chars was the longest line in the first 8 KB, while the real one —
107,648 chars — started at byte 16,687. It was admitted here and failed
at the embedding endpoint instead."
```

---

### Task 4: O módulo de quarentena (defeito A, parte 1 de 2)

O módulo puro, sem nenhum consumidor ainda. Segue o idioma de `core/jobs.py` e `core/lease.py`:
JSON por repo, escrita atômica via `os.replace`, `OSError` tolerado, sem protocolo entre
processos.

**Files:**
- Create: `core/quarantine.py`
- Test: `tests/test_quarantine.py`

**Interfaces:**
- Consumes: `knobs.state_dir()` e `names.safe()` — e nada mais.
- Produces, usado pela Task 5:
  - `dir() -> Path`
  - `load(repo: str) -> dict`
  - `record(repo: str, path: str, reason: str) -> None`
  - `held(repo: str) -> set[str]`
  - `clear(repo: str, paths) -> None`

- [ ] **Step 1: Escrever os testes que falham**

Criar `tests/test_quarantine.py`:

```python
"""Files that can never be indexed, remembered so they are not tried forever.

WHAT THIS PREVENTS, measured on 2026-09-14: the watcher re-queued the same 22 files every ~40 s
indefinitely, because a file that fails to index is a file the archive has no chunk for, and the
next poll reports it as missing all over again. The resulting load on the shared embedding
endpoint pushed automatic recall from 0.04 s to 1.98 s against a 2.00 s ceiling, so the user saw
"[automatic recall — UNAVAILABLE]" with no visible connection to indexing.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import quarantine  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


def a_file(text: str = "content\n") -> str:
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


class TestHoldingAndReleasing(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_recorded_path_is_held(self):
        path = a_file()
        quarantine.record("alpha", path, "nothing indexable")
        self.assertIn(path, quarantine.held("alpha"))

    def test_an_unknown_repo_holds_nothing(self):
        self.assertEqual(quarantine.held("never-seen"), set())

    def test_the_reason_is_kept_verbatim_for_the_user_to_read(self):
        path = a_file()
        quarantine.record("alpha", path, "HTTP 500: input (83086 tokens) is too large")
        self.assertIn("83086", quarantine.load("alpha")[path]["reason"])

    def test_a_repo_does_not_see_another_repos_quarantine(self):
        path = a_file()
        quarantine.record("alpha", path, "nothing indexable")
        self.assertEqual(quarantine.held("beta"), set())

    def test_clear_releases_a_path(self):
        path = a_file()
        quarantine.record("alpha", path, "nothing indexable")
        quarantine.clear("alpha", [path])
        self.assertEqual(quarantine.held("alpha"), set())

    def test_clearing_a_path_that_was_never_held_is_not_an_error(self):
        quarantine.clear("alpha", ["/nonexistent/never-recorded.py"])
        self.assertEqual(quarantine.held("alpha"), set())


class TestChangedContentIsRetried(unittest.TestCase):
    """The quarantine describes a CONTENT, not a path. This is what makes a 0-byte file that
    later gains content come back on its own, with no manual command — which is the state 20 of
    the 22 looping files were in."""

    def setUp(self):
        a_state_dir()

    def test_a_file_whose_size_changed_is_no_longer_held(self):
        path = a_file("")
        quarantine.record("alpha", path, "nothing indexable")
        with open(path, "w") as fh:
            fh.write("it has content now\n")
        self.assertNotIn(path, quarantine.held("alpha"))

    def test_a_file_that_did_not_change_stays_held(self):
        path = a_file("same content\n")
        quarantine.record("alpha", path, "some failure")
        self.assertIn(path, quarantine.held("alpha"))

    def test_a_file_that_vanished_is_not_held(self):
        path = a_file()
        quarantine.record("alpha", path, "some failure")
        os.unlink(path)
        self.assertNotIn(path, quarantine.held("alpha"),
                         "a path that is gone must not be held by a stale entry")


class TestFailureDegradesToHoldingNothing(unittest.TestCase):
    """The rule this project already follows: a tolerated failure must never become a LIE.
    Holding nothing means re-indexing something we might have skipped — today's behaviour.
    Claiming everything is indexed would be the lie."""

    def setUp(self):
        a_state_dir()

    def test_corrupt_json_reads_as_nothing_held(self):
        quarantine.dir().mkdir(parents=True, exist_ok=True)
        (quarantine.dir() / "alpha.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(quarantine.held("alpha"), set())

    def test_an_unwritable_state_dir_does_not_raise(self):
        with mock.patch("core.quarantine._write", return_value=False):
            quarantine.record("alpha", a_file(), "some failure")   # must not raise

    def test_an_oserror_while_reading_does_not_raise(self):
        with mock.patch("pathlib.Path.read_text", side_effect=OSError("boom")):
            self.assertEqual(quarantine.held("alpha"), set())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rodar os testes e confirmar que falham**

```bash
python3 -m unittest tests.test_quarantine -v
```

Esperado: FAIL com `ModuleNotFoundError: No module named 'core.quarantine'`.

- [ ] **Step 3: Escrever o módulo**

Criar `core/quarantine.py`:

```python
"""Files that could not be indexed, remembered so they are not tried forever.

WHY THIS EXISTS. A file that fails to index is a file the archive has no chunk for, so the next
poll reports it as missing and queues it again — forever. Measured on 2026-09-14: the same 22
files were re-queued every ~40 s indefinitely, and the load that put on the shared embedding
endpoint pushed automatic recall from 0.04 s to 1.98 s against its 2.00 s ceiling. The user saw
"[automatic recall — UNAVAILABLE]", which names nothing about indexing.

IT REMEMBERS A CONTENT, NOT A PATH. Every entry carries the `mtime` and `size` the file had when
it failed, and `held` returns only the paths that still match. A 0-byte file that later gains
content is therefore retried on its own, with no command to run — which is the state 20 of those
22 files were in. A quarantine keyed by path alone would need a human to clear it, and nothing
would ever tell them to.

THE REASON IS OPAQUE DATA. It is whatever the failing layer said, stored verbatim and shown to
the user. This module never enumerates reasons: a new failure mode — a new server error, a new
format — must not require a change here.

SAME IDIOM AS `jobs.py` AND `lease.py`: a JSON file per repository, written atomically, with
OSError tolerated. There is no protocol between processes; the daemon writes, the CLI reads.
"""
import json
import os
import time
from pathlib import Path

from . import names
from .knobs import state_dir


def dir() -> Path:                                  # noqa: A001 — the name says what it holds
    return state_dir() / "quarantine"


def load(repo: str) -> dict:
    """`{path: {reason, mtime, size, at}}`, or `{}` when there is nothing readable.

    Corrupt or unreadable state reads as EMPTY, never raises: holding nothing means retrying a
    file we could have skipped, which is exactly today's behaviour. The opposite failure —
    claiming a file is handled when nothing knows that — is the one this plugin refuses.
    """
    try:
        found = json.loads(_path(repo).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    return found if isinstance(found, dict) else {}


def record(repo: str, path: str, reason: str) -> None:
    """Remembers that `path` could not be indexed, with the content it had when it failed.

    A path that cannot be `stat`ed is NOT recorded: with no mtime/size there is nothing to
    compare later, so the entry could never be released and would be a permanent exclusion
    written by a transient error.
    """
    try:
        st = os.stat(path)
    except OSError:
        return
    entry = load(repo)
    entry[str(path)] = {"reason": str(reason), "mtime": st.st_mtime, "size": st.st_size,
                        "at": time.time()}
    _write(repo, entry)


def held(repo: str) -> set[str]:
    """The paths whose content STILL matches what failed — the ones to skip.

    An entry whose file changed, or vanished, is not held and is dropped from disk on the way
    through: the next attempt is the point, and a stale entry that no longer describes anything
    is just noise in what the user reads.
    """
    entry = load(repo)
    still, changed = set(), False
    for path, meta in list(entry.items()):
        try:
            st = os.stat(path)
        except OSError:
            del entry[path]
            changed = True
            continue
        if st.st_mtime == meta.get("mtime") and st.st_size == meta.get("size"):
            still.add(path)
            continue
        del entry[path]
        changed = True
    if changed:
        _write(repo, entry)

    return still


def clear(repo: str, paths) -> None:
    """Forgets `paths`. Clearing something never held is not an error — the caller indexes a
    batch and releases all of it, without having to know which members had failed before."""
    entry = load(repo)
    removed = False
    for path in paths:
        if entry.pop(str(path), None) is not None:
            removed = True
    if removed:
        _write(repo, entry)


def _path(repo: str) -> Path:
    return dir() / f"{names.safe(repo)}.json"


def _write(repo: str, entry: dict) -> bool:
    """Writes atomically, the way `jobs._write` does. False on failure, never raises: state
    that cannot be written is state the reader will not find, which every caller handles."""
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = _path(repo)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

        return True
    except OSError:
        return False
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

```bash
python3 -m unittest tests.test_quarantine -v
python3 -m unittest discover -s tests
```

Esperado: PASS (12 testes), suíte inteira verde.

- [ ] **Step 5: Commit**

```bash
git add core/quarantine.py tests/test_quarantine.py
git commit -m "feat: remember which files could not be indexed, keyed by their content

A file that fails to index has no chunk in the archive, so the next poll
reports it missing and queues it again — forever. Measured: the same 22
files re-queued every ~40 s, and that load pushed automatic recall from
0.04 s to 1.98 s against a 2.00 s ceiling.

Entries carry the mtime and size the file had when it failed, so a
0-byte file that later gains content is retried on its own — no command
to run, which matters because nothing would ever tell the user to run it."
```

---

### Task 5: O watcher para de reenfileirar o que já falhou (defeito A, parte 2 de 2)

Liga a quarentena ao `indexer`. É este passo que fecha o loop: `indexer.work()` passa a **ler** o
retorno de `add_files`, que hoje descarta em `core/indexer.py:45`.

**Files:**
- Modify: `core/indexer.py` (`work`, `watcher`/`_new_tracked_paths`)
- Test: `tests/test_watch.py`

**Interfaces:**
- Consumes: `quarantine.record()`, `quarantine.held()`, `quarantine.clear()` da Task 4;
  `refresh()` devolvendo `{"path", "action": "skipped", "reason"}` da Task 1.
- Produces: comportamento. Nenhuma assinatura pública nova.

- [ ] **Step 1: Dar ao `FakeIndex` a capacidade de falhar**

Em `tests/test_watch.py`, substituir `FakeIndex.__init__`, `add_files` e `refresh` por:

```python
    def __init__(self, changed=(), checkouts=("/nonexistent/alpha",), indexed=(), fails=()):
        self._changed = list(changed)
        self._checkouts = list(checkouts)
        self._indexed = set(indexed)
        # LISKOV: a fake that only knows how to SUCCEED is not a substitute for the real
        # index, and that gap is exactly what let the re-queue loop through review. `fails`
        # maps a path to the reason `add_files` reports for it, the way the real one does.
        self._fails = dict(fails)
        self.refreshed = []
        self.indexed_calls = []
        self.added = []

    def add_files(self, repo, paths, **kwargs):
        skipped = [(p, self._fails[p]) for p in paths if p in self._fails]
        stored = [p for p in paths if p not in self._fails]
        self.added.extend(paths)
        self._indexed.update(stored)

        return {"repo": repo, "files": len(stored), "chunks": len(stored), "skipped": skipped}

    def refresh(self, repo, should_stop=None):
        self.refreshed.append(repo)
        report = []
        for path in self._changed:
            if path in self._fails:
                report.append({"path": path, "action": "skipped",
                               "reason": self._fails[path]})
                continue
            report.append({"path": path, "action": "reindexed", "chunks": 1})

        return report
```

- [ ] **Step 2: Escrever os testes que falham**

Em `tests/test_watch.py`, no fim do arquivo:

```python
class TestAFileThatCannotBeIndexedIsNotRetriedForever(unittest.TestCase):
    """Measured on 2026-09-14: the same 22 files were re-queued every ~40 s indefinitely,
    because `work()` discarded the `skipped` report `add_files` already returns. The load that
    put on the shared embedding endpoint pushed automatic recall from 0.04 s to 1.98 s against
    its 2.00 s ceiling, and the user saw UNAVAILABLE blocks naming nothing about indexing."""

    def setUp(self):
        a_state_dir()

    def test_a_file_that_fails_to_index_is_not_enqueued_again(self):
        root = a_git_repo()
        doomed = track(root, "empty.json", text="")
        doomed = os.path.abspath(doomed)
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set(),
                       fails={doomed: "nothing indexable (empty file, or whitespace only)"})
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)

        watch(); watch()                       # debounce: enqueued on the second sighting
        job = jobs.load("alpha")
        self.assertIsNotNone(job, "the new file was never enqueued in the first place")
        run_job(job)                           # the daemon runs it; add_files reports a skip

        jobs.update("alpha", state=jobs.DONE)
        watch(); watch()
        self.assertIsNone(jobs.load("alpha"),
                          "a file that can never be indexed was queued all over again")

    def test_a_file_that_succeeds_is_not_quarantined(self):
        root = a_git_repo()
        good = os.path.abspath(track(root, "fine.py"))
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set())
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)
        watch(); watch()
        run_job(jobs.load("alpha"))
        self.assertEqual(quarantine.held("alpha"), set())

    def test_the_reason_is_kept_so_the_user_can_read_it(self):
        root = a_git_repo()
        doomed = os.path.abspath(track(root, "huge.json", text="x"))
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set(),
                       fails={doomed: "HTTP 500: input (83086 tokens) is too large"})
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)
        watch(); watch()
        run_job(jobs.load("alpha"))
        self.assertIn("83086", quarantine.load("alpha")[doomed]["reason"])

    def test_a_refresh_that_skips_a_file_also_quarantines_it(self):
        """The `refresh` path needs this as much as `index`: a file that changed and fails to
        re-embed stays in `changed` forever, so it is queued on every cycle."""
        doomed = "/nonexistent/alpha/broken.py"
        ix = FakeIndex(changed=[doomed], fails={doomed: "nothing indexable"})
        run_job = indexer.work(index=ix)
        run_job({"repo": "alpha", "kind": "refresh", "paths": []})
        self.assertIn("nothing indexable", quarantine.load("alpha")[doomed]["reason"])


class TestQuarantineReleasesWhenTheContentChanges(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_quarantined_file_is_queued_again_once_it_changes(self):
        root = a_git_repo()
        path = os.path.abspath(track(root, "was_empty.py", text=""))
        ix = FakeIndex(changed=[], checkouts=[root], indexed=set(),
                       fails={path: "nothing indexable (empty file, or whitespace only)"})
        watch = indexer.watcher(index=ix)
        run_job = indexer.work(index=ix)
        watch(); watch()
        run_job(jobs.load("alpha"))
        jobs.update("alpha", state=jobs.DONE)

        with open(path, "w") as fh:            # the file gains content
            fh.write("def real_code():\n    return 1\n")
        ix._fails.clear()                      # and now it indexes fine

        watch(); watch()
        job = jobs.load("alpha")
        self.assertIsNotNone(job, "a file that gained content was never retried")
        self.assertIn(path, job["paths"])
```

Acrescentar `quarantine` ao import no topo do arquivo, e o parâmetro `text` em `track`:

```python
from core import indexer, jobs, quarantine  # noqa: E402
```

A função `track` já aceita `text` como terceiro parâmetro — os testes acima chamam com
`text=""`, que é compatível com a assinatura atual `track(root, name, text="x = 1\n")`.

- [ ] **Step 3: Rodar os testes e confirmar que falham**

```bash
python3 -m unittest tests.test_watch.TestAFileThatCannotBeIndexedIsNotRetriedForever -v
```

Esperado: FAIL — `a file that can never be indexed was queued all over again`.

- [ ] **Step 4: Ligar a quarentena ao indexer**

Em `core/indexer.py`, trocar o import do topo:

```python
from . import jobs, quarantine, scan
```

Substituir `run_job` inteira por:

```python
    def run_job(job: dict) -> None:
        target = index if index is not None else _build(cfg)
        repo = job["repo"]
        if job.get("kind") == "refresh":
            report = target.refresh(repo, should_stop=lambda: jobs.cancel_requested(repo))
            # A file that changed and then fails to re-embed stays in `changed` forever, so it
            # is queued again on every cycle — the same loop the `index` path had, reached by
            # the other door. `missing` is deliberately NOT quarantined: a file that is gone
            # costs no embedding, and `refresh` reports it on purpose for the user to see.
            for item in report or ():
                if item.get("action") == "skipped":
                    quarantine.record(repo, item["path"], item.get("reason", "unindexable"))

            return
        paths = list(job.get("paths") or [])
        done = 0
        for start in range(0, len(paths), batch):
            if jobs.cancel_requested(repo):
                return
            chunk = paths[start:start + batch]
            # THE RETURN VALUE IS THE POINT. `add_files` has always reported which paths it
            # skipped and why; discarding it here is what made the watcher queue the same
            # unindexable files every ~40 s forever, saturating the embedding endpoint that
            # automatic recall shares (measured: 0.04 s idle, 1.98 s during a batch, against a
            # 2.00 s ceiling).
            out = target.add_files(repo, chunk) or {}
            skipped = dict(out.get("skipped") or ())
            for path, why in skipped.items():
                quarantine.record(repo, path, why)
            # Anything that went in is released, so a file repaired between two runs stops
            # being held without anyone having to say so.
            quarantine.clear(repo, [p for p in chunk if p not in skipped])
            done += len(chunk)
            jobs.update(repo, only_if=job.get("id"), done=done, current=chunk[-1])
```

E em `_new_tracked_paths`, trocar a linha final `return {p for p in eligible if p not in indexed}`
por:

```python
    # Held paths are dropped here rather than at the enqueue site because this is the function
    # that decides what COUNTS as new — a file the archive will never accept is not new, it is
    # known-bad. Leaving the filter to the caller would put the loop back the moment a second
    # caller appeared.
    return {p for p in eligible if p not in indexed and p not in quarantine.held(repo)}
```

- [ ] **Step 5: Rodar os testes e confirmar que passam**

```bash
python3 -m unittest tests.test_watch -v
python3 -m unittest discover -s tests
```

Esperado: PASS, suíte inteira verde.

- [ ] **Step 6: Commit**

```bash
git add core/indexer.py tests/test_watch.py
git commit -m "fix: stop re-queueing files the archive can never accept

work() discarded the return value of add_files, which has always
reported which paths were skipped and why. With no record of the
failure, the next poll saw a file with no chunks and queued it again —
measured, the same 22 files every ~40 s, indefinitely.

The load that put on the shared embedding endpoint pushed automatic
recall from 0.04 s to 1.98 s against its 2.00 s ceiling, so the symptom
reached the user as UNAVAILABLE memory, naming nothing about indexing.

FakeIndex could only succeed, which is why no test caught this; it can
now fail the way the real index fails."
```

---

### Task 6: A quarentena fica visível no `status`

Sem isso a correção é silenciosa, e um arquivo retido por engano nunca seria descoberto.

**Files:**
- Modify: `cli/qctx.py` (`cmd_repos_status`, a partir da linha 1473)
- Test: `tests/test_cli_repos.py`

**Interfaces:**
- Consumes: `quarantine.load()` da Task 4.
- Produces: saída no terminal e a chave `"quarantine"` no `--json`.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_cli_repos.py`, no fim do arquivo. Usa o `CLICase` que já existe neste arquivo
(com seu helper `self.rendered`) — é o harness onde `cmd_repos_status` já é testado, em
`TestStatusReapsBeforeRendering`; `tests/test_cli_daemon.py` não importa o CLI e não serve:

```python
class TestStatusShowsWhatIsQuarantined(CLICase):
    """A silent fix is one nobody can audit: a file held by mistake would never be found.

    It is printed even when every job reads `done`, because that is exactly the state a held
    file produces — the job succeeded, and some of its files were skipped on purpose."""

    def test_status_reports_the_held_files_with_their_reason(self):
        from core import jobs, quarantine

        fd, path = tempfile.mkstemp()
        os.close(fd)
        jobs.enqueue("alpha", "index", [path])
        jobs.update("alpha", state=jobs.DONE)
        quarantine.record("alpha", path, "HTTP 500: input (83086 tokens) is too large")
        text = self.rendered(self.cli.cmd_repos_status)
        self.assertIn("quarantine", text.lower())
        self.assertIn("83086", text, "the reason must reach the user, not just the count")

    def test_the_json_form_carries_the_quarantine_too(self):
        from core import jobs, quarantine

        fd, path = tempfile.mkstemp()
        os.close(fd)
        jobs.enqueue("alpha", "index", [path])
        quarantine.record("alpha", path, "nothing indexable")
        payload = json.loads(self.rendered(self.cli.cmd_repos_status, json=True))
        self.assertIn(path, payload["quarantine"]["alpha"])

    def test_nothing_held_prints_no_quarantine_section(self):
        from core import jobs

        jobs.enqueue("alpha", "index", ["/a.py"])
        self.assertNotIn("quarantine", self.rendered(self.cli.cmd_repos_status).lower(),
                         "an empty quarantine must not add noise to every status")
```

`os`, `json`, `tempfile` e `unittest.mock` já estão importados no topo deste arquivo — nada a
acrescentar.

- [ ] **Step 2: Rodar o teste e confirmar que falha**

```bash
python3 -m unittest tests.test_cli_repos.TestStatusShowsWhatIsQuarantined -v
```

Esperado: FAIL — `'quarantine' not found in ...`.

- [ ] **Step 3: Imprimir a quarentena no status**

Em `cli/qctx.py`, dentro de `cmd_repos_status`, trocar o import do topo da função:

```python
    from core import daemon, jobs, lease, quarantine
```

No ramo `--json`, incluir a quarentena:

```python
    held = {r: quarantine.load(r) for r in {j["repo"] for j in jobs.all_jobs()}}
    held = {r: v for r, v in held.items() if v}
    if args.json:
        output({"daemon": running, "jobs": rows, "leases": lease.live(),
                "quarantine": held}, True)

        return
```

E, depois do laço que imprime os jobs, acrescentar:

```python
    # Printed even when every job reads `done`, because that is exactly the state a held file
    # produces: the job succeeded, and some of its files were skipped on purpose.
    for repo, entries in sorted(held.items()):
        print(f"  {repo}: {len(entries)} file(s) in quarantine, not being retried")
        for path, meta in sorted(entries.items()):
            print(f"      {path}: {meta.get('reason', '')[:100]}")
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

```bash
python3 -m unittest tests.test_cli_repos -v
python3 -m unittest discover -s tests
```

Esperado: PASS, suíte inteira verde.

- [ ] **Step 5: Commit**

```bash
git add cli/qctx.py tests/test_cli_repos.py
git commit -m "feat: show what is held in quarantine, with the reason, in repos status

A silent skip is one nobody can audit: a file held by mistake would
never be found. Printed even when every job reads done, because that is
exactly the state a held file produces — the job succeeded and some of
its files were deliberately skipped."
```

---

### Task 7: Verificação em produção

Nada de código. É a prova de que os quatro defeitos saíram do caminho na máquina real — o padrão
que o usuário exige: observação mensurável, não impressão de melhora.

**Files:** nenhum.

- [ ] **Step 1: Suíte inteira verde**

```bash
cd /home/me/memories-plugin && python3 -m unittest discover -s tests
```

- [ ] **Step 2: Sincronizar o plugin instalado e reiniciar o daemon**

```bash
cp -r core cli hooks hosts $HERMES_HOME/plugins/memories/
find $HERMES_HOME/plugins/memories -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
python3 -c "
import sys; sys.path.insert(0,'$HERMES_HOME/plugins/memories')
from core import daemon
print('stopped:', daemon.stop())
"
```

- [ ] **Step 3: Confirmar que os jobs param de rotacionar**

```bash
cd ~/.memories-plugin/state/jobs && for i in 1 2 3 4 5 6; do
  date +%H:%M:%S
  python3 -c "
import json,glob
for f in sorted(glob.glob('*.json')):
    d=json.load(open(f))
    print('   ', f[:28].ljust(28), d.get('state'), d.get('done'),'/',d.get('total'))
"
  sleep 20
done
```

Esperado: os mesmos jobs **não** voltam a `pending`. Antes da correção, voltavam a cada ~40 s.

- [ ] **Step 4: Confirmar que o embed não tem mais picos**

```bash
python3 - <<'EOF'
import json, time, urllib.request
U="http://127.0.0.1:8003/v1/embeddings"
worst=0
for i in range(40):
    body=json.dumps({"model":"bge-m3","input":["sonda de latencia"]}).encode()
    req=urllib.request.Request(U, data=body, headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=10) as r: r.read()
    dt=time.time()-t0
    worst=max(worst,dt)
    time.sleep(2)
print(f"pior latencia em 80s: {worst:.2f}s  (teto do recall: 2.00s)")
EOF
```

Esperado: pior caso bem abaixo de 0,5 s. Antes: picos de 1,0-1,3 s a cada ~40 s.

- [ ] **Step 5: Confirmar que a quarentena reteve os 22 arquivos, com motivo**

```bash
cd /home/me/memories-plugin && python3 cli/qctx.py repos status
```

Esperado: os arquivos vazios listados com `nothing indexable`. Os dois `api.json` **não** devem
aparecer — passam a ser recusados antes, como `minified` (Task 3).

- [ ] **Step 6: Confirmar que nenhum bloco UNAVAILABLE novo aparece**

```bash
cd $HERMES_HOME && python3 -c "
import sqlite3, datetime
db = sqlite3.connect('file:state.db?mode=ro', uri=True)
cur = db.cursor()
cur.execute(\"SELECT MAX(timestamp) FROM messages WHERE api_content LIKE '%UNAVAILABLE for this prompt%'\")
ts = cur.fetchone()[0]
print('ultimo UNAVAILABLE:', datetime.datetime.fromtimestamp(ts) if ts else 'nenhum')
"
```

Esperado: o timestamp deve ser **anterior** ao restart do daemon, e não avançar depois dele.

---

## Self-review

**1. Cobertura da spec.** Os quatro defeitos têm tarefa: A → Tasks 4+5, B → Task 2, C → Task 3,
D → Task 1. As três camadas da spec estão cobertas (camada 1 → Tasks 4+5, camada 2 → Task 2,
camada 3 → Task 3). Os 7 testes nomeados na tabela da spec aparecem: A1/A2 (Task 5), A3 (Tasks 4
e 5), A4 (Task 4), B1 (Task 2), C1 (Task 3), D1 (Task 1). A visibilidade no status, decidida na
tabela "O que se decidiu", é a Task 6. As 4 observações da seção "Verificação em produção" da
spec são os Steps 3-6 da Task 7. A decisão de não mexer em `EMBED_BATCH` é respeitada: nenhuma
tarefa o toca.

**2. Placeholders.** Nenhum "TBD"/"TODO"/"tratar erros apropriadamente". Todo passo de código
traz o código; todo passo de teste traz a asserção.

**3. Consistência de tipos.** `quarantine.record(repo, path, reason)`, `held(repo) -> set`,
`load(repo) -> dict`, `clear(repo, paths)` e `dir() -> Path` são definidos na Task 4 e usados com
exatamente esses nomes e aridades nas Tasks 5 e 6. O shape `{"path", "action", "reason"}` que a
Task 1 garante é o que a Task 5 lê. O `skipped` como lista de tuplas `(path, reason)` é
consistente entre `core/repos.py:191` (real), o `FakeIndex` da Task 5 e o consumo em
`indexer.work`. `_write` é o nome mockado no teste da Task 4 e o nome definido no módulo.
