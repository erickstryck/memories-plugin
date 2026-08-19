# Daemon de indexação de repositório — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** indexar um repositório inteiro em segundo plano — cancelável, com status legível — e
manter o índice em dia sozinho enquanto o host estiver vivo.

**Architecture:** um daemon por usuário, iniciado sob demanda. Todo o estado é JSON em
`~/.memories-plugin/state/`, então `status` lê arquivo em vez de falar com o processo. O daemon
encerra quando nenhum lease de host continua vivo, e o teste de vida é `(pid, starttime)` — cobre
saída limpa e `kill -9` igualmente.

**Tech Stack:** Python 3, só stdlib. `git ls-files` para a lista de arquivos, `/proc` para a
árvore de processos, polling para vigiar.

**Spec:** `docs/superpowers/specs/2026-08-18-repo-daemon-design.md`

## Base

Branch `repo-daemon`, sobre `383e151`. Baseline: **1035 testes, OK (skipped=19)**.

**Antes de qualquer rodada de teste:**

```bash
export TMPDIR=$HOME/.cache/qctx-test-tmp && mkdir -p "$TMPDIR"
```

A suíte vaza temporários e o `/tmp` é um tmpfs de 16 GB que já encheu e matou todo shell desta
máquina. Limpe com `rm -rf $TMPDIR/tmp*` ao terminar, e `find . -name __pycache__ -type d -prune
-exec rm -rf {} +` antes de cada rodada.

## Global Constraints

- **Só a biblioteca padrão.** Nenhuma dependência nova. É a restrição REAL do projeto (README
  linha 104, com o motivo): dependência faltando dentro de um hook vira perda silenciosa.
- **Código, comentários, docstrings, mensagens e commits em INGLÊS.** Spec e plano em português.
- **A suíte nunca fica vermelha em nenhum commit.**
- **Nenhum teste sobe daemon de verdade nem escreve em coleção real do Qdrant.** O executor de
  trabalho é injetável, como `probe` em `refresh_window`.
- **Nada escreve em `~/.claude/`, `~/.hermes/` ou `~/.config/memories-plugin/` durante testes.**
- **Nenhuma falha abre para "não há nada".** Busca que falha e devolve vazio é indistinguível de
  ausência real.
- **Toda guarda precisa de sonda de mutação que MORDA**, com contagem verificada e escopo
  declarado.
- **Implementadores nunca rodam `git commit --amend`, `git reset` ou `git rebase`.**

## Disciplina de sonda — os oito modos de falha já sofridos aqui

1. Verifique que o BACKUP existe depois de criá-lo.
2. Afirme a contagem de ocorrências E confirme por `grep` que a substituição ATERROU.
3. Confirme que existe linha `Ran N` — substituição que quebra sintaxe não produz saída.
4. Tire o veredito da linha de RESUMO, nunca contando linhas `FAIL:`.
5. Âncora não-única é ACHADO, não motivo para alargar o padrão.
6. Duas mutações com o mesmo conjunto de falhas: no máximo uma é a que você pensa.
7. Fake que guarda por referência faz comparação antes/depois virar identidade.
8. **Teste que afirma ATRAVÉS de uma guarda irmã não distingue as duas** — se apagar uma guarda
   deixa a suíte verde, o teste cobre a outra.

## Estrutura de arquivos

```
core/scan.py        NOVO — quais arquivos de um repo entram
core/lease.py       NOVO — quem está vivo: (pid, starttime) por sessão
core/jobs.py        NOVO — trabalhos em disco: fila, progresso, cancelamento
core/daemon.py      NOVO — o processo: laço, executa, vigia, encerra
hooks/lease.py      NOVO — SessionStart do claude-code
cli/qctx.py         + repos init | add-all | status | cancel | daemon
hosts/hermes/__init__.py   initialize() escreve o lease
core/githook.py     REMOVIDO (e o comando, e o teste)
```

Cada módulo tem uma responsabilidade e nenhum conhece o host. `scan` não sabe o que é um trabalho;
`jobs` não sabe indexar; `daemon` não sabe quais arquivos entram.

---

### Task 1: Quais arquivos entram

**Files:**
- Create: `core/scan.py`
- Test: `tests/test_scan.py`

**Interfaces:**
- Consumes: nada do plugin. Chama `git ls-files` por `subprocess`.
- Produces:
  - `scan.MAX_FILE_BYTES = 1_048_576`
  - `scan.LOCKFILES` — nomes exatos
  - `scan.eligible(root: str, max_bytes: int = MAX_FILE_BYTES) -> dict` com as chaves
    `{"tracked": int, "eligible": list[str], "skipped": dict[str, int]}`

`eligible` devolve caminhos ABSOLUTOS, porque `add_files` recebe caminhos e o daemon roda com
outro cwd.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_scan.py
"""O que entra no acervo quando alguém pede "indexe este projeto".

A fonte é `git ls-files` e não uma varredura do disco. Isso faz o `.gitignore` ser respeitado
POR DEFINIÇÃO em vez de por uma reimplementação nossa, e `node_modules`, build e cache ficam de
fora sem regra especial — eles não estão versionados.
"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import scan  # noqa: E402


def a_repo(**files) -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    for name, content in files.items():
        path = os.path.join(root, name.replace("__", "/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(path, mode) as fh:
            fh.write(content)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, timeout=60)

    return root


class TestTheSourceIsGIT(unittest.TestCase):
    def test_only_tracked_files_are_eligible(self):
        root = a_repo(**{"kept.py": "x = 1\n"})
        with open(os.path.join(root, "untracked.py"), "w") as fh:
            fh.write("y = 2\n")
        out = scan.eligible(root)
        names = [os.path.basename(p) for p in out["eligible"]]
        self.assertEqual(names, ["kept.py"])

    def test_a_gitignored_file_is_absent_without_us_parsing_gitignore(self):
        """O ponto de usar `git ls-files`: nunca reimplementamos a semântica de `.gitignore`,
        que tem negação, precedência e regras por diretório."""
        root = a_repo(**{".gitignore": "build/\n", "app.py": "x = 1\n"})
        os.makedirs(os.path.join(root, "build"), exist_ok=True)
        with open(os.path.join(root, "build", "out.js"), "w") as fh:
            fh.write("console.log(1)\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, timeout=60)
        names = [os.path.basename(p) for p in scan.eligible(root)["eligible"]]
        self.assertIn("app.py", names)
        self.assertNotIn("out.js", names)

    def test_the_paths_are_ABSOLUTE(self):
        """`add_files` recebe caminhos e o daemon roda com outro cwd — um caminho relativo
        resolveria contra o diretório errado, em silêncio."""
        root = a_repo(**{"a.py": "x = 1\n"})
        for path in scan.eligible(root)["eligible"]:
            self.assertTrue(os.path.isabs(path), path)

    def test_a_directory_that_is_not_a_repo_yields_nothing_and_does_not_raise(self):
        out = scan.eligible(tempfile.mkdtemp())
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["tracked"], 0)


class TestTheFourDiscards(unittest.TestCase):
    """Cada um isolado, porque um teste com dois motivos de descarte não prova nenhum."""

    def test_a_binary_file_is_skipped(self):
        root = a_repo(**{"logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["binary"], 1)

    def test_a_lockfile_is_skipped(self):
        root = a_repo(**{"package-lock.json": '{"lockfileVersion": 3}\n'})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["lockfile"], 1)

    def test_a_minified_file_is_skipped(self):
        """Uma linha de 40 KB não responde pergunta nenhuma e domina o acervo do repo."""
        root = a_repo(**{"bundle.js": "var a=1;" * 6000 + "\n"})
        out = scan.eligible(root)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["minified"], 1)

    def test_a_file_above_the_ceiling_is_skipped(self):
        root = a_repo(**{"huge.txt": "line\n" * 300})
        out = scan.eligible(root, max_bytes=100)
        self.assertEqual(out["eligible"], [])
        self.assertEqual(out["skipped"]["too_big"], 1)

    def test_an_ordinary_source_file_survives_ALL_FOUR(self):
        """A guarda das guardas: se um filtro ficar largo demais, isto cai."""
        root = a_repo(**{"core__app.py": "def main():\n    return 1\n"})
        out = scan.eligible(root)
        self.assertEqual(len(out["eligible"]), 1)
        self.assertEqual(sum(out["skipped"].values()), 0)


class TestTheFunnelIsREPORTED(unittest.TestCase):
    def test_it_counts_what_was_seen_and_what_was_dropped(self):
        """Uma contagem que só aparece no fim é uma contagem que ninguém usa para decidir."""
        root = a_repo(**{"a.py": "x = 1\n", "package-lock.json": "{}\n",
                         "logo.png": b"\x00\x01\x02binary"})
        out = scan.eligible(root)
        self.assertEqual(out["tracked"], 3)
        self.assertEqual(len(out["eligible"]), 1)
        self.assertEqual(out["skipped"]["lockfile"], 1)
        self.assertEqual(out["skipped"]["binary"], 1)

    def test_every_discard_reason_is_present_even_when_zero(self):
        """Uma chave ausente obriga todo consumidor a usar `.get`, e um deles vai esquecer."""
        out = scan.eligible(a_repo(**{"a.py": "x = 1\n"}))
        self.assertEqual(set(out["skipped"]), {"binary", "minified", "lockfile", "too_big",
                                               "unreadable"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `export TMPDIR=$HOME/.cache/qctx-test-tmp; mkdir -p "$TMPDIR"; python3 -m unittest tests.test_scan 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.scan'`

- [ ] **Step 3: Implemente `core/scan.py`**

```python
"""Which files of a repository go into the archive.

WHY `git ls-files` AND NOT A WALK OF THE DISK. It makes `.gitignore` respected BY DEFINITION
rather than by a reimplementation of ours — that file has negation, precedence and per-directory
rules, and getting them subtly wrong would index a `build/` nobody wanted, silently. It also
means `node_modules`, caches and artefacts need no special rule: they are not tracked.

The consequence, stated rather than discovered: a file that exists but was never `git add`ed is
NOT indexed. That is the same rule, seen from the other side.

WHY FOUR DISCARDS AND NOT NONE. The archive answers questions; a minified bundle and a lockfile
answer none, and they are large enough to dominate every search of that repository. A binary
would be refused later by `_read_source` anyway — discarding it here just avoids paying the read.
"""
import os
import subprocess

#: A file larger than this becomes hundreds of chunks and dominates the repository's archive.
MAX_FILE_BYTES = 1_048_576

#: A single line longer than this is machine-written: minified bundles, embedded data URIs.
MINIFIED_LINE_CHARS = 2_000

#: Generated, enormous, and nobody searches them by meaning.
LOCKFILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json",
    "poetry.lock", "Pipfile.lock", "Cargo.lock", "composer.lock", "Gemfile.lock",
    "go.sum", "flake.lock", "uv.lock",
})

#: Read once per file, and enough to decide both "binary" and "minified".
_SNIFF_BYTES = 8192


def tracked_files(root: str) -> list[str]:
    """The repository's tracked paths, relative to `root`. Empty when `root` is not a repo."""
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-z"],
                             capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    return [name for name in out.stdout.decode("utf-8", "replace").split("\0") if name]


def eligible(root: str, max_bytes: int = MAX_FILE_BYTES) -> dict:
    """`{"tracked": int, "eligible": [absolute paths], "skipped": {reason: count}}`.

    Paths come back ABSOLUTE: `RepoIndex.add_files` takes paths and the daemon runs from a
    different working directory, so a relative one would resolve against the wrong place —
    silently, because a missing file is merely skipped and reported.

    Every discard reason is present even at zero, so no consumer has to guess whether a key
    exists.
    """
    names = tracked_files(root)
    skipped = {"binary": 0, "minified": 0, "lockfile": 0, "too_big": 0, "unreadable": 0}
    keep = []
    for name in names:
        path = os.path.join(root, name)
        if os.path.basename(name) in LOCKFILES:
            skipped["lockfile"] += 1
            continue
        try:
            size = os.stat(path).st_size
        except OSError:
            skipped["unreadable"] += 1
            continue
        if size > max_bytes:
            skipped["too_big"] += 1
            continue
        reason = _sniff(path)
        if reason:
            skipped[reason] += 1
            continue
        keep.append(os.path.abspath(path))

    return {"tracked": len(names), "eligible": sorted(keep), "skipped": skipped}


def _sniff(path: str) -> str | None:
    """`"binary"`, `"minified"`, `"unreadable"` or None — decided from one read of the head.

    A NUL byte is the same test `_read_source` uses, applied earlier so an image is not read in
    full only to be refused. The line length is measured on the same buffer, so a minified file
    costs no extra I/O.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return "unreadable"
    if b"\0" in head:
        return "binary"
    # A head with no newline at all is one very long line, which is the minified case; a head
    # WITH newlines is judged by its longest complete line.
    longest = max((len(part) for part in head.split(b"\n")[:-1]), default=len(head))
    if longest > MINIFIED_LINE_CHARS:
        return "minified"

    return None
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_scan 2>&1 | tail -3`
Expected: `Ran 12 tests`, `OK`

- [ ] **Step 5: Prove que cada filtro morde SOZINHO**

Quatro mutações, uma por vez, restaurando entre elas. Para cada uma: backup verificado, contagem
afirmada, `grep` confirmando, linha `Ran N` presente.

```bash
cp core/scan.py /tmp/scan.bak && [ -s /tmp/scan.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
p = "core/scan.py"; s = open(p).read()
old = '        if os.path.basename(name) in LOCKFILES:\n'
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "        if False:\n", 1))
PY
grep -c "if False:" core/scan.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_scan 2>&1 | grep -E "^FAIL: test|^Ran|^OK|^FAILED"
cp /tmp/scan.bak core/scan.py
```
Expected: cai `test_a_lockfile_is_skipped` **e só ele**.

Repita para: `if size > max_bytes:` → `if False:` (espera `test_a_file_above_the_ceiling_is_skipped`);
`if b"\0" in head:` → `if False:` (espera `test_a_binary_file_is_skipped`);
`if longest > MINIFIED_LINE_CHARS:` → `if False:` (espera `test_a_minified_file_is_skipped`).

**Se qualquer uma derrubar mais de um teste**, os filtros não estão isolados e isso é o achado.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/scan.py tests/test_scan.py
git commit -F - <<'MSG'
feat: decide what enters the archive from git, not from a walk of the disk

`git ls-files` makes .gitignore respected by definition instead of by a reimplementation of
ours — that file has negation, precedence and per-directory rules, and getting them subtly
wrong would index a build/ nobody wanted, silently. node_modules and caches then need no rule
at all: they are not tracked.

Four discards, each measured against one fixture so a widened filter fails alone: a lockfile is
generated and enormous, a minified bundle is one line of machine output, a binary would be
refused later anyway, and a file above the ceiling becomes hundreds of chunks that dominate
every search of that repository.

The funnel is counted and reported, because a number that appears only at the end is a number
nobody used to decide. Every discard reason is present even at zero, so no consumer has to
guess whether a key exists.

Paths come back absolute: the indexer takes paths and the daemon runs from a different working
directory, where a relative one resolves against the wrong place — and a missing file is only
skipped and reported, so it would be silent.
MSG
```

---

### Task 2: Quem está vivo

**Files:**
- Create: `core/lease.py`
- Test: `tests/test_lease.py`

**Interfaces:**
- Consumes: `core.knobs.state_dir() -> Path`.
- Produces:
  - `lease.dir() -> Path` — `state_dir()/leases`
  - `lease.process_start(pid: int) -> str | None`
  - `lease.find_host_pid(names: tuple = ("claude", "hermes")) -> tuple[int, str] | None`
  - `lease.write(session_id: str, host: str, pid: int | None = None) -> dict`
  - `lease.alive(entry: dict) -> bool`
  - `lease.live() -> list[dict]` — varre e REMOVE os mortos

**O `starttime` é o que separa este desenho de um ingênuo.** Um pid sozinho pode ter sido
reciclado pelo sistema; guardar `(pid, starttime)` distingue o processo original de outro que
recebeu o mesmo número.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_lease.py
"""Quem ainda está usando o daemon.

Um lease é um bilhete de "estou vivo": o pid do processo do HOST e o instante em que ele
começou. O daemon confere os bilhetes a cada ciclo e encerra quando nenhum resta.

POR QUE `(pid, starttime)` E NÃO SÓ O PID: o sistema recicla números de processo. Um pid
sozinho faz um processo qualquer segurar o daemon para sempre, e a falha seria invisível —
o daemon apenas nunca encerraria.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import lease  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestReadingAProcess(unittest.TestCase):
    def test_our_own_start_time_is_readable(self):
        self.assertTrue(lease.process_start(os.getpid()))

    def test_a_pid_that_does_not_exist_reads_as_None(self):
        """Sem levantar: o daemon chama isto a cada ciclo, para pids que ele ESPERA ver morrer."""
        self.assertIsNone(lease.process_start(4_000_000))

    def test_the_start_time_is_STABLE_across_reads(self):
        """Se variasse, todo lease pareceria reciclado e o daemon encerraria com o host vivo."""
        self.assertEqual(lease.process_start(os.getpid()), lease.process_start(os.getpid()))


class TestWritingAndCheckingALease(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_lease_for_a_live_process_is_alive(self):
        entry = lease.write("s1", "claude", pid=os.getpid())
        self.assertTrue(lease.alive(entry))

    def test_a_lease_for_a_DEAD_process_is_not_alive(self):
        done = subprocess.run([sys.executable, "-c", "pass"], timeout=60)
        entry = lease.write("s1", "claude", pid=done.pid)
        self.assertFalse(lease.alive(entry))

    def test_a_RECYCLED_pid_is_not_alive(self):
        """O caso que o pid sozinho não pega: o número existe, mas é outro processo."""
        entry = lease.write("s1", "claude", pid=os.getpid())
        entry["starttime"] = "1"          # como se o processo original tivesse começado antes
        self.assertFalse(lease.alive(entry))

    def test_a_lease_missing_its_fields_is_not_alive(self):
        """Arquivo truncado ou de uma versão anterior: não segura o daemon."""
        for broken in ({}, {"pid": os.getpid()}, {"starttime": "1"}, {"pid": "x"}):
            with self.subTest(broken=broken):
                self.assertFalse(lease.alive(broken))


class TestSweeping(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_live_returns_the_living_and_REMOVES_the_dead(self):
        done = subprocess.run([sys.executable, "-c", "pass"], timeout=60)
        lease.write("alive", "claude", pid=os.getpid())
        lease.write("dead", "hermes", pid=done.pid)
        live = lease.live()
        self.assertEqual([e["session_id"] for e in live], ["alive"])
        self.assertEqual(sorted(p.name for p in lease.dir().glob("*.json")), ["alive.json"])

    def test_no_leases_at_all_is_an_empty_list_and_not_an_error(self):
        """O estado normal antes da primeira sessão, e o estado que faz o daemon encerrar."""
        self.assertEqual(lease.live(), [])

    def test_a_corrupt_lease_file_is_removed_rather_than_crashing_the_sweep(self):
        lease.dir().mkdir(parents=True, exist_ok=True)
        (lease.dir() / "junk.json").write_text("{not json")
        self.assertEqual(lease.live(), [])
        self.assertFalse((lease.dir() / "junk.json").exists())


class TestFindingTheHOST(unittest.TestCase):
    def test_it_walks_up_and_finds_a_named_ancestor(self):
        """No claude-code o hook é subprocesso: python3 -> bash -> claude. Medido nesta
        máquina em 2026-08-18."""
        found = lease.find_host_pid(names=(os.path.basename(sys.executable),))
        self.assertIsNotNone(found, "não achou o próprio interpretador na árvore")
        pid, start = found
        self.assertEqual(lease.process_start(pid), start)

    def test_an_ancestor_that_is_not_there_yields_None(self):
        self.assertIsNone(lease.find_host_pid(names=("nao-existe-este-processo",)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_lease 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.lease'`

- [ ] **Step 3: Implemente `core/lease.py`**

```python
"""Who is still using the daemon.

A lease is a note saying "I am alive": the HOST process's pid and the moment it started. The
daemon checks the notes each cycle and exits when none is left. The user asked for exactly
this — "o daemon deve ser morto quando o claude ou hermes sai/morre".

WHY THE TEST IS THE PROCESS AND NOT A MESSAGE. A host that exits cleanly could tell us; a host
killed with -9 cannot. Checking whether the process still exists covers both with one mechanism,
and the one it covers worse — a machine that lost power — leaves a stale file that the next
sweep removes anyway.

WHY `(pid, starttime)` AND NOT THE PID ALONE. The system reuses process numbers. With the pid
alone, an unrelated process that inherited the number would hold the daemon open forever, and
the failure would be invisible: nothing breaks, the daemon merely never exits. `starttime` is
field 22 of `/proc/<pid>/stat` and is stable for the life of a process.

PLATFORM: reading `/proc` is Linux. Where it is absent, `process_start` returns None and every
lease reads as dead, so the daemon exits rather than lingering — the safe direction. Adding a
portable process table would mean a dependency, which this project refuses for a reason.
"""
import json
import os
from pathlib import Path

from .knobs import state_dir


def dir() -> Path:                                  # noqa: A001 — the name says what it holds
    return state_dir() / "leases"


def process_start(pid: int) -> str | None:
    """Field 22 of `/proc/<pid>/stat`, or None when the process is gone or unreadable.

    Parsed from the LAST `)` because the second field is the executable name in parentheses and
    may itself contain spaces or parentheses — splitting the whole line on whitespace is the
    classic way to read this file wrong.
    """
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8", errors="replace") as fh:
            after = fh.read().rsplit(")", 1)[1]
    except (OSError, ValueError, IndexError):
        return None
    fields = after.split()

    return fields[19] if len(fields) > 19 else None


def find_host_pid(names: tuple = ("claude", "hermes")) -> tuple[int, str] | None:
    """The nearest ancestor whose executable name is one of `names`, as `(pid, starttime)`.

    On claude-code a hook is a subprocess — measured: `python3 -> bash -> claude -> bash ->
    ptyxis` — so the host is up the tree. On hermes the provider runs INSIDE the host, and the
    caller passes `os.getpid()` instead of calling this.
    """
    pid = os.getpid()
    for _ in range(12):                             # deep enough for any real tree, bounded
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            name = raw.split("(", 1)[1].rsplit(")", 1)[0]
            ppid = int(raw.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return None
        if name in names:
            start = process_start(pid)

            return (pid, start) if start else None
        if ppid <= 1:
            return None
        pid = ppid

    return None


def write(session_id: str, host: str, pid: int | None = None) -> dict:
    """Records that `session_id` on `host` is alive. Returns the entry it wrote.

    `pid` defaults to this process, which is right for hermes — the provider IS the host. On
    claude-code the caller resolves the host with `find_host_pid` first.
    """
    pid = os.getpid() if pid is None else int(pid)
    entry = {"session_id": session_id, "host": host, "pid": pid,
             "starttime": process_start(pid) or "", "written_at": _now()}
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = dir() / f"{_safe(session_id)}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # A lease that cannot be written must not break a session start. The cost is a daemon
        # that exits sooner than it needed to, which is the safe direction.
        pass

    return entry


def alive(entry: dict) -> bool:
    """Whether the process this lease names is still the process it named."""
    if not isinstance(entry, dict):
        return False
    try:
        pid = int(entry.get("pid"))
    except (TypeError, ValueError):
        return False
    recorded = entry.get("starttime")
    if not recorded:
        return False

    return process_start(pid) == recorded


def live() -> list[dict]:
    """Every living lease, removing the files of the dead ones on the way through."""
    found = []
    try:
        paths = sorted(dir().glob("*.json"))
    except OSError:
        return []
    for path in paths:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            entry = None
        if entry is not None and alive(entry):
            found.append(entry)
            continue
        try:
            path.unlink()
        except OSError:
            pass

    return found


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name or "default"))


def _now() -> float:
    import time

    return time.time()
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_lease 2>&1 | tail -3`
Expected: `Ran 13 tests`, `OK`

- [ ] **Step 5: Prove que o `starttime` morde**

```bash
cp core/lease.py /tmp/lease.bak && [ -s /tmp/lease.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
p = "core/lease.py"; s = open(p).read()
old = "    return process_start(pid) == recorded\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "    return process_start(pid) is not None\n", 1))
PY
grep -c "is not None$" core/lease.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_lease 2>&1 | grep -E "^FAIL: test|^Ran|^OK|^FAILED"
cp /tmp/lease.bak core/lease.py
```
Expected: cai `test_a_RECYCLED_pid_is_not_alive` **e só ele** — a mutação deixa o pid sozinho
decidir, que é exatamente o desenho ingênuo.

Segunda sonda: trocar `path.unlink()` por `pass` e esperar
`test_live_returns_the_living_and_REMOVES_the_dead`.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/lease.py tests/test_lease.py
git commit -F - <<'MSG'
feat: a lease says which host is alive, by pid AND start time

The daemon has to end when the last host does, and a host killed with -9 cannot tell us. So the
test is whether the process still exists, which covers a clean exit and a kill identically.

The start time is what separates this from the naive version. The system reuses process
numbers, and with the pid alone an unrelated process that inherited one would hold the daemon
open forever — a failure with no symptom, because nothing breaks and the daemon merely never
exits. Field 22 of /proc/<pid>/stat is stable for the life of a process, and it is parsed from
the LAST `)` because the executable name sits in parentheses and can contain spaces.

Where /proc is absent every lease reads as dead, so the daemon exits rather than lingering. That
is the safe direction, and a portable process table would mean a dependency this project refuses
for a documented reason.
MSG
```

---

### Task 3: Trabalhos em disco

**Files:**
- Create: `core/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `core.knobs.state_dir()`.
- Produces:
  - `jobs.dir() -> Path`, `jobs.PENDING = "pending"`, `jobs.RUNNING = "running"`,
    `jobs.DONE = "done"`, `jobs.CANCELLED = "cancelled"`, `jobs.FAILED = "failed"`
  - `jobs.enqueue(repo: str, kind: str, paths: list[str], total: int | None = None) -> dict`
  - `jobs.load(repo: str) -> dict | None`, `jobs.all_jobs() -> list[dict]`
  - `jobs.update(repo: str, **fields) -> dict | None`
  - `jobs.request_cancel(repo: str) -> bool`, `jobs.cancel_requested(repo: str) -> bool`
  - `jobs.next_pending() -> dict | None`
  - `jobs.reap(pid_alive) -> list[str]` — marca como interrompido o que diz `running` sob um
    daemon morto

**O estado é o contrato entre os comandos e o daemon.** Não há socket: `add-all` escreve, o
daemon lê no ciclo seguinte, `status` lê o mesmo arquivo. É por isso que `status` funciona com o
daemon morto — e é por isso que `reap` existe.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_jobs.py
"""O estado dos trabalhos, em disco, que é o único contrato entre os comandos e o daemon.

Não há socket nem pipe: `add-all` escreve um arquivo, o daemon lê no ciclo seguinte, e `status`
lê o mesmo arquivo. Isso é o que faz `status` responder mesmo com o daemon morto — e o que
obriga a existir um `reap`, porque um arquivo que diz `running` sob um daemon que morreu está
mentindo.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import jobs  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestTheQueue(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_an_enqueued_job_comes_back_pending(self):
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py"])
        job = jobs.load("alpha")
        self.assertEqual(job["state"], jobs.PENDING)
        self.assertEqual(job["total"], 2)
        self.assertEqual(job["done"], 0)

    def test_next_pending_returns_it_and_None_when_there_is_nothing(self):
        self.assertIsNone(jobs.next_pending())
        jobs.enqueue("alpha", "index", ["/a.py"])
        self.assertEqual(jobs.next_pending()["repo"], "alpha")

    def test_next_pending_SKIPS_what_is_already_running(self):
        """Um trabalho por vez: dois processos no mesmo repo duplicariam o trabalho."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING)
        self.assertIsNone(jobs.next_pending())

    def test_enqueuing_the_same_repo_again_REPLACES_the_job(self):
        """Pedir de novo é pedir o estado atual do disco, não somar à fila antiga."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py"])
        self.assertEqual(jobs.load("alpha")["total"], 2)
        self.assertEqual(len(jobs.all_jobs()), 1)

    def test_a_repo_with_no_job_loads_as_None(self):
        self.assertIsNone(jobs.load("never-queued"))


class TestProgressAndCancellation(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_progress_is_readable_while_it_runs(self):
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py", "/c.py"])
        jobs.update("alpha", state=jobs.RUNNING, done=2, current="/b.py")
        job = jobs.load("alpha")
        self.assertEqual((job["done"], job["total"]), (2, 3))
        self.assertEqual(job["current"], "/b.py")

    def test_a_cancel_request_is_visible_to_whoever_is_running_it(self):
        """O cancelamento é um pedido em disco, não um sinal: o daemon o lê entre arquivos, o
        que o deixa parar num ponto onde o que já entrou está consistente."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        self.assertFalse(jobs.cancel_requested("alpha"))
        self.assertTrue(jobs.request_cancel("alpha"))
        self.assertTrue(jobs.cancel_requested("alpha"))

    def test_cancelling_a_repo_with_no_job_answers_False_instead_of_raising(self):
        self.assertFalse(jobs.request_cancel("never-queued"))

    def test_update_on_a_missing_job_answers_None_instead_of_creating_one(self):
        """Um `update` que cria um trabalho do nada faria um cancelamento tardio ressuscitar
        um trabalho já terminado."""
        self.assertIsNone(jobs.update("never-queued", done=1))


class TestAJobThatOUTLIVEDItsDaemon(unittest.TestCase):
    """Um arquivo dizendo `running` sob um daemon morto mente, e um estado que mente é pior
    que um estado ausente: o `status` mostraria progresso parado como se fosse atividade."""

    def setUp(self):
        a_state_dir()

    def test_reap_marks_a_running_job_of_a_dead_daemon_as_interrupted(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, daemon_pid=4_000_000, done=1)
        reaped = jobs.reap(lambda pid: False)
        self.assertEqual(reaped, ["alpha"])
        job = jobs.load("alpha")
        self.assertEqual(job["state"], jobs.FAILED)
        self.assertIn("interrupted", job["error"])
        self.assertEqual(job["done"], 1, "o progresso foi perdido ao marcar a interrupção")

    def test_reap_leaves_a_job_of_a_LIVING_daemon_alone(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.update("alpha", state=jobs.RUNNING, daemon_pid=os.getpid())
        self.assertEqual(jobs.reap(lambda pid: True), [])
        self.assertEqual(jobs.load("alpha")["state"], jobs.RUNNING)

    def test_reap_leaves_a_PENDING_job_alone(self):
        """Pendente não é responsabilidade de daemon nenhum ainda — é o estado normal de um
        trabalho enfileirado antes de o daemon subir."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        self.assertEqual(jobs.reap(lambda pid: False), [])
        self.assertEqual(jobs.load("alpha")["state"], jobs.PENDING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_jobs 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.jobs'`

- [ ] **Step 3: Implemente `core/jobs.py`**

```python
"""What the daemon is doing, on disk, where every command can read it.

THERE IS NO PROTOCOL BETWEEN PROCESSES, and that is the design. A command writes a file; the
daemon reads it on its next cycle; `status` reads the same file. A socket would be more
immediate and would bring negotiation, a wire format and a new failure mode — a daemon that is
alive but not listening. For a queue that changes once a minute, the cost does not pay.

The consequence to hold onto: a file saying `running` under a daemon that died is LYING, and a
state that lies is worse than one that is absent — `status` would render stalled progress as
activity. `reap` is what turns that into an honest "interrupted".
"""
import json
import os
import time
from pathlib import Path

from .knobs import state_dir

PENDING = "pending"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"


def dir() -> Path:                                  # noqa: A001 — the name says what it holds
    return state_dir() / "jobs"


def enqueue(repo: str, kind: str, paths: list[str], total: int | None = None) -> dict:
    """Queues work for `repo`, REPLACING whatever job it had.

    Replacing rather than appending: asking again means asking about the state of the disk NOW,
    and an older list of paths is a description of a repository that has since changed.
    """
    job = {"repo": repo, "kind": kind, "paths": list(paths),
           "total": len(paths) if total is None else int(total),
           "done": 0, "current": "", "state": PENDING, "error": "",
           "cancel": False, "daemon_pid": 0, "queued_at": time.time()}
    _write(repo, job)

    return job


def load(repo: str) -> dict | None:
    try:
        return json.loads(_path(repo).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def all_jobs() -> list[dict]:
    out = []
    try:
        paths = sorted(dir().glob("*.json"))
    except OSError:
        return []
    for path in paths:
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue

    return out


def update(repo: str, **fields) -> dict | None:
    """Merges `fields` into the job. None when there is no job — it never CREATES one.

    Creating one here would let a late update resurrect a job that had already finished, which
    is exactly the kind of state that outlives the thing it describes.
    """
    job = load(repo)
    if job is None:
        return None
    job.update(fields)
    _write(repo, job)

    return job


def request_cancel(repo: str) -> bool:
    """Asks for `repo`'s job to stop. False when there is nothing to cancel.

    A flag on disk rather than a signal: the daemon reads it BETWEEN files, so it stops at a
    point where what is already indexed is consistent.
    """
    return update(repo, cancel=True) is not None


def cancel_requested(repo: str) -> bool:
    job = load(repo)

    return bool(job and job.get("cancel"))


def next_pending() -> dict | None:
    """The oldest pending job, or None. Skips anything already running: one job at a time."""
    pending = [j for j in all_jobs() if j.get("state") == PENDING]
    if any(j.get("state") == RUNNING for j in all_jobs()):
        return None

    return min(pending, key=lambda j: j.get("queued_at", 0)) if pending else None


def reap(pid_alive) -> list[str]:
    """Marks as interrupted every RUNNING job whose daemon is gone. Returns the repos touched.

    `pid_alive` is injected so a test can decide without spawning anything.
    """
    touched = []
    for job in all_jobs():
        if job.get("state") != RUNNING:
            continue
        if pid_alive(job.get("daemon_pid") or 0):
            continue
        update(job["repo"], state=FAILED,
               error="interrupted: the daemon running this job is gone")
        touched.append(job["repo"])

    return touched


def _path(repo: str) -> Path:
    return dir() / f"{_safe(repo)}.json"


def _write(repo: str, job: dict) -> None:
    try:
        dir().mkdir(parents=True, exist_ok=True)
        path = _path(repo)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(job, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # State that cannot be written is state the reader will not find — the same answer as
        # no job at all, which every caller already handles.
        pass


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name or "default"))
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_jobs 2>&1 | tail -3`
Expected: `Ran 12 tests`, `OK`

- [ ] **Step 5: Prove que `reap` e o "um por vez" mordem**

```bash
cp core/jobs.py /tmp/jobs.bak && [ -s /tmp/jobs.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
p = "core/jobs.py"; s = open(p).read()
old = "    if any(j.get(\"state\") == RUNNING for j in all_jobs()):\n        return None\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "", 1))
PY
grep -c "== RUNNING for j in all_jobs" core/jobs.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_jobs 2>&1 | grep -E "^FAIL: test|^Ran|^OK|^FAILED"
cp /tmp/jobs.bak core/jobs.py
```
Expected: cai `test_next_pending_SKIPS_what_is_already_running`.

Segunda sonda: em `update`, trocar `if job is None: return None` por criar um job vazio, e
esperar `test_update_on_a_missing_job_answers_None_instead_of_creating_one`.

Terceira: em `reap`, remover a linha `if job.get("state") != RUNNING: continue` e esperar
`test_reap_leaves_a_PENDING_job_alone`.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/jobs.py tests/test_jobs.py
git commit -F - <<'MSG'
feat: the queue is a directory of files, not a protocol

A command writes a file, the daemon reads it next cycle, and status reads the same file. A
socket would be more immediate and would bring negotiation, a wire format and a failure mode
that does not exist here — a daemon alive but not listening. For a queue that changes once a
minute the cost does not pay, and this way `status` answers with the daemon dead.

Which is exactly why `reap` exists: a file saying `running` under a daemon that died is lying,
and a state that lies is worse than one that is absent, because status would render stalled
progress as activity. Reaping turns it into an honest "interrupted" without discarding the
progress that really happened.

Cancellation is a flag on disk rather than a signal, so the daemon reads it BETWEEN files and
stops where what is already indexed is consistent. And `update` never creates a job: doing so
would let a late update resurrect one that had already finished.
MSG
```

---

### Task 4: O daemon

**Files:**
- Create: `core/daemon.py`
- Test: `tests/test_daemon.py`

**Interfaces:**
- Consumes: `core.lease.live()`, `core.jobs.*`, `core.knobs.state_dir()`.
- Produces:
  - `daemon.CYCLE_S = 5.0`
  - `daemon.record() -> dict | None` — o daemon corrente, se o pid dele vive
  - `daemon.start(argv: list[str] | None = None) -> dict` — sobe destacado, uma vez só
  - `daemon.stop() -> bool`
  - `daemon.run(work, *, cycles: int | None = None, sleep=time.sleep) -> str` — O LAÇO. `work`
    é injetado; devolve o motivo da saída.

**`run` recebe `work` e `cycles` para ser testável sem subir processo e sem rede.** Essa é a
mesma escolha de `refresh_window(probe=...)`.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_daemon.py
"""O laço: executa o que está na fila, vigia, e encerra quando ninguém mais está usando.

NENHUM TESTE AQUI SOBE UM DAEMON DE VERDADE nem toca o Qdrant. `run` recebe o executor de
trabalho e um número de ciclos, então o laço é exercitado inteiro em processo — a mesma escolha
que `refresh_window(probe=...)` já usa.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import daemon, jobs, lease  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


def a_live_lease() -> dict:
    return lease.write("s1", "claude", pid=os.getpid())


class TestItEndsWithTheLastHost(unittest.TestCase):
    """O requisito do usuário, textualmente: "o daemon deve ser morto quando o claude ou
    hermes sai/morre"."""

    def setUp(self):
        a_state_dir()

    def test_with_NO_live_lease_it_exits_on_the_first_cycle(self):
        self.assertEqual(daemon.run(lambda job: None, cycles=10, sleep=lambda s: None),
                         "no live lease")

    def test_with_a_live_lease_it_keeps_going(self):
        a_live_lease()
        self.assertEqual(daemon.run(lambda job: None, cycles=3, sleep=lambda s: None),
                         "cycles exhausted")

    def test_it_exits_when_the_lease_DIES_mid_run(self):
        """O caso real: o host fecha enquanto o daemon roda."""
        a_live_lease()
        seen = []

        def kill_the_lease_after_one(seconds):
            seen.append(1)
            if len(seen) == 1:
                for path in lease.dir().glob("*.json"):
                    path.unlink()

        self.assertEqual(daemon.run(lambda job: None, cycles=10,
                                    sleep=kill_the_lease_after_one), "no live lease")
        self.assertEqual(len(seen), 1, "não encerrou no ciclo seguinte à morte do lease")


class TestItRunsWhatIsQueued(unittest.TestCase):
    def setUp(self):
        a_state_dir()
        a_live_lease()

    def test_a_pending_job_is_handed_to_the_worker(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        seen = []
        daemon.run(lambda job: seen.append(job["repo"]), cycles=1, sleep=lambda s: None)
        self.assertEqual(seen, ["alpha"])

    def test_the_job_is_marked_RUNNING_with_the_daemon_pid_while_it_runs(self):
        """Sem o pid, `reap` não teria como saber se o dono do trabalho ainda existe."""
        jobs.enqueue("alpha", "index", ["/a.py"])
        during = {}

        def worker(job):
            during.update(jobs.load("alpha"))

        daemon.run(worker, cycles=1, sleep=lambda s: None)
        self.assertEqual(during["state"], jobs.RUNNING)
        self.assertEqual(during["daemon_pid"], os.getpid())

    def test_a_finished_job_is_marked_done(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        daemon.run(lambda job: None, cycles=1, sleep=lambda s: None)
        self.assertEqual(jobs.load("alpha")["state"], jobs.DONE)

    def test_a_worker_that_RAISES_marks_the_job_failed_and_the_daemon_survives(self):
        """Um trabalho que explode não pode derrubar o daemon: os outros repos continuam."""
        jobs.enqueue("alpha", "index", ["/a.py"])

        def broken(job):
            raise RuntimeError("qdrant is down")

        out = daemon.run(broken, cycles=2, sleep=lambda s: None)
        self.assertEqual(out, "cycles exhausted")
        job = jobs.load("alpha")
        self.assertEqual(job["state"], jobs.FAILED)
        self.assertIn("qdrant is down", job["error"])

    def test_a_cancelled_job_is_marked_cancelled_and_not_run_again(self):
        jobs.enqueue("alpha", "index", ["/a.py"])
        jobs.request_cancel("alpha")
        seen = []
        daemon.run(lambda job: seen.append(1), cycles=3, sleep=lambda s: None)
        self.assertEqual(seen, [], "um trabalho cancelado foi executado")
        self.assertEqual(jobs.load("alpha")["state"], jobs.CANCELLED)


class TestOnlyONEDaemon(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_the_record_of_a_dead_daemon_reads_as_none(self):
        daemon._write_record({"pid": 4_000_000, "starttime": "1"})
        self.assertIsNone(daemon.record())

    def test_the_record_of_a_live_daemon_reads_back(self):
        daemon._write_record({"pid": os.getpid(), "starttime": lease.process_start(os.getpid())})
        self.assertIsNotNone(daemon.record())

    def test_starting_when_one_is_already_running_does_not_start_a_second(self):
        daemon._write_record({"pid": os.getpid(), "starttime": lease.process_start(os.getpid())})
        spawned = []
        out = daemon.start(spawn=lambda argv: spawned.append(argv) or 999)
        self.assertEqual(spawned, [], "subiu um segundo daemon")
        self.assertEqual(out["action"], "already")

    def test_starting_with_no_daemon_spawns_one(self):
        spawned = []
        out = daemon.start(spawn=lambda argv: spawned.append(argv) or 4_000_000)
        self.assertEqual(len(spawned), 1)
        self.assertEqual(out["action"], "started")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_daemon 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.daemon'`

- [ ] **Step 3: Implemente `core/daemon.py`**

```python
"""The background process: runs what is queued, watches what is indexed, and ends with its hosts.

WHY A DAEMON AT ALL. Indexing a real project is minutes of work — measured at ~0.13 s per file,
so 1,800 files is several minutes — and doing it in the caller's terminal means no progress, no
cancelling, and a frozen prompt. There is no prohibition on background processes in this project;
an earlier note about how the temporary archive's TTL expires was mistaken for one.

WHY IT ENDS WITH THE HOSTS. A process that outlives the tool that started it is a process the
user did not ask for and will not think to stop. Each cycle it checks the leases, and no living
lease means nobody is using it.

WHY `run` TAKES ITS WORKER AND ITS CYCLE COUNT. So the whole loop can be exercised in-process,
with no spawning and no network — the same choice `refresh_window(probe=...)` already makes.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import jobs, lease
from .knobs import state_dir

#: How long the loop sleeps between cycles. Short enough that a cancel or a file change is
#: noticed while the user is still looking at the screen; long enough to be free.
CYCLE_S = 5.0


def path() -> Path:
    return state_dir() / "daemon.json"


def record() -> dict | None:
    """The running daemon, or None. A record whose process is gone reads as none.

    Same `(pid, starttime)` test the leases use: a recycled pid must not make a dead daemon look
    alive, or nothing would ever start one again.
    """
    try:
        entry = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict):
        return None
    if not lease.alive(entry):
        return None

    return entry


def start(spawn=None, argv: list[str] | None = None) -> dict:
    """Starts the daemon if none is running. `{"action": "started" | "already", "pid": int}`.

    `spawn` is injected for tests; by default it launches a detached `qctx repos daemon run`.
    """
    existing = record()
    if existing:
        return {"action": "already", "pid": existing.get("pid", 0)}
    launch = spawn or _spawn
    command = argv or [sys.executable, _qctx_path(), "repos", "daemon", "run"]
    pid = launch(command)
    _write_record({"pid": pid, "starttime": lease.process_start(pid) or "",
                   "started_at": time.time()})

    return {"action": "started", "pid": pid}


def stop() -> bool:
    """Asks the running daemon to end. False when there was none."""
    entry = record()
    if not entry:
        return False
    try:
        os.kill(int(entry["pid"]), 15)
    except (OSError, ValueError, KeyError):
        return False
    try:
        path().unlink()
    except OSError:
        pass

    return True


def run(work, *, cycles: int | None = None, sleep=time.sleep, watch=None) -> str:
    """THE LOOP. Returns why it stopped.

    Each cycle, in this order and for these reasons:

      1. no living lease → exit. Checked FIRST so a daemon whose hosts are gone does not start
         one more job before noticing.
      2. reap jobs left `running` by a daemon that died, so `status` never shows stalled
         progress as activity.
      3. run one pending job, if any. One at a time: two workers on one repository duplicate
         the work without finishing sooner.
      4. otherwise watch, which is where changed files become new jobs.

    A worker that raises marks its job failed and the loop CONTINUES — one broken repository
    must not end the daemon for the others.
    """
    seen = 0
    while cycles is None or seen < cycles:
        if not lease.live():
            return "no live lease"
        jobs.reap(lambda pid: lease.process_start(pid) is not None)
        job = jobs.next_pending()
        if job is not None:
            _run_one(job, work)
        elif watch is not None:
            watch()
        seen += 1
        if cycles is None or seen < cycles:
            sleep(CYCLE_S)

    return "cycles exhausted"


def _run_one(job: dict, work) -> None:
    repo = job["repo"]
    if jobs.cancel_requested(repo):
        jobs.update(repo, state=jobs.CANCELLED)

        return
    jobs.update(repo, state=jobs.RUNNING, daemon_pid=os.getpid(), error="")
    try:
        work(job)
    except Exception as exc:                        # noqa: BLE001 — see the docstring of `run`
        jobs.update(repo, state=jobs.FAILED, error=f"{type(exc).__name__}: {exc}"[:400])

        return
    if jobs.cancel_requested(repo):
        jobs.update(repo, state=jobs.CANCELLED)

        return
    jobs.update(repo, state=jobs.DONE, current="")


def _write_record(entry: dict) -> None:
    try:
        path().parent.mkdir(parents=True, exist_ok=True)
        tmp = path().with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path())
    except OSError:
        pass


def _spawn(argv: list[str]) -> int:
    """Launches `argv` fully detached, so it survives the terminal that started it."""
    out = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, start_new_session=True)

    return out.pid


def _qctx_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        "cli", "qctx.py")
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_daemon 2>&1 | tail -3`
Expected: `Ran 12 tests`, `OK`

- [ ] **Step 5: Prove as três guardas do laço**

```bash
cp core/daemon.py /tmp/daemon.bak && [ -s /tmp/daemon.bak ] || { echo "BACKUP FALHOU"; exit 1; }
# A) o daemon deixa de conferir os leases
python3 - <<'PY'
p = "core/daemon.py"; s = open(p).read()
old = "        if not lease.live():\n            return \"no live lease\"\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "", 1))
PY
grep -c "no live lease" core/daemon.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_daemon 2>&1 | grep -E "^FAIL: test|^Ran|^FAILED"
cp /tmp/daemon.bak core/daemon.py
```
Expected: caem os três de `TestItEndsWithTheLastHost`.

```bash
# B) o worker que explode passa a derrubar o daemon
python3 - <<'PY'
p = "core/daemon.py"; s = open(p).read()
old = "    except Exception as exc:                        # noqa: BLE001 — see the docstring of `run`\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "    except ValueError as exc:\n", 1))
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_daemon 2>&1 | grep -E "^FAIL: test|^ERROR: test|^Ran|^FAILED"
cp /tmp/daemon.bak core/daemon.py

# C) dois daemons passam a ser possiveis
python3 - <<'PY'
p = "core/daemon.py"; s = open(p).read()
old = "    existing = record()\n    if existing:\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "    existing = None\n    if existing:\n", 1))
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_daemon 2>&1 | grep -E "^FAIL: test|^Ran|^FAILED"
cp /tmp/daemon.bak core/daemon.py
```
Expected em C: cai `test_starting_when_one_is_already_running_does_not_start_a_second`.

**As mutações A e C não podem derrubar o mesmo conjunto.**

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/daemon.py tests/test_daemon.py
git commit -F - <<'MSG'
feat: the loop — run what is queued, and end with the last host

Indexing a real project is minutes of work, measured at ~0.13s per file, so doing it in the
caller's terminal means no progress, no cancelling and a frozen prompt. There is no prohibition
on background processes here; a note about how the temporary archive's TTL expires had been
mistaken for one.

The lease check comes FIRST in the cycle, so a daemon whose hosts are gone does not start one
more job before noticing. Then reaping, so status never renders stalled progress as activity.
Then one job — one at a time, because two workers on one repository duplicate the work without
finishing sooner.

A worker that raises marks its job failed and the loop continues: one broken repository must not
end the daemon for the others.

`run` takes its worker and a cycle count, so the whole loop is exercised in-process with no
spawning and no network — the same choice refresh_window(probe=...) already makes. Not one test
here starts a real daemon.
MSG
```

---

### Task 5: Os comandos

**Files:**
- Modify: `cli/qctx.py`
- Create: `core/indexer.py`
- Test: `tests/test_cli_daemon.py`

**Interfaces:**
- Consumes: `scan.eligible`, `jobs.*`, `daemon.*`, `RepoIndex.add_files`, `RepoIndex.refresh`.
- Produces:
  - `indexer.work(cfg) -> callable` — devolve o `work(job)` que o daemon executa
  - CLI: `repos add-all`, `repos status`, `repos cancel`, `repos daemon start|stop|run`

**`core/indexer.py` existe para o daemon não conhecer o Qdrant.** `daemon.run` recebe uma função;
este módulo é quem a constrói, e é onde o cancelamento é verificado **entre arquivos**.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_cli_daemon.py
"""A superfície que o usuário toca, e o executor que o daemon roda.

O executor é testado com um índice falso: o que se mede é que ele indexa em lotes, atualiza o
progresso e PARA quando o cancelamento chega — não que o Qdrant funcione.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import indexer, jobs  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class FakeIndex:
    def __init__(self):
        self.added = []

    def add_files(self, repo, paths):
        self.added.append(list(paths))

        return {"repo": repo, "files": len(paths), "chunks": len(paths), "skipped": []}


class TestTheWorker(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_it_indexes_every_path_of_the_job(self):
        ix = FakeIndex()
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py", "/c.py"])
        indexer.work(index=ix)(jobs.load("alpha"))
        self.assertEqual(sorted(p for batch in ix.added for p in batch),
                         ["/a.py", "/b.py", "/c.py"])

    def test_progress_advances_as_it_goes(self):
        ix = FakeIndex()
        jobs.enqueue("alpha", "index", ["/a.py", "/b.py"])
        indexer.work(index=ix, batch=1)(jobs.load("alpha"))
        self.assertEqual(jobs.load("alpha")["done"], 2)

    def test_it_STOPS_when_a_cancel_arrives_mid_job(self):
        """O cancelamento é lido entre lotes, então ele para num ponto onde o que entrou está
        consistente — e o que já entrou FICA."""
        ix = FakeIndex()
        jobs.enqueue("alpha", "index", [f"/f{i}.py" for i in range(10)])

        original = ix.add_files

        def cancel_after_first(repo, paths):
            out = original(repo, paths)
            jobs.request_cancel("alpha")

            return out

        ix.add_files = cancel_after_first
        indexer.work(index=ix, batch=1)(jobs.load("alpha"))
        self.assertEqual(len(ix.added), 1, "não parou no cancelamento")
        self.assertEqual(jobs.load("alpha")["done"], 1, "o que já entrou foi descartado")

    def test_a_refresh_job_calls_refresh_and_not_add_files(self):
        class RefreshIndex(FakeIndex):
            def __init__(self):
                super().__init__()
                self.refreshed = []

            def refresh(self, repo):
                self.refreshed.append(repo)

                return []

        ix = RefreshIndex()
        jobs.enqueue("alpha", "refresh", [])
        indexer.work(index=ix)(jobs.load("alpha"))
        self.assertEqual(ix.refreshed, ["alpha"])
        self.assertEqual(ix.added, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_cli_daemon 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.indexer'`

- [ ] **Step 3: Implemente `core/indexer.py`**

```python
"""What the daemon actually runs. Kept apart so the daemon never has to know about Qdrant.

`daemon.run` takes a callable and knows nothing else about the work; this module builds it. The
separation is what lets every daemon test run with no network at all.

CANCELLATION IS CHECKED BETWEEN BATCHES, not inside one. A batch is already indexed or not, so
stopping between them leaves the archive consistent, and what was indexed STAYS — a partial
index answers questions about the part it has, and re-running skips whatever did not change.
"""
from . import jobs

#: How many files go to `add_files` at once. Small enough that progress moves visibly and a
#: cancel is honoured quickly; large enough not to pay the call overhead per file.
BATCH = 8


def work(cfg=None, index=None, batch: int = BATCH):
    """Returns the `work(job)` the daemon calls.

    `index` is injected by the tests; in production it is built from `cfg` on first use, so
    importing this module costs no connection.
    """
    def run_job(job: dict) -> None:
        target = index if index is not None else _build(cfg)
        repo = job["repo"]
        if job.get("kind") == "refresh":
            target.refresh(repo)

            return
        paths = list(job.get("paths") or [])
        done = 0
        for start in range(0, len(paths), batch):
            if jobs.cancel_requested(repo):
                return
            chunk = paths[start:start + batch]
            target.add_files(repo, chunk)
            done += len(chunk)
            jobs.update(repo, done=done, current=chunk[-1])

    return run_job


def _build(cfg):
    import core

    return core.build_repos(cfg if cfg is not None else core.load())
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_cli_daemon 2>&1 | tail -3`
Expected: `Ran 4 tests`, `OK`

- [ ] **Step 5: Acrescente os comandos ao CLI**

Em `cli/qctx.py`, junto dos outros `cmd_repos_*`:

```python
def cmd_repos_add_all(args, cfg):
    from core import daemon, jobs, scan

    root = bindings.git_root(args.path or os.getcwd())
    if not root:
        raise SystemExit("not inside a git working copy — pass --path")
    found = scan.eligible(root)
    dropped = ", ".join(f"{n} {k}" for k, n in sorted(found["skipped"].items()) if n)
    print(f"{found['tracked']} tracked → {len(found['eligible'])} eligible"
          + (f" ({dropped})" if dropped else ""))
    if not found["eligible"]:
        print("nothing to index")

        return
    jobs.enqueue(args.repo, "index", found["eligible"])
    started = daemon.start()
    print(f"queued under {args.repo!r}; daemon {started['action']} (pid {started['pid']})")
    print("watch it with:  qctx repos status")


def cmd_repos_status(args, cfg):
    from core import daemon, jobs, lease

    running = daemon.record()
    rows = jobs.all_jobs()
    if args.json:
        output({"daemon": running, "jobs": rows, "leases": lease.live()}, True)

        return
    # Said first and plainly: every number below was written by a process that may be gone, and
    # a reader who does not know that reads stalled progress as activity.
    print(f"daemon: {'running (pid %d)' % running['pid'] if running else 'not running'}")
    if not rows:
        print("no indexing jobs")

        return
    for job in rows:
        pct = f"{100 * job['done'] // job['total']}%" if job.get("total") else "—"
        line = f"  {job['repo']:<24} {job['state']:<10} {job['done']}/{job['total']} {pct}"
        print(line + (f"  {job['error']}" if job.get("error") else ""))


def cmd_repos_cancel(args, cfg):
    from core import jobs

    if jobs.request_cancel(args.repo):
        print(f"cancel requested for {args.repo!r} — what is already indexed stays")
    else:
        print(f"no job for {args.repo!r}")


def cmd_repos_daemon(args, cfg):
    from core import daemon, indexer

    if args.action == "stop":
        print("daemon stopped" if daemon.stop() else "no daemon was running")

        return
    if args.action == "start":
        out = daemon.start()
        print(f"daemon {out['action']} (pid {out['pid']})")

        return
    # `run` is what the detached process executes. It is a command rather than a flag so the
    # daemon is startable by hand when something needs to be watched directly.
    daemon.run(indexer.work(cfg), watch=None)
```

E os subparsers, junto de `repos refresh`:

```python
    p = repsub.add_parser("add-all", help="index a whole repository, in the background")
    p.add_argument("repo")
    p.add_argument("--path", help="the working copy (default: the current directory)")
    p.set_defaults(fn=cmd_repos_add_all)

    p = repsub.add_parser("status", help="what is being indexed, and whether the daemon is up")
    p.set_defaults(fn=cmd_repos_status)

    p = repsub.add_parser("cancel", help="stop indexing a repository; what is indexed stays")
    p.add_argument("repo")
    p.set_defaults(fn=cmd_repos_cancel)

    p = repsub.add_parser("daemon", help="the background indexer")
    p.add_argument("action", choices=["start", "stop", "run"])
    p.set_defaults(fn=cmd_repos_daemon)
```

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/indexer.py cli/qctx.py tests/test_cli_daemon.py
git commit -F - <<'MSG'
feat: add-all, status, cancel — and the worker that keeps the daemon ignorant of Qdrant

`daemon.run` takes a callable and knows nothing else about the work, and this module is what
builds it. That separation is what lets every daemon test run with no network at all.

Cancellation is checked BETWEEN batches rather than inside one: a batch is either indexed or it
is not, so stopping between them leaves the archive consistent — and what was indexed stays,
because a partial index answers questions about the part it has and re-running skips what did
not change.

`status` says whether the daemon is running before it prints any number, because every number
below that line was written by a process that may be gone, and a reader who does not know that
reads stalled progress as activity.
MSG
```

---

### Task 6: Vigiar

**Files:**
- Modify: `core/daemon.py`, `core/indexer.py`
- Test: `tests/test_watch.py`

**Interfaces:**
- Consumes: `scan.eligible`, `RepoIndex.list_repos`, `bindings`, `jobs.enqueue`.
- Produces: `indexer.watcher(cfg=None, index=None) -> callable` — o `watch()` que `daemon.run`
  chama quando não há trabalho pendente.

**O debounce é o que separa isto de reindexar a cada tecla.** Um arquivo só entra na fila quando
estiver estável por um ciclo.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_watch.py
"""Manter o índice em dia sem que ninguém peça.

O daemon percorre os repos indexados e compara `mtime` com o que está no acervo. Medido em
2026-08-18: 16 ms para 2.000 arquivos — barato o bastante para dispensar `inotify`, que exigiria
dependência externa ou código por plataforma.

O DEBOUNCE É O QUE SEPARA ISTO DE REINDEXAR A CADA TECLA: um arquivo só entra na fila depois de
ficar estável por um ciclo.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import indexer, jobs  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class FakeIndex:
    """Um índice que sabe dizer o que mudou, sem Qdrant."""

    def __init__(self, changed=()):
        self._changed = list(changed)
        self.refreshed = []

    def list_repos(self):
        return [{"repo": "alpha", "checkouts": ["/tmp/alpha"]}]

    def changed_paths(self, repo):
        return list(self._changed)

    def refresh(self, repo):
        self.refreshed.append(repo)

        return []


class TestWatching(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_stable_change_becomes_a_refresh_job_on_the_SECOND_sighting(self):
        """Primeira vez: anotado. Segunda vez, ainda mudado: enfileirado."""
        ix = FakeIndex(changed=["/tmp/alpha/a.py"])
        watch = indexer.watcher(index=ix)
        watch()
        self.assertIsNone(jobs.load("alpha"), "enfileirou na primeira vez, sem debounce")
        watch()
        self.assertEqual(jobs.load("alpha")["kind"], "refresh")

    def test_nothing_changed_means_no_job_at_all(self):
        watch = indexer.watcher(index=FakeIndex(changed=[]))
        watch()
        watch()
        self.assertEqual(jobs.all_jobs(), [])

    def test_it_does_not_queue_a_second_job_while_one_is_running(self):
        """Sem isto, cada ciclo empilharia um refresh sobre o anterior."""
        ix = FakeIndex(changed=["/tmp/alpha/a.py"])
        watch = indexer.watcher(index=ix)
        watch(); watch()
        jobs.update("alpha", state=jobs.RUNNING)
        watch(); watch()
        self.assertEqual(jobs.load("alpha")["state"], jobs.RUNNING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_watch 2>&1 | tail -4`
Expected: FAIL — `AttributeError: module 'core.indexer' has no attribute 'watcher'`

- [ ] **Step 3: Implemente `watcher` em `core/indexer.py`**

```python
def watcher(cfg=None, index=None):
    """Returns the `watch()` the daemon calls when no job is pending.

    WHY POLLING AND NOT `inotify`. Measured on 2026-08-18: stat over 2,000 files costs 16 ms, so
    a watch cycle is free — and `inotify` would mean either an external dependency, which this
    project refuses for a documented reason, or Linux-only code.

    WHY A CHANGE MUST BE SEEN TWICE. A file being written is a file that will change again in a
    moment; queueing on the first sighting reindexes on every keystroke of a long save. Seen
    twice with the same content, it is done being written.
    """
    seen: dict = {}

    def watch() -> None:
        target = index if index is not None else _build(cfg)
        for entry in target.list_repos():
            repo = entry["repo"]
            job = jobs.load(repo)
            if job and job.get("state") in (jobs.PENDING, jobs.RUNNING):
                # Already queued or running: a second job would only stack behind the first and
                # describe a disk that has moved on by the time it ran.
                continue
            changed = set(target.changed_paths(repo))
            if not changed:
                seen.pop(repo, None)
                continue
            if seen.get(repo) == changed:
                jobs.enqueue(repo, "refresh", [])
                seen.pop(repo, None)
                continue
            seen[repo] = changed

    return watch
```

E acrescente `changed_paths` a `core/repos.py`, junto de `refresh`:

```python
    def changed_paths(self, repo: str) -> list[str]:
        """The indexed paths of `repo` whose file changed on disk. The cheap half of `refresh`.

        Split out because the watcher asks this question every cycle and must not pay for the
        expensive half — `refresh` re-embeds, this only compares metadata.
        """
        out = []
        for path, md in sorted(self._indexed_sources(repo).items()):
            if source_changed(path, md.get("src_mtime"), md.get("src_size"),
                              md.get("src_digest")):
                out.append(path)

        return out
```

E ligue no comando do daemon, em `cli/qctx.py`:

```python
    daemon.run(indexer.work(cfg), watch=indexer.watcher(cfg))
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_watch 2>&1 | tail -3`
Expected: `Ran 3 tests`, `OK`

- [ ] **Step 5: Prove que o debounce morde**

```bash
cp core/indexer.py /tmp/indexer.bak && [ -s /tmp/indexer.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
p = "core/indexer.py"; s = open(p).read()
old = "            if seen.get(repo) == changed:\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "            if True:\n", 1))
PY
grep -c "if True:" core/indexer.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_watch 2>&1 | grep -E "^FAIL: test|^Ran|^OK|^FAILED"
cp /tmp/indexer.bak core/indexer.py
```
Expected: cai `test_a_stable_change_becomes_a_refresh_job_on_the_SECOND_sighting`.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/indexer.py core/repos.py cli/qctx.py tests/test_watch.py
git commit -F - <<'MSG'
feat: watch by polling, and only act on a change that stopped changing

Measured: stat over 2,000 files costs 16 ms, so a watch cycle is free. inotify would mean either
an external dependency, which this project refuses for a documented reason, or Linux-only code —
and it would buy nothing at this cost.

A change has to be seen twice before it becomes a job. A file being written is a file that will
change again in a moment, and queueing on the first sighting reindexes on every keystroke of a
long save; seen twice unchanged, it is done being written.

`changed_paths` is split out of `refresh` because the watcher asks that question every cycle and
must not pay for the expensive half: this compares metadata, refresh re-embeds.

This is also what covers the case git hooks could not — a file saved and not committed.
MSG
```

---

### Task 7: Detectar e oferecer

**Files:**
- Create: `hooks/lease.py`
- Modify: `cli/qctx.py`, `hosts/hermes/__init__.py`, `hosts/hermes/tools.py`, `hooks/hooks.json`
- Test: `tests/test_repos_init.py`

**Interfaces:**
- Consumes: `RepoIndex.candidates_for(root, remotes) -> {"bound","join","suggest","taken"}`,
  `bindings.git_root`, `bindings.remotes_of`, `lease.write`, `lease.find_host_pid`.
- Produces: CLI `repos init`; ferramenta `repos_candidates` no hermes; hook `SessionStart`.

**`candidates_for` já existe e não tem consumidor.** Esta task é o consumidor que faltava.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_repos_init.py
"""Detectar o repo e OFERECER — nunca indexar sozinho.

`candidates_for` existe no core desde o sub-projeto A e nenhum host a chamava: era código morto.
Isto é o consumidor que faltava, nos dois hosts.
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

QCTX = REPO / "cli" / "qctx.py"


def a_repo() -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)

    return root


def run_cli(*args, cwd=None):
    env = {**os.environ, "QCTX_STATE_DIR": tempfile.mkdtemp()}

    return subprocess.run([sys.executable, str(QCTX), *args], capture_output=True, text=True,
                          cwd=cwd, env=env, timeout=180)


class TestInitOffersAndDoesNotWrite(unittest.TestCase):
    def test_outside_a_repo_it_says_so_instead_of_guessing(self):
        out = run_cli("repos", "init", cwd=tempfile.mkdtemp())
        self.assertIn("not inside a git", (out.stdout + out.stderr).lower())

    def test_inside_a_fresh_repo_it_reports_what_it_WOULD_do(self):
        """Sem TTY não pergunta e não escreve — a mesma regra que `setup` já segue, porque um
        prompt esperando resposta que nunca vem trava a chamada."""
        out = run_cli("repos", "init", "--json", cwd=a_repo())
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        payload = json.loads(out.stdout)
        self.assertIn("suggest", payload)
        self.assertFalse(payload.get("indexed"), "indexou sem consentimento")


class TestTheLeaseHookWritesALease(unittest.TestCase):
    def test_the_hook_writes_a_lease_that_reads_back_alive(self):
        from core import lease
        box = tempfile.mkdtemp()
        env = {**os.environ, "QCTX_STATE_DIR": box}
        out = subprocess.run([sys.executable, str(REPO / "hooks" / "lease.py")],
                             input=json.dumps({"session_id": "s-1"}), capture_output=True,
                             text=True, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        os.environ["QCTX_STATE_DIR"] = box
        live = lease.live()
        self.assertEqual(len(live), 1, f"nenhum lease escrito: {out.stderr[-200:]}")
        self.assertEqual(live[0]["session_id"], "s-1")

    def test_the_hook_never_fails_the_session_even_with_a_broken_payload(self):
        """Um hook que falha o SessionStart é um hook que o usuário desinstala."""
        out = subprocess.run([sys.executable, str(REPO / "hooks" / "lease.py")],
                             input="{not json", capture_output=True, text=True,
                             env={**os.environ, "QCTX_STATE_DIR": tempfile.mkdtemp()},
                             timeout=120)
        self.assertEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_repos_init 2>&1 | tail -4`
Expected: FAIL — o comando `init` não existe e `hooks/lease.py` não existe.

- [ ] **Step 3: Implemente o hook e o comando**

`hooks/lease.py`:

```python
#!/usr/bin/env python3
"""SessionStart: records that this claude-code session is alive.

WHY IT IS ITS OWN HOOK AND NOT A LINE IN THE RECALL ONE. Recall is memory search and a lease is
a sign of life; coupling them would make disabling recall end the daemon under a live host. The
user decided this on 2026-08-18.

WHY SessionStart AND NOT EVERY PROMPT. The lease carries the host's pid and start time, and the
daemon tests whether that process still exists — so it needs writing once, not refreshing. A
per-prompt heartbeat would add work to every turn to answer a question the pid already answers.

IT NEVER FAILS THE SESSION. A hook that can break SessionStart is a hook that gets uninstalled.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    try:
        from core import lease

        found = lease.find_host_pid(names=("claude",))
        pid = found[0] if found else os.getppid()
        lease.write(str(payload.get("session_id") or "default"), "claude", pid=pid)
    except Exception:                               # noqa: BLE001 — see the module docstring
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
```

Registre em `hooks/hooks.json`, no mesmo formato dos outros:

```json
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/lease.py\"",
            "shell": "bash",
            "timeout": 5
          }
        ]
      }
    ],
```

O comando `repos init`, em `cli/qctx.py`:

```python
def cmd_repos_init(args, cfg):
    from core import bindings

    root = bindings.git_root(args.path or os.getcwd())
    if not root:
        raise SystemExit("not inside a git working copy — run this from a project, or pass --path")
    found = core.build_repos(cfg).candidates_for(root, bindings.remotes_of(root))
    found["root"] = root
    found["indexed"] = False
    if args.json:
        output(found, True)

        return
    if found["bound"]:
        print(f"already indexed as {found['bound']!r}")

        return
    if found["join"]:
        names = ", ".join(sorted(r["repo"] for r in found["join"]))
        print(f"this working copy shares a remote with: {names}")
    if found["taken"]:
        # Named rather than merged: two unrelated checkouts with the same directory name are not
        # the same repository, and deciding otherwise by an accident of naming is what the
        # declared-identity rule exists to refuse.
        print(f"the name {found['suggest']!r} already belongs to another repository")
    print(f"index this working copy as:  qctx repos add-all {found['suggest']}")
```

E em `hosts/hermes/__init__.py`, dentro de `initialize`, antes do fim:

```python
        # The provider runs INSIDE hermes, so this process IS the host: no tree to walk.
        try:
            from core import lease

            lease.write(self._session_id, "hermes")
        except Exception:                           # noqa: BLE001
            pass                                    # a missing lease costs a daemon that ends
                                                    # sooner, never a broken session
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_repos_init 2>&1 | tail -3`
Expected: `Ran 4 tests`, `OK`

- [ ] **Step 5: Prove que `init` não escreve**

```bash
cp cli/qctx.py /tmp/qctx.bak && [ -s /tmp/qctx.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
p = "cli/qctx.py"; s = open(p).read()
old = '    found["indexed"] = False\n'
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, '    found["indexed"] = True\n', 1))
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_repos_init 2>&1 | grep -E "^FAIL: test|^Ran|^OK|^FAILED"
cp /tmp/qctx.bak cli/qctx.py
```
Expected: cai `test_inside_a_fresh_repo_it_reports_what_it_WOULD_do`.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add hooks/lease.py hooks/hooks.json cli/qctx.py hosts/hermes/__init__.py tests/test_repos_init.py
git commit -F - <<'MSG'
feat: detect the repository and OFFER — plus the lease each host writes

candidates_for has existed in the core since sub-project A and no host ever called it: dead code
with the consumer missing. This is the consumer, on both hosts, and it offers rather than acts —
indexing never starts without a yes.

The lease is its own hook rather than a line in the recall one, decided by the user: recall is
memory search and a lease is a sign of life, and coupling them would make disabling recall end
the daemon under a live host. It runs at SessionStart and not every prompt, because the lease
carries a pid whose existence the daemon tests directly — a heartbeat would add work to every
turn to answer a question the pid already answers.

On hermes there is no tree to walk: the provider runs inside the host process, so os.getpid() is
the answer. On claude-code the hook walks up to find `claude`.

Neither can fail a session. A hook that breaks SessionStart is a hook that gets uninstalled, and
a missing lease costs a daemon that ends sooner — never a broken session.
MSG
```

---

### Task 8: Tirar o hook de git, e documentar

**Files:**
- Delete: `core/githook.py`, `tests/test_githook.py`
- Modify: `cli/qctx.py`, `README.md`, `skills/repo-index/SKILL.md`,
  `tests/test_host_equivalence.py`
- Test: `tests/test_host_equivalence.py` (atualizar)

**Interfaces:** nenhuma nova.

**O `install-hook` sai porque o daemon o substitui.** Duas coisas indexando o mesmo arquivo não
é redundância que soma.

- [ ] **Step 1: Remova, e ajuste a lista declarada**

```bash
git rm -q core/githook.py tests/test_githook.py
```

Em `cli/qctx.py`, remova `cmd_repos_install_hook` e o subparser `install-hook`.

Em `tests/test_host_equivalence.py`, atualize os dois pontos: tire `repos_install_hook` de
`NOT_FOR_THE_MODEL` e acrescente os verbos novos a `cli_verbs`:

```python
    NOT_FOR_THE_MODEL = {"setup", "collections_list", "config_show", "config_set",
                         "config_detect", "repos_daemon", "repos_init", "repos_add_all",
                         "repos_status", "repos_cancel"}
```

```python
        cli_verbs = {"list", "search", "add", "drop", "register", "refresh"}
```

**Por que os novos não são ferramentas do modelo:** `daemon`, `add-all`, `status`, `cancel` e
`init` administram um processo e o consentimento de indexação. Quem digita decidiu; um modelo
chamando decidiu pelo usuário. É a mesma linha que já separa `config set`.

- [ ] **Step 2: Rode e veja o que quebra**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | grep -E "^(FAIL|ERROR): " | sed 's/ (.*//' | sort -u`
Expected: só os testes de equivalência, que a Step 1 já ajustou. Se algo mais cair, é achado.

- [ ] **Step 3: Documente**

No `README.md`, na tabela de comandos, substitua a linha do `install-hook`:

```markdown
| `repos init` | — *(CLI only)* | detect this working copy and offer to index it |
| `repos add-all` | — *(CLI only)* | index the whole repository, in the background |
| `repos status` | — *(CLI only)* | what is indexing, and whether the daemon is up |
| `repos cancel` | — *(CLI only)* | stop indexing; what is already indexed stays |
| `repos daemon` | — *(CLI only)* | start, stop, or run the background indexer |
```

E acrescente uma seção, depois da tabela:

```markdown
### Indexing a whole project

```bash
cd ~/dev/my-project
qctx repos init                  # detects the repo and offers a name
qctx repos add-all my-project    # queues it; the daemon does the work
qctx repos status                # progress, and whether the daemon is up
```

The daemon indexes in the background, so your terminal is free. It also **watches** the
repositories it indexed: a file you change is reindexed within a few seconds, including one you
have not committed. Cancelling keeps whatever was already indexed, and running `add-all` again
skips the files that did not change.

**It ends when you do.** Each session writes a lease with its host's pid; when the last claude or
hermes exits — cleanly or killed — the daemon notices within a cycle and stops. Nothing is left
running behind you.
```

Na skill `skills/repo-index/SKILL.md`, troque o parágrafo do `install-hook` por:

```markdown
To index a whole project, `qctx repos init` detects the repository and offers a name, and
`qctx repos add-all <name>` queues it. The work happens in a background daemon — the terminal
stays free, `qctx repos status` shows progress, and `qctx repos cancel <name>` stops it while
keeping what was already indexed. The daemon then watches that repository and reindexes changed
files by itself, so `[stale]` becomes rare rather than routine.
```

- [ ] **Step 4: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add -A
git commit -F - <<'MSG'
refactor: one mechanism — the daemon replaces the git hook

The post-commit hook shipped earlier on 2026-08-18 to give repositories an update trigger without
a background process. The daemon covers everything it covered and the case it could not — a file
saved and not committed — so keeping both would mean two things indexing the same file, which is
not redundancy that adds up. The user asked for one mechanism.

The new commands are CLI-only, on the same line that already withholds `config set`: they
administer a process and the consent to index a project. A person typing them has decided; a
model calling them has decided for the user.
MSG
```

---

## Auto-revisão do plano

**Cobertura da spec, seção por seção:**

| Item da spec | Task |
|---|---|
| B — `git ls-files` como fonte | T1 |
| B — os quatro descartes | T1 (um teste por descarte + sonda por filtro) |
| B — funil relatado antes de começar | T1, T5 (`add-all` imprime) |
| C — `candidates_for` ganha consumidor | T7 |
| C — CLI pergunta; sem TTY só relata | T7 |
| C — indexar nunca começa sem aceite | T7 (sonda) |
| D — daemon único, sob demanda | T4 (`start` com `record()`) |
| D — estado em disco, `status` sem daemon | T3, T5 |
| D — cancelar mantém o parcial | T3, T5 (teste do executor) |
| D — um trabalho por vez | T3 (`next_pending`) |
| D — trabalho órfão vira interrompido | T3 (`reap`) |
| E — polling, `git ls-files` + mtime | T6 |
| E — debounce de um ciclo | T6 (sonda) |
| E — pega edição não commitada | T6 (é consequência do polling; dito no commit) |
| Lease com `(pid, starttime)` | T2 |
| Lease escrito fora do recall | T7 (hook próprio + `initialize`) |
| Daemon morre com o último host | T2, T4 (três testes) |
| Remoção do githook | T8 |
| Só stdlib | todas — nenhum `import` fora da stdlib |
| Nenhum teste sobe daemon real | T4, T5, T6 (executor e ciclos injetados) |

**Placeholders:** nenhum "TBD"/"TODO"/"similar à Task N". Todo passo de código tem o código.

**Consistência de tipo:** `scan.eligible -> dict` com `eligible: list[str]` é consumido por
`jobs.enqueue(repo, kind, paths)` em T5; `jobs.load -> dict | None` é lido por `indexer.work` e
por `cmd_repos_status`; `lease.live() -> list[dict]` é chamado por `daemon.run` sem argumento;
`daemon.record() -> dict | None` é lido por `start` e por `cmd_repos_status`;
`indexer.work(cfg, index, batch) -> callable` e `indexer.watcher(cfg, index) -> callable` são
passados a `daemon.run(work, watch=...)`; `RepoIndex.changed_paths(repo) -> list[str]` (T6) é
chamado pelo watcher.

**Consistência de valor:** `CYCLE_S = 5.0`, `BATCH = 8`, `MAX_FILE_BYTES = 1_048_576`,
`MINIFIED_LINE_CHARS = 2_000` aparecem uma vez cada, no módulo que os define.
