# Wizard de instalação e verificação — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** uma máquina nova sai de "acabei de instalar o plugin" para "tudo verde" com um
comando, e o mesmo comando reverifica depois sem escrever nada.

**Architecture:** `scripts/install.sh` é bash fino e só faz `exec` no Python. `bin/qctx`
ganha um resolvedor e passa a ser copiado para `~/.local/bin`. `core/install.py` traz os
checks de encanamento como dados (host-neutro). `cli/qctx.py` ganha `qctx install`, que
renderiza, pergunta, e delega o estado de cada host aos dois cutovers que já existem.

**Tech Stack:** Python 3 **só stdlib** (`filecmp`, `shutil`, `getpass`, `subprocess`,
`argparse`, `unittest`), bash, e os manifestos que já existem.

**Spec:** `docs/superpowers/specs/2026-08-19-install-wizard-design.md`

## Global Constraints

- **Stdlib apenas.** Nenhuma dependência nova, em nenhum arquivo. O core roda dentro de
  hook disparado a cada interação, e dependência faltando vira perda silenciosa de função.
- **Inglês** em código, comentários, docstrings, mensagens ao usuário, README e commits.
  Este plano e a spec são em português; nada que eles contêm em português vai para o código.
- **`core/` não conhece host.** `core/install.py` nunca escreve "claude" nem "hermes".
  Detecção de host e chamada dos cutovers vivem em `cli/qctx.py`.
- **Segredo nunca no `config.json`.** `core.save()` já recusa (`SECRET_FIELDS`); nada neste
  plano contorna isso.
- **Testes offline**, sem rede, com `HOME` e `QCTX_CONFIG` apontando para diretório
  temporário. Rodar tudo com `python3 -m unittest discover -s tests`.
- **Os 15 campos** de `core.config.DEFAULTS` são a definição de "configuração completa":
  `qdrant_url`, `qdrant_api_key`, `api_base_url`, `api_key`, `embed_url`, `rerank_url`,
  `embed_model`, `rerank_model`, `memory_collection`, `docs_collection`,
  `library_collection`, `repos_collection`, `repos_registry_collection`, `vector_size`,
  `context_window`.

## Estrutura de arquivos

| arquivo | responsabilidade | tarefa |
|---|---|---|
| `bin/qctx` | resolver a árvore viva e fazer `exec`; `--root` imprime o que resolveu | 1 |
| `tests/test_qctx_launcher.py` | a ordem de resolução, com árvores falsas | 1 |
| `core/install.py` | checks de encanamento e a lista de campos, como dados | 2, 5 |
| `tests/test_install_core.py` | launcher, PATH, config sem shell, completude dos campos | 2, 5 |
| `core/setup.py` | `_check_context_window`, somado a `diagnose()` | 3 |
| `tests/test_setup_context_window.py` | avisa quando não declarado, cala quando declarado | 3 |
| `cli/qctx.py` | `cmd_install`: render, prompts, hosts, apply | 4, 5, 6 |
| `tests/test_cli_install.py` | `--check` não escreve; passadas; destino dos segredos; hosts | 4, 5, 6 |
| `scripts/cutover.sh` | ganha `CUTOVER_SKIP_SUITE`, espelhando o do hermes | 4 |
| `scripts/install.sh` | bootstrap: acha árvore + python, `exec` | 7 |
| `tests/test_install_script.py` | roda o script contra árvore falsa | 7 |
| `README.md` | seção de instalação reescrita, números corrigidos | 8 |
| `tests/test_readme_fidelity.py` | comandos citados existem; contagens batem | 8 |

## Desvio consciente da spec, declarado aqui

A spec diz que os dois cutovers ficam **inalterados**. A tarefa 4 faz **uma** mudança em
`scripts/cutover.sh`: uma variável `CUTOVER_SKIP_SUITE`, cópia exata da
`HERMES_CUTOVER_SKIP_SUITE` que `hermes_cutover.sh` já tem (pula a suíte no dry-run, e
`--apply` **recusa** rodar com ela ligada). Sem isso, montar o plano custa 41 s por host
rodando a suíte inteira só para exibir uma lista. Nenhuma outra linha dos cutovers muda.

---

### Task 1: o resolvedor em `bin/qctx`

**Files:**
- Modify: `bin/qctx`
- Test: `tests/test_qctx_launcher.py`

**Interfaces:**
- Consumes: nada.
- Produces: `qctx --root` imprime o diretório raiz resolvido e sai 0. Ordem de resolução:
  `$QCTX_HOME` > a árvore do próprio script (symlinks resolvidos) >
  `${HERMES_HOME:-$HOME/.hermes}/plugins/memories` > `installPath` de
  `$HOME/.claude/plugins/installed_plugins.json`. Uma árvore só conta se tiver
  `cli/qctx.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qctx_launcher.py
"""The launcher's resolution order.

`bin/qctx` is COPIED to `~/.local/bin/qctx` by the wizard, so it cannot rely on sitting
inside the tree it runs. It resolves the live install on every call instead — which is what
makes it survive `claude plugin update`, where the install path is a fresh directory named
after the new commit.
"""
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "bin" / "qctx"


def fake_tree(root: Path) -> Path:
    """The minimum a directory needs to count as an install."""
    (root / "cli").mkdir(parents=True)
    (root / "cli" / "qctx.py").write_text("raise SystemExit(0)\n")
    (root / "bin").mkdir()
    return root


def run_root(launcher: Path, env: dict) -> str:
    done = subprocess.run([str(launcher), "--root"], capture_output=True, text=True,
                          env=env, timeout=30)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


class LauncherResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.env = {"HOME": str(self.home), "PATH": os.environ["PATH"]}
        self.addCleanup(self.tmp.cleanup)

    def test_qctx_home_wins_over_everything(self):
        chosen = fake_tree(Path(self.tmp.name) / "chosen")
        fake_tree(self.home / ".hermes" / "plugins" / "memories")
        self.env["QCTX_HOME"] = str(chosen)
        self.assertEqual(run_root(LAUNCHER, self.env), str(chosen))

    def test_own_tree_when_no_override(self):
        # Invoked from the repository itself: the tree it lives in is the answer.
        self.assertEqual(run_root(LAUNCHER, self.env), str(REPO))

    def test_copy_outside_a_tree_finds_the_hermes_install(self):
        tree = fake_tree(self.home / ".hermes" / "plugins" / "memories")
        copy = self.home / ".local" / "bin" / "qctx"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(LAUNCHER.read_bytes())
        copy.chmod(0o755)
        self.assertEqual(run_root(copy, self.env), str(tree))

    def test_copy_falls_through_to_the_claude_install_path(self):
        tree = fake_tree(Path(self.tmp.name) / "cache" / "b8008f7dac88")
        registry = self.home / ".claude" / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"version": 2, "plugins": {
            "memories-plugin@memories-plugin": [{"installPath": str(tree)}]}}))
        copy = self.home / ".local" / "bin" / "qctx"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(LAUNCHER.read_bytes())
        copy.chmod(0o755)
        self.assertEqual(run_root(copy, self.env), str(tree))

    def test_nothing_to_resolve_fails_loudly(self):
        copy = self.home / ".local" / "bin" / "qctx"
        copy.parent.mkdir(parents=True)
        copy.write_bytes(LAUNCHER.read_bytes())
        copy.chmod(0o755)
        done = subprocess.run([str(copy), "--root"], capture_output=True, text=True,
                              env=self.env, timeout=30)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("could not find", done.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_qctx_launcher -v`
Expected: FAIL — `bin/qctx` não conhece `--root` nem `QCTX_HOME`; `run_root` estoura no
`assert done.returncode == 0`.

- [ ] **Step 3: Write minimal implementation**

Substituir o corpo de `bin/qctx` (mantendo o cabeçalho de comentário, atualizado):

```bash
#!/usr/bin/env bash
# CLI launcher, so `qctx` works from any directory AND from any install shape.
#
# It exists because the skills and the documentation need to cite ONE stable command.
# Without it, every invocation would have to carry the plugin's install path, which
# changes with the version — and an instruction with a fragile path is an instruction
# that stops working without anyone noticing.
#
# `qctx install` COPIES this file to ~/.local/bin/qctx, so it must not assume it sits
# inside the tree it runs: it resolves the live install on every call. That is what
# survives `claude plugin update`, which installs into a directory named after the new
# commit and leaves the old one behind.
set -euo pipefail

# The tree this script lives in, with symlinks resolved — the development install is a
# symlink, and following it is the whole reason this loop exists.
own_tree() {
  local target="${BASH_SOURCE[0]}" dest
  while [ -L "$target" ]; do
    dest="$(readlink "$target")"
    case "$dest" in
      /*) target="$dest" ;;
      *)  target="$(cd -P "$(dirname "$target")" && pwd)/$dest" ;;
    esac
  done
  cd -P "$(dirname "$target")/.." && pwd
}

# The claude-code install path is not guessable: the cache directory is named after the
# commit. The harness records the live one, so read it instead of globbing.
claude_tree() {
  local registry="$HOME/.claude/plugins/installed_plugins.json"
  [ -f "$registry" ] || return 0
  python3 - "$registry" <<'PY' 2>/dev/null || true
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
for entry in data.get("plugins", {}).get("memories-plugin@memories-plugin", []):
    if entry.get("installPath"):
        print(entry["installPath"])
        break
PY
}

resolve_root() {
  local candidate
  for candidate in \
      "${QCTX_HOME:-}" \
      "$(own_tree)" \
      "${HERMES_HOME:-$HOME/.hermes}/plugins/memories" \
      "$(claude_tree)"; do
    if [ -n "$candidate" ] && [ -f "$candidate/cli/qctx.py" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'qctx: could not find the plugin tree. Set QCTX_HOME to the checkout.\n' >&2
  return 1
}

root="$(resolve_root)"

# Prints what it resolved and exits — the one thing a person debugging a stale copy needs,
# and what the wizard's own check reports.
if [ "${1:-}" = "--root" ]; then
  printf '%s\n' "$root"
  exit 0
fi

exec python3 "$root/cli/qctx.py" "$@"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_qctx_launcher -v`
Expected: PASS, 5 testes.

- [ ] **Step 5: Commit**

```bash
git add bin/qctx tests/test_qctx_launcher.py
git commit -m "feat: the launcher resolves the live install instead of assuming its own"
```

---

### Task 2: os checks de encanamento em `core/install.py`

**Files:**
- Create: `core/install.py`
- Test: `tests/test_install_core.py`

**Interfaces:**
- Consumes: `core.setup.Check(name, ok, detail, fix_hint=None, warning=False)`;
  `core.config.load(path, env)`.
- Produces:
  - `LAUNCHER_NAME = "qctx"`
  - `target_dir(env: dict) -> Path` — `~/.local/bin` a partir de `env["HOME"]`
  - `launcher_check(root: Path, env: dict) -> Check`
  - `path_check(env: dict) -> Check`
  - `no_shell_check(path: Path | None = None) -> Check`
  - `plumbing(root: Path, env: dict, config_path: Path | None = None) -> list[Check]` —
    os três acima, nessa ordem.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_install_core.py
"""The plumbing the wizard installs, checked as data.

Host-neutral by contract: this module is the one the other host imports too, so a check
that named claude or hermes would be a check the other host cannot use. The host sections
live in the CLI.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import install  # noqa: E402


class Plumbing(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        (self.home / ".local" / "bin").mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def env(self, on_path=True):
        bindir = self.home / ".local" / "bin"
        return {"HOME": str(self.home), "PATH": str(bindir) if on_path else "/usr/bin"}

    def test_launcher_missing_is_a_blocker_and_names_where_it_goes(self):
        check = install.launcher_check(REPO, self.env())
        self.assertFalse(check.ok)
        self.assertFalse(check.warning)
        self.assertIn(str(self.home / ".local" / "bin"), check.fix_hint)

    def test_launcher_identical_copy_is_ok(self):
        copy = self.home / ".local" / "bin" / "qctx"
        copy.write_bytes((REPO / "bin" / "qctx").read_bytes())
        copy.chmod(0o755)
        self.assertTrue(install.launcher_check(REPO, self.env()).ok)

    def test_launcher_stale_copy_is_reported(self):
        copy = self.home / ".local" / "bin" / "qctx"
        copy.write_text("#!/usr/bin/env bash\necho old\n")
        copy.chmod(0o755)
        check = install.launcher_check(REPO, self.env())
        self.assertFalse(check.ok)
        self.assertIn("differs", check.detail)

    def test_a_symlink_to_the_tree_counts_as_current(self):
        link = self.home / ".local" / "bin" / "qctx"
        link.symlink_to(REPO / "bin" / "qctx")
        self.assertTrue(install.launcher_check(REPO, self.env()).ok)

    def test_bin_dir_off_the_path_is_reported(self):
        self.assertFalse(install.path_check(self.env(on_path=False)).ok)
        self.assertTrue(install.path_check(self.env()).ok)

    def test_no_shell_check_reads_the_file_only(self):
        cfg_path = Path(self.tmp.name) / "config.json"
        cfg_path.write_text('{"qdrant_url": "https://q", "api_base_url": "https://e/v1",'
                            ' "memory_collection": "mem"}')
        self.assertTrue(install.no_shell_check(cfg_path).ok)

    def test_no_shell_check_ignores_the_environment(self):
        """The failure this exists to catch: exported URLs, empty file."""
        import os
        cfg_path = Path(self.tmp.name) / "empty.json"
        cfg_path.write_text("{}")
        os.environ["QCTX_QDRANT_URL"] = "https://exported"
        self.addCleanup(os.environ.pop, "QCTX_QDRANT_URL", None)
        check = install.no_shell_check(cfg_path)
        self.assertFalse(check.ok)
        self.assertIn("qdrant_url", check.detail)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_install_core -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'core.install'`.

- [ ] **Step 3: Write minimal implementation**

```python
# core/install.py
"""Checks for the plumbing the wizard installs: the launcher, the PATH, and the file a
process with no shell reads.

DATA, not text — the same contract as `core/setup.py`, and it returns that module's
`Check` so one renderer serves both.

HOST-NEUTRAL, deliberately. `core/setup.py` states the rule and the reason; this module
obeys it. Which host is installed, and running its cutover, is the CLI's business, which
is where the two host names already appear in this repository.
"""
import filecmp
import shutil
from pathlib import Path

from .config import load
from .setup import COMMAND_PREFIX, Check

#: The command every skill and every page of documentation cites.
LAUNCHER_NAME = "qctx"

#: What a shell-less process needs to find in the FILE. The two keys are absent on
#: purpose: `save()` refuses them, so a config file is complete without them.
NO_SHELL_FIELDS = ("qdrant_url", "memory_collection")


def target_dir(env: dict) -> Path:
    """Where the launcher goes. One location, so the fix hint can name it."""
    return Path(env["HOME"]) / ".local" / "bin"


def launcher_check(root: Path, env: dict) -> Check:
    source = root / "bin" / LAUNCHER_NAME
    found = shutil.which(LAUNCHER_NAME, path=env.get("PATH", ""))
    if not found:
        return Check("launcher", False, f"{LAUNCHER_NAME} is not on PATH",
                     f"{COMMAND_PREFIX} install copies it to "
                     f"{target_dir(env) / LAUNCHER_NAME}")
    # Bytes, not a version marker: a number is one more thing to remember to bump, and
    # this one would go stale the same way the manifests' `version` did.
    if not filecmp.cmp(found, source, shallow=False):
        return Check("launcher", False, f"{found} differs from {source}",
                     f"{COMMAND_PREFIX} install refreshes the copy")

    return Check("launcher", True, str(found))


def path_check(env: dict) -> Check:
    wanted = target_dir(env)
    entries = [Path(p) for p in env.get("PATH", "").split(":") if p]
    if wanted in entries:
        return Check("PATH", True, f"{wanted} is on PATH")

    return Check("PATH", False, f"{wanted} is not on PATH",
                 f'add `export PATH="$HOME/.local/bin:$PATH"` to your shell rc')


def no_shell_check(path: Path | None = None) -> Check:
    """What a process that inherits NO environment would read.

    `config show` mixes the file and the environment, so it prints a complete picture over
    an empty file. A hermes started by systemd, by the gateway or by cron gets the file and
    nothing else, and the symptom of an empty file is not an error — it is an archive that
    looks simply empty.
    """
    cfg = load(path=path, env={})
    missing = [f for f in NO_SHELL_FIELDS if not getattr(cfg, f)]
    if not cfg.embed_url and not cfg.api_base_url:
        missing.append("embed_url or api_base_url")
    if missing:
        return Check("no-shell config", False,
                     f"the config FILE is missing: {', '.join(missing)}",
                     f"{COMMAND_PREFIX} config set … writes the file; exporting only sets "
                     f"it for your own shell")

    return Check("no-shell config", True, "the file alone is enough to run")


def plumbing(root: Path, env: dict, config_path: Path | None = None) -> list[Check]:
    return [launcher_check(root, env), path_check(env), no_shell_check(config_path)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_install_core -v`
Expected: PASS, 7 testes.

- [ ] **Step 5: Commit**

```bash
git add core/install.py tests/test_install_core.py
git commit -m "feat: the plumbing checks, as data and without naming a host"
```

---

### Task 3: `context_window` no diagnóstico

**Files:**
- Modify: `core/setup.py` (nova função + uma linha em `diagnose`)
- Test: `tests/test_setup_context_window.py`

**Interfaces:**
- Consumes: `core.config.Config`.
- Produces: `_check_context_window(cfg) -> Check` (aviso, nunca blocker), presente em
  `diagnose(cfg)["checks"]` com `name == "Context window"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_context_window.py
"""`context_window` is the one setting nothing checked.

Measured on 2026-08-17: the big-file guard resolves the window per model name, and a model
outside `core/windows.py::MODEL_WINDOWS` resolves to 0 — and 0 allows every read. The
hermes model of the day (`MiniMax-M2.7`) is exactly such a model. Until now only
`scripts/hermes_cutover.sh` reported it, so on a claude-only machine nothing said a word.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import config, setup  # noqa: E402


def cfg_with(**over):
    values = dict(config.DEFAULTS)
    values.update(over)
    return config.Config(**values)


class ContextWindowCheck(unittest.TestCase):
    def test_absent_is_a_warning_not_a_blocker(self):
        check = setup._check_context_window(cfg_with(context_window=0))
        self.assertFalse(check.ok)
        self.assertTrue(check.warning)
        self.assertIn("config set context-window", check.fix_hint)

    def test_declared_is_ok_and_says_the_value(self):
        check = setup._check_context_window(cfg_with(context_window=200000))
        self.assertTrue(check.ok)
        self.assertIn("200000", check.detail)

    def test_diagnose_includes_it(self):
        """Reachability is irrelevant here: the check must be present even with Qdrant
        down, which is what an offline suite gives us."""
        names = [c["name"] for c in setup.diagnose(cfg_with())["checks"]]
        self.assertIn("Context window", names)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_setup_context_window -v`
Expected: FAIL — `AttributeError: module 'core.setup' has no attribute
'_check_context_window'`.

- [ ] **Step 3: Write minimal implementation**

Em `core/setup.py`, logo depois de `_check_rerank`:

```python
def _check_context_window(cfg: Config) -> Check:
    """Not declared is not an error — it is a silence with a cost, so it is a WARNING.

    The big-file guard resolves the window from the model name when this is 0, and a model
    outside the table resolves to 0 too — which allows every read. Whoever runs a 1M
    variant is right not to declare it; whoever runs a 200k model, or any model the table
    does not know, has a guard that is installed and inert.
    """
    if cfg.context_window:
        return Check("Context window", True, f"{cfg.context_window} declared")

    return Check("Context window", False,
                 "not declared — resolved per model, and a model outside the table "
                 "resolves to 0, which allows every read",
                 f"{COMMAND_PREFIX} config set context-window <n> — only if your model is "
                 f"not a 1M variant", warning=True)
```

E em `diagnose`, na montagem da lista:

```python
    checks = [check_q, check_emb, _check_rerank(cfg), _check_context_window(cfg)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_setup_context_window -v`
Expected: PASS, 3 testes.

Depois: `python3 -m unittest discover -s tests` — algum teste existente pode contar os
checks de `diagnose`. Se contar, atualizar o número **e** conferir que ele conta por nome,
não por posição.

- [ ] **Step 5: Commit**

```bash
git add core/setup.py tests/test_setup_context_window.py
git commit -m "feat: diagnose reports an undeclared context window, which nothing did"
```

---

### Task 4: `qctx install --check`

**Files:**
- Modify: `cli/qctx.py` (novo `cmd_install`, novo `add_parser`)
- Modify: `scripts/cutover.sh` (só a variável de pular a suíte)
- Test: `tests/test_cli_install.py`

**Interfaces:**
- Consumes: `core.install.plumbing`, `core.setup.diagnose`, `_render_check`.
- Produces:
  - `HOST_SECTIONS: tuple` — `(("claude-code", "scripts/cutover.sh", "CUTOVER_SKIP_SUITE"),
    ("hermes", "scripts/hermes_cutover.sh", "HERMES_CUTOVER_SKIP_SUITE"))`
  - `_host_dry_run(name, script, skip_var, root) -> dict` com chaves
    `{"host", "exit_code", "text"}`
  - `cmd_install(args, cfg)` respondendo a `--check`, `--json`, `--yes`
  - subcomando `qctx install`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_install.py
"""`qctx install` — the wizard.

`--check` is the mode everything else is measured against: it must report the same picture
and write NOTHING. A wizard that repairs while you are looking cannot be used to find out
what state a machine is in.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "cli" / "qctx.py"


class CheckMode(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

    def run_cli(self, *argv):
        env = dict(os.environ)
        env.update({"HOME": str(self.home), "QCTX_CONFIG": str(self.config),
                    "QCTX_QDRANT_URL": "", "PATH": "/usr/bin:/bin"})
        return subprocess.run([sys.executable, str(CLI), "install", *argv],
                              capture_output=True, text=True, env=env, timeout=180,
                              stdin=subprocess.DEVNULL)

    def test_check_writes_nothing(self):
        before = self.config.read_text()
        self.run_cli("--check")
        self.assertEqual(self.config.read_text(), before)
        self.assertFalse((self.home / ".local").exists())

    def test_check_reports_the_plumbing_section(self):
        done = self.run_cli("--check")
        self.assertIn("launcher", done.stdout)
        self.assertIn("no-shell config", done.stdout)

    def test_json_is_parseable_and_carries_both_shapes(self):
        done = self.run_cli("--check", "--json")
        payload = json.loads(done.stdout)
        self.assertIn("checks", payload)
        self.assertIn("hosts", payload)
        for section in payload["hosts"]:
            self.assertEqual({"host", "exit_code", "text"}, set(section))

    def test_absent_host_is_skipped_not_failed(self):
        """A machine with only one of the two hosts is the normal case."""
        done = self.run_cli("--check")
        self.assertNotIn("Traceback", done.stderr)
        self.assertEqual(done.returncode, 0)

    def test_no_tty_never_blocks(self):
        """stdin closed, no --yes: it diagnoses and exits, like `qctx setup`."""
        done = self.run_cli()
        self.assertEqual(done.returncode, 0)
        self.assertIn("no interactive terminal", done.stdout)


class SkipSuiteVariable(unittest.TestCase):
    def test_cutover_refuses_to_apply_with_the_suite_skipped(self):
        """Copied from the hermes script, including the refusal — the flag exists to make
        the PLAN cheap, never to make an apply cheap."""
        env = dict(os.environ, CUTOVER_SKIP_SUITE="1")
        done = subprocess.run(["bash", str(REPO / "scripts" / "cutover.sh"), "--apply"],
                              capture_output=True, text=True, env=env, timeout=120)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("suite unverified", done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli_install -v`
Expected: FAIL — `argument command: invalid choice: 'install'`.

- [ ] **Step 3: Write minimal implementation**

Em `cli/qctx.py`, depois de `cmd_setup`:

```python
# ---- install ---------------------------------------------------------------

#: The host sections, and the variable each script reads to skip its own test suite.
#: The scripts stay the owners of host state: they already back up what they edit and
#: re-read it to confirm the write, and a second implementation of those checks here would
#: be a second source of truth that diverges at the first fix.
HOST_SECTIONS = (
    ("claude-code", "scripts/cutover.sh", "CUTOVER_SKIP_SUITE"),
    ("hermes", "scripts/hermes_cutover.sh", "HERMES_CUTOVER_SKIP_SUITE"),
)

#: The command that tells us a host is on this machine at all.
HOST_BINARIES = {"claude-code": "claude", "hermes": "hermes"}


def _host_dry_run(host: str, script: str, skip_var: str, root: Path) -> dict:
    """Runs a cutover in its report-only mode. Writes nothing; that is the script's
    contract with no `--apply`.

    The suite is skipped HERE and only here: both scripts run the full suite among their
    checks, which costs ~41s per host to draw a list. It runs once before any apply, and
    both scripts refuse to apply with the variable set.
    """
    env = dict(os.environ, **{skip_var: "1"})
    done = subprocess.run(["bash", str(root / script)], capture_output=True, text=True,
                          env=env, timeout=300)

    return {"host": host, "exit_code": done.returncode, "text": done.stdout + done.stderr}


def _host_sections(root: Path) -> list[dict]:
    sections = []
    for host, script, skip_var in HOST_SECTIONS:
        if not shutil.which(HOST_BINARIES[host]):
            sections.append({"host": host, "exit_code": 0,
                             "text": f"  ..    {HOST_BINARIES[host]} is not on PATH — "
                                     f"skipping this host\n"})
            continue
        sections.append(_host_dry_run(host, script, skip_var, root))

    return sections


def cmd_install(args, cfg):
    """The wizard: diagnose, then offer to fix, one group at a time.

    With `--check` it only reports. With no TTY and no `--yes` it also only reports — the
    same rule `cmd_setup` follows, and for the same reason: this command is called by
    agents and by scripts, and an `input()` waiting for an answer that never comes hangs
    the caller.
    """
    root = Path(__file__).resolve().parent.parent
    report = core.setup.diagnose(cfg)
    plumbing = [asdict(c) for c in core.install.plumbing(root, dict(os.environ))]
    hosts = _host_sections(root)

    if args.json:
        output({**report, "checks": plumbing + report["checks"], "hosts": hosts}, True)

        return

    print("plumbing:\n")
    for c in plumbing:
        _render_check(c)
    print("\nreachability and configuration:\n")
    for c in report["checks"]:
        _render_check(c)
    for section in hosts:
        print(f"\n{section['host']}:\n")
        print(section["text"].rstrip())

    if args.check:
        return
    if not sys.stdin.isatty() and not args.yes:
        print("\n(no interactive terminal — nothing was changed)")

        return
    print("\n(the writing pass lands in the next task)")
```

Imports novos no topo do arquivo: `import shutil`, `import subprocess`, `from dataclasses
import asdict`, `from pathlib import Path` (conferir quais já existem antes de acrescentar).

No `build_parser`, junto do `setup`:

```python
    p = sub.add_parser("install", help="install and verify everything, step by step")
    p.add_argument("--check", action="store_true", help="report only; never writes")
    p.add_argument("--yes", action="store_true", help="answer yes to every group")
    p.set_defaults(fn=cmd_install)
```

Em `scripts/cutover.sh`, substituir o bloco da suíte:

```bash
# Skipping the suite is for the PLAN phase, where running it costs 41 seconds to draw a
# list. It is never a way to make an apply cheaper — mirroring HERMES_CUTOVER_SKIP_SUITE.
if [ -n "${CUTOVER_SKIP_SUITE:-}" ]; then
  if [ "$APPLY" = "--apply" ]; then
    fail "CUTOVER_SKIP_SUITE is set — refusing to --apply with the suite unverified"
  else
    note "suite skipped (CUTOVER_SKIP_SUITE)"
  fi
elif python3 -m unittest discover -s "$ROOT/tests" >/dev/null 2>&1; then
  ok "offline suite passes"
else
  fail "the offline suite does not pass — do not flip the switch like this"
fi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli_install -v`
Expected: PASS, 6 testes.

- [ ] **Step 5: Commit**

```bash
git add cli/qctx.py scripts/cutover.sh tests/test_cli_install.py
git commit -m "feat: qctx install --check, with each host reported by its own cutover"
```

---

### Task 5: as duas passadas de configuração, e a prova de completude

**Files:**
- Modify: `core/install.py` (listas de campos + escrita de arquivo de credencial)
- Modify: `cli/qctx.py` (`cmd_install`: a passada que escreve)
- Test: `tests/test_install_core.py` (acrescentar), `tests/test_cli_install.py` (acrescentar)

**Interfaces:**
- Consumes: `core.save`, `core.config.DEFAULTS`, `core.config.SECRET_FIELDS`,
  `core.config.ENV_ALIASES`.
- Produces:
  - `core.install.REQUIRED_FIELDS: tuple[str, ...]` (5)
  - `core.install.OPTIONAL_FIELDS: tuple[str, ...]` (9)
  - `core.install.DETECTED_FIELDS: tuple[str, ...]` (`("vector_size",)`)
  - `core.install.write_env_file(path: Path, values: dict[str, str]) -> None` — cria com
    `0o600`, substitui a linha da variável se já existir, nunca duplica

- [ ] **Step 1: Write the failing test**

Acrescentar a `tests/test_install_core.py`:

```python
class FieldCoverage(unittest.TestCase):
    """Simple must not cost complete.

    The wizard is allowed to be short, and it is not allowed to leave a field of `Config`
    with no place to be set. `_check_collections` learned this the expensive way: it
    enumerated three of five roles and reported `ready` over a configuration where the
    repository archive sat on top of the memory one.
    """

    def test_every_field_is_reachable_exactly_once(self):
        from core import config
        walked = (install.REQUIRED_FIELDS + install.OPTIONAL_FIELDS
                  + install.DETECTED_FIELDS)
        self.assertEqual(len(walked), len(set(walked)), "a field is listed twice")
        self.assertEqual(set(walked), set(config.DEFAULTS),
                         "the wizard and Config disagree about what configuration is")

    def test_the_two_keys_are_asked_for(self):
        """SECRET_FIELDS governs WHERE a value is written, never whether it is asked."""
        from core import config
        for secret in config.SECRET_FIELDS:
            self.assertIn(secret, install.REQUIRED_FIELDS)


class CredentialFile(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".env"
        self.addCleanup(self.tmp.cleanup)

    def test_creates_with_owner_only_permissions(self):
        install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertIn("SERVER_API_KEY=abc", self.path.read_text())

    def test_replaces_instead_of_duplicating(self):
        install.write_env_file(self.path, {"SERVER_API_KEY": "old"})
        install.write_env_file(self.path, {"SERVER_API_KEY": "new"})
        body = self.path.read_text()
        self.assertEqual(body.count("SERVER_API_KEY="), 1)
        self.assertIn("SERVER_API_KEY=new", body)

    def test_leaves_other_lines_alone(self):
        self.path.write_text("OTHER=keep\n")
        install.write_env_file(self.path, {"SERVER_API_KEY": "abc"})
        self.assertIn("OTHER=keep", self.path.read_text())
```

Acrescentar a `tests/test_cli_install.py`:

```python
class WritingPass(unittest.TestCase):
    """The interactive pass, driven through stdin with --yes off."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

    def run_with_input(self, keystrokes: str):
        env = dict(os.environ)
        env.update({"HOME": str(self.home), "QCTX_CONFIG": str(self.config),
                    "PATH": "/usr/bin:/bin", "QCTX_INSTALL_FORCE_TTY": "1"})
        return subprocess.run([sys.executable, str(CLI), "install", "--config-only"],
                              input=keystrokes, capture_output=True, text=True,
                              env=env, timeout=180)

    def test_writes_the_urls_and_keeps_the_defaults_on_enter(self):
        answers = "\n".join([
            "https://q.example",        # qdrant_url
            "https://e.example/v1",     # api_base_url
            "qkey", "skey",             # the two keys
            "mem",                      # memory_collection
        ] + [""] * 9) + "\n"            # pass 2: Enter all the way
        self.run_with_input(answers)
        written = json.loads(self.config.read_text())
        self.assertEqual(written["qdrant_url"], "https://q.example")
        self.assertEqual(written["memory_collection"], "mem")

    def test_no_key_reaches_the_config_file_in_any_spelling(self):
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "skey", "mem",
        ] + [""] * 9) + "\n"
        self.run_with_input(answers)
        body = self.config.read_text()
        for forbidden in ("qkey", "skey", "qdrant_api_key", "api_key"):
            self.assertNotIn(forbidden, body)

    def test_the_key_value_is_never_echoed(self):
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "s3cr3t-value", "mem",
        ] + [""] * 9) + "\n"
        done = self.run_with_input(answers)
        self.assertNotIn("s3cr3t-value", done.stdout + done.stderr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_install_core tests.test_cli_install -v`
Expected: FAIL — `AttributeError: module 'core.install' has no attribute
'REQUIRED_FIELDS'`, e `--config-only` inexistente.

- [ ] **Step 3: Write minimal implementation**

Em `core/install.py`:

```python
#: Pass 1 — what blocks use. Asked outright, in this order.
REQUIRED_FIELDS = ("qdrant_url", "api_base_url", "qdrant_api_key", "api_key",
                   "memory_collection")

#: Pass 2 — everything else, with the current value shown and Enter keeping it.
#: `embed_url` and `rerank_url` are HERE and not derived away: with them empty, the config
#: builds them from `api_base_url`, and a setup that serves rerank on another port then
#: gets a URL that does not exist — measured on the author's own machine, embedding on
#: :8003 and rerank on :8004.
OPTIONAL_FIELDS = ("embed_url", "rerank_url", "embed_model", "rerank_model",
                   "docs_collection", "library_collection", "repos_collection",
                   "repos_registry_collection", "context_window")

#: Written from what the endpoint answered, never typed.
DETECTED_FIELDS = ("vector_size",)


def write_env_file(path: Path, values: dict) -> None:
    """Writes KEY=value lines into a credential file, replacing rather than appending.

    Appending is what a person does by hand, and it leaves two lines for one variable —
    whichever the loader reads last wins, which is a coin toss the operator cannot see.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []
    for name, value in values.items():
        replacement = f"{name}={value}"
        for i, line in enumerate(lines):
            if line.startswith(f"{name}="):
                lines[i] = replacement
                break
        else:
            lines.append(replacement)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)
```

Em `cli/qctx.py`, a passada que escreve, chamada por `cmd_install` quando há TTY:

```python
#: Set by the tests, which have no terminal and still need the interactive path.
def _interactive(args) -> bool:
    return bool(sys.stdin.isatty() or os.environ.get("QCTX_INSTALL_FORCE_TTY"))


def _read_secret(prompt: str) -> str:
    """Echo off when there is a terminal, plain read when there is not.

    `getpass` opens /dev/tty and falls back to stdin with a warning when it cannot — two
    different behaviours depending on where it runs, which is not something a test should
    have to guess at. Choosing explicitly keeps the piped case deterministic.
    """
    if sys.stdin.isatty():
        import getpass

        return getpass.getpass(prompt)

    return input(prompt)


def _ask_config(cfg) -> None:
    """The two passes. Blockers first, then everything else with Enter keeping."""
    from core import config as _config

    patch, secrets = {}, {}
    print("\n--- required (Enter keeps the current value) ---")
    for field in core.install.REQUIRED_FIELDS:
        current = getattr(cfg, field)
        if field in _config.SECRET_FIELDS:
            shown = f"set, {len(current)} chars" if current else "MISSING"
            entry = _read_secret(f"{field} [{shown}]: ").strip()
            if entry:
                secrets[_config.ENV_ALIASES[field][0]] = entry
            continue
        entry = input(f"{field} [{current or 'MISSING'}]: ").strip()
        if entry:
            patch[field] = entry

    print("\n--- everything else (Enter keeps) ---")
    for field in core.install.OPTIONAL_FIELDS:
        entry = input(f"{field} [{getattr(cfg, field)}]: ").strip()
        if entry:
            patch[field] = int(entry) if field == "context_window" else entry

    if patch:
        core.save(patch)
        print(f"\n  wrote {len(patch)} setting(s) to the config file")
    if secrets:
        _store_secrets(secrets)


def _store_secrets(secrets: dict) -> None:
    """The keys go to files that exist FOR credentials, and to no other file.

    Never echoes a value: this output gets pasted into issues and chats. And on a machine
    with no hermes there is no plugin-owned credential file at all — saying "done" there
    would be the same lie the README told.
    """
    home = Path(os.path.expanduser("~"))
    written_anywhere = False
    hermes_env = Path(os.environ.get("HERMES_HOME", home / ".hermes")) / ".env"
    if hermes_env.parent.is_dir():
        core.install.write_env_file(hermes_env, secrets)
        for name, value in secrets.items():
            print(f"  ok    {name} written to {hermes_env} (len {len(value)})")
        written_anywhere = True

    shell_file = home / ".secrets"
    if shell_file.exists():
        core.install.write_env_file(shell_file, secrets)
        print(f"  ok    also written to {shell_file}")
        written_anywhere = True

    if not written_anywhere:
        print("  ..    no credential file on this machine — add these to your shell rc:")
        for name in secrets:
            print(f"          export {name}=…")
        print("  PENDING: a key that lives only in this shell is gone in the next one")
```

E `cmd_install` passa a chamar `_ask_config(cfg)` no lugar do `print` provisório da tarefa
4, mais o argumento novo:

```python
    p.add_argument("--config-only", action="store_true",
                   help="only the configuration pass; touch no host")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_install_core tests.test_cli_install -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/install.py cli/qctx.py tests/test_install_core.py tests/test_cli_install.py
git commit -m "feat: two configuration passes, with a test that pins them to Config"
```

---

### Task 6: instalar no host, e o PATH

**Files:**
- Modify: `cli/qctx.py`
- Test: `tests/test_cli_install.py` (acrescentar)

**Interfaces:**
- Consumes: `core.install.target_dir`, `core.install.LAUNCHER_NAME`.
- Produces:
  - `claude_install_path(home: Path) -> str | None` — lê `installed_plugins.json`
  - `hermes_install_path(env: dict) -> Path | None` — `$HERMES_HOME/plugins/memories`
  - `HOST_INSTALL_COMMANDS: dict[str, tuple[str, ...]]`
  - `install_launcher(root: Path, env: dict) -> Path` — copia `bin/qctx` e dá `0o755`
  - `should_stop_before_hosts(report: dict) -> bool` — `True` quando `diagnose` ainda tem
    bloqueador
  - `MANUAL_STEPS: tuple[str, ...]` — o que só um humano pode fazer, impresso no fim

- [ ] **Step 1: Write the failing test**

Acrescentar a `tests/test_cli_install.py`:

```python
def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member.

    Copied verbatim from `tests/test_cli_render.py` — `import cli.qctx` does NOT work:
    there is no `cli/__init__.py`, deliberately, because the CLI is an entry point and
    not something the core imports.
    """
    import importlib.util
    path = REPO / "cli" / "qctx.py"
    spec = importlib.util.spec_from_file_location("qctx_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class HostDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()

    def test_claude_absent_reads_as_none(self):
        self.assertIsNone(self.qctx.claude_install_path(self.home))

    def test_claude_present_returns_the_live_path(self):
        registry = self.home / ".claude" / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"plugins": {"memories-plugin@memories-plugin": [
            {"installPath": "/somewhere/b8008f7dac88"}]}}))
        self.assertEqual(self.qctx.claude_install_path(self.home),
                         "/somewhere/b8008f7dac88")

    def test_the_hermes_command_carries_force_and_the_provider_switch(self):
        joined = " ".join(self.qctx.HOST_INSTALL_COMMANDS["hermes"])
        self.assertIn("--force", joined)
        self.assertIn("memory.provider memories", joined)


class LauncherInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()

    def test_copies_and_makes_it_executable(self):
        target = self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertTrue(os.access(target, os.X_OK))
        self.assertEqual(target.read_bytes(), (REPO / "bin" / "qctx").read_bytes())


class ClosingBehaviour(unittest.TestCase):
    """Two things the spec asks for that are easy to leave out, and both are about NOT
    pretending: stop when the thing underneath is still broken, and say out loud what the
    wizard cannot do for you."""

    def setUp(self):
        self.qctx = load_cli()

    def test_blockers_stop_the_run_before_the_host_steps(self):
        """No Qdrant, no point installing into a host. It names what did not answer and
        stops — instead of asking fifteen questions that cannot work."""
        blocked = {"ready": False, "blockers": [{"name": "Qdrant", "detail": "no answer",
                                                 "ok": False, "fix_hint": None,
                                                 "warning": False}]}
        self.assertTrue(self.qctx.should_stop_before_hosts(blocked))
        self.assertFalse(self.qctx.should_stop_before_hosts({"ready": True,
                                                             "blockers": []}))

    def test_the_manual_steps_are_named(self):
        text = "\n".join(self.qctx.MANUAL_STEPS)
        self.assertIn("hermes hooks list", text)
        self.assertIn("restart", text.lower())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli_install -v`
Expected: FAIL — `AttributeError: module 'cli.qctx' has no attribute
'claude_install_path'`.

- [ ] **Step 3: Write minimal implementation**

Em `cli/qctx.py`:

```python
#: What each host needs typed, when the plugin is not installed there yet. Shown before it
#: is run: `--force` is the user agreeing to what hermes' scanner flagged (this tree ships
#: two scripts that edit host configuration, which is the cutovers' declared job), and
#: pointing `memory.provider` here REPLACES whatever provider is named, because hermes
#: activates exactly one.
HOST_INSTALL_COMMANDS = {
    "claude-code": (
        "claude plugin marketplace add erickstryck/memories-plugin",
        "claude plugin install memories-plugin@memories-plugin",
    ),
    "hermes": (
        "hermes plugins install erickstryck/memories-plugin --enable --force",
        "hermes config set memory.provider memories",
    ),
}


def claude_install_path(home: Path) -> str | None:
    """The live install path, read from the harness' own record.

    Not guessable: the cache directory is named after the commit, and old ones stay behind
    — five of them on the machine this was measured on.
    """
    registry = home / ".claude" / "plugins" / "installed_plugins.json"
    try:
        data = json.loads(registry.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for entry in data.get("plugins", {}).get("memories-plugin@memories-plugin", []):
        if entry.get("installPath"):
            return entry["installPath"]

    return None


def hermes_install_path(env: dict) -> Path | None:
    """One level deep and no deeper: hermes' loader scans `$HERMES_HOME/plugins/<name>/`
    and never looks further down."""
    home = Path(env.get("HERMES_HOME") or Path(env["HOME"]) / ".hermes")
    candidate = home / "plugins" / "memories"

    return candidate if candidate.exists() else None


def install_launcher(root: Path, env: dict) -> Path:
    """Copies the launcher onto PATH. A copy, not a symlink: on a machine with only
    claude-code the only stable thing to link to does not exist — the tree is a directory
    named after a commit, replaced by the next update."""
    target_dir = core.install.target_dir(env)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / core.install.LAUNCHER_NAME
    shutil.copyfile(root / "bin" / core.install.LAUNCHER_NAME, target)
    target.chmod(0o755)

    return target
```

E, em `cmd_install`, entre a configuração e as seções de host, o grupo que pergunta:

```python
    for host, _script, _skip in HOST_SECTIONS:
        binary = HOST_BINARIES[host]
        if not shutil.which(binary):
            continue
        present = (claude_install_path(Path(os.path.expanduser("~")))
                   if host == "claude-code" else hermes_install_path(dict(os.environ)))
        if present:
            print(f"  ok    {host}: installed at {present}")
            continue
        print(f"\n  {host} is on this machine and the plugin is not installed there:")
        for command in HOST_INSTALL_COMMANDS[host]:
            print(f"      {command}")
        if args.yes or input("  run these? [y/N]: ").strip().lower() == "y":
            for command in HOST_INSTALL_COMMANDS[host]:
                subprocess.run(command, shell=True, check=False)
```

E as duas peças de fechamento, no mesmo arquivo:

```python
#: What no wizard can do for you, printed at the end of a run that wrote something.
#: Both are one-time and both fail SILENTLY when skipped, which is why they are printed
#: rather than merely documented.
MANUAL_STEPS = (
    "hermes: approve the read guard once at a TTY — a hermes with no terminal skips the "
    "hook silently until then. `hermes hooks list` shows it allowed afterwards.",
    "claude-code: open a new terminal, or restart — hooks are read at start-up.",
)


def should_stop_before_hosts(report: dict) -> bool:
    """A blocker means the archive itself does not answer. Installing into a host on top
    of that produces a host wired to nothing, and a green report about it."""
    return not report.get("ready", False)
```

E no fim de `cmd_install`, depois da passada de configuração:

```python
    after = core.setup.diagnose(core.load())
    if should_stop_before_hosts(after):
        print("\nstopping here — these must answer before a host install means anything:")
        for c in after["blockers"]:
            _render_check(c)

        return
    # … the host group from this task …
    print("\nwhat is left, and only you can do it:")
    for step in MANUAL_STEPS:
        print(f"  - {step}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli_install -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/qctx.py tests/test_cli_install.py
git commit -m "feat: offer the host install the cutovers do not do, and put qctx on PATH"
```

---

### Task 7: o bootstrap `scripts/install.sh`

**Files:**
- Create: `scripts/install.sh` (modo `0755`)
- Test: `tests/test_install_script.py`

**Interfaces:**
- Consumes: `bin/qctx`'s `own_tree` logic (repetida, não importada — é bash).
- Produces: `bash scripts/install.sh [args…]` → `exec python3 <root>/cli/qctx.py install
  [args…]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_install_script.py
"""The bootstrap.

The only piece that runs before `qctx` exists anywhere, and therefore the only piece in
bash. It decides nothing: everything past the `exec` is Python the offline suite reaches.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "install.sh"


class Bootstrap(unittest.TestCase):
    def test_it_is_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))

    def test_it_forwards_to_qctx_install(self):
        env = dict(os.environ, PATH="/usr/bin:/bin")
        done = subprocess.run(["bash", str(SCRIPT), "--check", "--json"],
                              capture_output=True, text=True, env=env, timeout=300,
                              stdin=subprocess.DEVNULL)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('"hosts"', done.stdout)

    def test_it_runs_from_any_directory(self):
        with TemporaryDirectory() as elsewhere:
            done = subprocess.run(["bash", str(SCRIPT), "--check"], cwd=elsewhere,
                                  capture_output=True, text=True, timeout=300,
                                  stdin=subprocess.DEVNULL)
            self.assertEqual(done.returncode, 0, done.stderr)

    def test_it_says_what_is_missing_when_python3_is_absent(self):
        env = dict(os.environ, PATH="/nonexistent")
        done = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                              env=env, timeout=60, stdin=subprocess.DEVNULL)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("python3", done.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_install_script -v`
Expected: FAIL — o arquivo não existe.

- [ ] **Step 3: Write minimal implementation**

```bash
#!/usr/bin/env bash
# The wizard's front door, and the only piece that can run before `qctx` is on PATH.
#
#     bash ~/.hermes/plugins/memories/scripts/install.sh        # installed by hermes
#     bash ~/.claude/plugins/cache/…/<SHA>/scripts/install.sh   # installed by claude
#     ./scripts/install.sh                                      # cloned
#
# It decides NOTHING. Everything past the `exec` is Python, where the offline suite
# reaches it; a decision taken in bash here would be a decision no test could see.
set -euo pipefail

target="${BASH_SOURCE[0]}"
while [ -L "$target" ]; do
  dest="$(readlink "$target")"
  case "$dest" in
    /*) target="$dest" ;;
    *)  target="$(cd -P "$(dirname "$target")" && pwd)/$dest" ;;
  esac
done
root="$(cd -P "$(dirname "$target")/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  printf 'install: python3 is required and was not found on PATH\n' >&2
  exit 1
fi

if [ ! -f "$root/cli/qctx.py" ]; then
  printf 'install: %s does not look like the plugin tree (no cli/qctx.py)\n' "$root" >&2
  exit 1
fi

exec python3 "$root/cli/qctx.py" install "$@"
```

Depois: `chmod 755 scripts/install.sh`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_install_script -v`
Expected: PASS, 4 testes.

- [ ] **Step 5: Commit**

```bash
git add scripts/install.sh tests/test_install_script.py
git commit -m "feat: a bootstrap that works before qctx exists, and decides nothing"
```

---

### Task 8: o README, e o teste que o mantém honesto

**Files:**
- Modify: `README.md`
- Modify: `hosts/hermes/__init__.py:160` e `plugin.yaml` (comentários com contagem velha)
- Test: `tests/test_readme_fidelity.py`

**Interfaces:**
- Consumes: `cli.qctx.build_parser`, `hooks/hooks.json`, `skills/*/SKILL.md`,
  `hosts.hermes.tools.SCHEMAS`.
- Produces: nada que outra tarefa consuma.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_readme_fidelity.py
"""The README must describe THIS tree.

It stopped doing so, measured on 2026-08-19: it said "three hooks" over four, gave a test
count 29 short, and printed two different numbers of hermes tools on the same page. Every
one of those was true when written. Prose does not rot on its own — it rots because
nothing reads it, so this reads it.

Counts are written as DIGITS in the README on purpose. A test that has to parse "three"
would be a test nobody keeps working.
"""
import json
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

README = (REPO / "README.md").read_text()


def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member — the same
    helper `tests/test_cli_render.py` uses, and for the same reason."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qctx_cli", REPO / "cli" / "qctx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def known_commands(parser, prefix=()):
    """Every (command, subcommand) pair argparse accepts."""
    import argparse
    found = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                found.add(prefix + (name,))
                found |= known_commands(sub, prefix + (name,))

    return found


class CitedCommandsExist(unittest.TestCase):
    def test_every_qctx_command_in_the_readme_is_real(self):
        known = known_commands(load_cli().build_parser())
        tops = {pair[0] for pair in known}
        for command, rest in re.findall(r"\bqctx ([a-z][a-z-]*)((?: [a-z][a-z-]*)?)",
                                        README):
            self.assertIn(command, tops, f"README cites `qctx {command}`, which does not "
                                         f"exist")
            second = rest.strip()
            subs = {pair[1] for pair in known if len(pair) == 2 and pair[0] == command}
            if subs and second:
                self.assertIn(second, subs,
                              f"README cites `qctx {command} {second}`, and {command} has "
                              f"no such subcommand")


class CountsMatchTheTree(unittest.TestCase):
    def counted(self, noun):
        """Every "<n> <noun>" the README states."""
        return {int(n) for n in re.findall(rf"\b(\d+) {noun}s?\b", README)}

    def test_hooks(self):
        declared = json.loads((REPO / "hooks" / "hooks.json").read_text())["hooks"]
        real = sum(len(matcher.get("hooks", []))
                   for event in declared.values() for matcher in event)
        self.assertEqual(self.counted("hook"), {real})

    def test_skills(self):
        real = len(list((REPO / "skills").glob("*/SKILL.md")))
        self.assertEqual(self.counted("skill"), {real})

    def test_tools(self):
        from hosts.hermes import tools
        self.assertEqual(self.counted("tool"), {len(tools.SCHEMAS)})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_readme_fidelity -v`
Expected: FAIL nos três testes de contagem — `{20, 22} != {22}` para tools, e o texto
"three hooks"/"three skills" não produz dígito nenhum, então o conjunto sai vazio.

- [ ] **Step 3: Write minimal implementation**

Três edições no README, e duas linhas de comentário fora dele:

1. **Contagens em dígitos e corretas.** Trocar "three skills, three hooks" por "3 skills,
   4 hooks"; trocar a única ocorrência de "20 tools" por "22 tools"; tirar os números da
   frase da suíte ("1140 collected, 1123 offline" → sem número: o teste que afirma o
   número de testes quebra a cada teste novo). Rodar
   `grep -nE '\b(one|two|three|four|20|22) (hook|skill|tool)' README.md` e converter o que
   sobrar.
2. **A seção de instalação, com o wizard na frente.** Substituir o bloco de `git clone` +
   `ln -s` por:

````markdown
### The one command

On a machine where the plugin is already installed by its host, or on a fresh clone:

```bash
bash ~/.hermes/plugins/memories/scripts/install.sh          # installed by hermes
bash ~/.claude/plugins/cache/memories-plugin/memories-plugin/*/scripts/install.sh
./scripts/install.sh                                        # cloned
```

It puts `qctx` on PATH, asks for what is missing, offers to install into whichever host is
on this machine, and re-checks. Nothing it writes is silent: every group asks first.

To verify a machine without changing it — which is also how you check a machine you set up
months ago:

```bash
qctx install --check      # reports; writes nothing
```

`--check` covers the plumbing, Qdrant, the embedding and re-rank endpoints, the five
collections, whether a shell-less process would find the configuration, and each host's
own cutover report.
````

3. **O caminho manual fica**, logo abaixo, sob "If you would rather do it by hand" — a
   tabela de 6 passos que já existe, mais o `ln -s`. Ele é o que se usa quando o wizard
   falha, e é o que documenta o que o wizard faz.
4. Fora do README: `hosts/hermes/__init__.py:160` diz "all 15 tools" e o comentário de
   cabeçalho de `plugin.yaml` diz "this plugin's 20 tools". Ambos passam a 22.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_readme_fidelity -v`
Expected: PASS, 4 testes.

- [ ] **Step 5: Commit**

```bash
git add README.md plugin.yaml hosts/hermes/__init__.py tests/test_readme_fidelity.py
git commit -m "docs: the wizard leads the install section, and a test reads the counts"
```

---

## Fechamento

- [ ] **Rodar a suíte inteira:** `python3 -m unittest discover -s tests`
      Esperado: OK, com os ~30 testes novos somados aos 1169 existentes.
- [ ] **Provar na mão o caminho que originou o trabalho** — uma árvore falsa sem `qctx` no
      PATH, o bootstrap rodando dela, e `qctx --root` respondendo depois:

```bash
tmp="$(mktemp -d)"; cp -r . "$tmp/tree"
env -i HOME="$tmp/home" PATH=/usr/bin:/bin bash -c "
  mkdir -p '$tmp/home'
  bash '$tmp/tree/scripts/install.sh' --check | head -20"
```

- [ ] **Conferir que `--check` continua mudo quanto a segredo:**
      `qctx install --check | grep -iE '(api|secret|key)=' ` — esperado: sem saída.
      Falsificar o predicado antes de acreditar nele: rodar o mesmo `grep` contra uma
      string conhecidamente violadora (`echo 'SERVER_API_KEY=abc' | grep -iE '(api|secret|key)='`)
      e confirmar que ele casa. Um grep vazio só vale como veredito depois disso.
