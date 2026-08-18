# Resolução da janela de contexto — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** fazer a guarda de arquivo grande descobrir sozinha a janela de contexto no host que a
conhece, em vez de exigir que o usuário a declare.

**Architecture:** uma cascata no `core` — declarado → valor sondado em cache → tabela de tetos → 0 —
onde cada adaptador contribui a fonte que o host dele oferece. A guarda NUNCA sonda: ela precisa da
janela para decidir, então quem preenche o cache são os pontos que já pagam rede por outro motivo.

**Tech Stack:** Python 3, só stdlib. `core/http.py` para HTTP, `core/knobs.py::state_dir` para o
cache.

**Spec:** `docs/superpowers/specs/2026-08-17-window-resolution-design.md`

## Base

Branch `repo-subcollections`, sobre `9fb1d7b`. Baseline: **889 testes, OK (skipped=18)**.

**Antes de qualquer rodada de teste**, e isto não é opcional — uma sessão inteira já travou por
isso hoje:

```bash
export TMPDIR=/home/erick/.cache/qctx-test-tmp && mkdir -p "$TMPDIR"
```

e `rm -rf $TMPDIR/tmp*` ao terminar. A suíte vaza ~519 arquivos temporários por rodada e o `/tmp` é
um tmpfs de 16 GB que encheu e matou todo shell.

## Global Constraints

- **Só a biblioteca padrão.** Nenhuma dependência nova.
- **Código, comentários, docstrings, mensagens e commits em INGLÊS.** Spec e plano em português.
- **A suíte nunca fica vermelha em nenhum commit.**
- **Toda guarda precisa de sonda de mutação que MORDA**, com a contagem de ocorrências verificada e
  o **escopo** de cada contagem declarado.
- **Reusar, nunca copiar.**
- **A guarda NUNCA faz chamada de rede.** Ela roda antes de cada leitura de arquivo; quem sonda são
  os hooks que já pagam rede.
- **Toda falha DESCE a cascata, nunca sobe.** Nenhum caminho novo pode produzir uma janela menor
  que a verdade sem passar pelo usuário.

## Disciplina de sonda — sete modos de falha que este projeto já sofreu

1. Verifique que o BACKUP existe depois de criá-lo; um `cp` para diretório inexistente deixa o
   restore falhar em silêncio e toda medição seguinte é sobre código mutado.
2. Afirme a contagem de ocorrências E confirme por `grep` que a substituição ATERROU.
3. Confirme que a suíte RODOU — existe uma linha `Ran N`. Substituição que quebra sintaxe não
   produz saída nenhuma.
4. Tire a contagem da linha de RESUMO do unittest, nunca contando linhas `FAIL:`/`ERROR:` — saída
   truncada esconde falhas porque os prefixos ordenam.
5. Refatoração pode tornar a âncora não-única. Se a contagem falhar, **isso é o achado** — não
   alargue o padrão até casar.
6. **Duas mutações com o mesmo conjunto de falhas significam que no máximo uma é a que você pensa.**
7. Teste que compara estado antes/depois contra fake em memória tem de **copiar** o antes; o fake
   guarda por referência e a comparação vira identidade.

E: **sonda que não morde é ACHADO, não encolher de ombros.** Cinco dos quinze defeitos do plano
anterior foram sondas certificando guarda que ninguém segurava.

## Estrutura de arquivos

```
core/windowcache.py    NOVO — persiste (endpoint, modelo) -> janela, com TTL. Puro, sem rede.
core/windowprobe.py    NOVO — pergunta a um endpoint /models e extrai a janela. Sem estado.
core/windows.py        a cascata passa a consultar o cache entre "declarado" e a tabela
hosts/hermes/__init__.py   `prefetch()` atualiza o cache quando está velho
```

A divisão é por responsabilidade: o cache só persiste, a sonda só busca, `windows.py` só decide.
Nenhum dos três conhece o outro host.

---

### Task 1: O cache — persistir sem inventar

**Files:**
- Create: `core/windowcache.py`
- Test: `tests/test_windowcache.py`

**Interfaces:**
- Consumes: `core.knobs.state_dir()`.
- Produces:
  - `windowcache.FILENAME` = `"model-windows.json"`
  - `windowcache.get(endpoint: str, model: str) -> tuple[int, bool]` — `(janela, esta_fresco)`.
    `(0, False)` quando não há entrada.
  - `windowcache.put(endpoint: str, model: str, window: int, ttl: float = TTL_SECONDS) -> None`
  - `windowcache.TTL_SECONDS` = `86400.0`

**Chavear por `(endpoint, modelo)` e não só pelo modelo** é requisito da spec: o mesmo nome de
modelo em servidores diferentes pode ter janelas diferentes, e chavear só pelo nome faz dois setups
se contaminarem.

**`get` devolve DOIS valores de propósito.** Um cache velho ainda é usado — valor velho é melhor
que a guarda dormir — mas quem chama precisa saber que está velho para decidir atualizar. Um
booleano único ("achei") esconderia essa diferença.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_windowcache.py
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import windowcache  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestReadingAndWriting(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_an_unknown_pair_reads_as_nothing(self):
        self.assertEqual(windowcache.get("http://x/v1", "m"), (0, False))

    def test_what_was_written_comes_back_fresh(self):
        windowcache.put("http://x/v1", "m", 524288)
        self.assertEqual(windowcache.get("http://x/v1", "m"), (524288, True))

    def test_the_same_model_on_a_DIFFERENT_endpoint_is_a_different_entry(self):
        """Two servers can serve one model name with different windows. Keying by the name
        alone lets one deployment answer for another."""
        windowcache.put("http://a/v1", "m", 100_000)
        windowcache.put("http://b/v1", "m", 200_000)
        self.assertEqual(windowcache.get("http://a/v1", "m")[0], 100_000)
        self.assertEqual(windowcache.get("http://b/v1", "m")[0], 200_000)

    def test_a_stale_entry_is_STILL_RETURNED_but_marked_stale(self):
        """A stale window beats no window: without it the guard falls to the ceiling table
        and sleeps. The caller needs the flag to know a refresh is worth attempting."""
        windowcache.put("http://x/v1", "m", 524288, ttl=-1)
        window, fresh = windowcache.get("http://x/v1", "m")
        self.assertEqual(window, 524288)
        self.assertFalse(fresh)


class TestItNeverRaises(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_corrupt_file_reads_as_empty(self):
        with open(os.path.join(windowcache.state_dir(), windowcache.FILENAME), "w") as fh:
            fh.write("{not json")
        self.assertEqual(windowcache.get("http://x/v1", "m"), (0, False))

    def test_a_corrupt_file_is_REPLACED_by_the_next_write(self):
        path = os.path.join(windowcache.state_dir(), windowcache.FILENAME)
        with open(path, "w") as fh:
            fh.write("{not json")
        windowcache.put("http://x/v1", "m", 1000)
        self.assertEqual(windowcache.get("http://x/v1", "m"), (1000, True))

    def test_a_window_of_zero_or_less_is_NOT_stored(self):
        """Storing 0 would make the cache assert 'this endpoint says unknown' — and the next
        step of the cascade would never be consulted."""
        windowcache.put("http://x/v1", "m", 0)
        windowcache.put("http://x/v1", "n", -5)
        self.assertEqual(windowcache.get("http://x/v1", "m"), (0, False))
        self.assertEqual(windowcache.get("http://x/v1", "n"), (0, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `export TMPDIR=/home/erick/.cache/qctx-test-tmp; mkdir -p "$TMPDIR"; python3 -m unittest tests.test_windowcache 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.windowcache'`

- [ ] **Step 3: Implemente `core/windowcache.py`**

```python
"""What an endpoint said its model's context window was, remembered between processes.

WHY A CACHE AT ALL. The guard needs the window to DECIDE, so it cannot be the thing that
fetches it: that would be circular, and it would put a network call on the path that runs
before every single file read. The hooks that already pay for network fill this; the guard
only ever reads it.

WHY THE KEY IS (ENDPOINT, MODEL). One model name means different windows on different
servers — the same `qwen3.8-27b` is 262,144 on one host and 524,288 on another, measured.
Keying by the name alone lets one deployment answer for another, silently.

WHY A STALE ENTRY IS STILL RETURNED. Falling back to the ceiling table because the value is
a day old makes the guard sleep on a window it already knows. Stale beats absent; the
freshness flag exists so the caller can decide to refresh, not so it can discard.
"""
import json
import os
import time

from .knobs import state_dir

FILENAME = "model-windows.json"

#: A day. The window of a model does not change often, and the cost of being a day late is
#: one refresh; the cost of refreshing constantly is a network call nobody asked for.
TTL_SECONDS = 86400.0


def _path() -> str:
    root = state_dir()
    os.makedirs(root, exist_ok=True)

    return os.path.join(root, FILENAME)


def _key(endpoint: str, model: str) -> str:
    return f"{(endpoint or '').strip()}|{(model or '').strip()}"


def _load() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def get(endpoint: str, model: str) -> tuple[int, bool]:
    """`(window, is_fresh)`. `(0, False)` when nothing is known.

    A stale entry comes back with its value and `False` — see the module docstring.
    """
    row = _load().get(_key(endpoint, model))
    if not isinstance(row, dict):
        return 0, False
    try:
        window = int(row.get("window") or 0)
        expires = float(row.get("expires_at") or 0)
    except (TypeError, ValueError):
        return 0, False
    if window <= 0:
        return 0, False

    return window, time.time() < expires


def put(endpoint: str, model: str, window: int, ttl: float = TTL_SECONDS) -> None:
    """Records a window. A value of zero or less is NOT stored.

    Storing zero would make this cache assert "that endpoint says it does not know", and the
    cascade would stop here instead of falling through to the ceiling table. Absence and
    "answered zero" have to stay different.
    """
    try:
        window = int(window)
    except (TypeError, ValueError):
        return
    if window <= 0:
        return
    data = _load()
    data[_key(endpoint, model)] = {"window": window, "expires_at": time.time() + ttl}
    tmp = _path() + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, _path())
    except OSError:
        # A cache that cannot be written is a cache miss, never an error the caller sees.
        return
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_windowcache 2>&1 | tail -3`
Expected: `Ran 7 tests`, `OK`

- [ ] **Step 5: Prove que as duas guardas mordem, uma por vez**

```bash
cp core/windowcache.py /tmp/wc.bak && [ -s /tmp/wc.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
p = "core/windowcache.py"; s = open(p).read()
old = "    if window <= 0:\n        return\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, ""))
PY
grep -c "if window <= 0:" core/windowcache.py    # era 2, agora 1 — a mutacao aterrou
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_windowcache 2>&1 | grep -E "^FAIL: test|^Ran|^OK|^FAILED"
cp /tmp/wc.bak core/windowcache.py && rm /tmp/wc.bak
```
Expected: `FAILED` nomeando `test_a_window_of_zero_or_less_is_NOT_stored`. Registre a contagem COM
o escopo (`tests.test_windowcache`).

Repita trocando `return window, time.time() < expires` por `return window, True` e espere
`test_a_stale_entry_is_STILL_RETURNED_but_marked_stale`.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/windowcache.py tests/test_windowcache.py
git commit -F - <<'MSG'
feat: remember what an endpoint said a model's window was

The guard needs the window to decide, so it cannot be the thing that fetches it — that is
circular, and it puts a network call on the path that runs before every file read. This is
what the hooks that already pay for network write into, and all the guard ever does is read.

Keyed by (endpoint, model), not by model: the same qwen3.8-27b is 262,144 on one host and
524,288 on another, measured. Keying by the name alone lets one deployment answer for another.

A stale entry still comes back, with a flag. Falling back to the ceiling table because the
value is a day old makes the guard sleep on a window it already knows — stale beats absent,
and the flag exists so the caller can refresh rather than discard.

Zero is never stored. Storing it would make the cache assert "this endpoint says it does not
know", and the cascade would stop there instead of falling through to the table.
MSG
```

---

### Task 2: A sonda — perguntar sem inventar

**Files:**
- Create: `core/windowprobe.py`
- Test: `tests/test_windowprobe.py`

**Interfaces:**
- Consumes: `core.http.request_json(url, *, method="GET", headers=None, timeout=30.0)` e
  `core.http.bearer(token) -> dict`.
- Produces:
  - `windowprobe.PATHS` — a lista ORDENADA de caminhos
  - `windowprobe.window_from(entry: dict) -> int` — extrai de UMA entrada de modelo
  - `windowprobe.probe(base_url: str, api_key: str, model: str, *, timeout: float = 5.0,
    fetch=None) -> int` — `0` quando não descobriu. `fetch` existe para teste.

**A ordem dos caminhos é a decisão desta task**, e a spec explica por quê: o OpenRouter publica dois
campos que discordam em 34 dos 414 modelos, e o agregado é grande demais POR CONSTRUÇÃO.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_windowprobe.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import windowprobe  # noqa: E402


def a_listing(*models):
    return {"data": list(models)}


class TestTheFourShapesMeasuredAgainstRealServERS(unittest.TestCase):
    """Every fixture here is a shape measured against a live server on 2026-08-17, not a
    shape imagined. The values are the ones those servers returned."""

    def test_vLLM_puts_it_at_the_top_as_max_model_len(self):
        entry = {"id": "Qwen3.8-27B", "max_model_len": 524288}
        self.assertEqual(windowprobe.window_from(entry), 524288)

    def test_openrouter_aggregate_is_context_length(self):
        entry = {"id": "qwen/qwen3.8-27b", "context_length": 262144}
        self.assertEqual(windowprobe.window_from(entry), 262144)

    def test_llama_cpp_NESTS_it_under_meta(self):
        """A search that only looks at the top level misses this one entirely."""
        entry = {"id": "bge-m3", "meta": {"n_ctx": 8192, "n_ctx_train": 8192}}
        self.assertEqual(windowprobe.window_from(entry), 8192)

    def test_an_unknown_shape_yields_nothing_rather_than_a_guess(self):
        self.assertEqual(windowprobe.window_from({"id": "x", "window_size_tokens": 4096}), 0)

    def test_a_non_numeric_or_absurd_value_yields_nothing(self):
        self.assertEqual(windowprobe.window_from({"id": "x", "max_model_len": "big"}), 0)
        self.assertEqual(windowprobe.window_from({"id": "x", "max_model_len": 0}), 0)
        self.assertEqual(windowprobe.window_from({"id": "x", "max_model_len": -1}), 0)


class TestTheOrderThatOpenRouterForces(unittest.TestCase):
    def test_when_the_two_openrouter_fields_DISAGREE_the_provider_one_wins(self):
        """Measured: OpenRouter's two fields disagree on 34 of its 414 models.
        `context_length` is the best across every provider serving that model;
        `top_provider.context_length` is the one the request actually reaches. Taking the
        aggregate makes the guard sleep exactly when it needed to wake — a fixture where the
        two agree proves nothing about the order."""
        entry = {"id": "nvidia/nemotron-3.5-lightning",
                 "context_length": 1_000_000,
                 "top_provider": {"context_length": 262_144}}
        self.assertEqual(windowprobe.window_from(entry), 262_144)

    def test_a_broken_top_provider_falls_to_the_aggregate_rather_than_to_nothing(self):
        entry = {"id": "x", "context_length": 262_144, "top_provider": {}}
        self.assertEqual(windowprobe.window_from(entry), 262_144)


class TestProbing(unittest.TestCase):
    def test_it_finds_the_entry_whose_id_matches_the_model(self):
        listing = a_listing({"id": "other", "max_model_len": 111},
                            {"id": "wanted", "max_model_len": 222})
        got = windowprobe.probe("http://x/v1", "k", "wanted", fetch=lambda *a, **k: listing)
        self.assertEqual(got, 222)

    def test_a_model_absent_from_the_listing_yields_nothing(self):
        listing = a_listing({"id": "other", "max_model_len": 111})
        self.assertEqual(windowprobe.probe("http://x/v1", "k", "wanted",
                                           fetch=lambda *a, **k: listing), 0)

    def test_an_unreachable_endpoint_yields_nothing_and_does_not_raise(self):
        def boom(*a, **k):
            raise OSError("connection refused")

        self.assertEqual(windowprobe.probe("http://x/v1", "k", "m", fetch=boom), 0)

    def test_a_listing_that_is_not_a_listing_yields_nothing(self):
        for junk in ({}, {"data": "nope"}, {"data": [None]}, [], "text"):
            with self.subTest(junk=junk):
                self.assertEqual(windowprobe.probe("http://x/v1", "k", "m",
                                                   fetch=lambda *a, **k: junk), 0)

    def test_it_asks_the_models_route_of_the_base_url(self):
        seen = {}

        def spy(url, **kw):
            seen["url"] = url

            return a_listing()

        windowprobe.probe("http://x/v1", "k", "m", fetch=spy)
        self.assertEqual(seen["url"], "http://x/v1/models")

    def test_a_base_url_with_a_trailing_slash_does_not_double_it(self):
        seen = {}

        def spy(url, **kw):
            seen["url"] = url

            return a_listing()

        windowprobe.probe("http://x/v1/", "k", "m", fetch=spy)
        self.assertEqual(seen["url"], "http://x/v1/models")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_windowprobe 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.windowprobe'`

- [ ] **Step 3: Implemente `core/windowprobe.py`**

```python
"""Ask the endpoint that serves the model how big its context window is.

WHY A LIST OF PATHS AND NOT A FIELD. Three servers were measured on 2026-08-17 and each put
it somewhere different: vLLM at `max_model_len`, OpenRouter at `context_length`, and
llama.cpp NESTED under `meta.n_ctx` — a search that only looks at the top level misses that
one entirely.

WHY THE ORDER, AND WHY IT INVERTS THIS FEATURE'S USUAL RULE. OpenRouter publishes two
context lengths that disagree on 34 of its 414 models: `context_length` is the largest across
every provider serving that model, and `top_provider.context_length` is the one the request
actually reaches (measured: 1,000,000 against 262,144 for nvidia/nemotron-3.5-lightning).
Everywhere else in this feature erring LARGE is the safe direction, because a window too large
only makes the guard sleep. Here the aggregate is too large BY CONSTRUCTION: asking for a
million when the provider delivers 262k makes the guard sleep exactly when it needed to wake.
So the rule refines rather than contradicts — prefer the value most likely to be REAL, and
leave ceilings to the step of the cascade whose job is ceilings.

THIS LIST IS INCOMPLETE, and saying so is the point. It came from measuring three servers. A
provider using another name, or another nesting level, yields nothing here and the cascade
falls through — which is the correct answer for "we do not know", and better than a heuristic
that estimates a window from something else.

NOT VERIFIED: OpenAI and Anthropic native endpoints, which need credentials that were not
available. Nothing here claims anything about them; they take the unknown-shape path.
"""
from . import http

#: Ordered. First path that yields a positive integer wins. See the module docstring for why
#: `top_provider` precedes the aggregate.
PATHS = (
    ("top_provider", "context_length"),   # OpenRouter — the provider actually reached
    ("max_model_len",),                   # vLLM
    ("context_length",),                  # OpenRouter aggregate, and others
    ("meta", "n_ctx"),                    # llama.cpp — nested, not at the top
)


def window_from(entry: dict) -> int:
    """The window this model entry declares, or 0 when it declares none we recognise."""
    if not isinstance(entry, dict):
        return 0
    for path in PATHS:
        node = entry
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, bool) or not isinstance(node, (int, float, str)):
            continue
        try:
            value = int(node)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value

    return 0


def probe(base_url: str, api_key: str, model: str, *, timeout: float = 5.0, fetch=None) -> int:
    """The window `model` has on the server at `base_url`, or 0 when it could not be learned.

    Returns 0 for every failure — unreachable, malformed, model absent, shape unrecognised —
    because 0 is what the cascade reads as "ask the next step". It never raises: a caller
    doing this opportunistically must not be interrupted by a server being down.

    `fetch` is injected only so tests can drive the shapes without a server.
    """
    url = (base_url or "").rstrip("/") + "/models"
    call = fetch or (lambda u, **kw: http.request_json(u, **kw))
    try:
        payload = call(url, headers=http.bearer(api_key), timeout=timeout)
    except Exception:                      # noqa: BLE001 — see docstring
        return 0
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return 0
    wanted = (model or "").strip()
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id", "")).strip() == wanted:
            return window_from(entry)

    return 0
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_windowprobe 2>&1 | tail -3`
Expected: `Ran 13 tests`, `OK`

- [ ] **Step 5: Prove que a ORDEM morde, e que ela morde sozinha**

```bash
cp core/windowprobe.py /tmp/wp.bak && [ -s /tmp/wp.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
p = "core/windowprobe.py"; s = open(p).read()
old = '    ("top_provider", "context_length"),   # OpenRouter — the provider actually reached\n'
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, ""))
PY
grep -c '"top_provider"' core/windowprobe.py    # era 1 na lista, agora 0 — aterrou
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_windowprobe 2>&1 | grep -E "^FAIL: test|^Ran|^OK|^FAILED"
cp /tmp/wp.bak core/windowprobe.py && rm /tmp/wp.bak
```
Expected: `FAILED (failures=1)` nomeando
`test_when_the_two_openrouter_fields_DISAGREE_the_provider_one_wins` — **e apenas ele**. Se cair
mais de um, os caminhos não estão isolados e isso é o achado.

Repita removendo a linha do `meta` e espere `test_llama_cpp_NESTS_it_under_meta`.

- [ ] **Step 6: Teste de integração contra um endpoint real, sob guarda**

```python
# acrescente ao fim de tests/test_windowprobe.py, antes do __main__
@unittest.skipUnless(os.environ.get("QCTX_INTEGRATION") == "1",
                     "integration: needs a reachable model endpoint")
class TestARealServerAnswers(unittest.TestCase):
    """The half a fake cannot prove: that a real listing has the shape this module expects.
    Uses whatever endpoint the config points at; skips when it names no model we can check."""

    def test_the_listing_has_data_entries_with_ids(self):
        from core import load
        cfg = load()
        if not cfg.api_base_url:
            self.skipTest("no api_base_url configured")
        url = cfg.api_base_url.rstrip("/") + "/models"
        payload = http.request_json(url, headers=http.bearer(cfg.api_key), timeout=10.0)
        self.assertIsInstance(payload.get("data"), list)
        for entry in payload["data"]:
            self.assertIn("id", entry)
```

Acrescente `from core import http` ao topo do arquivo de teste.

- [ ] **Step 7: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/windowprobe.py tests/test_windowprobe.py
git commit -F - <<'MSG'
feat: ask the endpoint, and accept that it may answer in a shape we do not know

Three servers were measured and each puts the window somewhere different: vLLM at
max_model_len, OpenRouter at context_length, llama.cpp NESTED under meta.n_ctx. A search for a
field rather than a path misses the third, on a server already running here.

The order matters more than the list. OpenRouter publishes two context lengths that disagree
on 34 of its 414 models — the aggregate across every provider serving the model, and the one
the request actually reaches. Everywhere else in this feature erring large is safe, because
too large only makes the guard sleep. Here the aggregate is too large BY CONSTRUCTION: asking
for a million when the provider delivers 262k makes the guard sleep exactly when it needed to
wake. The rule refines rather than contradicts — prefer the value most likely to be real.

Every failure returns 0, which is what the cascade reads as "ask the next step". Unreachable,
malformed, absent, unrecognised: all the same answer, and none of them raise, because the
caller is doing this opportunistically and must not be interrupted by a server being down.

The list of shapes is incomplete and says so. OpenAI and Anthropic native endpoints could not
be measured without credentials, and nothing here claims anything about them.
MSG
```

---

### Task 3: A cascata, e a prova de que a guarda nunca sonda

**Files:**
- Modify: `core/windows.py` (`window_for`)
- Test: `tests/test_windows.py` (acrescentar)

**Interfaces:**
- Consumes: `windowcache.get(endpoint, model) -> (int, bool)` (Task 1).
- Produces: `window_for(model: str, cfg, endpoint: str = "") -> int` — o terceiro parâmetro é
  NOVO e opcional, então os dois chamadores existentes seguem funcionando sem mudança.

**O parâmetro é opcional por desenho.** O claude-code não tem endpoint para oferecer; passar nada
tem de continuar dando exatamente o comportamento de hoje.

- [ ] **Step 1: Escreva o teste que falha**

```python
# acrescente a tests/test_windows.py
class TestTheCascade(unittest.TestCase):
    """Four steps, and each fixture satisfies EXACTLY ONE of them. A fixture that satisfies
    two proves neither — the lesson this repo paid for twice."""

    def setUp(self):
        import tempfile
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()

    def test_declared_beats_everything_including_a_cached_probe(self):
        from core import windowcache
        windowcache.put("http://x/v1", "claude-opus-5", 111_111)
        cfg = type("C", (), {"context_window": 333_000})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 333_000)

    def test_a_cached_probe_beats_the_ceiling_table(self):
        """The table says 1,000,000 for this name. A real endpoint saying 204,800 is closer
        to the truth than a ceiling, and the whole point of probing."""
        from core import windowcache
        windowcache.put("http://x/v1", "claude-opus-5", 204_800)
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 204_800)

    def test_the_table_answers_when_nothing_was_probed(self):
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 1_000_000)

    def test_an_unknown_model_with_no_probe_is_still_zero(self):
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("MiniMax-M2.7", cfg, "http://x/v1"), 0)

    def test_with_NO_endpoint_the_cache_is_not_consulted_at_all(self):
        """claude-code has no endpoint to offer, and passing none must behave exactly as it
        did before this cascade existed."""
        from core import windowcache
        windowcache.put("", "claude-opus-5", 42)
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg), 1_000_000)

    def test_a_cached_value_is_used_even_when_STALE(self):
        from core import windowcache
        windowcache.put("http://x/v1", "MiniMax-M2.7", 204_800, ttl=-1)
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("MiniMax-M2.7", cfg, "http://x/v1"), 204_800)


class TestTheResolverNeverReachesTheNetwork(unittest.TestCase):
    """The guard calls this before EVERY file read. A probe here would be a network call on
    the hot path, and the reason the cache exists at all."""

    def test_resolving_with_the_socket_broken_still_answers(self):
        import socket
        import tempfile
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        original = socket.socket.connect
        socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("window_for reached the network"))
        try:
            cfg = type("C", (), {"context_window": 0})()
            self.assertEqual(windows.window_for("claude-opus-5", cfg, "http://x/v1"), 1_000_000)
        finally:
            socket.socket.connect = original
```

Confirme que `import os` está no topo de `tests/test_windows.py`; acrescente se faltar.

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_windows 2>&1 | tail -4`
Expected: FAIL — `window_for() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: Implemente a cascata**

Em `core/windows.py`, troque a assinatura e insira o degrau do cache entre o declarado e a tabela:

```python
def window_for(model: str, cfg, endpoint: str = "") -> int:
    """Tokens the context window holds, or 0 when we do not know.

    FOUR STEPS, and each is consulted only when the one before it did not answer:

      1. `context_window` in the config — whoever declared it does not want a guess, and a
         host upgrade must not move what they fixed.
      2. A window an endpoint reported, from the cache. Closer to the truth than a ceiling,
         and the reason probing exists. Used even when STALE: falling back to the table
         because the value is a day old makes the guard sleep on a window it already knows.
      3. The ceiling table, by bare model name.
      4. 0 — and 0 is load-bearing: the caller must ALLOW the read when the window is
         unknown. Blocking on a guessed window is the one failure this guard must not produce.

    THIS FUNCTION NEVER REACHES THE NETWORK. It runs before every file read; the probe that
    fills the cache is run by the hooks that already pay for network. `endpoint` empty means
    "this host has no endpoint to offer" — claude-code — and skips step 2 entirely, leaving
    the behaviour exactly as it was before the cascade existed.
    """
    declared = getattr(cfg, "context_window", 0)
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        declared = 0
    if declared > 0:
        return declared

    if endpoint:
        from .windowcache import get as cached      # local: keeps this import off the

        window, _fresh = cached(endpoint, model)    # hot path for hosts that pass nothing
        if window > 0:
            return window

    return int(MODEL_WINDOWS.get((model or "").strip(), 0))
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_windows 2>&1 | tail -3`
Expected: `OK`, e a contagem sobe em 7

- [ ] **Step 5: Prove que cada degrau morde SOZINHO**

Quatro mutações, uma por vez, restaurando entre elas. Para cada uma: verifique o backup, afirme a
contagem, confirme por `grep` que aterrou, rode, e confirme que existe linha `Ran N`.

```bash
cp core/windows.py /tmp/wn.bak && [ -s /tmp/wn.bak ] || { echo "BACKUP FALHOU"; exit 1; }
# A) o degrau do cache desligado
python3 - <<'PY'
p = "core/windows.py"; s = open(p).read()
old = "    if endpoint:\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "    if False:\n"))
PY
grep -c "if False:" core/windows.py
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_windows 2>&1 | grep -E "^FAIL: test|^Ran|^FAILED"
cp /tmp/wn.bak core/windows.py
```
Expected: cai `test_a_cached_probe_beats_the_ceiling_table` e `test_a_cached_value_is_used_even_when_STALE`, e **não** cai `test_declared_beats_everything...`.

```bash
# B) o declarado deixa de vencer o cache
python3 - <<'PY'
p = "core/windows.py"; s = open(p).read()
old = "    if declared > 0:\n        return declared\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, ""))
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_windows 2>&1 | grep -E "^FAIL: test|^Ran|^FAILED"
cp /tmp/wn.bak core/windows.py

# C) o cache passa a ser consultado mesmo sem endpoint
python3 - <<'PY'
p = "core/windows.py"; s = open(p).read()
old = "    if endpoint:\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "    if True:\n"))
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_windows 2>&1 | grep -E "^FAIL: test|^Ran|^FAILED"
cp /tmp/wn.bak core/windows.py && rm /tmp/wn.bak
```
Expected em C: cai `test_with_NO_endpoint_the_cache_is_not_consulted_at_all`.

**As mutações A e C não podem derrubar o mesmo conjunto.** Se derrubarem, no máximo uma delas é a
mutação que você pensa que é — verifique antes de seguir.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/windows.py tests/test_windows.py
git commit -F - <<'MSG'
feat: four steps to a window, and the resolver never reaches the network

Declared wins, then a window an endpoint reported, then the ceiling table, then zero. Each
step is consulted only when the one before it did not answer, and zero still means ALLOW,
because blocking on a guessed window is the one failure this guard must not produce.

A cached value is used even when stale. Falling back to a ceiling because the number is a day
old makes the guard sleep on a window it already knows — stale beats absent.

The endpoint argument is optional, and empty means "this host has no endpoint to offer".
claude-code passes nothing and behaves exactly as it did before this cascade existed, which
is why the existing two call sites needed no change.

And this function never opens a socket: it runs before every file read, so the probe that
fills the cache belongs to the hooks that already pay for network. There is a test that
breaks socket.connect and requires an answer anyway.
MSG
```

---

### Task 4: O hermes descobre sozinho

**Files:**
- Create: `hosts/hermes/endpoint.py`
- Modify: `hosts/hermes/bigfile.py:236` (passar o endpoint), `hosts/hermes/__init__.py`
  (atualizar o cache no `prefetch`)
- Test: `tests/test_hermes_endpoint.py`

**Interfaces:**
- Consumes: `windowprobe.probe(...)` (Task 2), `windowcache.{get,put}` (Task 1),
  `window_for(model, cfg, endpoint)` (Task 3).
- Produces:
  - `endpoint.from_hermes_config(home: str | None = None) -> tuple[str, str]` —
    `(base_url, api_key)`, `("", "")` quando não deu.
  - `endpoint.refresh_window(model: str) -> int` — sonda se o cache está velho; devolve a janela
    conhecida (0 se nenhuma). Chamado do `prefetch`.

**O adaptador lê o config do hermes, e isso é acoplamento declarado.** Ele já depende do `state.db`
do hermes; a defesa é a mesma — teste que morde quando o formato muda, e falha que DESCE a cascata.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_hermes_endpoint.py
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hosts.hermes import endpoint  # noqa: E402

CONFIG = """\
model:
  provider: custom
  base_url: https://server.example/api/v1
  key_env: MY_KEY_VAR
memory:
  provider: memories
"""


def a_hermes_home(text: str = CONFIG) -> str:
    home = tempfile.mkdtemp()
    with open(os.path.join(home, "config.yaml"), "w") as fh:
        fh.write(text)

    return home


class TestReadingTheHermesConfig(unittest.TestCase):
    def setUp(self):
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        os.environ["MY_KEY_VAR"] = "secret-value"

    def test_it_reads_the_base_url_and_resolves_the_key_from_the_environment(self):
        """`key_env` names a VARIABLE, not a secret. The config holds the name; the value
        lives in the environment the hermes process already has."""
        base, key = endpoint.from_hermes_config(a_hermes_home())
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "secret-value")

    def test_a_missing_config_yields_nothing_rather_than_raising(self):
        self.assertEqual(endpoint.from_hermes_config(tempfile.mkdtemp()), ("", ""))

    def test_a_config_without_a_base_url_yields_nothing(self):
        self.assertEqual(endpoint.from_hermes_config(a_hermes_home("memory:\n  provider: x\n")),
                         ("", ""))

    def test_a_key_variable_absent_from_the_environment_still_yields_the_url(self):
        """An endpoint that needs no key is a real case, and refusing the URL because the
        variable is unset would turn a working setup into no setup."""
        os.environ.pop("MY_KEY_VAR", None)
        base, key = endpoint.from_hermes_config(a_hermes_home())
        self.assertEqual(base, "https://server.example/api/v1")
        self.assertEqual(key, "")

    def test_a_config_that_is_not_readable_yields_nothing(self):
        home = a_hermes_home()
        os.chmod(os.path.join(home, "config.yaml"), 0)
        try:
            self.assertEqual(endpoint.from_hermes_config(home), ("", ""))
        finally:
            os.chmod(os.path.join(home, "config.yaml"), 0o644)


class TestRefreshing(unittest.TestCase):
    def setUp(self):
        os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
        os.environ["HERMES_HOME"] = a_hermes_home()
        os.environ["MY_KEY_VAR"] = "secret-value"

    def test_a_fresh_cache_is_NOT_probed_again(self):
        from core import windowcache
        windowcache.put("https://server.example/api/v1", "m", 524288)
        calls = []
        got = endpoint.refresh_window("m", probe=lambda *a, **k: calls.append(1) or 999)
        self.assertEqual(got, 524288)
        self.assertEqual(calls, [], "a fresh cache was refreshed anyway")

    def test_an_empty_cache_is_probed_and_the_answer_stored(self):
        from core import windowcache
        got = endpoint.refresh_window("m", probe=lambda *a, **k: 524288)
        self.assertEqual(got, 524288)
        self.assertEqual(windowcache.get("https://server.example/api/v1", "m")[0], 524288)

    def test_a_probe_that_learns_nothing_stores_nothing(self):
        from core import windowcache
        self.assertEqual(endpoint.refresh_window("m", probe=lambda *a, **k: 0), 0)
        self.assertEqual(windowcache.get("https://server.example/api/v1", "m"), (0, False))

    def test_a_failed_probe_keeps_the_STALE_value(self):
        """The endpoint being down must not cost a window we already knew."""
        from core import windowcache
        windowcache.put("https://server.example/api/v1", "m", 204800, ttl=-1)
        got = endpoint.refresh_window("m", probe=lambda *a, **k: 0)
        self.assertEqual(got, 204800)

    def test_with_no_endpoint_configured_it_does_nothing_quietly(self):
        os.environ["HERMES_HOME"] = tempfile.mkdtemp()
        self.assertEqual(endpoint.refresh_window("m", probe=lambda *a, **k: 999), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_hermes_endpoint 2>&1 | tail -4`
Expected: FAIL — `ImportError: cannot import name 'endpoint'`

- [ ] **Step 3: Implemente `hosts/hermes/endpoint.py`**

```python
#!/usr/bin/env python3
"""Which endpoint serves the model, read from the config hermes already keeps.

WHY READ ANOTHER PROJECT'S CONFIG. The window is knowable on this host and nowhere else: the
endpoint that serves the model reports it, and hermes is the only place that records which
endpoint that is. Asking the user to declare it again, in our config, to describe something
his other config already describes, is a second source of truth for one fact.

THE COUPLING IS REAL AND DECLARED. This depends on a config format owned by another project,
which can change. It is the same coupling this adapter already has with hermes' state.db, and
the defence is the same: a test that bites when the shape changes, and a failure that DESCENDS
the cascade rather than breaking. Every path here returns "" or 0, never an exception.

`key_env` NAMES A VARIABLE, IT IS NOT THE SECRET. The config holds the name; the value lives
in the environment the hermes process already has, because it is hermes that loads this
plugin. An endpoint that needs no key is a real case, so a missing variable still yields the
URL.
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from core import windowcache, windowprobe  # noqa: E402

#: Short: this runs from `prefetch`, which the user is waiting on.
PROBE_TIMEOUT_S = 5.0


def _home(home: str | None = None) -> str:
    """The same resolution hermes uses for subprocesses."""
    return home or os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"),
                                                                 ".hermes")


def from_hermes_config(home: str | None = None) -> tuple[str, str]:
    """`(base_url, api_key)` from hermes' own config, or `("", "")` when it cannot be read.

    Parsed with a regex rather than a YAML library because this package ships stdlib only —
    and because the two lines wanted are flat scalars, not structure. A shape this does not
    recognise yields "", which the caller reads as "no endpoint", which descends the cascade.
    """
    path = os.path.join(_home(home), "config.yaml")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return "", ""
    base = re.search(r"^\s+base_url:\s*(\S+)\s*$", text, re.M)
    if not base:
        return "", ""
    var = re.search(r"^\s+key_env:\s*(\w+)\s*$", text, re.M)

    return base.group(1), (os.environ.get(var.group(1), "") if var else "")


def refresh_window(model: str, *, probe=None) -> int:
    """The window for `model`, probing only when the cache has nothing fresh.

    Called from `prefetch`, which already pays for network. Returns the best value known —
    including a stale one when the probe learns nothing, because an endpoint being down must
    not cost a window we already had.

    `probe` is injected only so tests can drive it without a server.
    """
    base, key = from_hermes_config()
    if not base:
        return 0
    known, fresh = windowcache.get(base, model)
    if fresh:
        return known
    call = probe or windowprobe.probe
    learned = call(base, key, model, timeout=PROBE_TIMEOUT_S)
    if learned > 0:
        windowcache.put(base, model, learned)

        return learned

    return known
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_hermes_endpoint 2>&1 | tail -3`
Expected: `Ran 10 tests`, `OK`

- [ ] **Step 5: Ligue nos dois pontos**

Em `hosts/hermes/bigfile.py`, o adaptador passa a oferecer o endpoint ao resolvedor. Troque:

```python
    budget = budget_from(db_path, session_id, lambda model: windows.window_for(model, cfg))
```

por:

```python
    from hosts.hermes.endpoint import from_hermes_config

    base, _key = from_hermes_config()
    budget = budget_from(db_path, session_id,
                         lambda model: windows.window_for(model, cfg, base))
```

E em `hosts/hermes/__init__.py`, dentro de `_prefetch`, depois de o bloco de recall estar montado e
antes do `return`, acrescente a atualização oportunista:

```python
        # The window is knowable on this host, and this is where the network is already being
        # paid for. The guard reads the cache and never probes: it runs before every file read.
        try:
            from .endpoint import refresh_window

            refresh_window(self._model_of(session_id))
        except Exception:                       # noqa: BLE001
            pass                                # a window we did not learn is the next step
                                                # of the cascade, never a failed prefetch
```

**Se `_model_of` não existir**, o modelo da sessão sai da mesma consulta que o adaptador da guarda
já faz — `select model from sessions where id = ?`, com o fallback por `session_id` que
`hosts/hermes/bigfile.py` implementa. Reuse aquela função em vez de escrever a consulta de novo; se
ela não estiver exposta, extraia-a e **relate a extração**.

- [ ] **Step 6: Prove que a guarda continua sem tocar a rede**

```bash
cp hosts/hermes/bigfile.py /tmp/hb.bak && [ -s /tmp/hb.bak ] || { echo "BACKUP FALHOU"; exit 1; }
python3 - <<'PY'
import socket, sys, os, tempfile
sys.path.insert(0, ".")
os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network!"))
from core import windows
cfg = type("C", (), {"context_window": 0})()
print("com socket quebrado, window_for devolveu:", windows.window_for("claude-opus-5", cfg, "http://x/v1"))
PY
rm -f /tmp/hb.bak
```
Expected: imprime `1000000` sem levantar — o resolvedor leu cache e tabela, e nunca a rede.

- [ ] **Step 7: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add hosts/hermes/endpoint.py hosts/hermes/bigfile.py hosts/hermes/__init__.py tests/test_hermes_endpoint.py
git commit -F - <<'MSG'
feat: on hermes the window is knowable, so stop asking the user for it

The endpoint that serves the model reports its window, and hermes is the only place that
records which endpoint that is. Asking for it again in our config would be a second source of
truth for one fact, so this reads hermes' own — a coupling that is real, declared, and the
same one this adapter already has with hermes' state.db. Every path returns "" or 0 rather
than raising, so a format change descends the cascade instead of breaking the guard.

The probe runs from prefetch, where network is already being paid for, and only when the cache
has nothing fresh. The guard itself never probes: it runs before every file read, and a probe
there would be both circular and slow.

A probe that learns nothing keeps the stale value. An endpoint being down must not cost a
window we already knew — the alternative is falling back to a ceiling and sleeping on a number
that was already measured.

key_env names a variable, not a secret: the config holds the name and the value lives in the
environment hermes already has. An endpoint needing no key still yields its URL.
MSG
```

---

### Task 5: Dizer o que mudou, e provar que o outro host não mudou

**Files:**
- Modify: `README.md` (a seção `### The big-file read guard`)
- Modify: `docs/superpowers/specs/2026-08-15-big-file-read-guard-design.md` (nota de emenda)
- Test: `tests/test_host_equivalence.py` (acrescentar)

**Interfaces:**
- Consumes: tudo das Tasks 1 a 4.
- Produces: nada de código.

**A seção da guarda no README é verificada por teste** (`TestTheREADMEDescribesTheGuardThatSHIPPED`,
em `tests/test_host_equivalence.py`), que exige os dois limiares com valor e percentual, os knobs, o
marcador de fuga, os tetos por leitura e os números da divergência. **Não remova nada disso** —
acrescente.

- [ ] **Step 1: Escreva o teste que falha**

```python
# acrescente a tests/test_host_equivalence.py, na classe TestTheREADMEDescribesTheGuardThatSHIPPED
    def test_it_says_where_the_window_comes_from_on_each_host(self):
        """The window resolution differs by host, and a reader who does not know that will
        declare `context_window` on a host that no longer needs it, or fail to declare it on
        the host that does."""
        section = guard_section()
        for needle in ("context_window", "/models", "ceiling"):
            with self.subTest(needle=needle):
                self.assertIn(needle, section)
```

E, numa classe própria, a prova de que o claude-code não regrediu:

```python
class TestTheClaudeCodeSideDidNotChange(unittest.TestCase):
    """The cascade added a step that only one host can fill. The other must resolve exactly
    as it did before — a silent change there would move the guard's threshold for everyone
    on the host that cannot even use the new step."""

    def test_resolving_without_an_endpoint_gives_the_table_value(self):
        from core import windows
        cfg = type("C", (), {"context_window": 0})()
        self.assertEqual(windows.window_for("claude-opus-5", cfg), 1_000_000)
        self.assertEqual(windows.window_for("claude-haiku-4-5", cfg), 200_000)
        self.assertEqual(windows.window_for("nao-existe", cfg), 0)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_host_equivalence 2>&1 | tail -4`
Expected: FAIL nomeando `test_it_says_where_the_window_comes_from_on_each_host`

- [ ] **Step 3: Escreva a seção do README**

Na seção `### The big-file read guard`, acrescente — sem remover nada do que já está lá:

```markdown
**Where the window comes from, and why it differs by host.** The guard decides by percentage
of what REMAINS, so it needs the window. It resolves in four steps, and each is consulted only
when the one before it did not answer:

1. `context_window` in your config — declaring it wins over everything.
2. A window the model's endpoint reported, cached. **hermes only**, because it is the only
   host that records which endpoint serves the model; the value is refreshed from `/models`
   by the hook that already talks to the network, never by the guard itself.
3. The **ceiling** table by model name — the LARGEST window any variant of that name can
   have, because the transcript records the bare name and a 200k variant is indistinguishable
   from a 1M one.
4. Zero, which ALLOWS: blocking on a window we are unsure of is the one failure this guard
   must not produce.

On claude-code, step 2 never fires: the host hands the window to its status line and not to
hooks, measured. So there the table decides — and it is right for a 1M variant and generous
for a 200k one. **If you run a 200k session, declare `context_window`**, or the guard will
believe there is five times more room than there is.
```

- [ ] **Step 4: Emende o spec da guarda**

Em `docs/superpowers/specs/2026-08-15-big-file-read-guard-design.md`, logo após a emenda de tetos,
acrescente:

```markdown
> **EMENDA de 2026-08-17.** A afirmação "a janela máxima NÃO é legível de disco em nenhum host"
> vale para o claude-code — os quatro caminhos foram medidos — e **deixou de valer para o
> hermes**: o endpoint que serve o modelo a informa por `/models`, e o adaptador passou a lê-la.
> A tabela de tetos continua sendo o degrau seguinte, não o primeiro. Ver
> `docs/superpowers/specs/2026-08-17-window-resolution-design.md`.
```

- [ ] **Step 5: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add README.md docs/superpowers/specs/2026-08-15-big-file-read-guard-design.md tests/test_host_equivalence.py
git commit -F - <<'MSG'
docs: the window resolves in four steps, and step two only exists on one host

A reader who does not know that will declare context_window on the host that no longer needs
it, or fail to declare it on the host that does — and the second mistake is silent, because a
guard that believes there is five times more room than there is simply never fires.

The big-file spec claimed the window is not readable from disk on either host. That was
measured, and it still holds for claude-code: the payload, the transcript, the on-disk state
and the environment were all checked, and only the status line receives it. It stopped holding
for hermes, and the spec now says which half changed rather than leaving a sentence that is
half true.

The claude-code path gets a test of its own. The cascade added a step that only one host can
fill, and a silent change on the other would move the guard's threshold for everyone who
cannot even use the new step.
MSG
```

---

## Auto-revisão do plano

**Cobertura da spec, item por item:**

| Item da spec | Task |
|---|---|
| Cascata de quatro degraus | T3 |
| Declarado vence sempre | T3 (teste + mutação B) |
| Cache chaveado por `(endpoint, modelo)` | T1 |
| Cache velho é usado, com flag | T1, T3 |
| Cache corrompido lê vazio | T1 |
| Zero nunca é armazenado | T1 |
| A guarda NUNCA sonda | T3 (socket quebrado), T4 Step 6 |
| Quem preenche é quem já paga rede | T4 (`prefetch`) |
| Os quatro caminhos, incluindo o aninhado | T2 |
| A ordem que o OpenRouter força | T2 (fixture DIVERGENTE) |
| Nome/lugar desconhecido cai um degrau | T2 |
| Não verificado: OpenAI/Anthropic | T2 (docstring) |
| Endpoint lido do config do hermes | T4 |
| `key_env` é nome, não segredo | T4 |
| Toda falha desce, nunca sobe | T1, T2, T4 (cada um com teste) |
| claude-code segue na tabela | T3 (endpoint vazio), T5 (teste dedicado) |
| Integração guardada contra servidor real | T2 Step 6 |
| Limite do claude-code documentado | T5 |

**Placeholders:** nenhum "TBD"/"TODO"/"similar à Task N". O único passo que descreve em vez de
mostrar é o Step 5 da Task 4 quanto ao `_model_of` — e é deliberado: o nome real da função depende
do que `hosts/hermes/bigfile.py` expõe hoje, e o plano manda **reusar e relatar** em vez de
adivinhar um nome que pode não existir.

**Consistência de tipo:** `windowcache.get` devolve `(int, bool)` em T1 e é lido assim em T3 e T4;
`windowprobe.probe(base, key, model, *, timeout, fetch)` em T2 é chamado com essa forma em T4;
`window_for(model, cfg, endpoint="")` em T3 é chamado com três argumentos em T4 e com dois em T5;
`from_hermes_config() -> (str, str)` em T4 é desempacotado assim nos dois pontos de ligação.

**Consistência de valor:** 524.288, 262.144, 1.000.000 e 204.800 são os números medidos e aparecem
idênticos na spec e aqui; `TTL_SECONDS` 86400 e `PROBE_TIMEOUT_S` 5.0 aparecem uma vez cada;
`model-windows.json` é o mesmo nome em T1 e na spec.
