# Sub-coleções por repositório — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** dar ao acervo um terceiro arquivo, agrupado por repositório, em que se possa perguntar
*"algum dos meus projetos menciona x"* numa única consulta.

**Architecture:** uma coleção de chunks com `repo` indexado no payload, um registro em coleção
própria que é autoritativo sobre quais repos existem, e `search_groups` no servidor para a
travessia entre projetos. `core/repos.py` é dono do acervo e do registro; `core/bindings.py` é
dono do vínculo local *caminho → repo*; o CLI só traduz argumentos.

**Tech Stack:** Python 3, só stdlib. Qdrant 1.18.2 via `core/qdrant.py`.

**Spec:** `docs/superpowers/specs/2026-08-17-repo-subcollections-design.md`

## Base, já resolvida

Esta branch foi **rebaseada em `cbfad7f`** (o merge do PR #1, a guarda de arquivo grande) em
2026-08-17, e a suíte foi reconferida: **737 testes, OK (skipped=17)**. Essa é a baseline. O
conflito em `core/config.py` que uma versão anterior deste plano previa **não aconteceu** — as
duas branches tocaram regiões diferentes.

**O que agora EXISTE nesta base e o plano PODE referenciar** (uma versão anterior dizia o
contrário, escrita antes do merge): `core/knobs.py` com `env(name, legacy, default)` e
`env_num(...)`; `core/bigfile.py`, `core/windows.py`, `core/inventory.py`, `hooks/bigfile.py`.
Os helpers de env **já estão unificados** em `core/knobs.py` — não os reescreva.

## Global Constraints

- **Só stdlib.** Nada de dependência externa em nada que rode dentro de hook.
- **Código, comentários, mensagens ao usuário, README e commits em INGLÊS.** Specs em português.
- **Falha NÃO abre.** Busca que não pode consultar levanta erro explícito; NUNCA devolve lista
  vazia, que seria indistinguível de "não há nada".
- **`scope=all` do `docs` não pode alcançar repos.** É regressão de funcionalidade entregue.
- **Reusar, não copiar:** `chunk_text` e `mode_for_suffix` de `core/chunk.py`; `content_digest`,
  `source_changed`, `doc_id_for`, `_point_id`, `_read_source`, `_iso` de `core/docs.py`.
- **As CINCO coleções têm de ser distintas** (memory, docs, library, repos, repos_registry).
- **Toda guarda precisa de sonda de mutação**: removê-la deixa a suíte vermelha, com a contagem
  de ocorrências verificada antes de substituir e o **escopo** de cada contagem declarado
  (módulo isolado ou suíte inteira).
- **A suíte nunca fica vermelha em nenhum commit.** Baseline: rode
  `python3 -m unittest discover -s tests 2>&1 | tail -3` e anote a contagem antes da Task 1.

---

### Task 1: `search_groups` — a travessia entre repos

**Files:**
- Modify: `core/ports.py` (protocolo `VectorStore`)
- Modify: `core/qdrant.py` (implementação HTTP)
- Modify: `tests/fakes.py` (`FakeVectorStore`)
- Test: `tests/test_search_groups.py`

**Interfaces:**
- Consumes: nada.
- Produces: `VectorStore.search_groups(name: str, vector: list[float], group_by: str,
  limit: int, group_size: int, filter_: dict | None = None,
  with_payload: bool = True) -> list[dict]`, devolvendo
  `[{"id": <valor do campo>, "hits": [<ponto>, ...]}, ...]`, grupos ordenados pelo melhor hit.

**Por que este método existe, e a frase tem de estar no docstring:** agrupar o top-K no CLIENTE
responde "quais itens estão no top-K", não "quais grupos casaram". Um repo com cinquenta chunks
parecidos ocupa os dez primeiros lugares e um repo com UMA menção real desaparece.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_search_groups.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

COLL = "repos_test"


def a_store_with(loud_chunks: int, quiet_chunks: int) -> tuple:
    """A store where one repo is LOUD (many near-identical chunks about the subject) and
    another is QUIET (one weaker mention). This asymmetry is the whole point: a balanced
    fixture passes under client-side grouping and proves nothing."""
    store, embedder = FakeVectorStore(), FakeEmbedder(dim=8)
    store.ensure_collection(COLL, 8)
    points = []
    for i in range(loud_chunks):
        points.append({"id": 1000 + i, "vector": embedder.embed_one("billing invoice charge"),
                       "payload": {"repo": "loud", "document": f"billing {i}"}})
    for i in range(quiet_chunks):
        points.append({"id": 2000 + i, "vector": embedder.embed_one("invoice"),
                       "payload": {"repo": "quiet", "document": f"invoice {i}"}})
    store.upsert(COLL, points)

    return store, embedder


class TestTheQuietRepoIsNotShadowed(unittest.TestCase):
    def test_both_repos_come_back_even_when_one_dominates(self):
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=10, group_size=3)
        self.assertEqual({g["id"] for g in groups}, {"loud", "quiet"},
                         "the single-mention repo was shadowed by the loud one")

    def test_a_plain_search_DOES_shadow_it(self):
        """The control. Without this, the test above could pass for the wrong reason and we
        would never learn that grouping is what fixed it."""
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        hits = store.search(COLL, embedder.embed_one("billing invoice charge"), limit=10)
        self.assertEqual({h["payload"]["repo"] for h in hits}, {"loud"},
                         "the premise of this whole method is false: top-K did NOT shadow")

    def test_group_size_caps_hits_per_group(self):
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=10, group_size=3)
        loud = next(g for g in groups if g["id"] == "loud")
        self.assertEqual(len(loud["hits"]), 3)

    def test_limit_caps_the_number_of_groups(self):
        store, embedder = a_store_with(loud_chunks=50, quiet_chunks=1)
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=1, group_size=3)
        self.assertEqual(len(groups), 1)

    def test_a_point_without_the_field_is_not_a_group(self):
        store, embedder = a_store_with(loud_chunks=2, quiet_chunks=1)
        store.upsert(COLL, [{"id": 9999, "vector": embedder.embed_one("billing invoice charge"),
                             "payload": {"document": "no repo key at all"}}])
        groups = store.search_groups(COLL, embedder.embed_one("billing invoice charge"),
                                     group_by="repo", limit=10, group_size=3)
        self.assertNotIn(None, {g["id"] for g in groups})


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_search_groups 2>&1 | tail -4`
Expected: FAIL — `AttributeError: 'FakeVectorStore' object has no attribute 'search_groups'`

- [ ] **Step 3: Acrescente o método ao protocolo**

Em `core/ports.py`, dentro de `class VectorStore(Protocol)`, depois de `search`:

```python
    def search_groups(self, name: str, vector: list[float], group_by: str, limit: int,
                      group_size: int, filter_: dict | None = ...,
                      with_payload: bool = ...) -> list[dict]:
        """Best hits per DISTINCT value of `group_by`, not the global top-K.

        This is not a convenience over `search`. Grouping the top-K on the client answers
        "which values are IN the top-K", which is a different question: one group with fifty
        near-identical chunks fills every slot and a group with a single genuine match
        disappears. Groups here form per distinct value, so the quiet one survives.

        `limit` counts GROUPS; `group_size` counts hits within a group. Returns
        `[{"id": <field value>, "hits": [point, ...]}, ...]`, best group first.

        It is best-effort over what the search reaches, NOT an exhaustive scan: a group whose
        best hit falls below the score horizon can be missing. No caller may read an empty
        result as proof of absence.
        """
        ...
```

- [ ] **Step 4: Implemente no cliente HTTP**

Em `core/qdrant.py`, depois de `search`:

```python
    def search_groups(self, name: str, vector: list[float], group_by: str, limit: int,
                      group_size: int, filter_: dict | None = None,
                      with_payload: bool = True) -> list[dict]:
        body = {"vector": vector, "group_by": group_by, "limit": limit,
                "group_size": group_size, "with_payload": with_payload}
        if filter_:
            body["filter"] = filter_
        res = self.request("POST", f"/collections/{name}/points/search/groups", body)

        return (res.get("result") or {}).get("groups", []) or []
```

- [ ] **Step 5: Implemente no fake, e implemente DE VERDADE**

Em `tests/fakes.py`, dentro de `FakeVectorStore`, depois de `search`. O fake não pode ser um
stub: é ele que sustenta o teste de ofuscação, e um fake que devolve tudo provaria o oposto.

```python
    def search_groups(self, name: str, vector: list[float], group_by: str, limit: int,
                      group_size: int, filter_: dict | None = None,
                      with_payload: bool = True) -> list[dict]:
        """Real grouping over the real cosine ranking, so the shadowing test means
        something. Over-fetches deliberately: grouping the top-K is the defect this method
        exists to avoid, so the fake must not reproduce it."""
        ranked = self.search(name, vector, limit=len(self.collections.get(name, {}).get("points", {})),
                             filter_=filter_, with_payload=True)
        groups: dict = {}
        for hit in ranked:
            key = (hit.get("payload") or {}).get(group_by)
            if key is None:
                continue
            groups.setdefault(key, []).append(hit)
        out = [{"id": key, "hits": hits[:group_size]} for key, hits in groups.items()]
        out.sort(key=lambda g: g["hits"][0]["score"], reverse=True)

        return out[:limit]
```

- [ ] **Step 6: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_search_groups 2>&1 | tail -3`
Expected: `Ran 5 tests`, `OK`

- [ ] **Step 7: Prove que o agrupamento é o que conserta**

O teste de controle (`test_a_plain_search_DOES_shadow_it`) já afirma a premissa. Prove agora que
o agrupamento morde, mutando o fake para agrupar só o top-K:

```bash
cp tests/fakes.py /tmp/fk.bak
python3 - <<'PY'
p = "tests/fakes.py"; s = open(p).read()
old = "        ranked = self.search(name, vector, limit=len(self.collections.get(name, {}).get(\"points\", {})),\n"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, "        ranked = self.search(name, vector, limit=limit,\n"))
print("mutation landed: the fake now groups only the top-K")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_search_groups 2>&1 | tail -3
cp /tmp/fk.bak tests/fakes.py && rm /tmp/fk.bak && python3 -m unittest tests.test_search_groups 2>&1 | tail -1
```
Expected: `FAILED` nomeando `test_both_repos_come_back_even_when_one_dominates`, e depois do
restore `OK`. Registre as duas contagens COM o escopo (`tests.test_search_groups`).

- [ ] **Step 8: Teste de integração contra o Qdrant real, sob guarda**

```python
# acrescente ao fim de tests/test_search_groups.py, antes do __main__
@unittest.skipUnless(os.environ.get("QCTX_INTEGRATION") == "1",
                     "integration: needs a reachable Qdrant")
class TestTheRealServerGroups(unittest.TestCase):
    """Measured on 1.18.2 while designing this: `search/groups` returned 4 distinct groups in
    one query. This pins the CONTRACT (shape of the response) against the real server, which
    is the half a fake cannot prove."""

    def test_the_response_shape_is_groups_with_hits(self):
        from core import load
        from core.qdrant import build_qdrant
        cfg = load()
        q = build_qdrant(cfg)
        name = cfg.require_docs_collection()
        info = q.collection_info(name)
        if not info or not info.get("points_count"):
            self.skipTest(f"{name} is empty: zero groups from an empty collection would "
                          f"look like a missing capability and prove nothing")
        dim = cfg.vector_size
        groups = q.search_groups(name, [0.01] * dim, group_by="doc_id", limit=4, group_size=2)
        self.assertTrue(groups, "the real server returned no groups for a populated collection")
        for g in groups:
            self.assertIn("id", g)
            self.assertTrue(g["hits"])
```

**A guarda de coleção vazia não é zelo:** na sondagem do desenho, a primeira tentativa foi contra
`memories_docs_library`, que está vazia, e voltou zero grupos — o que parece falta de capacidade
e não é.

- [ ] **Step 9: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK`, contagem = baseline + 5

- [ ] **Step 10: Commit**

```bash
git add core/ports.py core/qdrant.py tests/fakes.py tests/test_search_groups.py
git commit -F - <<'MSG'
feat: ask which repos matched, not which chunks are in the top ten

Grouping the top-K on the client answers the wrong question. Vector search returns the K
best by similarity, so a repo with fifty near-identical chunks fills every slot and a repo
with one genuine mention vanishes — a confident wrong answer to "does any project mention
this", which is the question the archive is being reshaped to answer.

Qdrant groups server-side, per DISTINCT value rather than by global rank, so the quiet repo
gets its own group. Measured on this deployment (1.18.2) before designing around it.

The fake implements real grouping over its real cosine ranking, and over-fetches on purpose:
a fake that grouped the top-K would reproduce the defect and its test would prove the
opposite of what it claims. The paired control test asserts that a plain search DOES shadow,
so the grouping test cannot pass for the wrong reason.

It is best-effort over what the search reaches, never an exhaustive scan, and the docstring
forbids reading an empty result as proof of absence.
MSG
```

---

### Task 2: as duas coleções na config, e as CINCO distintas

**Files:**
- Modify: `core/config.py` (`ENV_ALIASES`, `DEFAULTS`, `Config`, `_require_distinct`, dois
  `require_*`)
- Modify: `hosts/hermes/__init__.py` (dict `described` de `get_config_schema`)
- Modify: `README.md` (tabela de aliases de env)
- Test: `tests/test_config_repos.py`

**Interfaces:**
- Consumes: nada.
- Produces: `Config.repos_collection: str`, `Config.repos_registry_collection: str`,
  `Config.require_repos_collection() -> str`, `Config.require_repos_registry_collection() -> str`.

**DUAS ARMADILHAS MEDIDAS, e as duas já quebraram este repo antes.** (a) `get_config_schema` do
hermes itera `dataclasses.fields(Config)` e faz consulta CRUA num dict `described` — campo novo
sem entrada lá é `KeyError`, não silêncio. (b) Existe teste que DIFERENCIA a tabela de env do
README contra `ENV_ALIASES` campo por campo; esquecer o README deixa a suíte vermelha.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_config_repos.py
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as C  # noqa: E402
from core.config import ConfigError  # noqa: E402


def a_config(**over):
    base = dict(memory_collection="mem", docs_collection="tmp", library_collection="lib",
                repos_collection="repos", repos_registry_collection="reg")
    base.update(over)
    full = {f.name: getattr(C.Config, f.name, "") for f in __import__("dataclasses").fields(C.Config)}
    full.update({k: v for k, v in base.items()})

    return C.Config(**{k: v for k, v in full.items()})


class TestTheTwoNewCollections(unittest.TestCase):
    def test_the_defaults_are_named(self):
        self.assertEqual(C.DEFAULTS["repos_collection"], "memories_repos")
        self.assertEqual(C.DEFAULTS["repos_registry_collection"], "memories_repos_registry")

    def test_both_have_env_aliases(self):
        self.assertEqual(C.ENV_ALIASES["repos_collection"],
                         ("QCTX_REPOS_COLLECTION", "REPOS_COLLECTION"))
        self.assertEqual(C.ENV_ALIASES["repos_registry_collection"],
                         ("QCTX_REPOS_REGISTRY_COLLECTION", "REPOS_REGISTRY_COLLECTION"))

    def test_require_returns_the_name(self):
        cfg = a_config()
        self.assertEqual(cfg.require_repos_collection(), "repos")
        self.assertEqual(cfg.require_repos_registry_collection(), "reg")

    def test_an_unset_repos_collection_is_an_error_and_not_a_silent_default(self):
        with self.assertRaises(ConfigError):
            a_config(repos_collection="").require_repos_collection()


class TestTheFiveAreDistinct(unittest.TestCase):
    """Every collision degrades SILENTLY, which is why this raises instead of warning."""

    def test_repos_may_not_be_the_memory_collection(self):
        with self.assertRaises(ConfigError):
            a_config(repos_collection="mem").require_repos_collection()

    def test_repos_may_not_be_the_library(self):
        """The library is permanent and hand-picked; tens of thousands of automatic code
        chunks in it would drown exactly the archive that curation paid for."""
        with self.assertRaises(ConfigError):
            a_config(repos_collection="lib").require_repos_collection()

    def test_the_registry_may_not_be_the_chunk_archive(self):
        """They are separate so that no search ever has to filter registry rows out. Letting
        them collide reintroduces the filter this design exists to avoid."""
        with self.assertRaises(ConfigError):
            a_config(repos_registry_collection="repos").require_repos_registry_collection()

    def test_the_older_three_still_reject_the_new_two(self):
        with self.assertRaises(ConfigError):
            a_config(docs_collection="repos").require_docs_collection()


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_config_repos 2>&1 | tail -4`
Expected: FAIL — `KeyError: 'repos_collection'`

- [ ] **Step 3: Acrescente os dois campos**

Em `core/config.py`, nos três pontos (mantendo a ordem alfabética de vizinhança que o arquivo já
usa):

```python
# em ENV_ALIASES, junto de docs_collection / library_collection
    "repos_collection": ("QCTX_REPOS_COLLECTION", "REPOS_COLLECTION"),
    "repos_registry_collection": ("QCTX_REPOS_REGISTRY_COLLECTION", "REPOS_REGISTRY_COLLECTION"),

# em DEFAULTS
    "repos_collection": "memories_repos",
    "repos_registry_collection": "memories_repos_registry",

# no dataclass Config, depois de library_collection
    repos_collection: str
    repos_registry_collection: str
```

E os dois acessadores, ao lado de `require_library_collection`:

```python
    def require_repos_collection(self) -> str:
        return self._require_doc_collection("repos_collection", self.repos_collection)

    def require_repos_registry_collection(self) -> str:
        return self._require_doc_collection("repos_registry_collection",
                                            self.repos_registry_collection)
```

- [ ] **Step 4: Estenda a checagem de distinção para cinco**

Em `_require_distinct`, acrescente as duas entradas ao dict `others` e ACRESCENTE ao docstring o
que cada colisão nova custa — o docstring existente enumera a consequência de cada colisão, e
deixar as novas de fora tornaria a lista mentirosa por omissão:

```python
        others = {
            "memory_collection": self.memory_collection,
            "docs_collection": self.docs_collection,
            "library_collection": self.library_collection,
            "repos_collection": self.repos_collection,
            "repos_registry_collection": self.repos_registry_collection,
        }
```

Duas linhas novas no docstring, depois das que já existem:

```
        - a repo archive on top of the LIBRARY: tens of thousands of automatic code chunks
          drown the hand-picked documents, which is the same volume argument that keeps
          documents out of the memory collection, one level down.
        - the REGISTRY sharing the chunk collection: they are apart so that no search has to
          filter registry rows out, and a filter forgotten once turns a registry row into a
          search hit.
```

- [ ] **Step 5: A entrada do wizard do hermes (senão é KeyError)**

Em `hosts/hermes/__init__.py`, no dict `described` de `get_config_schema`:

```python
        "repos_collection": "Collection holding repository chunks, grouped by repo",
        "repos_registry_collection": "Collection holding one entry per indexed repository",
```

- [ ] **Step 6: A tabela do README (senão a suíte fica vermelha)**

Em `README.md`, na tabela de aliases de env, duas linhas no mesmo formato das vizinhas:
`QCTX_REPOS_COLLECTION` / `REPOS_COLLECTION` / `memories_repos`, e
`QCTX_REPOS_REGISTRY_COLLECTION` / `REPOS_REGISTRY_COLLECTION` / `memories_repos_registry`.

- [ ] **Step 7: Rode a suíte inteira**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK`. Se `test_hermes_provider` ou o teste da tabela do README caírem, é o Step 5 ou 6
faltando — não "conserte" o teste.

- [ ] **Step 8: Prove que a distinção morde**

```bash
cp core/config.py /tmp/cf.bak
python3 - <<'PY'
p = "core/config.py"; s = open(p).read()
old = '            "repos_collection": self.repos_collection,\n'
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, ""))
print("mutation landed: repos no longer checked for collision")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_config_repos 2>&1 | tail -3
cp /tmp/cf.bak core/config.py && rm /tmp/cf.bak
```
Expected: `FAILED` com pelo menos `test_repos_may_not_be_the_memory_collection` e
`test_repos_may_not_be_the_library`. Registre a contagem COM o escopo.

- [ ] **Step 9: Commit**

```bash
git add core/config.py hosts/hermes/__init__.py README.md tests/test_config_repos.py
git commit -F - <<'MSG'
feat: two collections for repositories, and five names that may not collide

The chunks live apart from the library for the reason documents live apart from memory: tens
of thousands of automatic code chunks drown a hand-picked archive on volume alone. The
registry lives apart from the chunks so that no search ever has to filter registry rows out
— a filter forgotten once turns a registry row into a search hit, and this repo has twice
measured that a guard held by no test gets deleted by a cleanup that passes CI.

Every collision here degrades silently, so the distinctness check raises. Its docstring
enumerates what each collision costs, and the two new lines were added there rather than
left out, because an enumeration that skips cases lies by omission.

Two places a new Config field always touches, both learned the hard way: the hermes wizard's
`described` table, where an omission is a KeyError and not silence, and the README env table,
which a dedicated test diffs field by field.
MSG
```

---

### Task 3: `core/repos.py` — o acervo, o registro, e a escrita

**Files:**
- Create: `core/repos.py`
- Modify: `core/__init__.py` (exportar `RepoIndex`, `RepoError`)
- Test: `tests/test_repos.py`

**Interfaces:**
- Consumes: `Config.require_repos_collection()`, `Config.require_repos_registry_collection()`
  (Task 2); `ports.VectorStore`, `ports.EmbeddingModel`.
- Produces:
  - `class RepoError(CoreError)`
  - `RepoIndex(qdrant, embedder, repos_collection: str, registry_collection: str,
    vector_size: int)`
  - `RepoIndex.add_files(repo: str, paths: list[str]) -> dict` — devolve
    `{"repo": str, "files": int, "chunks": int, "skipped": list[tuple[str, str]]}`
  - `RepoIndex.register(repo: str, label: str, remotes: list[str], checkout: str) -> dict`
  - `RepoIndex.get_repo(repo: str) -> dict | None`
  - `RepoIndex.list_repos() -> list[dict]`

**A escrita RECEBE a lista de arquivos; não a descobre.** Percorrer o repositório, respeitar
`.gitignore` e pular minificado é o sub-projeto B. Essa fronteira é o que torna esta task
testável com meia dúzia de arquivos.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_repos.py
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402

CHUNKS, REG = "repos_c", "repos_r"


def an_index() -> RepoIndex:
    return RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), CHUNKS, REG, 8)


def a_file(text: str, suffix: str = ".py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


class TestWriting(unittest.TestCase):
    def test_every_chunk_carries_the_repo(self):
        ix = an_index()
        ix.add_files("alpha", [a_file("def one():\n    return 1\n")])
        points = list(ix.q.scroll_all(CHUNKS))
        self.assertTrue(points)
        self.assertEqual({p["payload"]["repo"] for p in points}, {"alpha"})

    def test_the_repo_is_top_level_and_not_buried_in_metadata(self):
        """`group_by` and the payload index address a top-level key, exactly as `doc_id`
        already does. Burying it under metadata would work for reading and break both."""
        ix = an_index()
        ix.add_files("alpha", [a_file("x = 1\n")])
        payload = next(iter(ix.q.scroll_all(CHUNKS)))["payload"]
        self.assertIn("repo", payload)
        self.assertNotIn("repo", payload.get("metadata", {}))

    def test_reindexing_the_same_file_replaces_instead_of_accumulating(self):
        """Without this the old version and the new one coexist and one search mixes chunks
        from two states of the same file."""
        ix = an_index()
        path = a_file("first content here\n")
        ix.add_files("alpha", [path])
        before = len(list(ix.q.scroll_all(CHUNKS)))
        with open(path, "w") as fh:
            fh.write("second content, entirely different\n")
        ix.add_files("alpha", [path])
        after = list(ix.q.scroll_all(CHUNKS))
        self.assertEqual(len(after), before)
        self.assertIn("second", " ".join(p["payload"]["document"] for p in after))

    def test_an_empty_file_is_skipped_and_reported_not_raised(self):
        """One unindexable file in a list of eight hundred must not abort the other 799."""
        ix = an_index()
        out = ix.add_files("alpha", [a_file("   \n\n"), a_file("real = 1\n")])
        self.assertEqual(out["files"], 1)
        self.assertEqual(len(out["skipped"]), 1)

    def test_an_unreadable_file_is_skipped_and_reported(self):
        ix = an_index()
        gone = os.path.join(tempfile.mkdtemp(), "never-existed.py")
        out = ix.add_files("alpha", [gone, a_file("real = 1\n")])
        self.assertEqual(out["files"], 1)
        self.assertEqual([p for p, _ in out["skipped"]], [gone])

    def test_the_digest_is_stored_so_staleness_can_be_judged_later(self):
        """`source_changed` compares by digest because mtime and size both lie: cp -p,
        rsync --times and any restore preserve mtime, and a one-character edit preserves
        size. Storing it is what lets the watcher (sub-project E) exist at all."""
        ix = an_index()
        ix.add_files("alpha", [a_file("content = 1\n")])
        md = next(iter(ix.q.scroll_all(CHUNKS)))["payload"]["metadata"]
        self.assertTrue(md["src_digest"])
        self.assertIn("src_mtime", md)
        self.assertIn("src_size", md)


class TestTheRegistry(unittest.TestCase):
    def test_registering_makes_the_repo_listable(self):
        ix = an_index()
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        self.assertEqual([r["repo"] for r in ix.list_repos()], ["alpha"])

    def test_the_registry_is_authoritative_over_WHICH_repos_exist(self):
        """Chunks own content; the registry owns existence. A repo with chunks and no entry
        is a divergence, not a repo — and it must be visible as one."""
        ix = an_index()
        ix.add_files("ghost", [a_file("x = 1\n")])
        self.assertEqual(ix.list_repos(), [])
        self.assertIsNone(ix.get_repo("ghost"))

    def test_registering_the_same_repo_twice_accumulates_checkouts_without_duplicating(self):
        ix = an_index()
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha-2")
        ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        entry = ix.get_repo("alpha")
        self.assertEqual(sorted(entry["checkouts"]), ["/home/me/alpha", "/home/me/alpha-2"])
        self.assertEqual(len(ix.list_repos()), 1)

    def test_the_registry_never_lands_in_the_chunk_collection(self):
        ix = an_index()
        ix.register("alpha", "Alpha", [], "/home/me/alpha")
        self.assertEqual(list(ix.q.scroll_all(CHUNKS)), [])

    def test_an_empty_repo_name_is_refused(self):
        ix = an_index()
        with self.assertRaises(RepoError):
            ix.register("", "Alpha", [], "/home/me/alpha")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_repos 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.repos'`

- [ ] **Step 3: Implemente `core/repos.py`**

```python
"""Repository archive: code chunks grouped by repo, with a registry of what exists.

WHY A THIRD ARCHIVE. `core/docs.py` keeps documents out of the memory collection because a
file chunk competing with a curated fact wins on volume and drowns it. One level down, the
same argument: the library is hand-picked reference material, and a repository is tens of
thousands of automatic chunks. It gets its own collection or it drowns the library.

WHY A REGISTRY, AND WHY IN ITS OWN COLLECTION. `ports.VectorStore` has no facet, distinct or
count, so answering "which repos do I have" from the chunks would scroll every point in the
archive. The registry answers it from a handful of rows. It sits in a separate collection so
that no search has to filter registry rows out: a filter forgotten once turns a registry row
into a search hit, and a guard that only exists by vigilance gets deleted by a cleanup that
passes CI.

WHO OWNS WHAT. The registry is authoritative over WHICH REPOS EXIST, their labels and their
checkouts. The chunks are authoritative over CONTENT, and the `repo` on a chunk is derived
from the registry, never the source of it. Chunks without an entry are a DIVERGENCE, and
`list_repos` deliberately does not invent an entry for them — a repo you cannot list is a
repo you cannot drop, and inventing one would hide that.

WHAT THIS MODULE DOES NOT DO. It never discovers files. `add_files` indexes exactly the paths
it is handed; walking a repository, honouring .gitignore and skipping minified bundles belong
to the bulk pipeline, which is a separate concern with a separate failure mode (scale).
"""
import os
import time

from . import ports
from .chunk import chunk_text, mode_for_suffix
from .docs import (CoreError, _iso, _point_id, _read_source, content_digest,
                   doc_id_for)


class RepoError(CoreError):
    """Something about a repository archive operation could not be done."""


#: The registry stores no meaning in its vector — it is a key-value table that happens to
#: live in Qdrant, read by scroll and never by similarity. Size 1 says so out loud, and a
#: unit vector avoids the zero-norm that Cosine has no answer for.
REGISTRY_VECTOR_SIZE = 1
REGISTRY_VECTOR = [1.0]


class RepoIndex:
    def __init__(self, qdrant: ports.VectorStore, embedder: ports.EmbeddingModel,
                 repos_collection: str, registry_collection: str, vector_size: int):
        self.q = qdrant
        self.embedder = embedder
        self.chunks_name = repos_collection
        self.registry_name = registry_collection
        self.vector_size = vector_size

    # ---- collections -------------------------------------------------------

    def ensure(self) -> None:
        if self.q.ensure_collection(self.chunks_name, self.vector_size):
            self.q.ensure_payload_index(self.chunks_name, "repo", "keyword")
            self.q.ensure_payload_index(self.chunks_name, "doc_id", "keyword")
        if self.q.ensure_collection(self.registry_name, REGISTRY_VECTOR_SIZE):
            self.q.ensure_payload_index(self.registry_name, "repo", "keyword")

    # ---- writing -----------------------------------------------------------

    def add_files(self, repo: str, paths: list[str]) -> dict:
        """Indexes exactly `paths` under `repo`. Never raises for one bad file.

        A list of eight hundred paths with one empty file in it must index the other 799:
        aborting the batch for an unindexable member would make the bulk pipeline's job
        impossible, so the failure is REPORTED per path instead.
        """
        if not repo:
            raise RepoError("a repository name is required")
        self.ensure()
        files = chunks = 0
        skipped: list = []
        for path in paths:
            try:
                added = self._write_one(repo, path)
            except (OSError, RepoError, ValueError) as exc:
                skipped.append((path, str(exc)))
                continue
            files += 1
            chunks += added

        return {"repo": repo, "files": files, "chunks": chunks, "skipped": skipped}

    def _write_one(self, repo: str, path: str) -> int:
        path, st, content = _read_source(path)
        pieces = chunk_text(content)
        if not pieces:
            raise RepoError("nothing indexable (empty file, or whitespace only)")
        doc_id = doc_id_for(path)
        digest = content_digest(content)
        mode = mode_for_suffix(os.path.splitext(path)[1])
        now = time.time()

        # Reindexing REPLACES: without this the old version and the new one coexist and one
        # search mixes chunks from two states of the same file.
        self.q.delete_by_filter(self.chunks_name,
                                {"must": [{"key": "doc_id", "match": {"value": doc_id}}]})

        vectors = self.embedder.embed([p.text for p in pieces])
        points = []
        for ix, (piece, vector) in enumerate(zip(pieces, vectors)):
            points.append({
                "id": _point_id(doc_id, ix),
                "vector": vector,
                "payload": {
                    "document": piece.text,
                    "doc_id": doc_id,
                    # Top level, like doc_id: the payload index and group_by address it.
                    "repo": repo,
                    "metadata": {
                        "path": path, "start_line": piece.start_line,
                        "end_line": piece.end_line, "mode": mode,
                        "chunk_ix": ix, "n_chunks": len(pieces),
                        "indexed_at": _iso(now),
                        "src_mtime": st.st_mtime, "src_size": st.st_size,
                        "src_digest": digest,
                    },
                },
            })
        self.q.upsert(self.chunks_name, points)

        return len(points)

    # ---- registry ----------------------------------------------------------

    def register(self, repo: str, label: str, remotes: list[str], checkout: str) -> dict:
        """Creates or updates the entry. Checkouts and remotes ACCUMULATE without repeating:
        the same repository legitimately lives in several working copies at once."""
        if not repo:
            raise RepoError("a repository name is required")
        self.ensure()
        entry = self.get_repo(repo) or {"repo": repo, "label": label or repo,
                                        "remotes": [], "checkouts": []}
        entry["label"] = label or entry.get("label") or repo
        for value, key in ((checkout, "checkouts"), *[(r, "remotes") for r in remotes or []]):
            if value and value not in entry[key]:
                entry[key].append(value)
        entry["indexed_at"] = _iso(time.time())
        self.q.upsert(self.registry_name, [{"id": _point_id(f"registry:{repo}", 0),
                                            "vector": list(REGISTRY_VECTOR),
                                            "payload": entry}])

        return entry

    def get_repo(self, repo: str) -> dict | None:
        point = self.q.get_point(self.registry_name, _point_id(f"registry:{repo}", 0))

        return (point or {}).get("payload") if point else None

    def list_repos(self) -> list[dict]:
        """Every registered repo, from the REGISTRY and never from the chunks.

        Deriving this from the chunks would mean scrolling the whole archive, and it would
        also invent entries for divergent chunks — hiding a repo that cannot be dropped.
        """
        try:
            rows = [p.get("payload") or {} for p in self.q.scroll_all(self.registry_name)]
        except Exception as exc:                       # noqa: BLE001
            raise RepoError(f"the repository registry could not be read: {exc}") from exc

        return sorted((r for r in rows if r.get("repo")), key=lambda r: r["repo"])
```

- [ ] **Step 4: Exporte no pacote**

Em `core/__init__.py`, junto dos exports de `docs`:

```python
from .repos import RepoError, RepoIndex
```
e acrescente `"RepoError", "RepoIndex",` ao `__all__`.

- [ ] **Step 5: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_repos 2>&1 | tail -3`
Expected: `Ran 11 tests`, `OK`

- [ ] **Step 6: Prove que a substituição na reindexação morde**

```bash
cp core/repos.py /tmp/rp.bak
python3 - <<'PY'
p = "core/repos.py"; s = open(p).read()
old = """        self.q.delete_by_filter(self.chunks_name,
                                {"must": [{"key": "doc_id", "match": {"value": doc_id}}]})
"""
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, ""))
print("mutation landed: reindexing now accumulates")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_repos 2>&1 | tail -3
cp /tmp/rp.bak core/repos.py && rm /tmp/rp.bak
```
Expected: `FAILED` nomeando `test_reindexing_the_same_file_replaces_instead_of_accumulating`.
Registre a contagem COM o escopo.

- [ ] **Step 7: Prove que o registro é o dono da existência**

```bash
cp core/repos.py /tmp/rp.bak
python3 - <<'PY'
p = "core/repos.py"; s = open(p).read()
old = '        return sorted((r for r in rows if r.get("repo")), key=lambda r: r["repo"])'
assert s.count(old) == 1
new = ('        seen = {(p.get("payload") or {}).get("repo")\n'
       '                for p in self.q.scroll_all(self.chunks_name)}\n'
       '        rows = rows + [{"repo": r} for r in seen if r]\n'
       '        return sorted((r for r in rows if r.get("repo")), key=lambda r: r["repo"])')
open(p, "w").write(s.replace(old, new))
print("mutation landed: list_repos now invents entries from chunks")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_repos 2>&1 | tail -3
cp /tmp/rp.bak core/repos.py && rm /tmp/rp.bak
```
Expected: `FAILED` nomeando `test_the_registry_is_authoritative_over_WHICH_repos_exist`.

- [ ] **Step 8: Rode a suíte inteira e commite**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK`

```bash
git add core/repos.py core/__init__.py tests/test_repos.py
git commit -F - <<'MSG'
feat: a repository archive, and a registry that owns which repos exist

A third archive, for the reason documents already live apart from memory: volume drowns
curation. The library is hand-picked reference material; a repository is tens of thousands of
automatic chunks.

The registry exists because ports.VectorStore has no facet, distinct or count — answering
"which repos do I have" from the chunks would scroll the whole archive. It is authoritative
over EXISTENCE, labels and checkouts; the chunks are authoritative over CONTENT, and the
repo on a chunk is derived from the registry rather than a second source of it. list_repos
refuses to invent an entry for chunks that have none, because a repo you cannot list is a
repo you cannot drop, and inventing one hides the divergence instead of showing it.

add_files indexes exactly the paths it is handed and never discovers any: walking a
repository is a different concern with a different failure mode. One unindexable file is
reported, not raised — a list of eight hundred paths must index the other 799.

`repo` sits at the top of the payload, like doc_id, because the payload index and group_by
address a top-level key. Burying it under metadata would read fine and break both.
MSG
```

---

### Task 4: `core/bindings.py` — qual repo é este checkout, e quem decide

**Files:**
- Create: `core/bindings.py`
- Modify: `core/knobs.py` (recebe `state_dir`)
- Modify: `core/inventory.py:62` (passa a importar `state_dir` em vez de defini-lo)
- Modify: `core/repos.py` (`candidates_for`)
- Test: `tests/test_bindings.py`

**`state_dir()` JÁ TEM DONO, e ele está na camada errada.** `core/inventory.py:62` o define — mas
`inventory` é o módulo que fala com o Qdrant, importado TARDIAMENTE no caminho raro da guarda
justamente para o caminho comum não pagar o import. Um helper que só lê uma variável de ambiente
e monta um caminho não pertence a um módulo de rede.

`Ruling: MOVER `state_dir` para `core/knobs.py` (puro, eager, já dono da leitura de ambiente) e
fazer `inventory` importá-lo de lá. `bindings` também importa de lá. — Motivo: a alternativa é
`bindings` definir o seu, e aí seriam TRÊS donos do mesmo caminho (inventory, bindings, e os dois
`STATE_DIR` dos hooks) — a ruling F5 desta base, um dono por invariante, e desta vez seria eu
introduzindo. Mover em vez de copiar também tira um helper puro de dentro de um módulo de rede.
— Custo se errado: um import a mais em `inventory`, e o `state_dir` fica num módulo que já é
importado por todo mundo de qualquer forma.`

**NÃO** unifique os dois `STATE_DIR` de `hooks/recall.py:111` e `hooks/checkpoint.py:56`. É
duplicação pré-existente que este trabalho não toca, e mexer nela é refatoração não relacionada.

**Interfaces:**
- Consumes: `RepoIndex.list_repos()` (Task 3).
- Produces:
  - `knobs.state_dir() -> Path` (MOVIDO de `core/inventory.py`, honra `QCTX_STATE_DIR`)
  - `bindings.get(checkout: str) -> str | None`
  - `bindings.bind(checkout: str, repo: str) -> None`
  - `bindings.forget_repo(repo: str) -> list[str]` — devolve os checkouts desvinculados
  - `bindings.git_root(path: str) -> str | None`
  - `bindings.remotes_of(root: str) -> list[str]`
  - `bindings.normalize_remote(url: str) -> str`
  - `bindings.slug_for(name: str) -> str`
  - `RepoIndex.candidates_for(root: str, remotes: list[str]) -> dict` — devolve
    `{"bound": str | None, "join": list[dict], "suggest": str}`

**Identidade é DECLARADA, e este módulo só prepara a escolha.** Quem PERGUNTA é o sub-projeto C.
A decisão do usuário (2026-08-17) foi por escolher na indexação porque ele mantém clones
paralelos do mesmo remote de propósito — identidade por remote fundiria checkouts que podem
estar em branches diferentes.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_bindings.py
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bindings  # noqa: E402
from core.repos import RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


def a_git_repo(remote: str | None = None) -> str:
    root = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    if remote:
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=root, check=True)

    return root


class TestTheLocalBinding(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_an_unbound_checkout_reads_as_None(self):
        self.assertIsNone(bindings.get("/home/me/never-seen"))

    def test_binding_survives_a_fresh_read(self):
        bindings.bind("/home/me/alpha", "alpha")
        self.assertEqual(bindings.get("/home/me/alpha"), "alpha")

    def test_two_checkouts_may_bind_to_the_same_repo(self):
        bindings.bind("/home/me/alpha", "alpha")
        bindings.bind("/home/me/alpha-2", "alpha")
        self.assertEqual(bindings.get("/home/me/alpha-2"), "alpha")

    def test_forgetting_a_repo_unbinds_every_checkout_of_it_and_names_them(self):
        """Dropping a repo must invalidate the bindings, or a checkout keeps claiming to
        belong to a repo that no longer exists and the next index writes to a phantom."""
        bindings.bind("/home/me/alpha", "alpha")
        bindings.bind("/home/me/alpha-2", "alpha")
        bindings.bind("/home/me/beta", "beta")
        freed = bindings.forget_repo("alpha")
        self.assertEqual(sorted(freed), ["/home/me/alpha", "/home/me/alpha-2"])
        self.assertIsNone(bindings.get("/home/me/alpha"))
        self.assertEqual(bindings.get("/home/me/beta"), "beta")

    def test_a_corrupt_state_file_reads_as_no_bindings_instead_of_raising(self):
        with open(os.path.join(bindings.state_dir(), bindings.FILENAME), "w") as fh:
            fh.write("{not json")
        self.assertIsNone(bindings.get("/home/me/alpha"))


class TestGitFacts(unittest.TestCase):
    def test_the_root_of_a_subdirectory_is_the_repo_root(self):
        root = a_git_repo()
        deep = os.path.join(root, "a", "b")
        os.makedirs(deep)
        self.assertEqual(os.path.realpath(bindings.git_root(deep)), os.path.realpath(root))

    def test_a_plain_directory_has_no_root(self):
        self.assertIsNone(bindings.git_root(tempfile.mkdtemp()))

    def test_remotes_are_read_and_a_repo_without_one_is_fine(self):
        self.assertEqual(bindings.remotes_of(a_git_repo()), [])
        self.assertEqual(bindings.remotes_of(a_git_repo("git@host:me/alpha.git")),
                         ["git@host:me/alpha.git"])

    def test_the_same_repo_over_ssh_and_https_normalizes_alike(self):
        """Otherwise the join offer misses, and the user is asked to name a repo that is
        already registered under the other URL form."""
        self.assertEqual(bindings.normalize_remote("git@github.com:me/alpha.git"),
                         bindings.normalize_remote("https://github.com/me/alpha"))

    def test_the_slug_is_stable_and_filesystem_shaped(self):
        self.assertEqual(bindings.slug_for("My Repo!"), "my-repo")
        self.assertEqual(bindings.slug_for("awesome-cv3"), "awesome-cv3")


class TestTheChoiceOffered(unittest.TestCase):
    def setUp(self):
        a_state_dir()
        self.ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)

    def test_a_matching_remote_is_offered_to_JOIN(self):
        self.ix.register("alpha", "Alpha", ["git@github.com:me/alpha.git"], "/home/me/alpha")
        out = self.ix.candidates_for("/home/me/alpha-2", ["https://github.com/me/alpha"])
        self.assertEqual([c["repo"] for c in out["join"]], ["alpha"])

    def test_with_no_match_the_suggestion_is_the_directory_name(self):
        out = self.ix.candidates_for("/home/me/brand-new", [])
        self.assertEqual(out["join"], [])
        self.assertEqual(out["suggest"], "brand-new")

    def test_an_already_bound_checkout_reports_its_binding_and_asks_nothing(self):
        self.ix.register("alpha", "Alpha", [], "/home/me/alpha")
        bindings.bind("/home/me/alpha", "alpha")
        self.assertEqual(self.ix.candidates_for("/home/me/alpha", [])["bound"], "alpha")

    def test_a_binding_to_a_repo_that_no_longer_exists_is_NOT_reported_as_bound(self):
        """The healing path. A stale binding must behave as unbound, or the checkout writes
        into a repo the registry does not know."""
        bindings.bind("/home/me/alpha", "deleted-repo")
        self.assertIsNone(self.ix.candidates_for("/home/me/alpha", [])["bound"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_bindings 2>&1 | tail -4`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.bindings'`

- [ ] **Step 3: Mova `state_dir` para a camada certa**

Corte a função de `core/inventory.py:62` e cole em `core/knobs.py`, acrescentando ao docstring
por que ela mora lá:

```python
def state_dir() -> Path:
    """Where this plugin keeps state, honouring QCTX_STATE_DIR.

    It lives here and not beside the code that first needed it: reading one env var and
    building a path is exactly what this module is for, and its previous home talks to Qdrant
    and is imported LAZILY so the common path does not pay for it. A pure helper inside a
    network module forces every caller to choose between an unwanted import and a copy.
    """
    return Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))
```

Em `core/inventory.py`, troque a definição por `from .knobs import state_dir` (mantendo o uso
inalterado) e confirme `from pathlib import Path` no `core/knobs.py`.

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK`, 737 — o move não muda comportamento, e se algum teste cair é porque o import
ficou faltando em algum lugar.

- [ ] **Step 4: Implemente `core/bindings.py`**

```python
"""Which repository is THIS working copy, locally and without asking twice.

Identity here is DECLARED, not derived, and that is a user decision from 2026-08-17: he keeps
parallel clones of the same remote on purpose, and worktrees. Deriving identity from the
remote would merge working copies that can sit on different branches, so the plugin OFFERS a
choice and remembers the answer. This module prepares and remembers; the host adapter asks.

The binding is keyed by absolute path, so MOVING a checkout loses it. That is not a leak: the
next detection asks again, the remote match makes "join the existing repo" the default, and
the archive is reached again instead of orphaned. It heals.

State lives beside the other plugin state so both hosts share it, and a corrupt file reads as
"no bindings" rather than raising — a broken cache must not make the tool unusable.
"""
import json
import os
import re
import subprocess

from .knobs import state_dir

FILENAME = "repo-bindings.json"


def _path() -> str:
    """The binding file, under the ONE state directory this plugin has.

    `state_dir` is imported and not redefined: it already had an owner, and a third copy of
    "where does state live" is how the three of them start disagreeing.
    """
    root = state_dir()
    os.makedirs(root, exist_ok=True)

    return os.path.join(root, FILENAME)


def _load() -> dict:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    tmp = _path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, _path())


def get(checkout: str) -> str | None:
    return _load().get(os.path.realpath(checkout)) or None


def bind(checkout: str, repo: str) -> None:
    data = _load()
    data[os.path.realpath(checkout)] = repo
    _save(data)


def forget_repo(repo: str) -> list[str]:
    """Unbinds every checkout of `repo` and names them, so the caller can say what changed."""
    data = _load()
    freed = [path for path, name in data.items() if name == repo]
    for path in freed:
        del data[path]
    if freed:
        _save(data)

    return freed


def git_root(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None

    return out.stdout.strip() or None if out.returncode == 0 else None


def remotes_of(root: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", root, "remote", "-v"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    urls = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in urls:
            urls.append(parts[1])

    return urls


def normalize_remote(url: str) -> str:
    """`git@github.com:me/alpha.git` and `https://github.com/me/alpha` are the same repo.

    Without this the join offer misses and the user is asked to name a repository that is
    already registered under the other URL form.
    """
    u = (url or "").strip()
    u = re.sub(r"^[a-z+]+://", "", u)
    u = re.sub(r"^[^@/]+@", "", u)
    u = u.replace(":", "/", 1) if "/" not in u.split(":", 1)[0] else u
    u = re.sub(r"\.git$", "", u)

    return u.strip("/").lower()


def slug_for(name: str) -> str:
    """A stable id from a human name: the FILTER KEY, so it may never change afterwards."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())

    return s.strip("-") or "repo"
```

- [ ] **Step 5: Acrescente `candidates_for` ao `RepoIndex`**

Em `core/repos.py`, depois de `list_repos`:

```python
    def candidates_for(self, root: str, remotes: list[str]) -> dict:
        """The choice to offer for this working copy. Presenting it is the host's job.

        `bound` is set only when the binding points at a repo the registry still knows: a
        stale binding must behave as unbound, or this checkout writes into a phantom repo.
        """
        from . import bindings

        known = self.list_repos()
        by_name = {r["repo"] for r in known}
        bound = bindings.get(root)
        if bound and bound not in by_name:
            bound = None
        wanted = {bindings.normalize_remote(r) for r in remotes or [] if r}
        join = [r for r in known
                if wanted & {bindings.normalize_remote(x) for x in r.get("remotes") or []}]

        return {"bound": bound, "join": join,
                "suggest": bindings.slug_for(os.path.basename(os.path.realpath(root)))}
```

- [ ] **Step 6: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_bindings 2>&1 | tail -3`
Expected: `Ran 13 tests`, `OK`

- [ ] **Step 7: Prove que as duas guardas mordem, uma por vez**

```bash
cp core/repos.py /tmp/rp.bak
python3 - <<'PY'
p = "core/repos.py"; s = open(p).read()
old = "        if bound and bound not in by_name:\n            bound = None\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, ""))
print("mutation 1: a stale binding is now trusted")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_bindings 2>&1 | tail -3
cp /tmp/rp.bak core/repos.py

cp core/bindings.py /tmp/bd.bak
python3 - <<'PY'
p = "core/bindings.py"; s = open(p).read()
old = "    u = re.sub(r\"\\.git$\", \"\", u)\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, ""))
print("mutation 2: the .git suffix is no longer normalized away")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_bindings 2>&1 | tail -3
cp /tmp/bd.bak core/bindings.py && rm /tmp/bd.bak /tmp/rp.bak
```
Expected: a primeira nomeia `test_a_binding_to_a_repo_that_no_longer_exists...`; a segunda
nomeia `test_the_same_repo_over_ssh_and_https_normalizes_alike`. Registre as duas COM o escopo.

- [ ] **Step 8: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/bindings.py core/repos.py tests/test_bindings.py
git commit -F - <<'MSG'
feat: which repository this working copy is, remembered so nothing asks twice

Identity is declared, not derived, and that is a measured user decision: he keeps parallel
clones of the same remote on purpose, and worktrees. Deriving identity from the remote would
merge working copies that can sit on different branches.

The binding is keyed by absolute path, so moving a checkout loses it. That is the healing
path, not a leak: detection asks again, the remote match makes "join the existing repo" the
default, and the archive is reached instead of orphaned. A binding pointing at a repo the
registry no longer knows behaves as UNBOUND, because trusting it would write into a phantom.

Remote normalization exists so ssh and https forms of the same repository match; without it
the join offer misses and the user is asked to name something already registered under the
other URL form.

A corrupt state file reads as "no bindings" instead of raising: a broken cache must not make
the tool unusable.
MSG
```

---

### Task 5: busca — escopada por padrão, ampla a um passo

**Files:**
- Modify: `core/repos.py` (`search`, `RepoHit`)
- Test: `tests/test_repos_search.py`

**Interfaces:**
- Consumes: `VectorStore.search_groups` (Task 1); `RepoIndex` (Task 3); `bindings` (Task 4).
- Produces:
  - `RepoIndex.search(query: str, repo: str | None = None, across: bool = False,
    limit: int = 8, group_size: int = 3) -> dict` devolvendo
    `{"scope": "repo" | "across", "repo": str | None, "groups": [{"repo": str,
    "hits": [RepoHit, ...]}, ...], "truncated": bool}`
  - `dataclass RepoHit(score, repo, path, start_line, end_line, mode, text, indexed_at,
    stale)`

**Regra que não se negocia:** falha aqui NÃO abre. Busca que não pode consultar levanta
`RepoError`; devolver lista vazia seria indistinguível de "não há nada", e o modelo concluiria
ausência a partir de infraestrutura fora do ar.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_repos_search.py
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402


def a_file(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def an_index_with_two_repos() -> RepoIndex:
    ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
    ix.register("loud", "Loud", [], "/tmp/loud")
    ix.register("quiet", "Quiet", [], "/tmp/quiet")
    ix.add_files("loud", [a_file("billing invoice charge total\n" * 40)])
    ix.add_files("quiet", [a_file("invoice\n")])

    return ix


class TestScopedSearch(unittest.TestCase):
    def test_a_scoped_search_returns_only_that_repo(self):
        out = an_index_with_two_repos().search("billing invoice", repo="quiet")
        self.assertEqual(out["scope"], "repo")
        self.assertEqual({g["repo"] for g in out["groups"]}, {"quiet"})

    def test_an_unknown_repo_is_an_error_and_not_an_empty_result(self):
        """Empty would read as "this repo has nothing about it", which is a different and
        false statement."""
        with self.assertRaises(RepoError):
            an_index_with_two_repos().search("billing", repo="no-such-repo")

    def test_a_hit_carries_the_location_and_not_only_the_text(self):
        out = an_index_with_two_repos().search("invoice", repo="quiet")
        hit = out["groups"][0]["hits"][0]
        self.assertTrue(hit.path)
        self.assertGreaterEqual(hit.start_line, 1)

    def test_a_changed_file_is_reported_stale_instead_of_answered_as_current(self):
        ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
        ix.register("alpha", "Alpha", [], "/tmp/alpha")
        path = a_file("original invoice content\n")
        ix.add_files("alpha", [path])
        with open(path, "w") as fh:
            fh.write("something else entirely\n")
        hit = ix.search("invoice", repo="alpha")["groups"][0]["hits"][0]
        self.assertTrue(hit.stale)


class TestTheTraversal(unittest.TestCase):
    def test_across_returns_the_quiet_repo_too(self):
        """The whole point of the feature. A repo with one mention must not be shadowed by a
        repo with forty."""
        out = an_index_with_two_repos().search("billing invoice charge total", across=True)
        self.assertEqual(out["scope"], "across")
        self.assertEqual({g["repo"] for g in out["groups"]}, {"loud", "quiet"})

    def test_it_asks_for_as_many_groups_as_the_registry_knows(self):
        """The registry is what makes this number a fact instead of a guess."""
        ix = an_index_with_two_repos()
        seen = {}
        real = ix.q.search_groups

        def spy(name, vector, group_by, limit, group_size, **kw):
            seen["limit"] = limit

            return real(name, vector, group_by, limit, group_size, **kw)

        ix.q.search_groups = spy
        ix.search("invoice", across=True)
        self.assertEqual(seen["limit"], len(ix.list_repos()))

    def test_across_with_no_repos_registered_is_an_error(self):
        ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
        with self.assertRaises(RepoError):
            ix.search("anything", across=True)


class TestFailureDoesNotOpen(unittest.TestCase):
    def test_an_unreachable_store_raises_instead_of_returning_nothing(self):
        """The inverse of the big-file guard, on purpose: there, blocking on a doubtful
        number kept the user from reading, so failure had to allow. Here an empty result is
        indistinguishable from "there is nothing", so failure must SAY so."""
        ix = an_index_with_two_repos()

        def boom(*a, **kw):
            raise OSError("connection refused")

        ix.q.search_groups = boom
        with self.assertRaises(RepoError):
            ix.search("invoice", repo="quiet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_repos_search 2>&1 | tail -4`
Expected: FAIL — `AttributeError: 'RepoIndex' object has no attribute 'search'`

- [ ] **Step 3: Implemente a busca**

No topo de `core/repos.py`, acrescente aos imports existentes:

```python
from dataclasses import dataclass

from .docs import source_changed
```

E o dataclass, depois de `REGISTRY_VECTOR`:

```python
@dataclass
class RepoHit:
    score: float
    repo: str
    path: str
    start_line: int
    end_line: int
    mode: str
    text: str
    indexed_at: str
    stale: str | None    # the reason, when the file changed since it was indexed
```

E o método, depois de `candidates_for`:

```python
    def search(self, query: str, repo: str | None = None, across: bool = False,
               limit: int = 8, group_size: int = 3) -> dict:
        """Grouped hits: one repo by default, every repo when `across`.

        FAILURE DOES NOT OPEN HERE, and that is the inverse of the read guard on purpose. A
        search that cannot reach the archive and returns [] is indistinguishable from "there
        is nothing about this", and a caller would conclude absence from an outage. It raises.

        `across` asks for as many groups as the registry knows, which is exactly why the
        registry exists: without it, how many groups to ask for would be a guess. It remains
        best-effort over what the search reaches — never a proof of absence.
        """
        known = {r["repo"] for r in self.list_repos()}
        if across:
            if not known:
                raise RepoError("no repository is indexed yet")
            group_limit, filter_ = len(known), None
        else:
            if not repo:
                raise RepoError("name a repository, or pass across=True")
            if repo not in known:
                raise RepoError(f"repository {repo!r} is not indexed")
            group_limit = 1
            filter_ = {"must": [{"key": "repo", "match": {"value": repo}}]}

        vector = self.embedder.embed_one(query)
        try:
            raw = self.q.search_groups(self.chunks_name, vector, group_by="repo",
                                       limit=group_limit, group_size=group_size,
                                       filter_=filter_)
        except RepoError:
            raise
        except Exception as exc:                        # noqa: BLE001
            raise RepoError(f"the repository archive could not be searched: {exc}") from exc

        groups = []
        for group in raw[:limit]:
            hits = [self._to_hit(h) for h in group.get("hits", [])]
            if hits:
                groups.append({"repo": group.get("id"), "hits": hits})

        return {"scope": "across" if across else "repo", "repo": None if across else repo,
                "groups": groups, "truncated": len(raw) > limit}

    def _to_hit(self, raw: dict) -> RepoHit:
        payload = raw.get("payload") or {}
        md = payload.get("metadata") or {}
        path = md.get("path", "?")

        return RepoHit(
            score=float(raw.get("score") or 0.0),
            repo=payload.get("repo", "?"),
            path=path,
            start_line=int(md.get("start_line") or 0),
            end_line=int(md.get("end_line") or 0),
            mode=md.get("mode", "snapshot"),
            text=payload.get("document", ""),
            indexed_at=md.get("indexed_at", "?"),
            # Reported, never hidden: a chunk from an older state of the file must degrade to
            # "this changed" instead of answering as if it were current.
            stale=source_changed(path, md.get("src_mtime"), md.get("src_size"),
                                 md.get("src_digest")),
        )
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_repos_search 2>&1 | tail -3`
Expected: `Ran 9 tests`, `OK`

- [ ] **Step 5: Prove que falha-não-abre morde, e que o `limit` vem do registro**

```bash
cp core/repos.py /tmp/rp.bak
python3 - <<'PY'
p = "core/repos.py"; s = open(p).read()
old = """        except Exception as exc:                        # noqa: BLE001
            raise RepoError(f"the repository archive could not be searched: {exc}") from exc
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "        except Exception:                              # noqa: BLE001\n            raw = []\n"))
print("mutation 1: an unreachable store now returns nothing at all")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_repos_search 2>&1 | tail -3
cp /tmp/rp.bak core/repos.py

python3 - <<'PY'
p = "core/repos.py"; s = open(p).read()
old = "            group_limit, filter_ = len(known), None\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "            group_limit, filter_ = 10, None\n"))
print("mutation 2: the group limit is a hardcoded guess")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_repos_search 2>&1 | tail -3
cp /tmp/rp.bak core/repos.py && rm /tmp/rp.bak
```
Expected: a primeira nomeia `test_an_unreachable_store_raises_instead_of_returning_nothing`;
a segunda nomeia `test_it_asks_for_as_many_groups_as_the_registry_knows`.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/repos.py tests/test_repos_search.py
git commit -F - <<'MSG'
feat: search one repo by default, and every repo one step away

Scoped is the default because the answer is almost always in the project you are standing in,
and a broad search there returns handleWebhook from five connectors. Across is one flag away.

Failure does NOT open here, and that is the inverse of the big-file read guard on purpose.
There, blocking on a number we were unsure of kept the user from reading files, so failure had
to allow. Here a search that cannot reach the archive and returns [] is indistinguishable from
"there is nothing about this" — the caller would conclude absence from an outage. It raises.

Across asks for exactly as many groups as the registry knows, which is the registry's second
keep: without it that number is a guess. It stays best-effort over what the search reaches and
never a proof of absence.

A hit carries the location, and a chunk whose file changed since indexing is reported stale
rather than answered as if it were current.
MSG
```

---

### Task 6: listar e apagar, na ordem que não corrompe

**Files:**
- Modify: `core/repos.py` (`drop_repo`, `divergent_repos`)
- Test: `tests/test_repos_drop.py`

**Interfaces:**
- Consumes: `RepoIndex` (Task 3), `bindings.forget_repo` (Task 4).
- Produces:
  - `RepoIndex.drop_repo(repo: str) -> dict` devolvendo
    `{"repo": str, "unbound": list[str]}`
  - `RepoIndex.divergent_repos() -> list[str]` — repos com chunks e sem item no registro

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_repos_drop.py
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import bindings  # noqa: E402
from core.repos import RepoError, RepoIndex  # noqa: E402
from tests.fakes import FakeEmbedder, FakeVectorStore  # noqa: E402


def a_file(text: str = "content = 1\n") -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)

    return path


def a_populated_index() -> RepoIndex:
    os.environ["QCTX_STATE_DIR"] = tempfile.mkdtemp()
    ix = RepoIndex(FakeVectorStore(), FakeEmbedder(dim=8), "c", "r", 8)
    for name in ("alpha", "beta"):
        ix.register(name, name.title(), [], f"/tmp/{name}")
        ix.add_files(name, [a_file()])
        bindings.bind(f"/tmp/{name}", name)

    return ix


class TestDropping(unittest.TestCase):
    def test_it_removes_the_chunks_the_entry_and_the_bindings(self):
        ix = a_populated_index()
        out = ix.drop_repo("alpha")
        self.assertEqual(out["unbound"], ["/tmp/alpha"])
        self.assertIsNone(ix.get_repo("alpha"))
        self.assertEqual({(p["payload"]["repo"]) for p in ix.q.scroll_all("c")}, {"beta"})
        self.assertEqual(bindings.get("/tmp/alpha"), None)

    def test_it_leaves_the_other_repos_alone(self):
        ix = a_populated_index()
        ix.drop_repo("alpha")
        self.assertEqual([r["repo"] for r in ix.list_repos()], ["beta"])
        self.assertEqual(bindings.get("/tmp/beta"), "beta")

    def test_dropping_an_unknown_repo_is_an_error(self):
        with self.assertRaises(RepoError):
            a_populated_index().drop_repo("no-such-repo")

    def test_the_chunks_go_FIRST_so_a_failure_leaves_a_visible_remainder(self):
        """Order is load-bearing. Registry first with the chunks failing leaves chunks that no
        listing can reach and that still compete in an across search — unreachable garbage.
        Chunks first with the registry failing leaves an entry pointing at zero chunks:
        visible, and a second run finishes the job."""
        ix = a_populated_index()
        calls = []
        real_delete = ix.q.delete_by_filter

        def spy_delete(name, filter_):
            calls.append(("delete", name))

            return real_delete(name, filter_)

        real_points = ix.q.delete_points

        def spy_points(name, ids):
            calls.append(("delete_points", name))

            return real_points(name, ids)

        ix.q.delete_by_filter, ix.q.delete_points = spy_delete, spy_points
        ix.drop_repo("alpha")
        touched = [name for _, name in calls]
        self.assertEqual(touched.index("c"), 0, f"chunks were not touched first: {calls}")

    def test_a_failure_deleting_chunks_does_NOT_remove_the_entry(self):
        """Otherwise the second run has nothing to finish from."""
        ix = a_populated_index()

        def boom(*a, **kw):
            raise OSError("connection refused")

        ix.q.delete_by_filter = boom
        with self.assertRaises(RepoError):
            ix.drop_repo("alpha")
        self.assertIsNotNone(ix.get_repo("alpha"))


class TestDivergence(unittest.TestCase):
    def test_chunks_without_a_registry_entry_are_reported(self):
        """The price of honesty for having two sources of truth about which repos exist. The
        registry is authoritative; this is how the copy is caught diverging."""
        ix = a_populated_index()
        ix.add_files("ghost", [a_file()])
        self.assertEqual(ix.divergent_repos(), ["ghost"])

    def test_a_healthy_archive_reports_no_divergence(self):
        self.assertEqual(a_populated_index().divergent_repos(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_repos_drop 2>&1 | tail -4`
Expected: FAIL — `AttributeError: 'RepoIndex' object has no attribute 'drop_repo'`

- [ ] **Step 3: Implemente**

Em `core/repos.py`, depois de `_to_hit`:

```python
    def drop_repo(self, repo: str) -> dict:
        """Deletes a repository: chunks, then the entry, then the local bindings.

        THE ORDER IS LOAD-BEARING. Registry first with the chunks failing leaves chunks no
        listing can reach, which still compete in an `across` search — unreachable garbage.
        Chunks first with the registry failing leaves an entry pointing at zero chunks:
        visible, and a second run finishes the job. Prefer the recoverable remainder.
        """
        from . import bindings

        if not self.get_repo(repo):
            raise RepoError(f"repository {repo!r} is not indexed")
        try:
            self.q.delete_by_filter(self.chunks_name,
                                    {"must": [{"key": "repo", "match": {"value": repo}}]})
        except Exception as exc:                        # noqa: BLE001
            raise RepoError(f"the chunks of {repo!r} could not be deleted, so its entry was "
                            f"kept for a second attempt: {exc}") from exc
        self.q.delete_points(self.registry_name, [_point_id(f"registry:{repo}", 0)])
        # Last, and only once the archive agrees: a checkout must never claim to belong to a
        # repo that is gone, because the next index would write into a phantom.
        unbound = bindings.forget_repo(repo)

        return {"repo": repo, "unbound": unbound}

    def divergent_repos(self) -> list[str]:
        """Repos with chunks and no registry entry.

        This exists because the design has TWO sources of truth about which repos exist: the
        registry, which is authoritative, and the `repo` field on every chunk, which is
        derived. Naming the owner is half the defence; this is the other half — without it the
        copy is free to diverge unobserved.
        """
        known = {r["repo"] for r in self.list_repos()}
        seen = {(p.get("payload") or {}).get("repo")
                for p in self.q.scroll_all(self.chunks_name)}

        return sorted(r for r in seen if r and r not in known)
```

- [ ] **Step 4: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_repos_drop 2>&1 | tail -3`
Expected: `Ran 7 tests`, `OK`

- [ ] **Step 5: Prove que a ORDEM morde**

```bash
cp core/repos.py /tmp/rp.bak
python3 - <<'PY'
p = "core/repos.py"; s = open(p).read()
chunks = """        try:
            self.q.delete_by_filter(self.chunks_name,
                                    {"must": [{"key": "repo", "match": {"value": repo}}]})
        except Exception as exc:                        # noqa: BLE001
            raise RepoError(f"the chunks of {repo!r} could not be deleted, so its entry was "
                            f"kept for a second attempt: {exc}") from exc
"""
registry = '        self.q.delete_points(self.registry_name, [_point_id(f"registry:{repo}", 0)])\n'
assert s.count(chunks) == 1 and s.count(registry) == 1
open(p, "w").write(s.replace(chunks + registry, registry + chunks))
print("mutation landed: the registry is now deleted BEFORE the chunks")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_repos_drop 2>&1 | tail -3
cp /tmp/rp.bak core/repos.py && rm /tmp/rp.bak
```
Expected: `FAILED` nomeando `test_the_chunks_go_FIRST_so_a_failure_leaves_a_visible_remainder`
e `test_a_failure_deleting_chunks_does_NOT_remove_the_entry`.

- [ ] **Step 6: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add core/repos.py tests/test_repos_drop.py
git commit -F - <<'MSG'
feat: drop a repository in the order that leaves a recoverable remainder

Chunks first, then the entry, then the local bindings — and the order is load-bearing.
Registry first with the chunks failing leaves chunks no listing can reach and that still
compete in an across search: unreachable garbage. Chunks first with the registry failing
leaves an entry pointing at zero chunks, which is visible and which a second run finishes.
When something must break, break toward the state a human can see and a rerun can fix.

Bindings go last and only once the archive agrees, because a checkout that claims a repo that
is gone makes the next index write into a phantom.

divergent_repos is the price of honesty for the design's one duplicated fact: the registry and
the repo field on every chunk both answer "which repos exist". The registry is authoritative;
naming the owner is half the defence and this is the other half, because a copy nobody checks
is free to diverge.
MSG
```

---

### Task 7: os verbos do CLI, as ferramentas do hermes, e a equivalência

**Files:**
- Modify: `cli/qctx.py` (grupo `repos`)
- Modify: `hosts/hermes/tools.py` (as mesmas operações como ferramentas)
- Modify: `tests/test_host_equivalence.py`
- Test: `tests/test_cli_repos.py`

**Interfaces:**
- Consumes: tudo das Tasks 3 a 6.
- Produces: `qctx repos {list,search,add,drop}` e as ferramentas equivalentes no hermes.

**A equivalência é a exigência central deste plugin** ("eles devem ser equivalentes entre si —
funções e configurações"). Um verbo que exista num host e não no outro é o defeito que
`tests/test_host_equivalence.py` existe para pegar.

- [ ] **Step 1: Escreva o teste que falha**

```python
# tests/test_cli_repos.py
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*args, env_extra=None):
    env = dict(os.environ)
    env["QCTX_STATE_DIR"] = env.get("QCTX_STATE_DIR") or tempfile.mkdtemp()
    env.update(env_extra or {})

    return subprocess.run([sys.executable, os.path.join(ROOT, "cli", "qctx.py"), *args],
                          capture_output=True, text=True, cwd=ROOT, env=env)


class TestTheVerbsExist(unittest.TestCase):
    """Argument wiring only: these must not reach Qdrant, so they are checked by --help."""

    def test_repos_has_the_four_verbs(self):
        out = run_cli("repos", "--help")
        self.assertEqual(out.returncode, 0, out.stderr)
        for verb in ("list", "search", "add", "drop"):
            self.assertIn(verb, out.stdout)

    def test_search_takes_repo_and_all(self):
        out = run_cli("repos", "search", "--help")
        self.assertIn("--repo", out.stdout)
        self.assertIn("--all", out.stdout)

    def test_drop_requires_confirmation_by_default(self):
        """Deleting a permanent archive on a bare command would be a footgun."""
        out = run_cli("repos", "drop", "--help")
        self.assertIn("--yes", out.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

E em `tests/test_host_equivalence.py`, na classe que compara superfícies:

```python
    def test_both_hosts_expose_the_same_repository_operations(self):
        """Equivalence is this plugin's central requirement. A verb on one host and not the
        other is exactly the divergence this file exists to catch."""
        from hosts.hermes import tools as hermes_tools

        cli_verbs = {"list", "search", "add", "drop"}
        hermes_names = {t["name"] for t in hermes_tools.TOOLS
                        if t["name"].startswith("repos_")}
        self.assertEqual({n.split("repos_", 1)[1] for n in hermes_names}, cli_verbs)
```

- [ ] **Step 2: Rode e veja falhar**

Run: `python3 -m unittest tests.test_cli_repos 2>&1 | tail -4`
Expected: FAIL — `invalid choice: 'repos'`

- [ ] **Step 3: Acrescente o grupo ao CLI**

Em `cli/qctx.py`, ao lado de onde `docs` é registrado:

```python
    rep = sub.add_parser("repos", help="repository archives, grouped by repo")
    repsub = rep.add_subparsers(dest="repos_cmd", required=True)

    repsub.add_parser("list", help="every indexed repository").set_defaults(fn=cmd_repos_list)

    p = repsub.add_parser("search", help="search one repository, or every one")
    p.add_argument("query")
    p.add_argument("--repo", help="the repository to search (default: the one you are in)")
    p.add_argument("--all", action="store_true", dest="across",
                   help="search every indexed repository")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(fn=cmd_repos_search)

    p = repsub.add_parser("add", help="index the given files under a repository")
    p.add_argument("repo")
    p.add_argument("paths", nargs="+")
    p.set_defaults(fn=cmd_repos_add)

    p = repsub.add_parser("drop", help="delete a repository archive, permanently")
    p.add_argument("repo")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    p.set_defaults(fn=cmd_repos_drop)
```

E as quatro funções, no estilo dos comandos existentes (que constroem o índice a partir de
`load()` e imprimem tabela ou JSON conforme `args.json`):

```python
def _repo_index(cfg):
    from core.repos import RepoIndex

    return RepoIndex(build_qdrant(cfg), build_embedder(cfg),
                     cfg.require_repos_collection(),
                     cfg.require_repos_registry_collection(), cfg.vector_size)


def cmd_repos_list(args, cfg):
    ix = _repo_index(cfg)
    rows = ix.list_repos()
    divergent = ix.divergent_repos()
    if args.json:
        return {"repos": rows, "divergent": divergent}
    for r in rows:
        print(f"{r['repo']:<24} {r.get('label', ''):<24} "
              f"{len(r.get('checkouts') or [])} checkout(s)  {r.get('indexed_at', '?')}")
    for name in divergent:
        # Named out loud: it cannot be listed, so it cannot be dropped by name either.
        print(f"{name:<24} (chunks with no registry entry — run `repos drop {name}`)")

    return None


def cmd_repos_search(args, cfg):
    from core import bindings

    ix = _repo_index(cfg)
    repo = args.repo
    if not repo and not args.across:
        root = bindings.git_root(os.getcwd())
        repo = bindings.get(root) if root else None
        if not repo:
            # Never a silent broad search: that is the noise the default exists to avoid.
            raise SystemExit("not inside an indexed repository — pass --repo <name> or --all")
    out = ix.search(args.query, repo=repo, across=args.across, limit=args.limit)
    if args.json:
        return out
    for group in out["groups"]:
        print(f"\n=== {group['repo']}")
        for hit in group["hits"]:
            flag = f"  [{hit.stale}]" if hit.stale else ""
            print(f"  {hit.score:.3f}  {hit.path}:{hit.start_line}-{hit.end_line}{flag}")

    return None


def cmd_repos_add(args, cfg):
    ix = _repo_index(cfg)
    out = ix.add_files(args.repo, args.paths)
    if args.json:
        return out
    print(f"{out['files']} file(s), {out['chunks']} chunk(s) under {out['repo']}")
    for path, why in out["skipped"]:
        print(f"  skipped {path}: {why}")

    return None


def cmd_repos_drop(args, cfg):
    ix = _repo_index(cfg)
    if not args.yes:
        raise SystemExit(f"this permanently deletes the archive of {args.repo!r}. "
                         f"Re-run with --yes to confirm.")
    out = ix.drop_repo(args.repo)
    if args.json:
        return out
    print(f"dropped {out['repo']}; unbound {len(out['unbound'])} checkout(s)")

    return None
```

- [ ] **Step 4: As mesmas operações no hermes**

Em `hosts/hermes/tools.py`, acrescente quatro ferramentas seguindo o formato exato das que já
existem (nome, descrição, `input_schema`, handler): `repos_list`, `repos_search` (com
`query`, `repo`, `across`), `repos_add` (`repo`, `paths`), `repos_drop` (`repo`, `yes`).
Cada handler chama o MESMO `RepoIndex` — nenhuma lógica de decisão vive no adaptador.

- [ ] **Step 5: Rode e veja passar**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest tests.test_cli_repos tests.test_host_equivalence 2>&1 | tail -3`
Expected: `OK`

- [ ] **Step 6: Prove que a equivalência morde**

```bash
cp hosts/hermes/tools.py /tmp/tl.bak
python3 - <<'PY'
p = "hosts/hermes/tools.py"; s = open(p).read()
needle = '"repos_drop"'
assert s.count(needle) >= 1, s.count(needle)
open(p, "w").write(s.replace(needle, '"repos_delete"', 1))
print("mutation landed: one host renamed a repository operation")
PY
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
python3 -m unittest tests.test_host_equivalence 2>&1 | tail -3
cp /tmp/tl.bak hosts/hermes/tools.py && rm /tmp/tl.bak
```
Expected: `FAILED` nomeando `test_both_hosts_expose_the_same_repository_operations`.

- [ ] **Step 7: Prove que `docs --scope all` NÃO alcança repos**

```python
# acrescente a tests/test_repos_search.py
class TestTheDocumentArchivesAreUntouched(unittest.TestCase):
    def test_the_repos_collection_is_not_one_of_the_docs_scopes(self):
        """A regression here floods every existing `docs search --scope all` with tens of
        thousands of code chunks — shipped behaviour, silently degraded."""
        from core.docs import SCOPES
        self.assertEqual(set(SCOPES), {"all", "tmp", "library"})
```

Run: `python3 -m unittest tests.test_repos_search 2>&1 | tail -3`
Expected: `OK`

- [ ] **Step 8: Rode a suíte inteira e commite**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + ; python3 -m unittest discover -s tests 2>&1 | tail -3
git add cli/qctx.py hosts/hermes/tools.py tests/test_cli_repos.py tests/test_host_equivalence.py tests/test_repos_search.py
git commit -F - <<'MSG'
feat: the repository verbs, on both hosts, with equivalence held by a test

Equivalence is this plugin's central requirement, in the user's words: the two hosts must be
equivalent in functions and configuration. A verb on one host and not the other is precisely
the divergence test_host_equivalence exists to catch, so the operation names are derived from
both surfaces and required to match rather than kept in step by memory.

`repos search` with no scope resolves the repository from the working directory and REFUSES
when there is none, naming the remedy. Falling back to a broad search would be the noise the
scoped default was chosen to avoid — and it would be silent, which is worse than wrong.

`repos drop` demands --yes: the archive it deletes is permanent by design and there is no
sweep to undo it.

`repos list` names divergent repos out loud. A repo with chunks and no registry entry cannot
be listed, therefore cannot be dropped by name — printing it is what makes it fixable.
MSG
```

---

## Auto-revisão do plano

**Cobertura da spec, item por item:**

| Item da spec | Task |
|---|---|
| Uma coleção + campo `repo` indexado | T3 (índice de payload no `ensure`) |
| Registro em coleção própria | T2 (config), T3 (escrita e leitura) |
| Cinco coleções distintas | T2 |
| `search_groups` (a travessia) | T1 |
| Identidade declarada, com casamento por remote | T4 (`candidates_for`) |
| Colisão de slug é erro, não fusão | **LACUNA — ver abaixo** |
| Vínculo local caminho → repo | T4 |
| Cicatrização ao mover checkout | T4 (teste de vínculo obsoleto) |
| Escopo padrão = repo atual; `--all` a um passo | T5, T7 |
| Falha explícita fora de repo indexado | T7 (`cmd_repos_search`) |
| Nunca afirmar ausência | T1 (docstring), T5 (`truncated`) |
| Localização em vez de retrato; `stale` | T5 (`_to_hit`) |
| Deleção manual, chunks antes do registro | T6 |
| Invalidar vínculos ao apagar | T6 |
| `drop` não alcança os outros acervos | T6 (teste), T2 (distinção) |
| `scope=all` não alcança repos | T7 Step 7 |
| Divergência registro × chunks | T6 (`divergent_repos`) |
| Equivalência entre hosts | T7 |
| Escrita recebe a lista, não descobre | T3 |
| Reusar e não copiar | T3 (imports de `docs`/`chunk`) |

**LACUNA ENCONTRADA E FECHADA:** a spec exige que **colisão de slug seja erro com o conflito
nomeado, não fusão** — nenhuma task cobria isso, e é exatamente a classe de requisito órfão que
uma review por task não veria. Acrescente à Task 4, entre os Steps 4 e 5:

```python
# acrescente a tests/test_bindings.py, na classe TestTheChoiceOffered
    def test_a_taken_slug_is_reported_as_a_conflict_and_not_silently_joined(self):
        """Merging on slug collision would decide identity by an accident of naming, which is
        what the declared-identity decision exists to reject."""
        self.ix.register("alpha", "Alpha", ["git@host:me/alpha.git"], "/home/me/alpha")
        out = self.ix.candidates_for("/home/me/some-other/alpha", [])
        self.assertEqual(out["suggest"], "alpha")
        self.assertTrue(out["taken"], "a suggestion that already exists must be flagged")
```

e no `candidates_for`, no dicionário devolvido:

```python
                "taken": bindings.slug_for(os.path.basename(os.path.realpath(root))) in by_name,
```

**Varredura de placeholder:** nenhum "TBD"/"TODO"/"similar à Task N". O único passo em prosa em
vez de código é o Step 4 da Task 7 (ferramentas do hermes), e é deliberado: o formato exato do
dict de ferramenta tem de ser copiado das vizinhas do arquivo, e transcrevê-lo aqui de memória
introduziria justamente o erro de valor que este projeto já mediu.

**Consistência de valor** (o check que este projeto aprendeu depois de a spec dizer três
`dirname` e o plano escrever dois): `memories_repos` e `memories_repos_registry` aparecem
idênticos na spec e aqui; `group_size` 3 e `limit` = nº de repos batem com a spec; `repo` no
TOPO do payload (não em `metadata`) é consistente entre T3, T5, T6 e o índice de payload;
`_point_id(f"registry:{repo}", 0)` é a mesma expressão em T3 e T6; `QCTX_STATE_DIR` é o mesmo
nome nas Tasks 4, 6 e 7.

**Consistência de tipo:** `add_files` devolve `{"repo", "files", "chunks", "skipped"}` em T3 e é
lido assim em T7; `search` devolve `{"scope", "repo", "groups", "truncated"}` em T5 e é lido
assim em T7; `drop_repo` devolve `{"repo", "unbound"}` em T6 e é lido assim em T7;
`candidates_for` devolve `{"bound", "join", "suggest", "taken"}` depois do fechamento da lacuna.
