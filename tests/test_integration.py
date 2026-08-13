"""Testes de INTEGRAÇÃO — exigem Qdrant e endpoints de modelo reais.

Ficam fora da suíte padrão de propósito: `python3 -m unittest discover -s tests`
tem de rodar offline e em milissegundos. Para rodar estes:

    QCTX_INTEGRATION=1 python3 -m unittest tests.test_integration -v

REGRA DE SEGURANÇA, e não é negociável: o acervo de memória configurado é tratado
como PRODUÇÃO. Toda escrita vai para uma coleção descartável criada e apagada pelo
próprio teste; no acervo real só há LEITURA. Um teste que apaga memória de verdade
é pior que nenhum teste.

O que estes testes provam, e que teste offline não consegue: que o núcleo lê o
payload escrito pelo servidor MCP antigo sem conversão, e que o payload que ele
escreve tem exatamente as mesmas chaves — é o que torna a substituição segura.
"""
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core

LIGADO = os.environ.get("QCTX_INTEGRATION") == "1"
COLECAO_DESCARTAVEL = f"qctx_test_{uuid.uuid4().hex[:8]}"

#: Chaves que o servidor MCP anterior gravava. O núcleo tem de ler e escrever
#: exatamente isto, senão o acervo existente vira ilegível ou fica inconsistente.
CHAVES_PAYLOAD = {"document", "metadata", "created_at", "updated_at"}


def read_config():
    return core.load()


def write_config():
    """Config apontando a memória para a coleção descartável."""
    base = core.load()
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields["memory_collection"] = COLECAO_DESCARTAVEL

    return core.Config(**fields)


@unittest.skipUnless(LIGADO, "defina QCTX_INTEGRATION=1")
class TestReadingTheRealArchive(unittest.TestCase):
    """SOMENTE LEITURA. Prova que o núcleo entende o que já está gravado."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = read_config()
        if not cls.cfg.memory_collection:
            raise unittest.SkipTest("memory_collection não configurada")
        cls.store = core.build_memory(cls.cfg)

    def test_archive_has_points(self):
        total = self.store.count()
        self.assertIsNotNone(total)
        self.assertGreater(total, 0, "o acervo real deveria ter memórias")

    def test_payload_written_by_the_old_mcp_is_readable(self):
        pagina = self.store.list_page(limit=5)
        self.assertGreater(pagina["count"], 0)
        for m in pagina["memories"]:
            self.assertIsInstance(m["document"], str)
            self.assertTrue(m["document"].strip(), "documento não pode vir vazio")
            self.assertIsInstance(m["metadata"], dict)

    def test_get_by_id_returns_the_four_keys(self):
        pagina = self.store.list_page(limit=1)
        mid = pagina["memories"][0]["id"]
        m = self.store.get(mid)
        self.assertNotEqual(m.get("status"), "not_found")
        for key in ("document", "metadata", "created_at", "updated_at"):
            self.assertIn(key, m, f"{key} tem de existir no payload legado")

    def test_find_returns_descending_scores(self):
        hits = self.store.find("memória de longo prazo", limit=5)
        self.assertTrue(hits, "busca densa no acervo real não devolveu nada")
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_two_gate_recall_against_the_real_archive(self):
        policy = core.Policy(0.45, 0.58, 0.10, 6, veto=True)
        hits, outcome = self.store.recall(["memória de longo prazo e recall automático"],
                                       policy, top_k=20)
        self.assertGreater(outcome.candidates, 0)
        for h in hits:
            self.assertIsInstance(h, core.Recalled)
            self.assertIn(h.origin, ("CE", "denso"))
            self.assertGreaterEqual(h.dense_score, 0.0)
        if outcome.by_rerank:
            self.assertTrue(all(h.origin == "CE" for h in hits))

    def test_recall_with_several_angles_fuses_by_id(self):
        hits, _ = self.store.recall(
            ["como funciona o hook de recall",
             "hook recall funciona como",
             "recall automático a cada prompt"],
            core.Policy(0.45, 0.58, 0.10, 6, veto=True), top_k=10)
        ids = [h.id for h in hits]
        self.assertEqual(len(ids), len(set(ids)), "fusão por id não pode duplicar")


@unittest.skipUnless(LIGADO, "defina QCTX_INTEGRATION=1")
class TestCrudInAThrowawayCollection(unittest.TestCase):
    """Escrita SÓ aqui. A coleção é criada e destruída pelo próprio teste."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = write_config()
        cls.store = core.build_memory(cls.cfg)
        cls.q = core.build_qdrant(cls.cfg)
        assert cls.cfg.memory_collection == COLECAO_DESCARTAVEL

    @classmethod
    def tearDownClass(cls):
        try:
            cls.q.delete_collection(COLECAO_DESCARTAVEL)
        except Exception:
            pass

    def test_full_cycle(self):
        criado = self.store.store("O poll do conector trunca em 100 itens por página.",
                                  {"type": "reference", "date": "2026-08-13"})
        self.assertEqual(criado["status"], "created")
        mid = criado["id"]

        lido = self.store.get(mid)
        self.assertIn("trunca em 100", lido["document"])
        self.assertEqual(lido["metadata"]["type"], "reference")

        achado = self.store.find("paginação do poll", limit=5)
        self.assertIn(mid, [h["id"] for h in achado])

        atualizado = self.store.update(mid, information="Corrigido: trunca em 50 itens.")
        self.assertEqual(atualizado["status"], "updated")
        relido = self.store.get(mid)
        self.assertIn("50 itens", relido["document"])
        self.assertEqual(relido["metadata"]["type"], "reference",
                         "update sem metadata tem de PRESERVAR a metadata anterior")
        self.assertEqual(relido["created_at"], lido["created_at"],
                         "created_at não pode ser reescrito por um update")
        self.assertNotEqual(relido["updated_at"], lido["updated_at"])

        self.store.delete(mid)
        self.assertEqual(self.store.get(mid)["status"], "not_found")

    def test_written_payload_has_the_same_keys_as_the_old_mcp(self):
        criado = self.store.store("fato para conferir a forma do payload", {"type": "test"})
        point = self.q.get_point(COLECAO_DESCARTAVEL, criado["id"])
        self.assertEqual(set(point["payload"].keys()), CHAVES_PAYLOAD,
                         "o payload tem de ser idêntico ao do servidor anterior, "
                         "senão o acervo existente fica inconsistente")
        self.store.delete(criado["id"])

    def test_store_many_is_all_or_nothing(self):
        items = [{"information": f"fato de lote número {i}", "metadata": {"type": "test"}}
                 for i in range(5)]
        res = self.store.store_many(items)
        self.assertEqual(res["count"], 5)
        for mid in res["ids"]:
            self.assertNotEqual(self.store.get(mid).get("status"), "not_found")
        for mid in res["ids"]:
            self.store.delete(mid)

    def test_store_many_refuses_an_invalid_item_before_writing(self):
        antes = self.store.count() or 0
        with self.assertRaises(core.memory.MemoryError_):
            self.store.store_many([{"information": "válido"}, {"information": "  "}])
        time.sleep(0.2)
        self.assertEqual(self.store.count() or 0, antes,
                         "validação tem de acontecer ANTES de qualquer escrita")

    def test_empty_store_is_refused(self):
        with self.assertRaises(core.memory.MemoryError_):
            self.store.store("   ")


@unittest.skipUnless(LIGADO, "defina QCTX_INTEGRATION=1")
class TestDocsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = core.load()
        fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
        fields["docs_collection"] = f"{COLECAO_DESCARTAVEL}_tmp"
        fields["library_collection"] = f"{COLECAO_DESCARTAVEL}_lib"
        cls.cfg = core.Config(**fields)
        cls.idx = core.build_docs(cls.cfg)
        cls.q = core.build_qdrant(cls.cfg)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.file_path = Path(cls.tmpdir.name) / "manual.md"
        # Documento realista: cada seção precisa ser grande o bastante para o
        # fatiamento ter trabalho, senão o teste "acha a seção certa" é vacuidade
        # — com um trecho único, acertar é inevitável.
        recheio = ("Detalhe operacional relevante para esta seção, repetido para dar "
                   "corpo ao documento sem mudar o assunto dela. ")
        cls.file_path.write_text("\n".join([
            "# Autenticação",
            "",
            "Para autenticar, envie o header Authorization com um token Bearer.",
            "O token expira em uma hora e precisa ser renovado pelo endpoint de refresh.",
            recheio * 12,
            "",
            "# Paginação",
            "",
            "As listagens devolvem no máximo 100 itens por página.",
            "Use o cursor devolvido em next_page para buscar a página seguinte.",
            recheio * 12,
            "",
            "# Limites de uso",
            "",
            "O limite é de 5000 requisições por hora, com janela deslizante de 180 segundos.",
            recheio * 12,
        ]) + "\n")

    @classmethod
    def tearDownClass(cls):
        for name in (f"{COLECAO_DESCARTAVEL}_tmp", f"{COLECAO_DESCARTAVEL}_lib"):
            try:
                cls.q.delete_collection(name)
            except Exception:
                pass
        cls.tmpdir.cleanup()

    def test_index_then_search_locates_the_right_section(self):
        res = self.idx.index_file(str(self.file_path), ttl_seconds=600)
        self.assertGreater(res["chunks"], 1, "documento com 3 seções longas tem de virar vários trechos")
        self.assertEqual(res["mode"], "locator")

        hits, info = self.idx.search("qual o limite de requisições por hora?",
                                     scope="tmp", limit=3)
        self.assertTrue(hits, "deveria achar a seção de limites")
        top_text = hits[0].text.lower()
        self.assertIn("5000", top_text)
        self.assertGreater(hits[0].start_line, 0)
        self.assertGreaterEqual(hits[0].end_line, hits[0].start_line)

    def test_line_range_points_at_the_real_content(self):
        self.idx.index_file(str(self.file_path), ttl_seconds=600)
        hits, _ = self.idx.search("como paginar?", scope="tmp", limit=1)
        lines = self.file_path.read_text().splitlines()
        slice_text = "\n".join(lines[hits[0].start_line - 1:hits[0].end_line])
        self.assertEqual(slice_text.strip("\n"), hits[0].text,
                         "o contrato do modo localizador é que essas linhas "
                         "reproduzam exatamente o trecho indexado")

    def test_library_never_expires_and_temporary_does(self):
        self.idx.keep_file(str(self.file_path))
        self.idx.index_file(str(self.file_path), ttl_seconds=600)
        docs = {d["scope"]: d for d in self.idx.list_docs("all")}
        self.assertIn("library", docs)
        self.assertIn("tmp", docs)
        self.assertIsNone(docs["library"]["expires_at_ts"])
        self.assertIsNotNone(docs["tmp"]["expires_at_ts"])

    def test_expired_ttl_disappears_from_search(self):
        self.idx.index_file(str(self.file_path), ttl_seconds=-1)  # já nasce vencido
        hits, _ = self.idx.search("autenticação", scope="tmp", limit=3)
        self.assertEqual(hits, [], "trecho vencido não pode aparecer")

    def test_purging_temporary_preserves_the_library(self):
        self.idx.keep_file(str(self.file_path))
        self.idx.index_file(str(self.file_path), ttl_seconds=600)
        self.idx.drop_all_tmp()
        docs = self.idx.list_docs("all")
        scopes = {d["scope"] for d in docs}
        self.assertIn("library", scopes, "a biblioteca TEM de sobreviver ao purge")
        self.assertNotIn("tmp", scopes)

    def test_reindexing_replaces_instead_of_duplicating(self):
        first = self.idx.keep_file(str(self.file_path))
        second = self.idx.keep_file(str(self.file_path))
        self.assertEqual(first["doc_id"], second["doc_id"])
        docs = [d for d in self.idx.list_docs("library") if d["doc_id"] == second["doc_id"]]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["chunks"], second["chunks"],
                         "não pode sobrar trecho da indexação anterior")


if __name__ == "__main__":
    unittest.main(verbosity=2)
