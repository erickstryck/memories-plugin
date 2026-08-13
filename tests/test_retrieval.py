"""Testes do pipeline de recuperação, com DUBLÊS em vez de rede.

É o ganho concreto de ter contrato declarado: esta lógica — a mais delicada do
pacote — antes só dava para exercitar em teste de integração, que ninguém roda
enquanto edita. Agora roda em milissegundos, offline, e cobre os caminhos que
importam justamente por serem os que degradam em SILÊNCIO.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import retrieval
from core.retrieval import CE, CE_FRACO, DENSO, Policy, fuse_by_id, needs_rerank, two_stage


def hit(id_: str, score: float, texto: str = "texto"):
    return {"id": id_, "score": score, "document": texto}


TEXTO = lambda h: h["document"]  # noqa: E731
ID = lambda h: h["id"]           # noqa: E731


class RerankerFake:
    """Dublê que devolve os scores combinados. Não precisa herdar de nada — o
    contrato é estrutural (Protocol)."""

    def __init__(self, scores=None, ok=True, erro=None, era_logit=False):
        self.scores = scores          # lista de score por índice de entrada
        self.ok = ok
        self.erro = erro
        self.era_logit = era_logit
        self.chamadas = []

    def rank(self, query, documentos):
        self.chamadas.append((query, list(documentos)))
        if not self.ok:
            return [], {"ok": False, "erro": self.erro or "falhou", "era_logit": False}
        scores = self.scores if self.scores is not None else [1.0] * len(documentos)
        pares = sorted(enumerate(scores[:len(documentos)]), key=lambda p: -p[1])

        return pares, {"ok": True, "erro": None, "era_logit": self.era_logit}


POLITICA_MEMORIA = Policy(dense_floor=0.45, strict_floor=0.58, min_score=0.10,
                          max_results=6, veto=True)
# Espelha `core/docs.py`: os DOIS pisos iguais, de propósito. Em documentos não há
# veto e o objetivo é nunca devolver silêncio, então "voltar ao corte estrito" tem
# de ser um no-op — se fosse 0.58, o colapso cross-lingual (denso ~0.46) devolveria
# vazio, que é exatamente o que essa política existe para evitar.
POLITICA_DOCS = Policy(dense_floor=0.30, strict_floor=0.30, min_score=0.10,
                       max_results=5, veto=False, order_matters=True)


class TestFusao(unittest.TestCase):
    def test_mantem_o_maior_score_de_cada_id(self):
        fundido = fuse_by_id([[hit("a", 0.5), hit("b", 0.9)],
                              [hit("a", 0.8), hit("c", 0.3)]], ID)
        por_id = {h["id"]: h["score"] for h in fundido}
        self.assertEqual(por_id, {"a": 0.8, "b": 0.9, "c": 0.3},
                         "id repetido em dois ângulos fica com o MAIOR score")

    def test_sai_ordenado_por_score(self):
        fundido = fuse_by_id([[hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)]], ID)
        self.assertEqual([h["id"] for h in fundido], ["b", "c", "a"])

    def test_lotes_vazios(self):
        self.assertEqual(fuse_by_id([], ID), [])
        self.assertEqual(fuse_by_id([[], []], ID), [])


class TestQuandoChamarOSegundoEstagio(unittest.TestCase):
    def test_chama_quando_ha_mais_candidatos_que_vagas(self):
        candidatos = [hit(str(i), 0.9) for i in range(10)]
        self.assertTrue(needs_rerank(candidatos, POLITICA_MEMORIA))

    def test_chama_quando_ha_candidato_na_faixa_permissiva(self):
        candidatos = [hit("a", 0.90), hit("b", 0.50)]  # 0.50 < strict 0.58
        self.assertTrue(needs_rerank(candidatos, POLITICA_MEMORIA),
                        "0.50 só entrou porque alguém ia julgá-lo")

    def test_NAO_chama_quando_tudo_cabe_e_esta_acima_do_estrito(self):
        candidatos = [hit("a", 0.90), hit("b", 0.70)]
        self.assertFalse(needs_rerank(candidatos, POLITICA_MEMORIA),
                         "reordenar não mudaria o que sai — a chamada seria inútil")

    def test_lista_vazia_nao_chama(self):
        self.assertFalse(needs_rerank([], POLITICA_MEMORIA))
        self.assertFalse(needs_rerank([], POLITICA_DOCS))

    def test_um_candidato_so_nao_precisa_de_ordem(self):
        self.assertFalse(needs_rerank([hit("a", 0.9)], POLITICA_DOCS),
                         "não há o que ordenar com um resultado")

    def test_quando_a_ordem_e_o_produto_chama_mesmo_cabendo_tudo(self):
        candidatos = [hit("a", 0.90), hit("b", 0.70)]
        self.assertFalse(needs_rerank(candidatos, POLITICA_MEMORIA),
                         "memória injeta o conjunto: a ordem é cosmética")
        self.assertTrue(needs_rerank(candidatos, POLITICA_DOCS),
                        "documento é lido de cima para baixo: a ordem é o produto")


class TestSemSegundoEstagio(unittest.TestCase):
    """O caso mais importante: sem julgamento, o piso permissivo precisa voltar
    ao estrito, senão o modo COM re-rank fica pior que o modo sem."""

    def test_reranker_ausente_aplica_corte_estrito(self):
        candidatos = [hit("a", 0.90), hit("b", 0.60), hit("c", 0.50), hit("d", 0.47)]
        fora = two_stage(candidatos, "q", None, POLITICA_MEMORIA, TEXTO)
        self.assertEqual([s.item["id"] for s in fora.scored], ["a", "b"])
        self.assertTrue(all(s.origin == DENSO for s in fora.scored))
        self.assertFalse(fora.reranked)

    def test_reranker_que_falha_aplica_corte_estrito(self):
        candidatos = [hit("a", 0.90), hit("b", 0.53), hit("c", 0.49)]
        rr = RerankerFake(ok=False, erro="timeout")
        fora = two_stage(candidatos, "q", rr, POLITICA_MEMORIA, TEXTO)
        self.assertEqual([s.item["id"] for s in fora.scored], ["a"],
                         "0.53 e 0.49 só eram aceitáveis com o cross-encoder para limpar")
        self.assertEqual(fora.rerank_error, "timeout")
        self.assertFalse(fora.reranked)

    def test_resposta_sem_hits_utilizaveis_tambem_cai_no_estrito(self):
        rr = RerankerFake(ok=False, erro="resposta sem hits utilizáveis")
        fora = two_stage([hit("a", 0.50)], "q", rr, POLITICA_MEMORIA, TEXTO)
        self.assertEqual(fora.scored, [])


class TestComVeto(unittest.TestCase):
    """Política de memória: falso positivo polui o contexto do agente."""

    def test_elimina_o_que_esta_abaixo_do_corte(self):
        # um candidato na faixa permissiva é o que obriga a chamada — com tudo
        # acima do estrito e cabendo nas vagas, a política de memória nem chamaria
        candidatos = [hit("a", 0.9), hit("b", 0.50), hit("c", 0.9)]
        rr = RerankerFake(scores=[0.95, 0.02, 0.40])
        fora = two_stage(candidatos, "q", rr, POLITICA_MEMORIA, TEXTO)
        self.assertEqual([s.item["id"] for s in fora.scored], ["a", "c"],
                         "0.02 é eliminado, não rebaixado")
        self.assertTrue(all(s.origin == CE for s in fora.scored))

    def test_o_cross_encoder_inverte_o_julgamento_denso(self):
        """Medido no acervo real: o CE matou um denso 0.59 e salvou um denso 0.47."""
        candidatos = [hit("alto_denso", 0.59), hit("baixo_denso", 0.47)]
        rr = RerankerFake(scores=[0.004, 0.11])
        fora = two_stage(candidatos, "q", rr, POLITICA_MEMORIA, TEXTO)
        self.assertEqual([s.item["id"] for s in fora.scored], ["baixo_denso"])

    def test_respeita_o_teto_de_resultados(self):
        candidatos = [hit(str(i), 0.9) for i in range(20)]
        rr = RerankerFake(scores=[0.9] * 20)
        fora = two_stage(candidatos, "q", rr, Policy(0.45, 0.58, 0.10, 3), TEXTO)
        self.assertEqual(len(fora.scored), 3)


class TestSemVeto(unittest.TestCase):
    """Política de documentos: quem pergunta já escolheu o documento, então
    silêncio é pior que ordem imperfeita."""

    def test_entrega_o_fraco_no_fim_marcado(self):
        candidatos = [hit("a", 0.9), hit("b", 0.9)]
        rr = RerankerFake(scores=[0.80, 0.02])
        fora = two_stage(candidatos, "q", rr, POLITICA_DOCS, TEXTO)
        self.assertEqual([s.item["id"] for s in fora.scored], ["a", "b"])
        self.assertEqual([s.origin for s in fora.scored], [CE, CE_FRACO])
        self.assertTrue(fora.scored[1].is_weak)

    def test_nunca_devolve_vazio_tendo_candidato(self):
        candidatos = [hit("a", 0.9), hit("b", 0.9)]
        rr = RerankerFake(scores=[0.05, 0.03])
        fora = two_stage(candidatos, "q", rr, POLITICA_DOCS, TEXTO)
        self.assertEqual(len(fora.scored), 2, "com veto isto viraria silêncio")
        self.assertTrue(all(s.is_weak for s in fora.scored))


class TestColapsoCrossLingual(unittest.TestCase):
    """Quando o cross-encoder colapsa, a ORDEM dele também é ruído."""

    def test_detecta_colapso_e_volta_para_a_ordem_densa(self):
        candidatos = [hit("melhor_denso", 0.60), hit("pior_denso", 0.46)]
        # invertido de propósito: o CE colapsado "prefere" o pior denso
        rr = RerankerFake(scores=[0.0004, 0.0009])
        fora = two_stage(candidatos, "q", rr, POLITICA_DOCS, TEXTO)
        self.assertTrue(fora.collapsed)
        self.assertEqual([s.item["id"] for s in fora.scored], ["melhor_denso", "pior_denso"],
                         "ordem densa, não a do CE colapsado")
        self.assertTrue(all(s.origin == DENSO for s in fora.scored))

    def test_colapso_em_memoria_RESTAURA_o_corte_estrito(self):
        """Regressão: o colapso descarta o julgamento, então o piso permissivo volta
        a não ter quem o limpe. Sem isto, o modo COM re-rank devolvia candidato que o
        modo SEM re-rank nunca devolveria — o defeito que o pipeline existe para não
        ter. Só aparece na política de MEMÓRIA, onde os dois pisos diferem."""
        candidatos = [hit("passa", 0.90), hit("permissivo", 0.50), hit("permissivo2", 0.46)]
        rr = RerankerFake(scores=[0.0004, 0.0009, 0.0002])
        fora = two_stage(candidatos, "q", rr, POLITICA_MEMORIA, TEXTO)
        self.assertTrue(fora.collapsed)
        self.assertEqual([s.item["id"] for s in fora.scored], ["passa"],
                         "0.50 e 0.46 só eram aceitáveis com o cross-encoder para julgar")

    def test_colapso_em_docs_NAO_devolve_silencio(self):
        """A outra metade do par: com os pisos iguais, restaurar o estrito não corta
        nada, e a pergunta em outra língua continua sendo respondida."""
        candidatos = [hit("a", 0.46), hit("b", 0.44)]  # faixa típica cross-lingual
        rr = RerankerFake(scores=[0.0004, 0.0002])
        fora = two_stage(candidatos, "q", rr, POLITICA_DOCS, TEXTO)
        self.assertTrue(fora.collapsed)
        self.assertEqual(len(fora.scored), 2, "silêncio aqui seria pior que ordem imperfeita")

    def test_score_baixo_mas_acima_do_limiar_nao_e_colapso(self):
        candidatos = [hit("a", 0.9), hit("b", 0.9)]
        rr = RerankerFake(scores=[0.05, 0.02])  # 0.05 > COLLAPSE_MAX
        fora = two_stage(candidatos, "q", rr, POLITICA_DOCS, TEXTO)
        self.assertFalse(fora.collapsed, "0.05 é relevância baixa, não colapso")

    def test_colapso_pode_ser_desligado_por_politica(self):
        politica = Policy(0.45, 0.58, 0.10, 5, veto=False, detect_collapse=False)
        rr = RerankerFake(scores=[0.0004, 0.0009])
        fora = two_stage([hit("a", 0.60), hit("b", 0.46)], "q", rr, politica, TEXTO)
        self.assertFalse(fora.collapsed)


class TestRastro(unittest.TestCase):
    def test_registra_conversao_de_escala(self):
        rr = RerankerFake(scores=[0.9, 0.5], era_logit=True)
        fora = two_stage([hit("a", 0.9), hit("b", 0.9)], "q", rr,
                         Policy(0.45, 0.58, 0.10, 1), TEXTO)
        self.assertTrue(fora.scale_converted)

    def test_registra_melhor_denso_mesmo_sem_resultado(self):
        fora = two_stage([hit("a", 0.30)], "q", None, POLITICA_MEMORIA, TEXTO)
        self.assertEqual(fora.scored, [])
        self.assertAlmostEqual(fora.best_dense, 0.30)

    def test_by_rerank_distingue_a_procedencia(self):
        rr = RerankerFake(scores=[0.9])
        com_ce = two_stage([hit("a", 0.5)], "q", rr, POLITICA_MEMORIA, TEXTO)
        sem_ce = two_stage([hit("a", 0.9)], "q", None, POLITICA_MEMORIA, TEXTO)
        self.assertTrue(com_ce.by_rerank)
        self.assertFalse(sem_ce.by_rerank)

    def test_query_chega_ao_reranker_e_os_textos_tambem(self):
        rr = RerankerFake(scores=[0.9, 0.8])
        two_stage([hit("a", 0.5, "texto A"), hit("b", 0.5, "texto B")],
                  "minha pergunta", rr, POLITICA_MEMORIA, TEXTO)
        query, docs = rr.chamadas[0]
        self.assertEqual(query, "minha pergunta")
        self.assertEqual(docs, ["texto A", "texto B"])


class TestPiso(unittest.TestCase):
    def test_piso_relaxa_so_quando_ha_segundo_estagio(self):
        self.assertAlmostEqual(POLITICA_MEMORIA.floor_for(True), 0.45)
        self.assertAlmostEqual(POLITICA_MEMORIA.floor_for(False), 0.58)


class TestContratoDoDuble(unittest.TestCase):
    def test_o_fake_satisfaz_o_protocolo_sem_herdar_nada(self):
        from core.ports import RerankModel
        self.assertIsInstance(RerankerFake(), RerankModel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
