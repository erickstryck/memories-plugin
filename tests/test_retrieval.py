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
from core.retrieval import CE, CE_WEAK, DENSE, Policy, fuse_by_id, needs_rerank, two_stage


def hit(id_: str, score: float, text: str = "texto"):
    return {"id": id_, "score": score, "document": text}


TEXT = lambda h: h["document"]  # noqa: E731
ID = lambda h: h["id"]           # noqa: E731


class FakeReranker:
    """Dublê que devolve os scores combinados. Não precisa herdar de nada — o
    contrato é estrutural (Protocol)."""

    def __init__(self, scores=None, ok=True, error=None, was_logit=False):
        self.scores = scores          # lista de score por índice de entrada
        self.ok = ok
        self.error = error
        self.was_logit = was_logit
        self.calls = []

    def rank(self, query, documents):
        self.calls.append((query, list(documents)))
        if not self.ok:
            return [], {"ok": False, "erro": self.error or "falhou", "era_logit": False}
        scores = self.scores if self.scores is not None else [1.0] * len(documents)
        pairs = sorted(enumerate(scores[:len(documents)]), key=lambda p: -p[1])

        return pairs, {"ok": True, "erro": None, "era_logit": self.was_logit}


MEMORY_POLICY = Policy(dense_floor=0.45, strict_floor=0.58, min_score=0.10,
                          max_results=6, veto=True)
# Espelha `core/docs.py`: os DOIS pisos iguais, de propósito. Em documentos não há
# veto e o objetivo é nunca devolver silêncio, então "voltar ao corte estrito" tem
# de ser um no-op — se fosse 0.58, o colapso cross-lingual (denso ~0.46) devolveria
# vazio, que é exatamente o que essa política existe para evitar.
DOCS_POLICY = Policy(dense_floor=0.30, strict_floor=0.30, min_score=0.10,
                       max_results=5, veto=False, order_matters=True)


class TestFusion(unittest.TestCase):
    def test_keeps_the_highest_score_for_each_id(self):
        fused = fuse_by_id([[hit("a", 0.5), hit("b", 0.9)],
                              [hit("a", 0.8), hit("c", 0.3)]], ID)
        by_id = {h["id"]: h["score"] for h in fused}
        self.assertEqual(by_id, {"a": 0.8, "b": 0.9, "c": 0.3},
                         "id repetido em dois ângulos fica com o MAIOR score")

    def test_comes_out_sorted_by_score(self):
        fused = fuse_by_id([[hit("a", 0.1), hit("b", 0.9), hit("c", 0.5)]], ID)
        self.assertEqual([h["id"] for h in fused], ["b", "c", "a"])

    def test_empty_batches(self):
        self.assertEqual(fuse_by_id([], ID), [])
        self.assertEqual(fuse_by_id([[], []], ID), [])


class TestWhenToCallTheSecondStage(unittest.TestCase):
    def test_calls_when_there_are_more_candidates_than_slots(self):
        candidates = [hit(str(i), 0.9) for i in range(10)]
        self.assertTrue(needs_rerank(candidates, MEMORY_POLICY))

    def test_calls_when_a_candidate_sits_in_the_permissive_band(self):
        candidates = [hit("a", 0.90), hit("b", 0.50)]  # 0.50 < strict 0.58
        self.assertTrue(needs_rerank(candidates, MEMORY_POLICY),
                        "0.50 só entrou porque alguém ia julgá-lo")

    def test_does_NOT_call_when_everything_fits_and_clears_the_strict_floor(self):
        candidates = [hit("a", 0.90), hit("b", 0.70)]
        self.assertFalse(needs_rerank(candidates, MEMORY_POLICY),
                         "reordenar não mudaria o que sai — a chamada seria inútil")

    def test_empty_list_does_not_call(self):
        self.assertFalse(needs_rerank([], MEMORY_POLICY))
        self.assertFalse(needs_rerank([], DOCS_POLICY))

    def test_a_single_candidate_needs_no_ordering(self):
        self.assertFalse(needs_rerank([hit("a", 0.9)], DOCS_POLICY),
                         "não há o que ordenar com um resultado")

    def test_when_order_is_the_product_it_calls_even_if_everything_fits(self):
        candidates = [hit("a", 0.90), hit("b", 0.70)]
        self.assertFalse(needs_rerank(candidates, MEMORY_POLICY),
                         "memória injeta o conjunto: a ordem é cosmética")
        self.assertTrue(needs_rerank(candidates, DOCS_POLICY),
                        "documento é lido de cima para baixo: a ordem é o produto")


class TestWithoutTheSecondStage(unittest.TestCase):
    """O caso mais importante: sem julgamento, o piso permissivo precisa voltar
    ao estrito, senão o modo COM re-rank fica pior que o modo sem."""

    def test_absent_reranker_applies_the_strict_cut(self):
        candidates = [hit("a", 0.90), hit("b", 0.60), hit("c", 0.50), hit("d", 0.47)]
        outcome = two_stage(candidates, "q", None, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a", "b"])
        self.assertTrue(all(s.origin == DENSE for s in outcome.scored))
        self.assertFalse(outcome.reranked)

    def test_failing_reranker_applies_the_strict_cut(self):
        candidates = [hit("a", 0.90), hit("b", 0.53), hit("c", 0.49)]
        rr = FakeReranker(ok=False, error="timeout")
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a"],
                         "0.53 e 0.49 só eram aceitáveis com o cross-encoder para limpar")
        self.assertEqual(outcome.rerank_error, "timeout")
        self.assertFalse(outcome.reranked)

    def test_response_without_usable_hits_also_falls_back_to_strict(self):
        rr = FakeReranker(ok=False, error="resposta sem hits utilizáveis")
        outcome = two_stage([hit("a", 0.50)], "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual(outcome.scored, [])


class TestWithVeto(unittest.TestCase):
    """Política de memória: falso positivo polui o contexto do agente."""

    def test_eliminates_what_sits_below_the_cutoff(self):
        # um candidato na faixa permissiva é o que obriga a chamada — com tudo
        # acima do estrito e cabendo nas vagas, a política de memória nem chamaria
        candidates = [hit("a", 0.9), hit("b", 0.50), hit("c", 0.9)]
        rr = FakeReranker(scores=[0.95, 0.02, 0.40])
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a", "c"],
                         "0.02 é eliminado, não rebaixado")
        self.assertTrue(all(s.origin == CE for s in outcome.scored))

    def test_the_cross_encoder_overturns_the_dense_verdict(self):
        """Medido no acervo real: o CE matou um denso 0.59 e salvou um denso 0.47."""
        candidates = [hit("alto_denso", 0.59), hit("baixo_denso", 0.47)]
        rr = FakeReranker(scores=[0.004, 0.11])
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["baixo_denso"])

    def test_respects_the_result_ceiling(self):
        candidates = [hit(str(i), 0.9) for i in range(20)]
        rr = FakeReranker(scores=[0.9] * 20)
        outcome = two_stage(candidates, "q", rr, Policy(0.45, 0.58, 0.10, 3), TEXT)
        self.assertEqual(len(outcome.scored), 3)


class TestWithoutVeto(unittest.TestCase):
    """Política de documentos: quem pergunta já escolheu o documento, então
    silêncio é pior que ordem imperfeita."""

    def test_delivers_the_weak_ones_last_and_marked(self):
        candidates = [hit("a", 0.9), hit("b", 0.9)]
        rr = FakeReranker(scores=[0.80, 0.02])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["a", "b"])
        self.assertEqual([s.origin for s in outcome.scored], [CE, CE_WEAK])
        self.assertTrue(outcome.scored[1].is_weak)

    def test_never_returns_empty_when_a_candidate_exists(self):
        candidates = [hit("a", 0.9), hit("b", 0.9)]
        rr = FakeReranker(scores=[0.05, 0.03])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertEqual(len(outcome.scored), 2, "com veto isto viraria silêncio")
        self.assertTrue(all(s.is_weak for s in outcome.scored))


class TestCrossLingualCollapse(unittest.TestCase):
    """Quando o cross-encoder colapsa, a ORDEM dele também é ruído."""

    def test_detects_collapse_and_falls_back_to_dense_order(self):
        candidates = [hit("melhor_denso", 0.60), hit("pior_denso", 0.46)]
        # invertido de propósito: o CE colapsado "prefere" o pior denso
        rr = FakeReranker(scores=[0.0004, 0.0009])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertTrue(outcome.collapsed)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["melhor_denso", "pior_denso"],
                         "ordem densa, não a do CE colapsado")
        self.assertTrue(all(s.origin == DENSE for s in outcome.scored))

    def test_collapse_in_memory_RESTORES_the_strict_cut(self):
        """Regressão: o colapso descarta o julgamento, então o piso permissivo volta
        a não ter quem o limpe. Sem isto, o modo COM re-rank devolvia candidato que o
        modo SEM re-rank nunca devolveria — o defeito que o pipeline existe para não
        ter. Só aparece na política de MEMÓRIA, onde os dois pisos diferem."""
        candidates = [hit("passa", 0.90), hit("permissivo", 0.50), hit("permissivo2", 0.46)]
        rr = FakeReranker(scores=[0.0004, 0.0009, 0.0002])
        outcome = two_stage(candidates, "q", rr, MEMORY_POLICY, TEXT)
        self.assertTrue(outcome.collapsed)
        self.assertEqual([s.item["id"] for s in outcome.scored], ["passa"],
                         "0.50 e 0.46 só eram aceitáveis com o cross-encoder para julgar")

    def test_collapse_in_docs_does_NOT_return_silence(self):
        """A outra metade do par: com os pisos iguais, restaurar o estrito não corta
        nada, e a pergunta em outra língua continua sendo respondida."""
        candidates = [hit("a", 0.46), hit("b", 0.44)]  # faixa típica cross-lingual
        rr = FakeReranker(scores=[0.0004, 0.0002])
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertTrue(outcome.collapsed)
        self.assertEqual(len(outcome.scored), 2, "silêncio aqui seria pior que ordem imperfeita")

    def test_low_score_above_the_threshold_is_not_collapse(self):
        candidates = [hit("a", 0.9), hit("b", 0.9)]
        rr = FakeReranker(scores=[0.05, 0.02])  # 0.05 > COLLAPSE_MAX
        outcome = two_stage(candidates, "q", rr, DOCS_POLICY, TEXT)
        self.assertFalse(outcome.collapsed, "0.05 é relevância baixa, não colapso")

    def test_collapse_detection_can_be_turned_off_by_policy(self):
        policy = Policy(0.45, 0.58, 0.10, 5, veto=False, detect_collapse=False)
        rr = FakeReranker(scores=[0.0004, 0.0009])
        outcome = two_stage([hit("a", 0.60), hit("b", 0.46)], "q", rr, policy, TEXT)
        self.assertFalse(outcome.collapsed)


class TestTrace(unittest.TestCase):
    def test_records_the_scale_conversion(self):
        rr = FakeReranker(scores=[0.9, 0.5], was_logit=True)
        outcome = two_stage([hit("a", 0.9), hit("b", 0.9)], "q", rr,
                         Policy(0.45, 0.58, 0.10, 1), TEXT)
        self.assertTrue(outcome.scale_converted)

    def test_records_the_best_dense_score_even_with_no_result(self):
        outcome = two_stage([hit("a", 0.30)], "q", None, MEMORY_POLICY, TEXT)
        self.assertEqual(outcome.scored, [])
        self.assertAlmostEqual(outcome.best_dense, 0.30)

    def test_by_rerank_distinguishes_the_origin(self):
        rr = FakeReranker(scores=[0.9])
        com_ce = two_stage([hit("a", 0.5)], "q", rr, MEMORY_POLICY, TEXT)
        without_ce_scores = two_stage([hit("a", 0.9)], "q", None, MEMORY_POLICY, TEXT)
        self.assertTrue(com_ce.by_rerank)
        self.assertFalse(without_ce_scores.by_rerank)

    def test_the_query_reaches_the_reranker_and_so_do_the_texts(self):
        rr = FakeReranker(scores=[0.9, 0.8])
        two_stage([hit("a", 0.5, "texto A"), hit("b", 0.5, "texto B")],
                  "minha pergunta", rr, MEMORY_POLICY, TEXT)
        query, docs = rr.calls[0]
        self.assertEqual(query, "minha pergunta")
        self.assertEqual(docs, ["texto A", "texto B"])


class TestFloor(unittest.TestCase):
    def test_floor_relaxes_only_when_a_second_stage_exists(self):
        self.assertAlmostEqual(MEMORY_POLICY.floor_for(True), 0.45)
        self.assertAlmostEqual(MEMORY_POLICY.floor_for(False), 0.58)


class TestFakeContract(unittest.TestCase):
    def test_the_fake_satisfies_the_protocol_without_inheriting_anything(self):
        from core.ports import RerankModel
        self.assertIsInstance(FakeReranker(), RerankModel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
