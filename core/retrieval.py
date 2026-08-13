"""Pipeline de recuperação em dois estágios, em UM só lugar.

Antes existiam três implementações da mesma ideia — memória, documentos e o
diagnóstico —, cada uma com a sua versão das regras. Isso já custou concreto: a
normalização de escala do re-rank existiu num consumidor e não no outro, e o
segundo herdaria o bug de volta no dia em que precisasse dela.

O ALGORITMO é comum; o que difere entre os consumidores é POLÍTICA, e política é
parâmetro:

    memória     -> o re-rank VETA. Precisão importa mais que alcance: falso
                   positivo entra no contexto do agente e o polui.
    documentos  -> o re-rank ORDENA e não veta. Quem pergunta já escolheu o
                   documento; silêncio é pior que ordem imperfeita.

Duas assimetrias medidas moldam o desenho, e ignorar qualquer uma degrada em
silêncio:

1. Se o segundo estágio NÃO roda, o piso permissivo do primeiro fica sem quem o
   limpe. É obrigatório voltar ao corte estrito — senão o modo COM re-rank fica
   pior que o modo sem, que é o oposto da intenção.

2. O cross-encoder COLAPSA em par cross-lingual (medido: 0.2073 com pergunta em
   inglês contra 0.0004 em português, no mesmo documento em inglês; o estágio denso
   ficou indiferente, 0.475 contra 0.460). Quando colapsa, a ORDEM dele também é
   ruído — então não basta parar de filtrar, é preciso detectar e voltar à ordem
   densa.
"""
from dataclasses import dataclass, field
from typing import Any, Callable

#: Melhor score do cross-encoder abaixo disto indica colapso, não irrelevância.
#: Separador largo: colapso medido em 4e-4, casos saudáveis em 0.21 e 0.53.
COLLAPSE_MAX = 0.01

#: Procedência do score de cada resultado. Quem apresenta precisa distinguir:
#: ordem densa NÃO é veredito de relevância, é proximidade de vetor.
CE = "CE"           # julgado pelo cross-encoder, acima do corte
CE_WEAK = "CE?"    # julgado, abaixo do corte — entregue sem veto, marcado
DENSE = "dense"     # sem julgamento do cross-encoder


def default_score(hit: Any) -> float:
    return float(hit.get("score") or 0.0)


@dataclass(frozen=True)
class Policy:
    """As decisões que variam por consumidor.

    `veto` é a única realmente estrutural: com ele o segundo estágio pode ELIMINAR
    candidatos; sem ele, só reordena. As duas escolhas são defensáveis, mas por
    motivos opostos, então explicitar evita que alguém "unifique" as duas por
    engano no futuro.
    """
    dense_floor: float          # piso do primeiro estágio quando há segundo
    strict_floor: float         # piso quando o primeiro está SOZINHO
    min_score: float            # corte do segundo estágio
    max_results: int
    veto: bool = True
    detect_collapse: bool = True
    order_matters: bool = False
    """Se a ORDEM do resultado é o produto, ou só o conjunto.

    Memória: os resultados vão TODOS para o contexto de uma vez, então reordenar não
    muda nada — o segundo estágio só se paga para escolher ou filtrar.

    Documentos: o resultado é uma lista lida de cima para baixo, e a ordem é
    exatamente o que se está comprando. Aí o segundo estágio vale mesmo quando tudo
    cabe e tudo passa.

    Aplicar a mesma regra aos dois foi um erro de desenho que só apareceu quando o
    pipeline virou um só e ganhou teste com dublê."""

    def floor_for(self, has_reranker: bool) -> float:
        """Piso do primeiro estágio. Só relaxa quando há segundo estágio para limpar."""
        return self.dense_floor if has_reranker else self.strict_floor


@dataclass(frozen=True)
class Scored:
    """Um resultado com o score FINAL e de onde ele veio."""
    item: Any
    score: float
    origin: str

    @property
    def is_weak(self) -> bool:
        return self.origin == CE_WEAK


@dataclass
class Outcome:
    """Resultados escolhidos e o RASTRO de como se chegou neles.

    O rastro não é luxo de log: quem consome precisa saber se o veredito é do
    cross-encoder ou só proximidade de vetor, e precisa poder dizer ao usuário que
    a ordem virou densa porque o julgamento colapsou.
    """
    scored: list[Scored] = field(default_factory=list)
    candidates: int = 0
    best_dense: float = 0.0
    best_rerank: float = 0.0
    reranked: bool = False
    collapsed: bool = False
    scale_converted: bool = False
    rerank_error: str | None = None
    dropped: int = 0
    """Candidatos que o segundo estágio nem viu, por teto de pares.

    Era calculado pelo cliente e lido por ninguém. Candidato descartado sem
    julgamento pode incluir um que passaria o corte estrito — quem apresenta precisa
    poder dizer que a lista não é exaustiva."""

    @property
    def items(self) -> list[Any]:
        return [s.item for s in self.scored]

    @property
    def by_rerank(self) -> bool:
        return any(s.origin in (CE, CE_WEAK) for s in self.scored)


def fuse_by_id(batches: list[list[Any]], id_of: Callable[[Any], str],
               score_of: Callable[[Any], float] = default_score) -> list[Any]:
    """Funde resultados de vários vetores, mantendo o MAIOR score de cada id.

    Ângulos diferentes da mesma pergunta pescam registros diferentes; um registro
    que aparece em dois ângulos não deve ser penalizado pelo pior deles.
    """
    fused: dict[str, Any] = {}
    for batch in batches:
        for hit in batch:
            key = id_of(hit)
            current = fused.get(key)
            if current is None or score_of(hit) > score_of(current):
                fused[key] = hit

    return sorted(fused.values(), key=lambda h: -score_of(h))


def needs_rerank(candidates: list[Any], policy: Policy,
                 score_of: Callable[[Any], float] = default_score) -> bool:
    """O segundo estágio tem dois papéis, e só o segundo obriga a chamada.

    ESCOLHER: há mais candidatos que vagas.
    FILTRAR: existe candidato na faixa permissiva, que só entrou porque alguém ia
    julgá-lo.

    Sem nenhum dos dois, reordenar não muda o que sai — e a chamada é trabalho
    inútil pago em latência, num caminho disparado a cada interação do usuário.

    A EXCEÇÃO é quando a ordem é o produto (`order_matters`): aí ORDENAR já é o
    terceiro motivo, e basta ter mais de um candidato.
    """
    if not candidates:
        return False
    if policy.order_matters and len(candidates) > 1:
        return True
    if len(candidates) > policy.max_results:
        return True

    return any(score_of(c) < policy.strict_floor for c in candidates)


def two_stage(candidates: list[Any], query: str, reranker, policy: Policy,
              text_of: Callable[[Any], str],
              score_of: Callable[[Any], float] = default_score) -> Outcome:
    """Aplica o segundo estágio sobre candidatos JÁ recuperados pelo primeiro.

    Recebe os candidatos em vez de buscá-los: o primeiro estágio difere entre
    consumidores (uma coleção com vários ângulos da pergunta, ou várias coleções com
    um vetor), e forçar os dois no mesmo molde criaria parâmetros que só um usa.

    Nunca muta a entrada.
    """
    outcome = Outcome(candidates=len(candidates),
                   best_dense=score_of(candidates[0]) if candidates else 0.0)
    if not candidates:
        return outcome

    if reranker is None or not needs_rerank(candidates, policy, score_of):
        outcome.scored = _strict_cut(candidates, policy, score_of)

        return outcome

    pairs, info = reranker.rank(query, [text_of(c) for c in candidates])
    outcome.reranked = bool(info.get("ok"))
    outcome.scale_converted = bool(info.get("era_logit"))
    outcome.rerank_error = info.get("erro")
    outcome.dropped = int(info.get("descartados") or 0)
    outcome.best_rerank = max((s for _, s in pairs), default=0.0)

    # Sem julgamento, o piso permissivo do primeiro estágio ficou sem quem o limpe.
    if not outcome.reranked:
        outcome.scored = _strict_cut(candidates, policy, score_of)

        return outcome

    # Colapso: o score é baixo por incompatibilidade de idioma, não por
    # irrelevância. A ordem do cross-encoder também é ruído aqui.
    if policy.detect_collapse and pairs and outcome.best_rerank < COLLAPSE_MAX:
        # Colapso é o julgamento sendo DESCARTADO, então o piso permissivo volta a
        # não ter quem o limpe — exatamente como quando o re-rank falha. Devolver a
        # ordem densa sem reaplicar o corte estrito deixava passar candidato que o
        # modo SEM re-rank nunca devolveria, que é o defeito que este pipeline
        # existe para não ter.
        outcome.collapsed = True
        outcome.scored = _strict_cut(candidates, policy, score_of)

        return outcome

    valid_keys = [(candidates[i], s) for i, s in pairs if 0 <= i < len(candidates)]
    above = [Scored(c, s, CE) for c, s in valid_keys if s >= policy.min_score]
    below = [Scored(c, s, CE_WEAK) for c, s in valid_keys if s < policy.min_score]
    chosen = above if policy.veto else above + below
    outcome.scored = chosen[:policy.max_results]

    return outcome


def _strict_cut(candidates: list[Any], policy: Policy,
             score_of: Callable[[Any], float]) -> list[Scored]:
    """Reaplica o corte estrito. Chamado sempre que o segundo estágio não julgou."""
    return [Scored(c, score_of(c), DENSE) for c in candidates
            if score_of(c) >= policy.strict_floor][:policy.max_results]
