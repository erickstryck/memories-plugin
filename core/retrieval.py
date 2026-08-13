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
CE_FRACO = "CE?"    # julgado, abaixo do corte — entregue sem veto, marcado
DENSO = "denso"     # sem julgamento do cross-encoder


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

    def floor_for(self, tem_reranker: bool) -> float:
        """Piso do primeiro estágio. Só relaxa quando há segundo estágio para limpar."""
        return self.dense_floor if tem_reranker else self.strict_floor


@dataclass(frozen=True)
class Scored:
    """Um resultado com o score FINAL e de onde ele veio."""
    item: Any
    score: float
    origin: str

    @property
    def is_weak(self) -> bool:
        return self.origin == CE_FRACO


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

    @property
    def items(self) -> list[Any]:
        return [s.item for s in self.scored]

    @property
    def by_rerank(self) -> bool:
        return any(s.origin in (CE, CE_FRACO) for s in self.scored)


def fuse_by_id(lotes: list[list[Any]], id_de: Callable[[Any], str],
               score_de: Callable[[Any], float] = default_score) -> list[Any]:
    """Funde resultados de vários vetores, mantendo o MAIOR score de cada id.

    Ângulos diferentes da mesma pergunta pescam registros diferentes; um registro
    que aparece em dois ângulos não deve ser penalizado pelo pior deles.
    """
    fundido: dict[str, Any] = {}
    for lote in lotes:
        for hit in lote:
            chave = id_de(hit)
            atual = fundido.get(chave)
            if atual is None or score_de(hit) > score_de(atual):
                fundido[chave] = hit

    return sorted(fundido.values(), key=lambda h: -score_de(h))


def needs_rerank(candidatos: list[Any], policy: Policy,
                 score_de: Callable[[Any], float] = default_score) -> bool:
    """O segundo estágio tem dois papéis, e só o segundo obriga a chamada.

    ESCOLHER: há mais candidatos que vagas.
    FILTRAR: existe candidato na faixa permissiva, que só entrou porque alguém ia
    julgá-lo.

    Sem nenhum dos dois, reordenar não muda o que sai — e a chamada é trabalho
    inútil pago em latência, num caminho disparado a cada interação do usuário.

    A EXCEÇÃO é quando a ordem é o produto (`order_matters`): aí ORDENAR já é o
    terceiro motivo, e basta ter mais de um candidato.
    """
    if not candidatos:
        return False
    if policy.order_matters and len(candidatos) > 1:
        return True
    if len(candidatos) > policy.max_results:
        return True

    return any(score_de(c) < policy.strict_floor for c in candidatos)


def two_stage(candidatos: list[Any], query: str, reranker, policy: Policy,
              texto_de: Callable[[Any], str],
              score_de: Callable[[Any], float] = default_score) -> Outcome:
    """Aplica o segundo estágio sobre candidatos JÁ recuperados pelo primeiro.

    Recebe os candidatos em vez de buscá-los: o primeiro estágio difere entre
    consumidores (uma coleção com vários ângulos da pergunta, ou várias coleções com
    um vetor), e forçar os dois no mesmo molde criaria parâmetros que só um usa.

    Nunca muta a entrada.
    """
    fora = Outcome(candidates=len(candidatos),
                   best_dense=score_de(candidatos[0]) if candidatos else 0.0)
    if not candidatos:
        return fora

    if reranker is None or not needs_rerank(candidatos, policy, score_de):
        fora.scored = _estrito(candidatos, policy, score_de)

        return fora

    pares, info = reranker.rank(query, [texto_de(c) for c in candidatos])
    fora.reranked = bool(info.get("ok"))
    fora.scale_converted = bool(info.get("era_logit"))
    fora.rerank_error = info.get("erro")
    fora.best_rerank = max((s for _, s in pares), default=0.0)

    # Sem julgamento, o piso permissivo do primeiro estágio ficou sem quem o limpe.
    if not fora.reranked:
        fora.scored = _estrito(candidatos, policy, score_de)

        return fora

    # Colapso: o score é baixo por incompatibilidade de idioma, não por
    # irrelevância. A ordem do cross-encoder também é ruído aqui.
    if policy.detect_collapse and pares and fora.best_rerank < COLLAPSE_MAX:
        fora.collapsed = True
        fora.scored = [Scored(c, score_de(c), DENSO)
                       for c in candidatos[:policy.max_results]]

        return fora

    validos = [(candidatos[i], s) for i, s in pares if 0 <= i < len(candidatos)]
    acima = [Scored(c, s, CE) for c, s in validos if s >= policy.min_score]
    abaixo = [Scored(c, s, CE_FRACO) for c, s in validos if s < policy.min_score]
    escolhidos = acima if policy.veto else acima + abaixo
    fora.scored = escolhidos[:policy.max_results]

    return fora


def _estrito(candidatos: list[Any], policy: Policy,
             score_de: Callable[[Any], float]) -> list[Scored]:
    """Reaplica o corte estrito. Chamado sempre que o segundo estágio não julgou."""
    return [Scored(c, score_de(c), DENSO) for c in candidatos
            if score_de(c) >= policy.strict_floor][:policy.max_results]
