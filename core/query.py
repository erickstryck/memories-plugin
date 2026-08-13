"""Preparação da pergunta antes da busca.

Mora no núcleo porque não tem nada de host: transformar o texto do usuário em
consultas é a mesma coisa em qualquer agente, e é uma decisão de qualidade de
busca, não de apresentação.
"""
import re

#: Prompt abaixo disto não descreve assunto nenhum — buscar seria pagar uma ida à
#: rede para devolver ruído.
MIN_CHARS = 12

#: Continuações e confirmações. Não vale ida à rede: o assunto é o do turno
#: anterior, que já foi buscado quando chegou.
TRIVIAIS = {
    # confirmação
    "ok", "okay", "beleza", "blz", "sim", "isso", "certo", "correto", "exato",
    "exatamente", "claro", "perfeito", "top", "boa", "legal", "ótimo", "otimo",
    "bom", "tudo", "bem", "não", "nao", "yes", "no",
    # seguir adiante — com as inflexões, porque o filtro exige que TODAS as
    # palavras sejam triviais e uma inflexão faltando deixa a frase passar
    "vai", "vamos", "bora", "continua", "continue", "continuar", "segue",
    "seguir", "prossiga", "prosseguir", "pode", "manda", "mandar", "faz",
    "faça", "faca", "fazer", "next", "go",
    # agradecimento
    "obrigado", "obrigada", "valeu", "thanks", "thx", "ty", "please",
}

#: Palavras sem carga semântica. Removidas no ângulo "só conteúdo" para o vetor
#: não ficar diluído pela estrutura da frase.
STOPWORDS = {
    # pt
    "a", "à", "às", "ao", "aos", "o", "os", "as", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos", "por",
    "para", "pra", "pro", "com", "sem", "sobre", "entre", "que", "qual",
    "quais", "quando", "onde", "como", "porque", "se", "e", "ou",
    "mas", "então", "entao", "já", "ja", "não", "nao", "sim", "eu", "você",
    "voce", "ele", "ela", "eles", "elas", "nós", "nos", "meu", "minha", "seu",
    "sua", "isso", "isto", "esse", "essa", "este", "esta", "aquele", "aquela",
    "é", "foi", "ser", "está", "estão", "estao", "tem", "ter",
    "há", "ha", "vai", "vou", "fazer", "faz", "feito", "muito", "mais",
    "menos", "também", "tambem", "só", "so", "ainda", "agora", "aqui", "ali",
    "lá", "la", "me", "te", "lhe", "nesse", "neste", "dessa", "desta", "num",
    "numa", "pelo", "pela", "até", "ate", "depois", "antes", "quero", "queria",
    "gostaria", "favor", "pode", "poderia", "deve", "deveria", "preciso",
    # en
    "the", "an", "of", "in", "on", "at", "to", "for", "with", "without",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "can", "could", "should", "would", "will", "shall",
    "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "my",
    "your", "our", "their", "him", "her", "them", "what", "which",
    "when", "where", "how", "why", "who", "please", "just", "also", "very",
    "about", "into", "from", "as", "so", "not", "no", "yes",
}

SO_SLASH = re.compile(r"/[\w:-]+\Z")
PALAVRA = re.compile(r"[\wÀ-ÿ_./-]{2,}")


def _so_confirmacao(texto: str) -> bool:
    """True quando TODAS as palavras do texto são de confirmação.

    Palavra por palavra, e não o texto inteiro comparado ao conjunto: comparar o
    texto inteiro deixava a checagem INALCANÇÁVEL, porque toda palavra do conjunto
    é mais curta que MIN_CHARS e o filtro de tamanho a pegava primeiro. O valor
    está justamente nas confirmações de várias palavras — "ok, pode continuar",
    "beleza, segue" — que passam do tamanho mínimo e não têm assunto nenhum.
    """
    palavras = [p for p in PALAVRA.findall(texto.lower()) if p.strip()]
    if not palavras:
        return False

    return all(p in TRIVIAIS for p in palavras)


def motivo_para_pular(prompt: str) -> str | None:
    """Devolve o motivo de não buscar, ou None quando vale buscar."""
    texto = (prompt or "").strip()
    if SO_SLASH.fullmatch(texto):
        return "comando sem argumento"
    if _so_confirmacao(texto):
        return "prompt trivial"
    if len(texto) < MIN_CHARS:
        return "prompt curto"

    return None


def palavras_conteudo(texto: str) -> str:
    vistas, saida = set(), []
    for p in PALAVRA.findall(texto.lower()):
        if p in STOPWORDS or p in vistas:
            continue
        vistas.add(p)
        saida.append(p)

    return " ".join(saida)


def frase_mais_longa(texto: str) -> str:
    partes = [p.strip() for p in re.split(r"[.!?\n;]+", texto) if p.strip()]
    if len(partes) < 2:
        return ""

    return max(partes, key=len)


def angulos(prompt: str, limite_chars: int = 1500) -> list[str]:
    """Três ângulos do mesmo texto, para uma única chamada de embeddings.

    A busca é semântica, e ângulos diferentes do mesmo prompt pescam registros
    diferentes: o texto cru carrega a intenção completa; só as palavras de
    conteúdo empurram o vetor para o assunto em vez da estrutura da frase; e a
    frase mais longa costuma ser onde a pergunta de fato está, quando o prompt
    tem preâmbulo. Duplicatas são descartadas — embedar o mesmo texto duas vezes
    é gasto sem retorno.
    """
    base = (prompt or "")[:limite_chars]
    saida = [base]
    conteudo = palavras_conteudo(base)
    if conteudo and conteudo != base.lower():
        saida.append(conteudo)
    longa = frase_mais_longa(base)
    if longa and longa != base and len(longa) > 20:
        saida.append(longa)

    return saida
