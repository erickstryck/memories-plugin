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
TRIVIAL_WORDS = {
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
WORD_RE = re.compile(r"[\wÀ-ÿ_./-]{2,}")


def _only_confirmation(text: str) -> bool:
    """True quando TODAS as palavras do texto são de confirmação.

    Palavra por palavra, e não o texto inteiro comparado ao conjunto: comparar o
    texto inteiro deixava a checagem INALCANÇÁVEL, porque toda palavra do conjunto
    é mais curta que MIN_CHARS e o filtro de tamanho a pegava primeiro. O valor
    está justamente nas confirmações de várias palavras — "ok, pode continuar",
    "beleza, segue" — que passam do tamanho mínimo e não têm assunto nenhum.
    """
    words = [p for p in WORD_RE.findall(text.lower()) if p.strip()]
    if not words:
        return False

    return all(p in TRIVIAL_WORDS for p in words)


def skip_reason(prompt: str) -> str | None:
    """Devolve o motivo de não buscar, ou None quando vale buscar."""
    text = (prompt or "").strip()
    if SO_SLASH.fullmatch(text):
        return "comando sem argumento"
    if _only_confirmation(text):
        return "prompt trivial"
    if len(text) < MIN_CHARS:
        return "prompt curto"

    return None


def content_words(text: str) -> str:
    seen_map, output = set(), []
    for p in WORD_RE.findall(text.lower()):
        if p in STOPWORDS or p in seen_map:
            continue
        seen_map.add(p)
        output.append(p)

    return " ".join(output)


def longest_sentence(text: str) -> str:
    parts = [p.strip() for p in re.split(r"[.!?\n;]+", text) if p.strip()]
    if len(parts) < 2:
        return ""

    return max(parts, key=len)


def angles(prompt: str, limite_chars: int = 1500) -> list[str]:
    """Três ângulos do mesmo texto, para uma única chamada de embeddings.

    A busca é semântica, e ângulos diferentes do mesmo prompt pescam registros
    diferentes: o texto cru carrega a intenção completa; só as palavras de
    conteúdo empurram o vetor para o assunto em vez da estrutura da frase; e a
    frase mais longa costuma ser onde a pergunta de fato está, quando o prompt
    tem preâmbulo. Duplicatas são descartadas — embedar o mesmo texto duas vezes
    é gasto sem retorno.
    """
    base = (prompt or "")[:limite_chars]
    output = [base]
    content = content_words(base)
    if content and content != base.lower():
        output.append(content)
    longa = longest_sentence(base)
    if longa and longa != base and len(longa) > 20:
        output.append(longa)

    return output
