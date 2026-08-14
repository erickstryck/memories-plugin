"""Preparing the question before the search.

It lives in the core because there is nothing host-specific about it: turning the
user's text into queries is the same thing in any agent, and it is a search-quality
decision, not a presentation one.
"""
import re

#: A prompt shorter than this describes no subject at all — searching would pay a
#: network round trip to get noise back.
MIN_CHARS = 12

#: Continuations and acknowledgements. Not worth a network round trip: the subject is
#: the previous turn's, which was already searched when it arrived.
#:
#: The CONTENTS of this set and of STOPWORDS stay in Portuguese on purpose. They are
#: linguistic data matched against what the user actually types, not prose —
#: translating them would silently disable both filters.
TRIVIAL_WORDS = {
    # acknowledgement
    "ok", "okay", "beleza", "blz", "sim", "isso", "certo", "correto", "exato",
    "exatamente", "claro", "perfeito", "top", "boa", "legal", "ótimo", "otimo",
    "bom", "tudo", "bem", "não", "nao", "yes", "no",
    # carry on — with the inflections, because the filter requires ALL words to be
    # trivial and one missing inflection lets the whole phrase through
    "vai", "vamos", "bora", "continua", "continue", "continuar", "segue",
    "seguir", "prossiga", "prosseguir", "pode", "manda", "mandar", "faz",
    "faça", "faca", "fazer", "next", "go",
    # thanks
    "obrigado", "obrigada", "valeu", "thanks", "thx", "ty", "please",
}

#: Words with no semantic load. Removed in the "content only" angle so the vector is
#: not diluted by the structure of the sentence.
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

BARE_SLASH_CMD = re.compile(r"/[\w:-]+\Z")
WORD_RE = re.compile(r"[\wÀ-ÿ_./-]{2,}")


def _only_confirmation(text: str) -> bool:
    """True when EVERY word in the text is an acknowledgement.

    Word by word, and not the whole text compared against the set: comparing the whole
    text made the check UNREACHABLE, because every word in the set is shorter than
    MIN_CHARS and the length filter caught it first. The value is precisely in
    multi-word acknowledgements — "ok, pode continuar", "beleza, segue" — which clear
    the minimum length and carry no subject at all.
    """
    words = [p for p in WORD_RE.findall(text.lower()) if p.strip()]
    if not words:
        return False
    # STOPWORDS count as trivial here, and composing the two sets is the point. They were
    # applied separately before, so an acknowledgement holding one structural word slipped
    # through: measured in production, "sim, pode ser" — `sim` and `pode` are TRIVIAL_WORDS,
    # `ser` is not, but it IS a STOPWORD, i.e. the package already asserts it carries no
    # meaning. The prompt cleared the 12-char minimum, bought a full search, and injected a
    # memory at CE 0.110 for a three-word "go ahead".
    #
    # This cannot make the filter too aggressive: a word in neither set is content, and one
    # content word is enough to keep the search. "ok, pode continuar o poll" still searches,
    # because `poll` is in neither.
    meaningful = [p for p in words if p not in STOPWORDS]
    if not meaningful:
        return True

    return all(p in TRIVIAL_WORDS for p in meaningful)


def skip_reason(prompt: str) -> str | None:
    """Returns the reason not to search, or None when searching is worth it."""
    text = (prompt or "").strip()
    if BARE_SLASH_CMD.fullmatch(text):
        return "command with no argument"
    if _only_confirmation(text):
        return "trivial prompt"
    if len(text) < MIN_CHARS:
        return "short prompt"

    return None


def content_words(text: str) -> str:
    seen, output = set(), []
    for p in WORD_RE.findall(text.lower()):
        if p in STOPWORDS or p in seen:
            continue
        seen.add(p)
        output.append(p)

    return " ".join(output)


def longest_sentence(text: str) -> str:
    parts = [p.strip() for p in re.split(r"[.!?\n;]+", text) if p.strip()]
    if len(parts) < 2:
        return ""

    return max(parts, key=len)


def angles(prompt: str, char_limit: int = 1500) -> list[str]:
    """Three angles on the same text, for a single embeddings call.

    The search is semantic, and different angles on the same prompt catch different
    records: the raw text carries the full intent; the content words alone push the
    vector toward the subject rather than the shape of the sentence; and the longest
    sentence is usually where the actual question is, when the prompt has a preamble.
    Duplicates are dropped — embedding the same text twice is spend with no return.
    """
    base = (prompt or "")[:char_limit]
    output = [base]
    content = content_words(base)
    if content and content != base.lower():
        output.append(content)
    longest = longest_sentence(base)
    if longest and longest != base and len(longest) > 20:
        output.append(longest)

    return output
