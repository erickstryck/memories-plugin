"""Fatiamento de documento em trechos indexáveis.

Funções puras, sem I/O e sem rede — é aqui que mora a decisão de qualidade do
índice, e é o único módulo que dá para testar inteiro sem infra.

POR QUE NÃO JANELA FIXA DE LINHAS: cortar no meio de uma função ou de uma seção
faz o vetor do trecho ser a média de dois assuntos, e um vetor médio não casa bem
com pergunta nenhuma. O mesmo motivo pelo qual um bom registro de memória é um
fato atômico e não um parágrafo. Então quebramos primeiro nas fronteiras que o
arquivo já oferece e só caímos na janela fixa quando um bloco estoura sozinho.
"""
import re
from dataclasses import dataclass

TARGET_CHARS = 2400   # ~800 tokens: o trecho se explica sozinho sem virar média
HARD_MAX_CHARS = 6000  # o par (pergunta, trecho) tem de caber com folga no reranker
OVERLAP_LINES = 2      # só na janela fixa, para não perder o que cai na costura

HEADING = re.compile(r"^\s{0,3}#{1,6}\s")

# Sufixos cujo conteúdo dá para reler por região com um leitor de arquivo, o que
# permite o modo LOCALIZADOR: a busca devolve `arquivo:linhas` e quem consome lê o
# conteúdo ATUAL, sem risco de operar sobre foto vencida. O resto vira modo FOTO.
LOCATABLE_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".sql", ".graphql",
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".bash",
    ".zsh", ".fish", ".lua", ".vim", ".el", ".scala", ".kt", ".swift", ".m",
    ".gradle", ".dockerfile", ".tf", ".hcl", ".proto", ".xml", ".html", ".css",
    ".scss", ".diff", ".patch", ".gremlin",
}


@dataclass(frozen=True)
class Chunk:
    start_line: int  # 1-based, inclusivo
    end_line: int    # 1-based, inclusivo
    text: str

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1


def is_probably_binary(amostra: str) -> bool:
    return "\x00" in amostra


def mode_for_suffix(sufixo: str) -> str:
    """`locator` quando dá para reler a região no arquivo; `snapshot` quando não."""
    return "locator" if sufixo.lower() in LOCATABLE_SUFFIXES else "snapshot"


def split_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Fronteiras estruturais. Devolve [(início, fim)] 0-based, fim exclusivo.

    As três fronteiras reconhecidas, todas baratas e sem parser por linguagem:
      - heading de markdown
      - início de parágrafo (linha com conteúdo logo depois de linha em branco)
      - definição de topo em código (linha na coluna 0 depois de linha indentada)
    """
    if not lines:
        return []
    limites = {0}
    for i in range(1, len(lines)):
        atual = lines[i]
        anterior = lines[i - 1]
        if HEADING.match(atual):
            limites.add(i)
            continue
        if not anterior.strip() and atual.strip():
            limites.add(i)
            continue
        comeca_na_coluna_zero = atual[:1] not in ("", " ", "\t", "\n")
        anterior_indentado = anterior[:1] in (" ", "\t")
        if comeca_na_coluna_zero and anterior_indentado:
            limites.add(i)
    ordenados = sorted(limites) + [len(lines)]

    return [(ordenados[i], ordenados[i + 1]) for i in range(len(ordenados) - 1)]


def _window(lines: list[str], ini: int, fim: int, target: int) -> list[tuple[int, int]]:
    """Janela fixa com sobreposição, para bloco que estoura o teto sozinho."""
    janelas = []
    passo = ini
    while passo < fim:
        acumulado = 0
        cursor = passo
        while cursor < fim and acumulado < target:
            acumulado += len(lines[cursor])
            cursor += 1
        janelas.append((passo, cursor))
        if cursor >= fim:
            break
        passo = max(passo + 1, cursor - OVERLAP_LINES)

    return janelas


def pack_chunks(lines: list[str], target: int = TARGET_CHARS,
                hard_max: int = HARD_MAX_CHARS) -> list[Chunk]:
    """Agrupa blocos estruturais em trechos de até `target` chars.

    Trecho vazio ou só com espaço em branco é descartado — indexar isso gasta uma
    chamada de embedding e polui a busca com um vetor sem significado.
    """
    if not lines:
        return []
    chunks: list[Chunk] = []

    def emitir(ini: int, fim: int) -> None:
        texto = "".join(lines[ini:fim]).strip("\n")
        if texto.strip():
            chunks.append(Chunk(ini + 1, fim, texto))

    aberto: int | None = None
    tamanho = 0
    for bi, bf in split_blocks(lines):
        bloco = sum(len(l) for l in lines[bi:bf])
        if bloco > hard_max:
            if aberto is not None:
                emitir(aberto, bi)
                aberto, tamanho = None, 0
            for ji, jf in _window(lines, bi, bf, target):
                emitir(ji, jf)
            continue
        if aberto is None:
            aberto, tamanho = bi, bloco
            continue
        if tamanho + bloco > target:
            emitir(aberto, bi)
            aberto, tamanho = bi, bloco
            continue
        tamanho += bloco
    if aberto is not None:
        emitir(aberto, len(lines))

    return chunks


def chunk_text(conteudo: str, target: int = TARGET_CHARS,
               hard_max: int = HARD_MAX_CHARS) -> list[Chunk]:
    return pack_chunks(conteudo.splitlines(keepends=True), target, hard_max)
