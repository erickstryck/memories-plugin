"""Slicing a document into indexable chunks.

Pure functions, no I/O and no network — this is where the index-quality decision
lives, and it is the only module that can be tested end to end with no infra.

WHY NOT A FIXED LINE WINDOW: cutting in the middle of a function or a section makes
the chunk's vector the average of two subjects, and an average vector matches no
question well. The same reason a good memory record is an atomic fact and not a
paragraph. So we break first on the boundaries the file already offers, and only fall
back to the fixed window when a single block overflows on its own.
"""
import re
from dataclasses import dataclass

TARGET_CHARS = 2400   # ~800 tokens: the chunk explains itself without becoming an average
HARD_MAX_CHARS = 6000  # the (query, chunk) pair has to fit the reranker with room to spare
OVERLAP_LINES = 2      # fixed window only, so nothing falling on the seam is lost

HEADING = re.compile(r"^\s{0,3}#{1,6}\s")

# Suffixes whose content can be re-read by region with a file reader, which enables
# LOCATOR mode: the search returns `file:lines` and the consumer reads the CURRENT
# content, with no risk of operating on a stale snapshot. Everything else becomes
# SNAPSHOT mode.
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
    start_line: int  # 1-based, inclusive
    end_line: int    # 1-based, inclusive
    text: str

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1


def is_probably_binary(sample: str) -> bool:
    return "\x00" in sample


def mode_for_suffix(suffix: str) -> str:
    """`locator` when the region can be re-read from the file; `snapshot` when not."""
    return "locator" if suffix.lower() in LOCATABLE_SUFFIXES else "snapshot"


def split_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Structural boundaries. Returns [(start, end)] 0-based, end exclusive.

    The three recognized boundaries, all cheap and with no per-language parser:
      - a markdown heading
      - the start of a paragraph (a line with content right after a blank line)
      - a top-level definition in code (a line at column 0 after an indented line)
    """
    if not lines:
        return []
    boundaries = {0}
    for i in range(1, len(lines)):
        current = lines[i]
        previous = lines[i - 1]
        if HEADING.match(current):
            boundaries.add(i)
            continue
        if not previous.strip() and current.strip():
            boundaries.add(i)
            continue
        starts_at_column_zero = current[:1] not in ("", " ", "\t", "\n")
        previous_is_indented = previous[:1] in (" ", "\t")
        if starts_at_column_zero and previous_is_indented:
            boundaries.add(i)
    ordered = sorted(boundaries) + [len(lines)]

    return [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]


def _window(lines: list[str], start: int, end: int, target: int) -> list[tuple[int, int]]:
    """Fixed window with overlap, for a block that overflows the ceiling on its own."""
    windows = []
    step = start
    while step < end:
        accumulated = 0
        cursor = step
        while cursor < end and accumulated < target:
            accumulated += len(lines[cursor])
            cursor += 1
        windows.append((step, cursor))
        if cursor >= end:
            break
        step = max(step + 1, cursor - OVERLAP_LINES)

    return windows


def pack_chunks(lines: list[str], target: int = TARGET_CHARS,
                hard_max: int = HARD_MAX_CHARS) -> list[Chunk]:
    """Groups structural blocks into chunks of up to `target` chars.

    An empty or whitespace-only chunk is discarded — indexing that spends an embedding
    call and pollutes the search with a meaningless vector.
    """
    if not lines:
        return []
    chunks: list[Chunk] = []

    def emit(start: int, end: int) -> None:
        text = "".join(lines[start:end]).strip("\n")
        if text.strip():
            chunks.append(Chunk(start + 1, end, text))

    open_start: int | None = None
    size = 0
    for bi, bf in split_blocks(lines):
        block = sum(len(l) for l in lines[bi:bf])
        if block > hard_max:
            if open_start is not None:
                emit(open_start, bi)
                open_start, size = None, 0
            for ji, jf in _window(lines, bi, bf, target):
                emit(ji, jf)
            continue
        if open_start is None:
            open_start, size = bi, block
            continue
        if size + block > target:
            emit(open_start, bi)
            open_start, size = bi, block
            continue
        size += block
    if open_start is not None:
        emit(open_start, len(lines))

    return chunks


def chunk_text(content: str, target: int = TARGET_CHARS,
               hard_max: int = HARD_MAX_CHARS) -> list[Chunk]:
    return pack_chunks(content.splitlines(keepends=True), target, hard_max)
