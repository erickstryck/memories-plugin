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


#: Above this share of replacement characters, the decode failed rather than succeeded.
#: Generous on purpose: a legitimate document may carry a stray bad byte, and refusing it
#: costs the user a file they wanted indexed.
REPLACEMENT_RATIO = 0.05


def is_probably_binary(sample: str) -> bool:
    """A NUL byte is the strong signal, but not the only one.

    The sample arrives already decoded with `errors="replace"`, so binary content that
    happens to contain no NUL survived the check and got indexed — 20 KB of high random
    bytes produced chunks of U+FFFD, spending embedding calls on noise and polluting the
    archive with vectors that mean nothing. The replacement ratio catches that without
    refusing a valid file over one bad byte.
    """
    if "\x00" in sample:
        return True
    if not sample:
        return False

    return sample.count("\ufffd") / len(sample) > REPLACEMENT_RATIO


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


#: Splits on "\n" and NOTHING else, which is what every reader of a file means by a line.
#:
#: `str.splitlines()` is the obvious call and it is wrong here. It also breaks on \x0b
#: \x0c \x1c \x1d \x1e \x85     — and no file reader, editor, `sed` or `wc -l`
#: treats any of those as ending a line. Every one of them therefore shifted our line
#: numbers by one relative to the file on disk, silently and cumulatively.
#:
#: That matters because the whole point of locator mode is the promise that
#: `path:start-end` reproduces the chunk: the search returns a location, and the reader
#: goes and reads THOSE lines to work on the current content. A form feed is the
#: conventional page break in C and Emacs-formatted sources, and \x1c/\x1d are the ASCII
#: File and Group Separators legacy exports use as delimiters — with `.py`, `.csv` and
#: `.log` all in LOCATABLE_SUFFIXES. Measured on a 39-line file with one form feed: half
#: the chunks reported the wrong first line, and the last one claimed line 40.
#:
#: The two tests that were supposed to guard the promise both sliced with `.splitlines()`
#: themselves, so they agreed with the bug by construction.
_LINE = re.compile(r"[^\n]*\n|[^\n]+")


def split_lines(content: str) -> list[str]:
    """Lines with their terminators, joinable back into the original text."""
    return _LINE.findall(content)


def chunk_text(content: str, target: int = TARGET_CHARS,
               hard_max: int = HARD_MAX_CHARS) -> list[Chunk]:
    return pack_chunks(split_lines(content), target, hard_max)
