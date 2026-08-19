"""Would reading this file cost more context than it is worth?

The criterion is RELATIVE to what remains, never the file's size alone: 171k tokens is
irrelevant in a freshly opened 1M window and fatal with 100k left. Both halves of the rule
come from the same measured incident — a 586 KB JSON read straight into context when a
search over 258 indexed chunks would have answered in ~6k tokens.

This module decides and nothing else. It does not learn the budget (the host adapters do),
it does not index (the model does, after reading the message), and it touches the network
never. That is what makes it testable with a fabricated Budget and no infrastructure.
"""
import os
from dataclasses import dataclass

from core.chunk import is_probably_binary

#: Fraction of the window that must REMAIN after the read.
FLOOR_PCT = 0.20
#: Fraction of the free space a single file may take.
SHARE_PCT = 0.40
#: Same ratio hermes uses in `agent/context_breakdown.py::_chars_to_tokens`. Bytes are not
#: characters under UTF-8, but the error is far below the precision a threshold needs.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Budget:
    """What the host could tell us about the context right now.

    `exact` is not decoration: claude-code reports measured token usage, hermes can only
    sum message bodies. The message says which one it is, because a number that looks
    precise and is a guess is worse than a guess that admits it.
    """
    window: int
    used: int
    exact: bool


@dataclass(frozen=True)
class Verdict:
    block: bool
    reason: str
    cost: int
    free: int


#: Enough bytes to catch a NUL without paying for the file.
_SNIFF_BYTES = 8192

# THE ESCAPE MARKER IS NOT DEFINED HERE, AND ITS ABSENCE IS LOAD-BEARING. The literal the
# user types to force a read through is CONFIGURABLE (the spec, "A palavra de escape" —
# the escape word),
# because it is text a user types and there are domains where `--full` occurs on its own —
# anyone working on a CLI with a `--full` flag would unlock the guard by accident, which is
# the false positive a literal marker exists to avoid.
#
# Configurable text used in TWO places — the message below, which teaches the user what to
# type, and the detection in each adapter, which reads the last user turn — is the shape
# ruling F5 settles: one owner. A default kept in this module would be a second one, free to
# teach a marker the detection rejects, and a guard that advertises a way out it then
# refuses is worse than one with no way out at all. So `decide` prints the marker its CALLER
# detects with, or none.


def sample_of(path: str) -> bytes | None:
    """The ONE bounded read of the file, or None when it could not be read at all.

    One read, two answers: is this indexable, and how long are its lines. They used to be
    two separate opens' worth of work — and this runs before EVERY file read, so a second
    open of the same file is a regression, not a detail. `None` is not an error either: a
    file we cannot open is a file we have nothing to offer about, and every caller turns
    that into an allow.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(_SNIFF_BYTES)
    except OSError:
        return None


def indexable_sample(sample: bytes | None) -> bool:
    """Whether `docs_index` could do anything with a file that starts like this.

    Delegates the binary/text call to `chunk.is_probably_binary` — the same policy
    `docs_index` itself applies to the file — instead of re-deciding it here. A NUL byte
    alone is not enough: high-entropy content with no NUL decodes into a wall of U+FFFD
    replacement characters, which is exactly the measured incident that policy exists to
    catch (see `chunk.is_probably_binary`'s docstring). Duplicating a weaker check here
    would let this guard wave through a file `docs_index` then rejects.
    """
    if sample is None:
        return False

    return not is_probably_binary(sample.decode("utf-8", errors="replace"))


def is_indexable(path: str) -> bool:
    """The same question asked about a PATH — one open, for callers outside `decide`."""
    return indexable_sample(sample_of(path))


def loaded_bytes(size: int, sample: bytes | None, *, read_lines=None,
                 read_bytes=None) -> int:
    """Bytes ONE read of this file would actually put in the context.

    THE DEFECT THIS EXISTS FOR. Pricing the whole file made `Read(a_684KB_file, limit=50)`
    cost 171k tokens and get BLOCKED, when the read would have loaded about 500 — a block
    based on a number that was simply wrong, which is the one failure this feature must
    never produce.

    ONE RULE, and it subsumes the special case it replaced. "If the request carries a
    limit, allow" would get `limit=50` right and `limit=1000000` wrong — that second one IS
    the whole file. Taking the MINIMUM of what the file holds and what the request can pull
    is right in both directions, and it is right for a request that carries no limit at all,
    because every host has a ceiling of its own.

    WHERE `bytes_per_line` COMES FROM, AND WHY IT COSTS NOTHING. The sample was already read
    to decide whether the file is binary; the newlines in it are free. A sample with NO
    newline at all — a minified bundle, one enormous line — makes the average the whole
    sample, so `read_lines` lines of it exceed the file and the minimum falls back to the
    file itself. That is the right answer for a one-line file: a line-limited read of it
    loads all of it.

    Both ceilings are OPTIONAL and independent: hosts that cap by lines pass `read_lines`,
    hosts that also truncate by bytes pass `read_bytes`, and a caller that passes neither
    gets the whole file, exactly as before.
    """
    ceilings = [size]
    if read_lines is not None and sample:
        newlines = sample.count(b"\n")
        bytes_per_line = len(sample) if newlines == 0 else len(sample) / newlines
        ceilings.append(int(read_lines * bytes_per_line))
    if read_bytes is not None:
        ceilings.append(int(read_bytes))

    return max(0, min(ceilings))


def size_of(path: str) -> int:
    """Bytes on disk, or 0 when `stat` could not say — the file is never opened."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def cost_of(path: str) -> int:
    """Tokens the WHOLE file would cost, from its SIZE — the file is never read.

    Reading it to measure it would spend exactly what the guard exists to save, and this
    property is proven by execution: a file with mode 000 still gets a cost, because
    nothing on this path does more than `stat`. It stayed that way when the price learned
    about read limits — the refinement that needs the file's contents lives in
    `loaded_bytes`, which is handed a sample somebody else already read.
    """
    return int(size_of(path) // CHARS_PER_TOKEN)


def decide(path: str, budget: Budget, *, indexed_ids: set | None = None,
           floor_pct: float = FLOOR_PCT, share_pct: float = SHARE_PCT,
           read_lines=None, read_bytes=None, escape: str = "") -> Verdict:
    """Block when either criterion fires, whichever comes first.

    `indexed_ids` is a set of doc ids ALREADY FETCHED by the caller — this function never
    reaches Qdrant itself. Knowing whether a file is indexed costs a round trip, and this
    guard runs before every file read; the common case is "small file, allow", so paying
    that round trip up front would tax the path that never needed it. The adapter fetches
    `indexed_ids` only after a first `decide(..., indexed_ids=None)` call has already said
    it would block — which is rare — and calls again just to enrich the message.

    `escape` is the literal the HOST detects in the user's turn to let a read through, and
    the message repeats it verbatim. It is passed in rather than known here so that the text
    the message teaches and the text the adapter looks for cannot be two different settings.

    `read_lines` / `read_bytes` are the HOST's ceilings on a single read (claude-code stops
    at 2000 lines by default, hermes truncates at ~100k chars). They only ever make the
    price SMALLER, which is why the size-only pass below can be trusted as a first filter:
    a file too small to block at full price cannot block at a fraction of it. That ordering
    is not just tidiness — it is what keeps the file CLOSED on the common path.
    """
    size = size_of(path)                    # ONE stat, and no open on the common path
    cost = int(size // CHARS_PER_TOKEN)
    free = max(0, budget.window - budget.used)
    after = budget.used + cost

    if budget.window <= 0 or budget.used >= budget.window:
        # THE TWO WAYS OF NOT KNOWING THE WINDOW, and they sit together because they end
        # the same way: allow.
        #
        # `window <= 0` — we could not learn it at all. Blocking here would mean blocking
        # on a guess, and the window is not derivable from disk on either host.
        #
        # `used >= window` — we thought we knew it and the facts REFUTED us. A session
        # cannot consume more of the window than the window holds, so a `used` at or above
        # it does not describe a full session; it proves the number came from
        # `windows.MODEL_WINDOWS` guessing low for a variant that outgrew its bare name.
        # Measured before this rule existed: a real 1M session read as 200_000/989,479,
        # `free` collapsed to 0, and the guard denied a 4 KB file — fail open inverted into
        # fail closed. Erring large costs a sleeping guard; erring small costs a cage.
        return Verdict(False, "", cost, free)

    floor_hit = after > budget.window * (1 - floor_pct)
    # No `free > 0` here on purpose. The early return above is the SINGLE OWNER of "the
    # window is not usable" — `free == 0` iff `used >= window` iff that return has already
    # fired — so a guard here would be a second owner of the same invariant, and two owners
    # do not stay agreeing. Ruling F5 settled the identical shape once already, when two
    # detectors both answered "is this indexable". Do not re-add it as a missing guard: it
    # protects no arithmetic (`free * share_pct` is fine at zero, and the division lives in
    # `pct` below, which the same early return guards).
    share_hit = cost > free * share_pct
    if not (floor_hit or share_hit):
        return Verdict(False, "", cost, free)

    # THE ONE READ, and everything that needs the file's contents hangs off it: whether it
    # is indexable, and how long its lines are. Reached only here, on the rare path where
    # the size alone was already enough to block.
    sample = sample_of(path)
    if not indexable_sample(sample):
        # Unreadable or binary. Nothing to offer instead, so blocking would only take away
        # an option — and an unreadable file must produce an answer, never an exception.
        return Verdict(False, "", cost, free)

    # SECOND PASS ON THE PRICE, now that we know what one read actually pulls. It can only
    # come down, so a verdict can only turn from block to allow here.
    cost = int(loaded_bytes(size, sample, read_lines=read_lines,
                            read_bytes=read_bytes) // CHARS_PER_TOKEN)
    after = budget.used + cost
    floor_hit = after > budget.window * (1 - floor_pct)
    share_hit = cost > free * share_pct
    if not (floor_hit or share_hit):
        return Verdict(False, "", cost, free)

    about = "≈" if not budget.exact else ""
    # `free > 0` is guaranteed here, not hoped for: `used >= window` returned above, so
    # reaching this line means `used < window`. The branch that used to phrase `free == 0`
    # as "nothing left in the window" went with it — that state can no longer reach the
    # message, and a branch no call can enter is a branch that rots and lies about what
    # was verified.
    pct = int(round(cost / free * 100))
    head = (f"reading this file would cost {about}{cost:,} tokens, "
            f"{pct}% of the {about}{free:,} you have left")

    from core.docs import doc_id_for          # local: keeps the pure path import-light
    doc_id = doc_id_for(path)
    if indexed_ids and doc_id in indexed_ids:
        what = f"it is already indexed as {doc_id} — search it with docs_search"
    else:
        what = "index it with docs_index and search the parts that answer"

    # The caller's marker, never one of ours — see the note where the constant is not.
    way_out = f" To read it anyway, put {escape} in your request." if escape else ""
    reason = f"{head}. Instead, {what}.{way_out}"

    return Verdict(True, reason, cost, free)
