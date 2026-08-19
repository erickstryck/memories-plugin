"""Turning an outside-supplied name into a filename.

ONE OWNER, BECAUSE FIVE COPIES DECIDED THE SAME FILENAME. `core/jobs.py`, `core/lease.py`,
`hosts/hermes/__init__.py` and, inline, `hooks/recall.py` and `hooks/checkpoint.py` each carried
a byte-identical copy of this expression. They were identical when this module was written --
verified against the same inputs before the move -- but nothing kept them that way, and two of
them derive the state-file name for the SAME session on the two hosts: the hermes provider and
the claude-code recall hook. A divergence between those two would not raise. It would silently
split one session's recall state into two files, and the symptom would be a session that keeps
re-injecting memories it already showed, with nothing anywhere pointing at the cause.
"""


def safe(name) -> str:
    """`name` reduced to characters that are safe in a filename. Empty or None becomes
    "default".

    NOT AN ASCII TRANSLITERATION, AND NOT MEANT TO BE. `str.isalnum()` is true for letters in
    any script, so "ção" and "проект" pass through unchanged -- which is correct here, because
    the goal is a path that cannot escape its directory or name a different file, not a name
    made of ASCII. What it does remove is every separator and every dot, so "../../etc/passwd"
    becomes "______etc_passwd": no traversal survives, and no name can reach outside the
    directory the caller joined it to.

    "default" rather than "" for an absent name, because an empty filename is not a filename --
    the write would fail, or worse, hit the directory itself.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name or "default"))
