"""The catalogue of skills this package ships: what exists, where it lives, what it is called.

WHY THIS IS IN THE CORE AND NOT IN AN ADAPTER. Which skills the package contains is a fact
about the PACKAGE, not about the host reading it -- exactly like `core/prompts.py`, which is
here because every host injects the same words. The hosts differ in what they DO with the
catalogue (claude-code auto-discovers `skills/` and needs nothing from us; hermes must be
handed each one through `ctx.register_skill`), and that difference belongs in the adapters.
The list itself does not.

WHY IT IS DERIVED AND NEVER WRITTEN DOWN. A hand-maintained list is one more place to
remember when a fourth skill lands, and this repo has already paid that bill twice:
`core/names.py` exists because five copies decided the same filename, `core/prompts.py`
because the same text was copied into two hosts. Five sites across four test files were
building `<repo>/skills/<name>/SKILL.md` by hand, and the hermes adapter now needs the same
fact -- counted at the commit this was written against, where the adapter had no skill code
at all. Three hand-built sites remain on purpose: they are the independent SIDE of a
comparison against this catalogue, and routing them through it would make them agree with
it by construction and pin nothing.

WHY IT KNOWS NOTHING ABOUT A HOST. No namespacing, no registration, no `skill_view`. A skill
here is a name and a path; `memories:memory` is hermes' spelling of it and lives in hermes'
adapter. A core that learned one host's naming would have to learn the other's next.
"""
import os
from typing import Iterator, NamedTuple

#: The file that makes a directory a skill. Both hosts agree on this name; it is the
#: contract, not a convention we chose.
SKILL_FILE = "SKILL.md"

#: `<repo>/skills`, resolved from this file rather than assumed from the working directory.
#: `realpath` AND NOT `abspath`, for a narrower reason than it looks: the hermes development
#: install symlinks the REPO ROOT into `$HERMES_HOME/plugins/memories`, and that case works
#: under either -- the path through the link (`.../plugins/memories/skills`) reaches the same
#: directory. MEASURED, rather than assumed: swapping in `abspath` left the whole suite green.
#: What `realpath` actually buys is the case where THIS FILE is reached through a link while
#: its parent tree is not (a packaging step that links individual modules into a staging dir):
#: `abspath` then yields the staging dir's `skills`, which does not exist, and the catalogue
#: comes back empty with nothing raised. It costs one syscall and removes a silent-empty mode.
SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "skills")


class Skill(NamedTuple):
    """One skill: the name a host registers it under, and the file to read.

    Deliberately two fields. A host that needs the description parses the file -- putting it
    here would make this module read and parse every skill on import, for callers that only
    wanted to know which ones exist.
    """

    name: str
    path: str


def find(skills_dir: str | None = None) -> Iterator[Skill]:
    """Yield every skill under *skills_dir* (default: this package's), sorted by name.

    A directory without a `SKILL.md` is skipped rather than raising: that is what both hosts
    do with it (claude-code's auto-discovery treats it as a no-op), and an empty or
    half-created directory must not take a plugin's registration down with it.

    An unreadable `skills/` yields nothing. The caller is a plugin being loaded, and a
    missing directory is a broken install, not an exception worth aborting a host's startup
    for -- the skills are an accessory to the memory provider, never a precondition of it.
    """
    root = SKILLS_DIR if skills_dir is None else skills_dir
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name, SKILL_FILE)
        if os.path.isfile(path):
            yield Skill(name=name, path=path)


def paths(skills_dir: str | None = None) -> dict[str, str]:
    """`{name: path}` for every skill. The mapping form, for callers that look one up."""
    return {skill.name: skill.path for skill in find(skills_dir)}


def names(skills_dir: str | None = None) -> list[str]:
    """Just the names, sorted. What a count or a comparison needs."""
    return [skill.name for skill in find(skills_dir)]
