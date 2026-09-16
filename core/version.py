"""The package's version. One owner, because the last hand-maintained one went stale.

WHY THIS FILE EXISTS RATHER THAN A STRING IN EACH MANIFEST. This package declares itself in
three manifests -- `plugin.yaml` (hermes, git install), `hosts/hermes/plugin.yaml` (hermes,
symlink install) and `.claude-plugin/plugin.json` (claude-code) -- and none of them can read
Python. So the number is physically written four times, and the ONLY thing that keeps the
four honest is a test that compares them against this constant, which is the declared owner.

WHY NOT GO ON WITHOUT A VERSION. The repo previously declared none on purpose, and the
reasoning was sound as far as it went: a hand-maintained number goes stale, and this one
already had (manifests said 0.3.0 while the installed record said 0.2.0, measured). The
commit SHA never lies.

What the SHA cannot do is say whether an upgrade is safe. `9ff5469 -> 5701914` carries no
information about compatibility; `1.0.0 -> 2.0.0` does. A user pinning `--ref <sha>` has to
read a git log to learn what they pinned. So the version comes back, and the staleness it
costs is paid for by `tests/test_installable_from_git.py::TestTheVersionIsONENumber`, which
fails the moment any manifest disagrees with this line -- the exact drift that produced the
0.3.0/0.2.0 split, now a red test instead of a silent lie.

HOW TO BUMP IT. Edit this line, run the suite, and let the test tell you which manifests
still disagree. Then tag the commit `v<version>`: the tag is what `--ref` pins.

SEMANTICS, so the number means something:
- MAJOR: a user's archives, config or invocation must change to keep working.
- MINOR: new capability; an existing install keeps working untouched.
- PATCH: a fix with no new surface.
"""

#: The one place this number is decided. Every manifest is checked against it.
__version__ = "1.0.0"
