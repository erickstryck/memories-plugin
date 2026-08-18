"""What `hermes plugins install` imports. The adapter itself lives in `hosts/hermes/`.

WHY THIS FILE EXISTS AT THE REPOSITORY ROOT. hermes' memory loader scans
`$HERMES_HOME/plugins/<name>/` — one level, flat — and imports the `__init__.py` it finds
there. `hermes plugins install owner/repo` clones a repository into exactly that place, so the
thing the loader imports is whatever sits at the repo's root. Without this file the clone
installs and then loads nothing, and the failure is SILENT: the loader swallows it at debug
level (measured — `_load_provider_from_dir` returned None with no message the user ever sees).

WHY NOT INSTALL THE SUBDIRECTORY INSTEAD. `hermes plugins install owner/repo/hosts/hermes` is
accepted syntax and it does not work here: the adapter imports `core`, which lives at the root,
and a subdirectory install brings neither. Measured against the real loader — None, silently,
for the same reason. So the whole repository is the installable unit, and this file is its door.

WHY THE sys.path LINE IS NOT DECORATION. The loader imports this file by PATH, under a synthetic
module name, so `hosts` is not importable when it runs — `ModuleNotFoundError: No module named
'hosts'`, measured. Putting this directory on the path first is what makes the import below
resolve, and it is the same bootstrap every file in `hosts/hermes/` already carries for the same
reason.

WHY THIS DOCSTRING NAMES TWO SYMBOLS IT DOES NOT USE. Before importing anything, the loader
greps the first 8192 bytes of this file for the literal string `register_memory_provider` or the
literal string `MemoryProvider` (`_is_memory_provider_dir`). A directory carrying neither is not
considered a provider at all, and the failure reads as "the provider does not exist".

Neither string appears anywhere else here, and that is not an oversight: what we export is
`MemoriesProvider` — which does NOT contain `MemoryProvider` as a substring, "ies" against "y" —
and `register`, whose `register_memory_provider` is a method on the ctx the loader passes IN, not
a function it looks up. So an alias would be a lie about what this module offers. The mention
above is the whole mechanism, exactly as `hosts/hermes/__init__.py` already relies on it, and
`tests/test_installable_from_git.py` fails if an editing pass ever removes it.

WHAT IT DELIBERATELY DOES NOT DO: any work. It re-exports and nothing else, so a `plugins list`
or a doctor run pays an import and not a connection.
"""
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hosts.hermes import MemoriesProvider, register  # noqa: E402,F401

__all__ = ["MemoriesProvider", "register"]
