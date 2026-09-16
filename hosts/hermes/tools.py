"""The hermes host's view of the shared operation table.

THE TABLE ITSELF IS IN `core/operations.py` — see its docstring for why. What stays here is
the two things that are genuinely this host's:

  1. THE BOOTSTRAP BELOW, which is load-bearing and cannot move. hermes' provider loader
     pre-execs every sibling `*.py` of the package BEFORE `__init__.py`, registering each in
     `sys.modules` first and swallowing any failure at `logger.debug`. Without these lines
     `import core` here raised ModuleNotFoundError during that pre-exec, the broken shell
     STAYED in `sys.modules`, the package's own `from . import tools` then succeeded and
     handed back a module with nothing in it. The provider failed to load entirely, taking
     recall and the checkpoint cadence with it, and the only symptom was one debug line.

     `realpath` and THREE `dirname` levels: the plugin is installed as a symlink, and
     `abspath` resolves to the symlink's directory.

  2. The re-export, so `hosts.hermes.tools.SCHEMAS` and `hosts.hermes.tools.dispatch` stay
     the names this host's package, its tests and its documentation already use.

Which operations are withheld from the model, and why, is documented in
`core/operations.py`; the enforced set is NOT_FOR_THE_MODEL in tests/test_host_equivalence.py.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.operations import (  # noqa: E402,F401
    ROUTES,
    SCHEMAS,
    DefaultTuning,
    ToolArgError,
    bind_tuning,
    dispatch,
)
