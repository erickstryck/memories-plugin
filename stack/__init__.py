"""The local stack: Qdrant and the two llama-servers, in containers, on this machine.

What this package is for: a step of `qctx install` that stands up, with Docker or Podman, the
three endpoints the plugin talks to (Qdrant, an embedding llama-server and a rerank
llama-server), and `qctx stack status|up|down|remove` to manage them afterwards.

THE DEPENDENCIES POINT ONE WAY, `cli -> stack -> core`:

- `core/`, `hooks/` and `hosts/` never import `stack`. The hooks load `core` on every prompt, and
  they must not so much as load the subprocess, container and download code this package is
  made of.
- `stack` never imports `hooks/`, `hosts/` or `cli/`, because it must not know which hosts exist.
  What differs per host (the latency budgets) must come in as a parameter from the CLI, which
  already names the hosts. It imports `core` and the stdlib, nothing else.

`tests/test_stack_boundaries.py` asserts both directions, and `stack` is in the `FORBIDDEN` set
of `tests/test_core_is_portable.py`.

WHY NOT IN `core/`. `core/` is the portable nucleus every host shares; the stack is the
infrastructure of one machine, its container runtime, GPUs, ports and model files. A host
recalling a memory needs the endpoints, never the code that stood them up.
"""
from core.errors import CoreError


class StackError(CoreError):
    """A step of the stack failed. It names the step and, when there is one, the fix.

    A `CoreError`, so the `except core.CoreError` that `cli/qctx.py::main` already has prints it
    with no new `except`: `error: runtime: no runtime (fix: install Docker or Podman)`.
    """

    def __init__(self, message: str, *, step: str, fix: str | None = None):
        super().__init__(f"{step}: {message}" + (f" (fix: {fix})" if fix else ""))
        self.step = step
        self.fix = fix
