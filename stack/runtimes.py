"""The door to the container runtimes: `discover` finds the engines that answer.

The contract and what both engines share live in `stack.engine`, each engine in its own module
(`stack.docker`, `stack.podman`), and the command runner in `stack.process`. Inside `stack`,
modules import from those directly; this module keeps the names the earlier tasks and the tests
import from `stack.runtimes` (review round R3, item R3-7).
"""
import shutil

from . import StackError
from .engine import as_which
# Re-exported: the earlier tasks and the tests import these names from `stack.runtimes`.
from .docker import Docker  # noqa: F401
from .engine import (ContainerRuntime, EngineInfo, Provider, ProviderInfo,  # noqa: F401
                     normalize_arch, parse_size, socket_alive)
from .podman import Podman  # noqa: F401
from .process import Completed, Runner, SubprocessRunner  # noqa: F401


def discover(runner: Runner, which=shutil.which,
             host_system: str = "linux") -> list[ContainerRuntime]:
    """Docker before Podman, only the engines whose binary exists and answers `info`. A binary
    present but whose `info` does not answer, or hangs, is skipped, not an error: the other
    engine may work (R2 item m1)."""
    which = as_which(which)
    found: list[ContainerRuntime] = []
    for factory in (lambda: Docker(runner, which=which),
                    lambda: Podman(runner, which=which, host_system=host_system)):
        runtime = factory()
        try:
            engine = runtime.engine()
        except StackError:
            engine = None  # a hung `info` (timeout) is not an answering engine
        if engine is not None:
            found.append(runtime)
    return found
