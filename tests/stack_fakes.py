"""The shared fakes for the `stack` tests: a `Runner` and a `ContainerRuntime`.

They exist so the stack modules are tested against recorded command output, never against a
live engine: a `FakeRunner` answers a `run` from a table, a `FakeRuntime` is a
`ContainerRuntime` in a box, and a `FakeTransport` answers a download from scripted bytes.
Tasks 4-11 import from here, so nothing in this file does real I/O and it depends only on
`stack.runtimes` (which exists by the time this file is imported: this test module is never
collected before it is written). The `FakeTransport` stands in for the `Transport` contract
of `stack.fetch`; like the other fakes it is checked against it structurally, so this file
does not need to import it.

None of the fakes inherits from a real class: the contracts are `Protocol`s, so having the
methods is enough (the same convention as `tests/fakes.py`).
"""
import sys
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack.runtimes import Completed  # noqa: E402


class FakeRunner:
    """A `Runner` that answers `run` from a table, matching by the LONGEST prefix.

    A key `(podman, info)` matches `run(["podman", "info", "--format", "json"])`, and
    `(podman, "info", "--format", "json")` matches the SAME argv as well, winning by being
    longer. This is what lets one table stand in for every subcommand of a binary. Each call
    is recorded in `.calls` as `(argv, timeout)`, so a test can assert the exact command was
    built (the contract test reads the argv this way: the real `compose` returns a
    `Completed` and hands the argv to the runner).

    No real subprocess is started. An argv no prefix answers raises `KeyError`, so a command
    a test forgot to register is a loud failure, not a silent empty answer.
    """

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list = []

    def run(self, argv: list, *, timeout: float, stream: bool = False) -> Completed:
        self.calls.append((list(argv), timeout))
        best = None
        for key in self.responses:
            if argv[:len(key)] == list(key) and (best is None or len(key) > len(best)):
                best = key
        if best is None:
            raise KeyError(f"no registered answer for: {argv}")
        return self.responses[best]


class FakeRuntime:
    """A `ContainerRuntime` in a box: `engine()` and `compose_provider()` return fixed
    answers, `compose` is scripted, and every `compose` is recorded in `.calls`.

    `compose` returns a `Completed` (always, mirroring the real runtimes, which delegate to the
    runner), in this order of priority for what that `Completed` carries:
      1. a `fail` entry, when one of its substrings appears in the arguments (a failed
         `up`/`pull`);
      2. for a `run ... --list-devices`, a `Completed(0, ...)` whose stdout is the `list_devices`
         answer registered under the STEM of the compose file (Ruling 2: the probe files are
         `<stack>/probe/<backend>.yaml` and `<stack>/probe/nvidia-<i>.yaml`, so the stem is
         `"intel"`, `"nvidia-0"`, `"cpu"` ...); an unregistered stem gives an empty device list;
      3. a `Completed(0)`, for everything else.

    `.calls` records `(project, file, args, timeout, stream)` for each `compose`, so a test
    can assert both the command and what it was run against.
    """

    def __init__(self, name: str, engine, provider_info,
                 list_devices: dict, fail: dict | None = None):
        self.name = name
        self._engine_info = engine
        self.provider_info = provider_info
        self.list_devices = list_devices
        self.fail = fail or {}
        self.calls: list = []

    def engine(self):
        return self._engine_info

    def compose_provider(self):
        return self.provider_info

    def compose(self, provider, project: str, file: Path, *args: str,
                timeout: float, stream: bool = False):
        # Build the same argv the real runtimes build, so a test can assert the
        # command across runtimes by reading `calls[-1][0]`. The rest is scripted.
        argv = [*provider.argv, "-p", project, "-f", str(file), *args]
        self.calls.append((argv, project, args, timeout, stream))
        joined = " ".join(args)
        for key, completed in self.fail.items():
            if key in joined:
                return completed
        if "run" in args and "--list-devices" in args:
            return Completed(0, self.list_devices.get(Path(file).stem, ""))
        return Completed(0)


class FakeTransport:
    """A `Transport` that answers `get` from scripted bytes, never the network.

    It sits at the TRANSPORT contract: `content` is the whole file and the bytes a
    download sees are `content[start:]`, because that is exactly what a
    `UrllibTransport` hands back for both server classes it absorbs (the 206 tail, and
    the 200 whole-file with its head read away: measured, module docstring of
    `stack/fetch.py`). `ignore_range` therefore records which server class the test
    stands in for without changing the bytes, and `cut_after` ends the stream after
    that many bytes (a connection that dropped).

    Every call is recorded in `.starts` as `(url, start)`, so a test asserts the
    resume point by reading the call. Tasks 10 and 11 reuse this fake for the
    installer's download steps (Ruling 1).
    """

    def __init__(self, content: bytes, cut_after: int | None = None,
                 ignore_range: bool = False):
        self.content = content
        self.cut_after = cut_after
        self.ignore_range = ignore_range
        self.starts: list[tuple[str, int]] = []

    def get(self, url: str, *, start: int = 0) -> Iterator[bytes]:
        self.starts.append((url, start))
        # The transport contract, whichever server class `ignore_range` stands in
        # for: the bytes start at `start`.
        body = self.content[start:]
        if self.cut_after is not None:
            body = body[:self.cut_after]
        # A memoryview, not slicing: the fake serves 10 MiB files, and slicing in a
        # loop is O(n^2) (measured: the focused suite took 59 s before this).
        view = memoryview(body)
        offset = 0
        while offset < len(view):
            yield bytes(view[offset:offset + 256])
            offset += 256
