"""The model download: resumable, hash-verified, with the progress bar (decision 16).

What it is FOR: `qctx install` takes each model from its revision-pinned Hugging Face
URL into `<file>.part`, checks the size and the sha256 of the catalogue, and only then
`os.replace`s it into place. An interrupted download (a Ctrl-C, a dropped connection)
leaves the `.part`, and the next run resumes from its size with a `Range` request; the
sha256 is computed incrementally over the part and over the new bytes, so a long resume
never re-reads or re-downloads what it already has, and a part is never trusted blindly.

WHY THE TRANSPORT IS INJECTED: the network is the only thing this module must not need
to be tested. `Transport` is the whole contract the download runs against, and the
tests script it. `UrllibTransport` is the real one: stdlib `urllib`, which follows the
redirects Hugging Face sends to its CDN by itself, so the integrity comes from the
hash, never from the host.

MEASURED 2026-10-07 (loopback probes, `/tmp/mp-range-probe3.py` and
`/tmp/mp-fetch-probe.py`, run on this machine): a server that honours
`Range: bytes=N-` answers 206 with the body starting at N; a server that does
not (Python's own http.server is one) answers 200 with the WHOLE body from
offset 0; `urlopen` follows a 302 `Location` on its own; and a refused
connection comes back as `URLError` (an OSError). A body cut before its
Content-Length is read here in bounded chunks, so it simply ends early and
`fetch` reports the short stream, keeping the `.part`; an UNBOUNDED read of
the same body would instead raise `http.client.IncompleteRead`, an
`HTTPException` and NOT an OSError, which is why the transport's error clause
catches both families.
"""
import hashlib
import http.client
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator, Protocol

from . import StackError
from .catalog import Model

#: The read size: 1 MiB, the unit the bar and the messages speak in.
_BLOCK = 1 << 20


class Transport(Protocol):
    """A byte source of a model file. `get` yields chunks of the file; with `start > 0`
    the bytes start at `start` — a server that honours `Range` answers 206 that way,
    and a server that does not has its head discarded by the transport (measured,
    module docstring)."""

    def get(self, url: str, *, start: int = 0) -> Iterator[bytes]:
        ...


class UrllibTransport:
    """`Transport` over stdlib urllib.

    `start > 0` sends `Range: bytes=start-`. A 206 answers with the tail directly; a
    200 answers with the whole file, so the first `start` bytes are read away before
    the first yield (measured, module docstring). A network failure, before or
    mid-stream, becomes `StackError(step="download")` with the run-again fix, because
    a dropped download is exactly what the `.part` resumes.
    """

    def __init__(self, timeout: float = 60.0, chunk: int = _BLOCK):
        self.timeout = timeout
        self.chunk = chunk

    def get(self, url: str, *, start: int = 0) -> Iterator[bytes]:
        request = urllib.request.Request(url)
        if start > 0:
            request.add_header("Range", f"bytes={start}-")
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except (urllib.error.URLError, OSError) as exc:
            raise StackError(
                f"cannot download {url}: {exc}", step="download",
                fix="run again; a partial .part file resumes where it stopped") from exc
        try:
            if start > 0 and response.status == 200:
                # The server ignored the Range and sent the whole file: read the head
                # away, or the `.part` would hold every byte twice.
                skipped = 0
                while skipped < start:
                    piece = response.read(min(self.chunk, start - skipped))
                    if not piece:
                        raise StackError(
                            f"the server answered the resume of {url} with a file that "
                            f"ends before byte {start}: the .part is not from this file",
                            step="download", fix="delete the .part and run again")
                    skipped += len(piece)
            while True:
                piece = response.read(self.chunk)
                if not piece:
                    return
                yield piece
        except (urllib.error.URLError, OSError,
                http.client.HTTPException) as exc:
            # OSError covers a refused or reset connection; HTTPException covers the
            # HTTP layer, where a body cut before its Content-Length raises
            # IncompleteRead (an HTTPException, NOT an OSError: measured 2026-10-07).
            # In this transport the read is BOUNDED, so a truncated body usually just
            # ends early instead and `fetch` reports the short stream; this clause is
            # the backstop for the layer errors a bounded read can still raise.
            raise StackError(
                f"download of {url} failed: {exc}", step="download",
                fix="run again; the .part resumes where it stopped") from exc
        finally:
            response.close()


class ProgressBar:
    """The download's line (decision 16): percent, MiB done/total, speed and ETA.

    The speed and the ETA count the bytes of THIS run only: a resume starts from a
    `.part`, and the part must not look like download speed (the bar starts at
    `start`, the run counter at zero). On a TTY every update rewrites the same line
    with `\\r`, and only `finish` breaks it; off a TTY one line is written per slice
    of `slices` (10 by default), so a log does not fill with rewritten lines.
    """

    def __init__(self, total: int, label: str, *, tty: bool,
                 write: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic,
                 start: int = 0, slices: int = 10):
        self.total = total
        self.label = label
        self.tty = tty
        self.write = write
        self.clock = clock
        self.start = start
        self.slices = slices
        self._done = start
        self._run = 0  # bytes of this run: the numerator of the speed and the ETA
        self._started = clock()
        self._last_slice = 0

    def update(self, done: int) -> None:
        self._run += done - self._done
        self._done = done
        if self.tty:
            self.write("\r" + self._line())
        else:
            slice_ = done * self.slices // self.total if self.total else self.slices
            if slice_ > self._last_slice:
                self._last_slice = slice_
                self.write(self._line() + "\n")

    def finish(self) -> None:
        if self.tty:
            self.write("\r" + self._line() + "\n")
        elif self._done < self.total:
            # Off a TTY the last slice may never have been crossed (a failed or a
            # tiny download): the final state still deserves its one line.
            self.write(self._line() + "\n")

    def _line(self) -> str:
        done, total = self._done, self.total
        pct = done * 100 // total if total else 100
        elapsed = self.clock() - self._started
        if self._run and elapsed > 0:
            speed = self._run / elapsed  # bytes/second of THIS run
            eta = (total - done) / speed
            eta_text = f"{int(eta // 60)}:{int(eta % 60):02d}"
        else:
            # No speed before this run has bytes AND the clock has moved: the ETA
            # would be a division by zero dressed as a number.
            speed = 0.0
            eta_text = "--"
        return (f"  {self.label}  {pct:3d}%  {done / 2**20:.1f}/{total / 2**20:.1f} MiB"
                f"  {speed / 2**20:.1f} MiB/s ETA {eta_text}")


def fetch(model: Model, models_dir: Path, transport: Transport, *,
          bar: Callable[[int, int], ProgressBar],
          log: Callable[[str], None]) -> str:
    """Make `model.filename` present in `models_dir`, verified against the catalogue.

    Returns what happened: `"present"` (the file was already the pinned one),
    `"downloaded"` (a fresh download) or `"resumed"` (one that continued a `.part`).
    The part becomes the model only after its size AND its hash match (`os.replace`):
    a wrong hash, an oversized stream or a part bigger than the model deletes the
    part, a stream that ended early keeps it, because that is exactly where the next
    run resumes. A Ctrl-C propagates with the part intact, the same way.
    """
    target = models_dir / model.filename
    part = target.with_name(target.name + ".part")
    if target.exists():
        if _file_digest(target) == model.sha256:
            log(f"{model.filename}: present, verified")
            return "present"
        # The bytes on disk are not the pinned model: download over them.
        target.unlink()
    done = 0
    digest = hashlib.sha256()
    resumed = part.exists()
    if resumed:
        done = part.stat().st_size
        if done > model.size:
            # A part past the pinned size only exists after a Ctrl-C in an oversized
            # stream: resuming from it asks for bytes past the end of the file and
            # can only grow the mistake, so it is deleted before the download starts.
            part.unlink()
            raise StackError(
                f"the .part of {model.filename} is {done} bytes, more than the "
                f"{model.size} bytes the catalogue pins: it is not from this file",
                step="download", fix="delete the .part and run again")
        # Hash the part into the digest before continuing: the final check is over
        # the whole file, and re-reading the part at the end of a long resume would
        # double the disk I/O the download already pays.
        with part.open("rb") as handle:
            while block := handle.read(_BLOCK):
                digest.update(block)
    progress = bar(model.size, done)
    oversized = False
    try:
        with part.open("ab") as handle:
            for chunk in transport.get(model.url(), start=done):
                done += len(chunk)
                digest.update(chunk)
                handle.write(chunk)
                progress.update(done)
                if done > model.size:
                    # More bytes than the catalogue pins: this is not the file, and
                    # resuming from it would only grow the mistake.
                    oversized = True
                    break
    finally:
        progress.finish()
    if oversized:
        part.unlink()
        raise StackError(
            f"download of {model.filename} returned more than the {model.size} bytes "
            f"the catalogue pins: not the expected file",
            step="download", fix="delete the .part and run again")
    if done < model.size:
        # A stream that ended early is unfinished, not corrupt: keep the part, the
        # next run resumes from its size.
        raise StackError(
            f"download of {model.filename} ended early at {done} of {model.size} "
            f"bytes; run again to resume",
            step="download")
    if digest.hexdigest() != model.sha256:
        part.unlink()
        raise StackError(
            f"the sha256 of {model.filename} is {digest.hexdigest()}, the catalogue "
            f"wants {model.sha256}",
            step="download", fix="delete the .part and run again")
    os.replace(part, target)
    return "resumed" if resumed else "downloaded"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()
