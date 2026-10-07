"""The model download: resume from a `.part`, verification before the rename, and the
progress bar (decision 16).

Every download runs against a `FakeTransport`, never the network: the bytes are scripted
(`content`, a `cut_after` that ends the stream), and every call is recorded with its
`start`, so a test asserts the resume point by reading the call, not by guessing.
`pin` is what the catalogue claims about the bytes: by default it is the `content`
itself, and a mismatch between the two is how the wrong-hash and the oversized cases
are made. The bar is checked line by line with a scripted clock, on a bar of exactly
10 MiB so the MiB numbers are whole. The suite does no real I/O at all; the server
behaviours the real transport absorbs (206, 200-ignoring-Range, 302, a refused
connection) were measured at development time against local HTTP servers on loopback,
and the dates and commands live in the module docstring of `stack/fetch.py`.
"""
import hashlib
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stack import StackError, catalog  # noqa: E402
from stack.fetch import ProgressBar, fetch  # noqa: E402
from tests.stack_fakes import FakeTransport  # noqa: E402

#: Exactly 10 MiB, so the MiB figures the bar prints are whole.
CONTENT = bytes(range(256)) * (10 * 2**20 // 256)


def model(content: bytes) -> catalog.Model:
    """A catalogue model whose size and sha256 are the real ones of `content`."""
    return catalog.Model(
        "embed", "acme/model", "0" * 40, "model.gguf",
        len(content), hashlib.sha256(content).hexdigest(), "MIT")


def bar_factory(*, tty: bool, clock=None, slices: int = 10):
    """A `bar` callable for `fetch`: one `ProgressBar` per model, lines to `out`."""
    out: list[str] = []

    def make(total: int, start: int) -> ProgressBar:
        return ProgressBar(total, "model.gguf", tty=tty, write=out.append,
                           clock=clock or time.monotonic, start=start,
                           slices=slices)

    return out, make


class ScriptedClock:
    """A clock that only moves when the test says so: `advance` jumps it.

    The bar reads it on every `update`, so one jump before an update is one speed
    sample; that is what makes the MiB/s and the ETA come out of the exact bytes, and
    what keeps the first line (clock not yet moved) at `0.0 MiB/s ETA --`.
    """

    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FetchCase(unittest.TestCase):
    """`fetch` with a fake transport.

    The `models/` directory comes from `tempfile.mkdtemp`, so it outlives the call:
    an error-path test reads it after the `StackError` came back. It lives in the
    suite's run directory, which `tests/__init__.py` removes at exit.
    """

    def fetch(self, content, *, pin=None, cut_after=None, ignore_range=False,
              part=None, present=None, tty=True, clock=None):
        models_dir = Path(tempfile.mkdtemp(prefix="stack-fetch-")) / "models"
        models_dir.mkdir()
        if part is not None:
            (models_dir / "model.gguf.part").write_bytes(part)
        if present is not None:
            (models_dir / "model.gguf").write_bytes(present)
        transport = FakeTransport(content, cut_after=cut_after,
                                  ignore_range=ignore_range)
        out, make = bar_factory(tty=tty, clock=clock)
        lines: list[str] = []
        self._models_dir = models_dir
        self._transport = transport
        result = fetch(model(pin or content), models_dir, transport, bar=make,
                       log=lines.append)
        return result, models_dir, transport, out, lines

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    @property
    def transport(self):
        return self._transport


class TheFreshDownload(_FetchCase):
    def test_a_fresh_download_lands_and_verifies(self):
        result, models_dir, transport, _, _ = self.fetch(CONTENT)
        self.assertEqual("downloaded", result)
        self.assertEqual(CONTENT, (models_dir / "model.gguf").read_bytes())
        self.assertEqual([], list(models_dir.glob("*.part")))
        self.assertEqual([(model(CONTENT).url(), 0)], transport.starts)


class TheResume(_FetchCase):
    def test_a_part_file_resumes_from_its_size(self):
        part = CONTENT[:4 * 2**20]
        result, models_dir, transport, _, _ = self.fetch(CONTENT, part=part)
        self.assertEqual("resumed", result)
        self.assertEqual(CONTENT, (models_dir / "model.gguf").read_bytes())
        self.assertEqual([(model(CONTENT).url(), len(part))], transport.starts)

    def test_a_part_bigger_than_the_model_is_deleted_before_resuming(self):
        # A part past the pinned size is what a Ctrl-C in an oversized stream leaves
        # behind: the catalogue pins 10 MiB, the part is 11 MiB, so it is deleted and
        # no download starts for it.
        with self.assertRaises(StackError) as caught:
            self.fetch(CONTENT, pin=CONTENT, part=CONTENT + b"x" * (1 << 20))
        self.assertEqual("download", caught.exception.step)
        self.assertEqual([], list(self.models_dir.glob("*.part")))
        self.assertFalse((self.models_dir / "model.gguf").exists())
        self.assertEqual([], self.transport.starts)

    def test_a_server_that_ignores_range_still_resumes_correctly(self):
        # The server behind the transport answered the Range request with 200 and the
        # whole file; the transport contract still hands back the tail, so the resume
        # is correct either way and the `.part` is not duplicated.
        part = CONTENT[:4 * 2**20]
        result, models_dir, transport, _, _ = self.fetch(
            CONTENT, part=part, ignore_range=True)
        self.assertEqual("resumed", result)
        self.assertEqual(CONTENT, (models_dir / "model.gguf").read_bytes())
        self.assertEqual([(model(CONTENT).url(), len(part))], transport.starts)


class TheBadBytes(_FetchCase):
    def test_a_wrong_hash_deletes_the_part_and_fails(self):
        # The catalogue pins the good file; the server hands back a file with one
        # flipped tail byte, same length, wrong digest.
        broken = bytearray(CONTENT)
        broken[-1] ^= 0x01
        with self.assertRaises(StackError) as caught:
            self.fetch(bytes(broken), pin=CONTENT)
        self.assertEqual("download", caught.exception.step)
        self.assertIn("sha256", str(caught.exception))
        self.assertEqual([], list(self.models_dir.glob("*.part")))
        self.assertFalse((self.models_dir / "model.gguf").exists())

    def test_a_short_stream_keeps_the_part_and_says_run_again(self):
        # A stream that ended early is not corrupt, it is unfinished: the `.part` stays
        # exactly as it arrived and the next run resumes from it.
        with self.assertRaises(StackError) as caught:
            self.fetch(CONTENT, cut_after=100)
        self.assertEqual("download", caught.exception.step)
        self.assertIn("run again", str(caught.exception))
        self.assertEqual(CONTENT[:100],
                         (self.models_dir / "model.gguf.part").read_bytes())
        self.assertFalse((self.models_dir / "model.gguf").exists())

    def test_an_oversized_stream_deletes_the_part(self):
        # More bytes than the catalogue pins: the file is not the one we want, so what
        # was downloaded is deleted, not kept for a resume that would only grow it.
        with self.assertRaises(StackError) as caught:
            self.fetch(CONTENT + b"extra", pin=CONTENT)
        self.assertEqual("download", caught.exception.step)
        self.assertEqual([], list(self.models_dir.glob("*.part")))
        self.assertFalse((self.models_dir / "model.gguf").exists())


class ThePresentFile(_FetchCase):
    def test_a_present_file_with_the_right_hash_is_skipped(self):
        result, models_dir, transport, _, lines = self.fetch(CONTENT, present=CONTENT)
        self.assertEqual("present", result)
        self.assertEqual(CONTENT, (models_dir / "model.gguf").read_bytes())
        self.assertEqual([], transport.starts)
        self.assertEqual(["model.gguf: present, verified"], lines)

    def test_a_present_file_with_the_wrong_hash_is_replaced(self):
        result, models_dir, transport, _, _ = self.fetch(
            CONTENT, present=CONTENT[:-1] + b"x")
        self.assertEqual("downloaded", result)
        self.assertEqual(CONTENT, (models_dir / "model.gguf").read_bytes())
        self.assertEqual([(model(CONTENT).url(), 0)], transport.starts)


class TheBar(unittest.TestCase):
    """The progress line itself: fields from the exact bytes, both modes of output."""

    def test_tty_bar_rewrites_one_line(self):
        out: list[str] = []
        clock = ScriptedClock()
        bar = ProgressBar(4, "m", tty=True, write=out.append, clock=clock)
        bar.update(1)
        clock.advance(1.0)
        for done in (2, 3, 4):
            bar.update(done)
        bar.finish()
        # Every update rewrites the same line; nothing breaks it until `finish`.
        self.assertNotIn("\n", "".join(out[:-1]))
        self.assertTrue(out[-1].endswith("\n"))
        self.assertTrue(out[0].startswith("\r"))
        # The clock has not moved on the first update: no speed, no ETA yet.
        self.assertEqual("  m   25%  0.0/0.0 MiB  0.0 MiB/s ETA --", out[0][1:])
        self.assertEqual("  m  100%  0.0/0.0 MiB  0.0 MiB/s ETA 0:00",
                         out[-1][1:].rstrip("\n"))

    def test_non_tty_bar_prints_one_line_per_slice(self):
        out: list[str] = []
        clock = ScriptedClock()
        bar = ProgressBar(10 * 2**20, "m", tty=False, write=out.append,
                          clock=clock)
        for percent in range(10, 101, 10):
            bar.update(percent * 2**20 // 10)
            clock.advance(1.0)
        bar.finish()
        # One line per 10% crossing: 10, 20, ..., 100, and `finish` does not repeat
        # the 100% line the last update already printed.
        self.assertEqual(10, len(out))
        self.assertTrue(out[0].startswith("  m   10%"))
        self.assertTrue(out[-1].startswith("  m  100%"))
        for line in out:
            self.assertTrue(line.endswith("\n"), line)

    def test_resumed_bar_starts_at_the_part_and_counts_speed_for_this_run_only(self):
        # The run downloaded 6 MiB of a 10 MiB file in 2 s, starting from a 4 MiB
        # part: the done figure counts the file, the speed and the ETA count this run
        # only.
        out: list[str] = []
        clock = ScriptedClock()
        bar = ProgressBar(10 * 2**20, "m", tty=True, write=out.append,
                          clock=clock, start=4 * 2**20)
        clock.advance(1.0)
        bar.update(4 * 2**20 + 3 * 2**20)
        clock.advance(1.0)
        bar.update(10 * 2**20)
        bar.finish()
        lines = [write[1:].rstrip("\n") for write in out]
        self.assertEqual("  m   70%  7.0/10.0 MiB  3.0 MiB/s ETA 0:01", lines[0])
        self.assertEqual("  m  100%  10.0/10.0 MiB  3.0 MiB/s ETA 0:00", lines[-1])


if __name__ == "__main__":
    unittest.main()
