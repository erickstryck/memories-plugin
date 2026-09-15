"""The adapter on the WIRE, and the port's signature.

Everything else offline drives `FakeVectorStore`, which implements ordering itself. So
the one place the feature meets HTTP was covered by nothing: an independent review
deleted `if order_by: body["order_by"] = order_by` from `core/qdrant.py` and the whole
1354-test suite stayed green, and made the real `ensure_payload_index` a no-op with the
same result. Both are the ONLY thing standing between an ordered listing and a scroll
the server paginates by id, which is the exact defect this branch exists to remove,
reachable again by a refactor with the suite passing.

The integration tests do catch both, and they are the reason this is a gap in the safety
net rather than a live bug. But they are gated behind `QCTX_INTEGRATION` and a reachable
Qdrant, so they do not run while someone edits the adapter.

WHY A REAL HTTP SERVER and not a mock of `request`: patching `Qdrant.request` would
assert that the adapter calls a method, and would keep passing if the body never
reached a socket. The property worth holding is that the JSON the SERVER receives carries
`order_by`, and the only way to know what the server received is to be the server. The
repo already settled this pattern; `tests/fakes.py` and `tests/test_bigfile_claude.py`
both stand up a loopback Qdrant for the same reason.
"""
import http.server
import inspect
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ports import VectorStore  # noqa: E402
from core.qdrant import Qdrant  # noqa: E402
from tests.fakes import FakeVectorStore  # noqa: E402


class _RecordingQdrant(http.server.BaseHTTPRequestHandler):
    """A Qdrant that answers plausibly and REMEMBERS what it was asked.

    The recording is the point: the assertions read the request bodies this collected,
    which is the only vantage point from which "the parameter reached the wire" is a
    fact rather than an inference.
    """

    #: Class-level, because `BaseHTTPRequestHandler` is instantiated per request. Each
    #: test binds a fresh subclass through `_serve`, so the list is never shared.
    seen: list = []

    def log_message(self, *_args):
        pass

    def _answer(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _record(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        type(self).seen.append({"method": self.command, "path": self.path, "body": body})

        return body

    def do_GET(self):
        self._record()
        self._answer({"result": {"config": {"params": {"vectors": {}}}, "points_count": 0}})

    def do_PUT(self):
        self._record()
        self._answer({"result": True})

    def do_POST(self):
        self._record()
        self._answer({"result": {"points": [], "next_page_offset": None}})


def _serve(case) -> tuple[str, list]:
    """A loopback Qdrant for `case`, as (base_url, recorded_requests).

    The handler subclass is per-test so two tests never read each other's traffic, and
    the shutdown is registered as cleanup rather than left to a `tearDown` each caller
    would have to remember.
    """
    seen: list = []
    handler = type("H", (_RecordingQdrant,), {"seen": seen})
    quiet = type("QuietServer", (http.server.ThreadingHTTPServer,),
                 {"handle_error": lambda self, request, addr: None})
    server = quiet(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    case.addCleanup(server.server_close)
    case.addCleanup(server.shutdown)

    return f"http://127.0.0.1:{server.server_address[1]}", seen


class TestTheAdapterPutsOrderByOnTheWire(unittest.TestCase):
    """`order_by` has to arrive in the POST body, and be ABSENT when nobody asked.

    Both halves matter. Losing it silently turns an ordered listing back into a scroll
    paged by point id, which is the original defect. Sending it unconditionally would
    change every existing caller's behaviour: the server makes `next_page_offset` null
    for an ordered scroll, so a `scroll_all` sweep would stop after one page.
    """

    def _scroll_body(self, seen):
        bodies = [r["body"] for r in seen if r["path"].endswith("/points/scroll")]
        self.assertEqual(len(bodies), 1, f"expected exactly one scroll, got {seen}")

        return bodies[0]

    def test_the_order_by_a_caller_passes_reaches_the_server(self):
        base, seen = _serve(self)
        order = {"key": "updated_at", "direction": "desc"}
        Qdrant(base).scroll("mem", limit=3, order_by=order)
        body = self._scroll_body(seen)
        self.assertEqual(body.get("order_by"), order,
                         "the adapter is the only thing that can put it on the wire")

    def test_the_body_carries_no_order_by_when_none_was_asked_for(self):
        """The spec's promise that no existing caller changes behaviour lives here."""
        base, seen = _serve(self)
        Qdrant(base).scroll("mem", limit=3)
        body = self._scroll_body(seen)
        self.assertNotIn("order_by", body,
                         "an unrequested order_by would null the cursor of every sweep")

    def test_an_ordered_scroll_sends_no_offset(self):
        """The server answers 400 to `order_by` and `offset` together, so the adapter
        must not smuggle a positional offset alongside an order."""
        base, seen = _serve(self)
        Qdrant(base).scroll("mem", limit=3,
                            order_by={"key": "updated_at", "direction": "desc"})
        body = self._scroll_body(seen)
        self.assertNotIn("offset", body)

    def test_an_unordered_scroll_still_sends_its_offset(self):
        """The negative above must not be achieved by dropping `offset` altogether."""
        base, seen = _serve(self)
        Qdrant(base).scroll("mem", limit=3, offset="abc")
        body = self._scroll_body(seen)
        self.assertEqual(body.get("offset"), "abc")


class TestTheAdapterCreatesThePayloadIndex(unittest.TestCase):
    """`ensure_payload_index` has to issue the PUT, and swallow only the server's refusal.

    The ordered listing NEEDS this index: without it the server answers 400 and the
    listing degrades to unordered. A no-op here degrades every listing on a fresh
    collection while every offline test still passes, because the fake records the index
    in a dict of its own.
    """

    def test_the_index_request_reaches_the_server(self):
        base, seen = _serve(self)
        Qdrant(base).ensure_payload_index("mem", "updated_at", "datetime")
        puts = [r for r in seen if r["method"] == "PUT" and "/index" in r["path"]]
        self.assertEqual(len(puts), 1, f"expected one index PUT, got {seen}")
        self.assertEqual(puts[0]["body"],
                         {"field_name": "updated_at", "field_schema": "datetime"})
        self.assertIn("wait=true", puts[0]["path"],
                      "an index created asynchronously is not there for the next call")

    def test_a_refusal_does_not_reach_the_caller(self):
        """A payload index is an optimization; failing to create one must not take down
        the write that asked for it. Proven against a server that refuses, not by
        reading the `except`."""
        class Refusing(_RecordingQdrant):
            def do_PUT(self):
                self._record()
                body = b'{"status":{"error":"nope"}}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        quiet = type("QuietServer", (http.server.ThreadingHTTPServer,),
                     {"handle_error": lambda self, request, addr: None})
        seen: list = []
        server = quiet(("127.0.0.1", 0), type("H", (Refusing,), {"seen": seen}))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        Qdrant(f"http://127.0.0.1:{server.server_address[1]}").ensure_payload_index(
            "mem", "updated_at", "datetime")
        self.assertTrue([r for r in seen if r["method"] == "PUT"],
                        "it has to have tried before swallowing")


class TestThePortAndItsImplementationsAgree(unittest.TestCase):
    """The `Protocol` is documentation unless something compares it to the real thing.

    `VectorStore` is a `Protocol`, and `runtime_checkable` only ever checks that the
    METHOD NAMES exist, never the parameters. So `order_by` was removable from the port
    with the suite green, and a fake could drift from the adapter it stands in for. Both
    are the drift class this branch exists to eliminate, one level up.
    """

    #: Compared for both implementations, so a divergence names which one drifted.
    IMPLEMENTATIONS = (("Qdrant", Qdrant), ("FakeVectorStore", FakeVectorStore))

    def test_every_port_method_exists_on_both_implementations(self):
        for name in [n for n in dir(VectorStore) if not n.startswith("_")]:
            for label, impl in self.IMPLEMENTATIONS:
                with self.subTest(method=name, impl=label):
                    self.assertTrue(callable(getattr(impl, name, None)),
                                    f"{label} does not implement {name}")

    def test_a_method_production_CALLS_cannot_leave_the_port(self):
        """The guard above iterates the PORT, so a port that shrinks takes its own test with
        it: deleting `facet` from `VectorStore` left the suite green while `core/repos.py`
        went on calling it. `facet` reached production without ever being declared, which is
        how a fake that only knew how to succeed hid a real refusal for months.

        Read from the SOURCE, so the list cannot rot: any `self.q.<name>(` in `core/`."""
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "core"
        called = set()
        for path in root.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute)
                        and f.value.attr == "q"):
                    called.add(f.attr)

        self.assertIn("facet", called, "guard vacuous: production stopped calling facet")
        declared = {n for n in dir(VectorStore) if not n.startswith("_")}
        self.assertEqual(called - declared, set(),
                         "core/ calls a store method the port does not declare")

    def test_every_port_parameter_exists_on_both_implementations(self):
        """A port parameter no implementation accepts is a contract nobody honours.

        The comparison is one-directional on purpose: an implementation may add an
        argument the port does not publish, but it may never LACK one the port promises,
        because callers are typed against the port.
        """
        for name in [n for n in dir(VectorStore) if not n.startswith("_")]:
            declared = inspect.signature(getattr(VectorStore, name)).parameters
            for label, impl in self.IMPLEMENTATIONS:
                actual = inspect.signature(getattr(impl, name)).parameters
                for param in declared:
                    with self.subTest(method=name, param=param, impl=label):
                        self.assertIn(param, actual,
                                      f"{label}.{name} is missing `{param}`")

    def test_the_port_publishes_the_ordering_parameter(self):
        """Named explicitly, and not left to the sweep above, because this is the row the
        ordered listing depends on: without `order_by` on the port there is no contract
        for an ordered walk at all."""
        self.assertIn("order_by", inspect.signature(VectorStore.scroll).parameters)


if __name__ == "__main__":
    unittest.main()
