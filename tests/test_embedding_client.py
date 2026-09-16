"""The embedding client's own contract, which nothing exercised.

A design review mutated three guards in `core/embedding.py` and the suite stayed green
on all three: the reordering by `index`, the incomplete-response refusal, and the batch
loop. The reason was structural rather than an oversight — no test imported `Embedder`
at all. Only `EmbeddingError` was imported, to be raised by a fake, so every consumer of
embeddings was tested against a stand-in and the client itself against nothing.

Two of those guards carry a measurement in their own docstring ("I have seen a server
return them out of order"), which is the worst possible state for a rule: prose claiming
it was validated, with no execution behind it. A vector in the wrong slot produces a
wrong search with no visible error, and half a stored batch is the state the archive can
never recover from by itself.

These drive the real client over a loopback server that misbehaves in the exact ways the
guards name, in the same shape `tests/fakes.py::qdrant_with_no_collections` already uses
for absence.
"""
import http.server
import json
import sys
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.embedding import EMBED_BATCH, Embedder, EmbeddingError  # noqa: E402


class _Endpoint(http.server.BaseHTTPRequestHandler):
    """An /embeddings endpoint whose behaviour each test sets on the class."""

    mode = "ok"
    seen: list = []

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        texts = json.loads(raw)["input"]
        type(self).seen.append(list(texts))
        data = [{"index": i, "embedding": [float(i), 0.0, 0.0]} for i in range(len(texts))]
        if type(self).mode == "reversed":
            data = list(reversed(data))
        elif type(self).mode == "short":
            data = data[:-1] if len(data) > 1 else []
        body = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class EmbedderCase(unittest.TestCase):
    def serving(self, mode: str) -> Embedder:
        _Endpoint.mode = mode
        _Endpoint.seen = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}"

        return Embedder(url, "a-model", "", timeout=5)


class TestTheOrderOfTheVectors(EmbedderCase):
    """`embed` returns vectors in the order of the TEXTS, whatever order they arrive in."""

    def test_a_server_that_answers_OUT_OF_ORDER_does_not_scramble_the_vectors(self):
        out = self.serving("reversed").embed(["a", "b", "c"])
        self.assertEqual([v[0] for v in out], [0.0, 1.0, 2.0],
                         "the vectors came back in the server's order, not the texts'")

    def test_the_ordinary_case_is_unchanged(self):
        out = self.serving("ok").embed(["a", "b", "c"])
        self.assertEqual([v[0] for v in out], [0.0, 1.0, 2.0])


class TestAnIncompleteAnswerStoresNothing(EmbedderCase):
    """Fewer vectors than texts RAISES: half a batch is worse than no batch.

    The caller pairs vectors with texts positionally, so a short answer silently shifts
    every text after the missing one onto the wrong vector."""

    def test_fewer_vectors_than_texts_raises(self):
        with self.assertRaises(EmbeddingError) as caught:
            self.serving("short").embed(["a", "b", "c"])
        self.assertIn("incomplete", str(caught.exception).lower())

    def test_it_names_both_numbers_so_the_user_can_act(self):
        with self.assertRaises(EmbeddingError) as caught:
            self.serving("short").embed(["a", "b", "c"])
        self.assertIn("2", str(caught.exception))
        self.assertIn("3", str(caught.exception))


class TestItSendsTextsInBATCHES(EmbedderCase):
    """EMBED_BATCH is a request-size limit, and a loop that ignores it is one request.

    The guard is invisible to any consumer: the vectors come back the same either way.
    What changes is whether a large archive is embedded in requests the endpoint accepts
    or in one it refuses."""

    def test_more_texts_than_the_batch_size_become_several_requests(self):
        embedder = self.serving("ok")
        texts = [f"t{i}" for i in range(EMBED_BATCH + 3)]
        out = embedder.embed(texts)
        self.assertEqual(len(out), len(texts), "not every text came back")
        self.assertEqual(len(_Endpoint.seen), 2, "the batch loop sent one request")
        self.assertEqual(len(_Endpoint.seen[0]), EMBED_BATCH)
        self.assertEqual(len(_Endpoint.seen[1]), 3)

    def test_an_empty_list_asks_the_endpoint_nothing(self):
        self.assertEqual(self.serving("ok").embed([]), [])
        self.assertEqual(_Endpoint.seen, [], "it called the endpoint for no texts")


class TestDimensionDetection(EmbedderCase):
    """`detect_dimension` exists so `vector_size` is not hand-typed."""

    def test_it_reports_the_width_the_endpoint_returns(self):
        self.assertEqual(self.serving("ok").detect_dimension(), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
