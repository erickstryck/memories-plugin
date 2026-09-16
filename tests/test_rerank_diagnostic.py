"""The re-rank client's success path and the diagnostic built on it.

Two guards were mutated by a design review and the suite stayed green on both:
`core/setup.py::_check_rerank`'s `top_is_right` — the check that tells a user their model
ranked the WRONG answer first, which is how someone learns they pointed the setting at an
embedding model instead of a cross-encoder — and the whole success path of
`Reranker.rank`.

Both were unreachable for the same structural reason: every test config passes
`rerank_url=""`, so `_check_rerank` always left through its `ConfigError` branch, and the
real `Reranker` was only ever constructed against 127.0.0.1:1 to be refused. The failure
half was well covered; the half that runs when the server WORKS had nothing.

`info["contract"]` is the sharpest case. The diagnostic reports it to the user as a fact —
"answers in sigmoid 0..1 via the jina contract" — and nothing verified that the name
matches the contract that actually parsed the answer.

These drive the real client and the real diagnostic over a loopback server speaking each
supported wire contract, in the shape `tests/fakes.py::qdrant_with_no_collections` already
uses.
"""
import http.server
import json
import sys
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.config import Config  # noqa: E402
from core.reranking import Reranker  # noqa: E402
from core.setup import _check_rerank  # noqa: E402


class _RerankServer(http.server.BaseHTTPRequestHandler):
    """A re-rank endpoint. `mode` decides which contract it speaks and how well it ranks."""

    mode = "jina"

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        sent = json.loads(raw)
        mode = type(self).mode
        # A server that is reachable but REFUSES. This is the common real failure: the
        # endpoint is up, the model named in the setting is not loaded on it.
        if mode == "http_error":
            body = json.dumps({"error": "model 'a-cross-encoder' not found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if mode == "garbage":
            body = b"<html>502 Bad Gateway</html>"
            self.send_response(200)          # 200 carrying something that is not JSON
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # The documents the caller sent, in order. `_check_rerank` sends the right answer
        # first, so "good" means index 0 wins and "wrong" means it does not.
        count = len(sent.get("documents") or [sent.get("text_2")])
        if mode == "wrong_first":
            scores = [0.1] + [0.9] * (count - 1)
        elif mode == "logit":
            scores = [4.0] + [-3.0] * (count - 1)       # raw logits, not 0..1
        else:
            scores = [0.9] + [0.1] * (count - 1)
        if mode == "score_contract":
            body = {"data": [{"index": i, "score": s} for i, s in enumerate(scores)]}
        else:
            body = {"results": [{"index": i, "relevance_score": s}
                                for i, s in enumerate(scores)]}
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


class RerankCase(unittest.TestCase):
    def serving(self, mode: str) -> str:
        _RerankServer.mode = mode
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RerankServer)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        return f"http://127.0.0.1:{server.server_address[1]}/rerank"

    def a_config(self, url: str) -> Config:
        return Config(qdrant_url="http://127.0.0.1:1", qdrant_api_key="", api_base_url="",
                      api_key="", embed_url="http://127.0.0.1:1", rerank_url=url,
                      embed_model="e", rerank_model="a-cross-encoder",
                      memory_collection="m", docs_collection="d", library_collection="l",
                      repos_collection="repos", repos_registry_collection="reg",
                      vector_size=1024)


class TestTheDiagnosticReadsTheRanking(RerankCase):
    """`qctx setup --check` has to say when the model ranked the wrong answer first.

    It is the only signal a user gets that they pointed the setting at something that is
    not a cross-encoder — an embedding endpoint answers, and answers plausibly, so nothing
    else in the report looks wrong."""

    def test_a_model_that_ranks_the_RIGHT_answer_first_passes(self):
        check = _check_rerank(self.a_config(self.serving("jina")))
        self.assertTrue(check.ok, check.detail)
        self.assertNotIn("WRONG", check.detail)

    def test_a_model_that_ranks_the_WRONG_answer_first_is_reported(self):
        check = _check_rerank(self.a_config(self.serving("wrong_first")))
        self.assertFalse(check.ok, "a model ranking the wrong answer first passed the check")
        self.assertIn("WRONG", check.detail)
        self.assertIn("cross-encoder", check.fix_hint or "",
                      "the hint does not name the likely cause")

    def test_a_wrong_ranking_is_a_WARNING_not_a_blocker(self):
        """Re-ranking is optional: a bad reranker must not make the package read as unusable."""
        check = _check_rerank(self.a_config(self.serving("wrong_first")))
        self.assertTrue(check.warning, "an optional component blocked the install")


class TestTheReportedScaleIsTheMeasuredOne(RerankCase):
    """The detail line states the scale as a fact, so it must come from the answer."""

    def test_a_server_answering_in_0_to_1_is_reported_as_sigmoid(self):
        check = _check_rerank(self.a_config(self.serving("jina")))
        self.assertIn("sigmoid 0..1", check.detail)
        self.assertNotIn("logit", check.detail)

    def test_a_server_answering_in_RAW_LOGITS_is_reported_as_normalized(self):
        check = _check_rerank(self.a_config(self.serving("logit")))
        self.assertIn("logit", check.detail,
                      "a raw-logit server was reported as answering in sigmoid")


class TestTheReportedContractIsTheONEThatParsed(RerankCase):
    """`info["contract"]` is shown to the user as fact; a stale name is a lie about wire."""

    def test_the_jina_shape_reports_the_jina_contract(self):
        _, info = Reranker(self.serving("jina"), "m", timeout=5).rank("q", ["a", "b"])
        self.assertTrue(info["ok"], info.get("error"))
        self.assertEqual(info["contract"], "jina")

    def test_the_score_shape_reports_the_score_contract(self):
        url = self.serving("score_contract").replace("/rerank", "/score")
        _, info = Reranker(url, "m", timeout=5).rank("q", ["a", "b"])
        self.assertTrue(info["ok"], info.get("error"))
        self.assertEqual(info["contract"], "score")


class TestRankOrdersByScore(RerankCase):
    """The pairs come back best-first, whatever order the server used."""

    def test_the_best_scoring_document_is_first(self):
        pairs, info = Reranker(self.serving("wrong_first"), "m", timeout=5).rank(
            "q", ["the right one", "a distractor"])
        self.assertTrue(info["ok"], info.get("error"))
        self.assertEqual(pairs[0][0], 1, "it did not order by score")
        self.assertGreater(pairs[0][1], pairs[-1][1])


class TestTheDiagnosticReportsAServerThatREFUSES(RerankCase):
    """A re-rank server that is up but will not answer must be reported as not working.

    THE HOLE THIS CLOSES, MEASURED: deleting `if not info["ok"]:` from `_check_rerank`, and
    separately flipping its Check to `ok=True`, both left the entire suite green. Every
    other test in this file drives the SUCCESS path, so the branch that handles failure was
    never entered — `qctx setup --check` would have reported a dead re-ranker as healthy,
    which is the exact opposite of what the command exists for.

    This is the most likely real failure, not an exotic one: the endpoint is reachable and
    the model named in the setting is simply not loaded on it."""

    def test_a_server_that_answers_404_is_reported_as_failing(self):
        check = _check_rerank(self.a_config(self.serving("http_error")))

        self.assertFalse(check.ok, f"a refusing server was reported as working: {check.detail}")
        self.assertIn("failed", check.detail.lower())

    def test_the_failure_says_WHERE_it_failed(self):
        """The URL has to be in the message: the setting is the thing most likely wrong."""
        url = self.serving("http_error")
        check = _check_rerank(self.a_config(url))

        port = url.rsplit(":", 1)[1].split("/")[0]
        self.assertIn(port, check.detail,
                      f"the report does not say which endpoint failed: {check.detail!r}")

    def test_a_failure_is_a_WARNING_not_a_blocker(self):
        """Re-rank is optional, and `setup --check` says so. A missing re-ranker must not
        make someone conclude the package is broken."""
        check = _check_rerank(self.a_config(self.serving("http_error")))

        self.assertTrue(check.warning,
                        "an optional component's failure was raised to a blocker")

    def test_a_server_answering_NON_JSON_is_reported_as_failing(self):
        """200 carrying an HTML error page, which is what a proxy in front of a dead
        backend returns. The status line says fine; the body is not a ranking."""
        check = _check_rerank(self.a_config(self.serving("garbage")))

        self.assertFalse(check.ok, f"a non-JSON answer was accepted: {check.detail}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
