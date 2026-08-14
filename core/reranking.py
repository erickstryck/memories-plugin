"""Re-rank (cross-encoder) client, with the wire contract as a STRATEGY.

Two things live here, and neither is the retrieval algorithm — that one is in
`retrieval`, shared. This is only the conversation with the server:

1. The WIRE CONTRACT, which varies per server implementation. It used to be an `if`
   inside the ranking method, looking at the URL suffix. Now each shape is a small
   class: adding a new server means writing a strategy and registering it, without
   touching the path that already works (OCP). The `if` also hid the fact that the two
   shapes differ in the BODY and in how the response is READ, not just in the path.

2. The SCALE NORMALIZATION, which is mandatory rather than optional: the SAME model
   returns a sigmoid (0..1) on one server and a raw logit on another. Measured with
   bge-reranker-v2-m3 on the same irrelevant document: 1.6e-05 on one server and
   -11.04 on the other, where -11.04 is exactly logit(1.6e-05). A cutoff calibrated on
   one scale is INERT on the other — 0.10 on a logit scale sits in the middle of the
   distribution and accepts nearly everything. The failure is silent: no error, just
   worse relevance.
"""
import math
from dataclasses import dataclass

from .errors import CoreError
from .http import HttpError, bearer, post_json


class RerankError(CoreError):
    pass


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)  # avoids overflow when x is very negative

    return e / (1.0 + e)


def normalize_scores(pairs: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], bool]:
    """Converts to the 0..1 range when the response came back as logits.

    It detects by RANGE and not by configuration: configuration is wrong silently when
    someone swaps the server and forgets to update it. A useful equivalence to keep in
    mind: sigmoid 0.10 corresponds to logit -2.197.

    Returns (normalized pairs, was_logit).
    """
    if not pairs:
        return [], False
    was_logit = any(s < 0.0 or s > 1.0 for _, s in pairs)
    if was_logit:
        pairs = [(i, sigmoid(s)) for i, s in pairs]

    return pairs, was_logit


# ---- wire contract strategies ----------------------------------------------

@dataclass(frozen=True)
class WireContract:
    """How to build the request and how to read the response of a re-rank server."""
    name: str

    def body(self, model: str, query: str, documents: list[str]) -> dict:
        raise NotImplementedError

    def parse(self, response: dict) -> list[tuple[int, float]]:
        raise NotImplementedError


class JinaContract(WireContract):
    """`{model, query, documents}` -> `results[].relevance_score`.

    The shape exposed by servers following JinaAI's rerank API, including the
    `/rerank` endpoint of llama.cpp and of vLLM.
    """

    def __init__(self):
        super().__init__(name="jina")

    def body(self, model: str, query: str, documents: list[str]) -> dict:
        return {"model": model, "query": query, "documents": documents}

    def parse(self, response: dict) -> list[tuple[int, float]]:
        return _pairs(response.get("results") or [], "relevance_score")


class ScoreContract(WireContract):
    """`{model, text_1, text_2}` -> `data[].score`.

    The shape of the `/score` endpoint, which exists alongside `/rerank` on some
    servers and is the only one available on others.
    """

    def __init__(self):
        super().__init__(name="score")

    def body(self, model: str, query: str, documents: list[str]) -> dict:
        return {"model": model, "text_1": query, "text_2": documents}

    def parse(self, response: dict) -> list[tuple[int, float]]:
        return _pairs(response.get("data") or [], "score")


def _pairs(rows: list, field: str) -> list[tuple[int, float]]:
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        value = row.get(field, row.get("relevance_score", row.get("score")))
        if not isinstance(idx, int) or value is None:
            continue
        output.append((idx, float(value)))

    return output


def contract_for(url: str) -> WireContract:
    """Picks the strategy from the address. A `/score` suffix uses the score shape;
    everything else uses the Jina shape, which is the more common one."""
    return ScoreContract() if url.rstrip("/").endswith("score") else JinaContract()


# ---- client -----------------------------------------------------------------

class Reranker:
    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 15.0,
                 max_docs: int = 12, doc_chars: int = 8000, query_chars: int = 2000,
                 contract: WireContract | None = None):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_docs = max_docs
        self.doc_chars = doc_chars
        self.query_chars = query_chars
        self.contract = contract or contract_for(url)

    def rank(self, query: str, documents: list[str]) -> tuple[list[tuple[int, float]], dict]:
        """Reorders `documents` for `query`. NEVER raises.

        Returns (pairs sorted by score desc, info carrying `ok`). Failure is
        DEGRADATION, not an exception: re-ranking improves the ordering and is never a
        prerequisite for it, and the caller has to be able to carry on with the first
        stage. But `ok` MUST be checked — a pipeline that relaxes the first cut
        counting on the second ends up worse than the single stage when the second
        fails silently.

        A cross-encoder does one forward pass per (query, document) pair and has no
        precomputable vector: the cost is linear in the total token count. Hence the
        ceilings — the cut applies to the JUDGEMENT, not to what the caller passes on.
        """
        info = {"ok": False, "was_logit": False, "dropped": 0,
                "error": None, "contract": self.contract.name}
        if not documents:
            return [], info

        info["dropped"] = max(0, len(documents) - self.max_docs)
        # Preparing the input stays inside the try for the same reason reading the
        # response does. `d[:self.doc_chars]` raises TypeError on anything that is not a
        # string, and "NEVER raises" has to hold against a caller's mistake too —
        # otherwise the promise is only kept while nothing goes wrong, which is when no
        # promise is needed. Reachable today only because `_flatten_hit` guarantees
        # strings; guarantees held by a distant collaborator are the ones that lapse.
        try:
            candidates = [d[:self.doc_chars] for d in documents[:self.max_docs]]
            response = post_json(self.url,
                                 self.contract.body(self.model, query[:self.query_chars], candidates),
                                 headers=bearer(self.api_key), timeout=self.timeout)
            pairs = [(i, s) for i, s in self.contract.parse(response)
                     if 0 <= i < len(candidates)]
        except HttpError as exc:
            info["error"] = str(exc)

            return [], info
        except Exception as exc:  # unexpected response shape, and whatever else comes
            info["error"] = f"{type(exc).__name__}: {exc}"

            return [], info
        if not pairs:
            info["error"] = f"response with no usable hits: {str(response)[:200]}"

            return [], info

        pairs, was_logit = normalize_scores(pairs)
        pairs.sort(key=lambda p: -p[1])
        info["ok"] = True
        info["was_logit"] = was_logit

        return pairs, info
