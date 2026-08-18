"""Ask the endpoint that serves the model how big its context window is.

WHY A LIST OF PATHS AND NOT A FIELD. Three servers were measured on 2026-08-17 and each put
it somewhere different: vLLM at `max_model_len`, OpenRouter at `context_length`, and
llama.cpp NESTED under `meta.n_ctx` — a search that only looks at the top level misses that
one entirely.

WHY THE ORDER, AND WHY IT INVERTS THIS FEATURE'S USUAL RULE. OpenRouter publishes two
context lengths that disagree on 34 of its 414 models: `context_length` is the largest across
every provider serving that model, and `top_provider.context_length` is the one the request
actually reaches (measured: 1,000,000 against 262,144 for nvidia/nemotron-3.5-lightning).
Everywhere else in this feature erring LARGE is the safe direction, because a window too large
only makes the guard sleep. Here the aggregate is too large BY CONSTRUCTION: asking for a
million when the provider delivers 262k makes the guard sleep exactly when it needed to wake.
So the rule refines rather than contradicts — prefer the value most likely to be REAL, and
leave ceilings to the step of the cascade whose job is ceilings.

THIS LIST IS INCOMPLETE, and saying so is the point. It came from measuring three servers. A
provider using another name, or another nesting level, yields nothing here and the cascade
falls through — which is the correct answer for "we do not know", and better than a heuristic
that estimates a window from something else.

NOT VERIFIED: OpenAI and Anthropic native endpoints, which need credentials that were not
available. Nothing here claims anything about them; they take the unknown-shape path.
"""
from . import http

#: Ordered. First path that yields a positive integer wins. See the module docstring for why
#: `top_provider` precedes the aggregate.
PATHS = (
    ("top_provider", "context_length"),   # OpenRouter — the provider actually reached
    ("max_model_len",),                   # vLLM
    ("context_length",),                  # OpenRouter aggregate, and others
    ("meta", "n_ctx"),                    # llama.cpp — nested, not at the top
)


def window_from(entry: dict) -> int:
    """The window this model entry declares, or 0 when it declares none we recognise."""
    if not isinstance(entry, dict):
        return 0
    for path in PATHS:
        node = entry
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, bool) or not isinstance(node, (int, float, str)):
            continue
        try:
            value = int(node)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value

    return 0


def probe(base_url: str, api_key: str, model: str, *, timeout: float = 5.0, fetch=None) -> int:
    """The window `model` has on the server at `base_url`, or 0 when it could not be learned.

    Returns 0 for every failure — unreachable, malformed, model absent, shape unrecognised —
    because 0 is what the cascade reads as "ask the next step". It never raises: a caller
    doing this opportunistically must not be interrupted by a server being down.

    `fetch` is injected only so tests can drive the shapes without a server.
    """
    url = (base_url or "").rstrip("/") + "/models"
    call = fetch or (lambda u, **kw: http.request_json(u, **kw))
    try:
        payload = call(url, headers=http.bearer(api_key), timeout=timeout)
    except Exception:                      # noqa: BLE001 — see docstring
        return 0
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return 0
    wanted = (model or "").strip()
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id", "")).strip() == wanted:
            return window_from(entry)

    return 0
