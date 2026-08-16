"""How large is the context window? Neither host can tell us from disk.

Measured on 2026-08-15: the claude-code transcript records the MODEL (`claude-opus-5`)
and not the window; hermes has `sessions.model`, a `model_config` whose `max_tokens` is
the OUTPUT cap, and a `context_length` that lives only on the in-process compressor.

So the table below is a guess and the operator's config is the truth. The guess exists so
the guard works out of the box; the override exists because the guess WILL be wrong — the
session this was designed in ran a 1M variant of a model whose bare name maps to 200k
here, a 5x error that would have been silent.
"""

#: Best-effort, by bare model name. Wrong for any variant that changes the window without
#: changing the name — which is exactly why `context_window` in the config wins.
MODEL_WINDOWS = {
    "claude-opus-5": 200_000,
    "claude-sonnet-5": 200_000,
    "claude-haiku-4-5": 200_000,
}


def window_for(model: str, cfg) -> int:
    """Tokens the context window holds, or 0 when we do not know.

    0 is load-bearing: the caller must ALLOW the read when the window is unknown. Blocking
    on a guessed window is the one failure this whole guard must not produce.
    """
    declared = getattr(cfg, "context_window", 0)
    try:
        declared = int(declared)
    except (TypeError, ValueError):
        declared = 0
    if declared > 0:
        return declared

    return int(MODEL_WINDOWS.get((model or "").strip(), 0))
