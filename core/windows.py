"""How large is the context window? Neither host can tell us from disk.

Measured on 2026-08-15: the claude-code transcript records the MODEL (`claude-opus-5`)
and not the window; hermes has `sessions.model`, a `model_config` whose `max_tokens` is
the OUTPUT cap, and a `context_length` that lives only on the in-process compressor.

So the table below is a fallback and the operator's config is the truth. But the fallback
is not a GUESS AT THE LIKELY WINDOW — it is a CEILING, and the difference is the whole
reason this file was amended on 2026-08-16.

WHY A CEILING AND NOT A BEST GUESS. The transcript records the BARE name: `claude-opus-5`
appears 82 times in the session that found this, and the `[1m]` suffix appears nowhere. So
a 200k variant and a 1M variant are INDISTINGUISHABLE here, and no amount of care picks
the right number for both. It is a choice, not an engineering problem — and the two
directions are not symmetric:

  * too LARGE only makes the guard sleep: it under-fires, and the read goes through, which
    is the same thing that happens when the window is unknown;
  * too SMALL INVERTS the feature. Measured with the original table: `window_for` returned
    200_000 in a real 1M session whose `used` was 989,479, so `free` came out 0 and the
    real hook denied a 4 KB file. Fail open had become fail closed.

Erring large is therefore the only side whose worst case is the failure mode this design
already accepted. `core.bigfile.decide` carries the other half of the amendment: `used >=
window` means the ceiling has been REFUTED by facts, and it falls back to "unknown".

DO NOT "FIX" THESE VALUES BACK DOWN to a model's nominal window. A smaller number here does
not make the guard sharper; it makes it a cage on the variant that outgrew the name.
"""

#: The LARGEST window any variant of that bare name can have — a ceiling, not a nominal
#: window. `context_window` in the config still wins, and is the only way to get the guard
#: to fire early for someone genuinely on a small variant.
#:
#: Each value comes from the claude v2.1.233 binary, not from recollection — but the three
#: rows do NOT rest on equally strong evidence, and saying so is the point:
#:   * `claude-opus-5` — the binary ships `claude-opus-5[1m]` as a model id. A 1M variant
#:     demonstrably EXISTS. Strongest of the three.
#:   * `claude-sonnet-5` — WEAKER, and deliberately kept at 1M anyway. There is NO
#:     `claude-sonnet-5[1m]` id in the binary (the only sonnet `[1m]` ids are
#:     `sonnet-4-5-20250929[1m]` and `sonnet-4-6[1m]`; all 64 mentions of sonnet-5 are of
#:     the bare name). What is true is weaker: sonnet-5 is NOT in `rHt`, the binary's
#:     "can never be 1M" predicate, and eligibility is then decided at RUNTIME by
#:     `oA(t)?.context?.supports_1m_beta` — data this file cannot see. So nothing rules a
#:     1M sonnet-5 out, and under the ceiling rule "not ruled out" is exactly what erring
#:     large means. Do not upgrade this justification to "the binary ships the id": it
#:     does not, and an earlier version of this comment claimed it did.
#:   * `claude-haiku-4-5` — strong again, and in the opposite direction: `rHt` enumerates it:
#:     `e.includes("claude-3-") || e==="claude-opus-4-0" || e==="claude-opus-4-1" ||
#:     e==="claude-opus-4-5" || e==="claude-haiku-4-5"`. Being named there is what makes
#:     200k a ceiling for it rather than a guess.
#: A name whose ceiling cannot be established this way belongs OUT of this table — absent
#: resolves to 0, which allows, and that is the correct answer for "we do not know".
MODEL_WINDOWS = {
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
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
