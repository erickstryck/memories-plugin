"""The final summary block: what the install did, where the stack is, and
what it runs with, in the lines `_finish` prints on every platform.

`render_summary` is PURE: it returns the lines and prints nothing, so the
installer's reporter stays the only thing that talks to the terminal and a
test can drive the block with a `Plan` and a plain config dict. The three
URLs come from the plan's ports (`verify.stack_urls`), not from the saved
config: when the user declines the config write the stack is still RUNNING
on the plan's ports, and the file's old URLs would be a false claim. The
api-key is the one thing read from the passed config (the re-read disk
file): `config_patch` never writes `qdrant_api_key`, so what is there is
the user's existing setting — and the truth in the decline case. The
engine-VM RAM line names, on windows (WSL2), that the RAM figure the
hardware check measured is the engine's VM (docker-desktop), not the
Windows host's: on WSL2 every distro is its own VM, separate from the one
the containers run in, so the number must not be mistaken for the host's.
"""
from . import compose, verify

#: The api-key line when the disk's config carries no key: the local Qdrant
#: runs keyless by default, and the block names that instead of showing an
#: empty value.
NO_KEY = "(nenhuma: Qdrant local sem chave)"


def render_summary(saved_config: dict, plan: "compose.Plan") -> list[str]:
    """The summary lines, in the order the installer prints them.

    `saved_config` is the re-read disk config (the installer hands it in
    from `deps.config.current_file()`); `plan` is the plan the stack was
    stood up from.
    """
    urls = verify.stack_urls(plan.ports)
    lines = [
        f"stack: running ({plan.runtime}, {plan.backend})",
        f"qdrant   {urls['qdrant_url']}",
        f"embed    {urls['api_base_url']}",
        f"rerank   {urls['rerank_url']}",
        f"api-key  {(saved_config.get('qdrant_api_key') or NO_KEY)}",
    ]
    if plan.platform == "windows":
        # WSL2: the RAM the step-3 check read is the engine's VM, not the
        # Windows host's — the line names it so the figure is not mistaken
        # for the host's RAM.
        lines.append(
            "ram      the RAM check measured the engine VM (docker-desktop), "
            "not the Windows host")
    return lines
