"""The statusLine command: claude-code's own context window, handed to the big-file guard.

WHY A STATUS LINE. claude-code computes the window of the model selected now
(`context_window.context_window_size`) and hands it to exactly one external process: the
statusLine command. Measured on 2.1.282, running an interactive session: it runs when the
REPL mounts, after the SessionStart hooks and before any prompt; ~300 ms after every
assistant message; and ~0.1 s after a `/model`, already with the new model and window. The
hook payloads carry no model and no window. So this command publishes what it receives
(`core.hostwindow`), and the guard reads it by the session id its own payload carries.

`claude -p` runs no statusLine at all; there, the guard falls back to the config.

NEVER LOUD. A status line that raises prints a traceback into the user's screen after every
message, so `main` prints a line and returns 0 whatever happens, and it is dispatched before
the config is loaded: a broken config must not cost the line.

WHY IT IS INSTALLED IN THE USER'S SETTINGS. A plugin cannot declare a statusLine; it is a
user setting. `install` adds ours to `~/.claude/settings.json` and nothing else, and never
replaces a status line somebody else put there.
"""
import json
import os
import sys
from pathlib import Path

from . import hostwindow

#: How the guard tells this host's reports apart from hermes'.
SOURCE = "claude-code"

#: The tail that identifies our command, whatever path the launcher has on that machine.
COMMAND_TAIL = "qctx statusline"


def _context(payload: dict) -> dict:
    context = payload.get("context_window")

    return context if isinstance(context, dict) else {}


def _window_of(payload: dict) -> int:
    size = _context(payload).get("context_window_size")
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return size

    return 0


def _model_of(payload: dict) -> str:
    model = payload.get("model")
    if isinstance(model, dict):
        return str(model.get("id") or "")

    return str(model or "")


def human(tokens: int) -> str:
    """1M, 1.5M, 200k: what a person reads in a status line."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"

    return f"{round(tokens / 1000)}k"


def render(payload: dict) -> str:
    """`ctx 23% · 1M`, or less when claude-code has not said that much yet."""
    line = "ctx"
    used = _context(payload).get("used_percentage")
    if isinstance(used, (int, float)) and not isinstance(used, bool):
        line += f" {round(used)}%"
    window = _window_of(payload)
    if window:
        line += f" · {human(window)}"

    return line


def publish_from(payload: dict) -> bool:
    """Record the window claude-code reported for this session. False when it gave none."""
    return hostwindow.publish(str(payload.get("session_id") or ""), _model_of(payload),
                              _window_of(payload), SOURCE)


def main(stdin=None, stdout=None) -> int:
    """Read the payload, publish, print one line. Always 0."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    line = "ctx"
    try:
        payload = json.loads(stdin.read() or "null")
        if isinstance(payload, dict):
            publish_from(payload)
            line = render(payload)
    except Exception:  # noqa: BLE001 -- see the module docstring: never loud
        pass
    try:
        stdout.write(line + "\n")
        stdout.flush()
    except Exception:  # noqa: BLE001
        pass

    return 0


def settings_path() -> Path:
    """claude-code's user settings, where a status line is configured."""
    return Path.home() / ".claude" / "settings.json"


def state(settings: Path | None = None) -> str | None:
    """What `qctx setup` reports about the status line, read without writing anything.

    None where claude-code is not set up on this machine (no `~/.claude`): a hermes-only
    machine must not be told to install a claude-code status line. Otherwise the state
    `install` would start from (`installed`, `missing`, `foreign`, `unreadable`).
    """
    settings = Path(settings) if settings is not None else settings_path()
    if not settings.parent.is_dir():
        return None

    return install(settings, "", apply=False)[0]


def is_ours(command) -> bool:
    return isinstance(command, str) and command.strip().endswith(COMMAND_TAIL)


def install(settings: Path, command: str, apply: bool) -> tuple[str, str]:
    """Put our statusLine in `settings`, or say why not. Returns (state, detail):

      * `installed`: ours is already there (detail: the command found);
      * `added`: it was missing and `apply` wrote it (detail: the command written);
      * `missing`: it is missing and this was a dry run (detail: what would be written);
      * `foreign`: another status line is configured, and it is left alone;
      * `unreadable`: the file is not a JSON object, and it is left alone;
      * `failed`: the write itself failed, and the file is as it was.

    The write goes to a temporary in the same directory and replaces the file in one step,
    keeping its mode and every other key.
    """
    settings = Path(settings)
    try:
        text = settings.read_text()
    except FileNotFoundError:
        text = None
    except OSError as exc:
        return "unreadable", f"{settings}: {exc}"
    data = {}
    if text is not None:
        try:
            data = json.loads(text)
        except ValueError:
            return "unreadable", f"{settings} is not valid JSON"
        if not isinstance(data, dict):
            return "unreadable", f"{settings} does not hold a JSON object"

    current = data.get("statusLine")
    if current is not None:
        found = current.get("command") if isinstance(current, dict) else current
        if is_ours(found):
            return "installed", str(found)

        return "foreign", str(found)

    if not apply:
        return "missing", command

    data["statusLine"] = {"type": "command", "command": command}
    mode = (settings.stat().st_mode & 0o777) if text is not None else 0o600
    temporary = settings.with_name(f".{settings.name}.qctx-{os.getpid()}.tmp")
    try:
        settings.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        os.chmod(temporary, mode)
        os.replace(temporary, settings)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        return "failed", f"{settings}: {exc}"

    return "added", command
