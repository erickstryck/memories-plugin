"""Records what the LIVE stack answers, so two versions can be compared answer for answer.

Not part of the suite: it needs the real Qdrant and the real model endpoints, the same ones
`tests/test_integration.py` needs. It only READS the configured archives. Everything it writes
(session state, logs, the hermes home) goes to temporary directories it creates, so running it
leaves the user's state untouched and cannot start a daemon.

    python3 scripts/regression_probe.py --root <tree> > probe.json

`--root` is the tree whose code answers, so the same script drives an older checkout in a
worktree and the current one. Run it twice against the same tree before trusting a diff: a line
that differs between two runs of ONE version is noise, not a regression.

What it covers is every surface a user or a model reaches: both hooks and the hermes provider
(the automatic recall), the CLI commands that read, and the tool list the model is offered.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

#: Prompts chosen to hit the three block states on this archive: two that find memories, one
#: that should find nothing, one the query builder skips as too short.
PROMPTS = [
    "como funciona o claim do daemon e a escrita atomica de estado no mnemosine",
    "qual a sequencia de update do plugin no claude code a partir do github",
    "receita de bolo de cenoura com cobertura de chocolate",
    "ok",
]
REPO_QUERY = ("connector-sdk", "how is the webhook body decoded")
DOCS_SCOPE = "all"


def _run(root: str, argv: list[str], stdin: str = "", env: dict | None = None) -> dict:
    done = subprocess.run(argv, cwd=root, input=stdin, capture_output=True, text=True,
                          timeout=120, env=env)

    return {"code": done.returncode, "stdout": done.stdout, "stderr_empty": not done.stderr}


def _isolated_env(**extra) -> dict:
    env = dict(os.environ)
    env["QCTX_STATE_DIR"] = tempfile.mkdtemp(prefix="probe-state-")
    env["QCTX_DAEMON_AUTOSTART_DISABLED"] = "1"
    env.update(extra)

    return env


def hook_blocks(root: str) -> dict:
    """What the claude-code recall hook injects for each prompt, from a fresh session."""
    out = {}
    env = _isolated_env()
    for prompt in PROMPTS:
        payload = json.dumps({"prompt": prompt, "session_id": "probe"})
        res = _run(root, [sys.executable, "hooks/recall.py"], payload, env)
        body = json.loads(res["stdout"]) if res["stdout"].strip() else None
        out[prompt] = {"code": res["code"],
                       "context": _scores((body or {}).get("hookSpecificOutput", {})
                                          .get("additionalContext"))}

    return out


def hermes_blocks(root: str) -> dict:
    """What the hermes provider returns from `prefetch`, through the class the loader builds."""
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {root!r})\n"
        "from hosts.hermes import MemoriesProvider\n"
        "p = MemoriesProvider()\n"
        "assert p.is_available(), p.unavailable_reason()\n"
        "prompts = json.loads(sys.stdin.read())\n"
        "print(json.dumps({q: p.prefetch(q, session_id='probe') for q in prompts}))\n"
        "print(json.dumps(sorted(s['name'] for s in p.get_tool_schemas())))\n"
    )
    env = _isolated_env(HERMES_HOME=tempfile.mkdtemp(prefix="probe-hermes-"))
    res = _run(root, [sys.executable, "-c", code], json.dumps(PROMPTS), env)
    lines = res["stdout"].splitlines()
    if res["code"] != 0 or len(lines) < 2:
        return {"code": res["code"], "stdout": res["stdout"]}

    blocks = {q: _scores(b) for q, b in json.loads(lines[-2]).items()}

    return {"code": 0, "blocks": blocks, "tools": json.loads(lines[-1])}


def cli_reads(root: str) -> dict:
    """The CLI commands that only read, through the real launcher."""
    qctx = [sys.executable, "cli/qctx.py"]
    repo, query = REPO_QUERY
    commands = {
        "repos list": qctx + ["repos", "list", "--json"],
        "repos search": qctx + ["repos", "search", query, "--repo", repo, "--json"],
        "memory find": qctx + ["memory", "find", PROMPTS[0], "--json"],
        "memory recall": qctx + ["memory", "recall", PROMPTS[1], "--json"],
        "memory list": qctx + ["memory", "list", "--limit", "5", "--json"],
        "docs list": qctx + ["docs", "list", "--scope", DOCS_SCOPE, "--json"],
    }
    env = _isolated_env()
    out = {}
    for name, argv in commands.items():
        res = _run(root, argv, env=env)
        try:
            parsed = json.loads(res["stdout"])
        except ValueError:
            parsed = res["stdout"]
        out[name] = {"code": res["code"], "out": _normalise(name, parsed)}

    return out


def _normalise(name: str, value):
    """Drops what legitimately moves between two runs of the SAME version: the indexing
    counters and timestamps a live daemon rewrites, and float noise in scores. Measured: two
    runs of one version differed only in a cross-encoder score at the third decimal."""
    if name == "repos list" and isinstance(value, dict):
        return sorted(r.get("repo") for r in value.get("repos", value.get("entries", [])))
    if name == "repos list" and isinstance(value, list):
        return sorted(r.get("repo") for r in value)
    if name == "repos search" and isinstance(value, dict):
        return [(g.get("repo"), [(h.get("path"), h.get("start_line"), h.get("end_line"))
                                 for h in g.get("hits", [])])
                for g in value.get("groups", [])]

    return json.loads(json.dumps(value, default=str), parse_float=lambda s: round(float(s), 2))


def _scores(text):
    """Block text with its scores rounded, for the same reason `_normalise` rounds floats."""
    if not isinstance(text, str):
        return text

    return re.sub(r"\b(CE|dense) (\d\.\d+)", lambda m: f"{m[1]} {float(m[2]):.2f}", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the tree whose code answers")
    root = os.path.abspath(ap.parse_args().root)
    print(json.dumps({"hook": hook_blocks(root), "hermes": hermes_blocks(root),
                      "cli": cli_reads(root)}, indent=1, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
