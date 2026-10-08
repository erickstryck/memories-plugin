"""The terminal seam for the stack: the `qctx stack` group and the step in `qctx install`.

This module OWNS the presentation/IO of the stack, and nothing else: it translates argparse
into the already-reviewed use cases and prints their answers in the wizard's style. `register`
adds the `stack` group, `add_install_flags` adds the three install flags, `install_step` is the
"Quando aparece" decision table (a dispatcher over provision / status / check_section, not a
reimplementation), `check_section` reads state and health only (never a runtime) for the
report and `--json`, and `section_lines` turns a section dict into report lines. It also owns
the three protocol implementations the use cases run against: the terminal questions
(`TerminalPrompter`), the terminal output (`TerminalReporter`) and the plugin's config file
(`CoreConfigSink`).

THE LAZY-IMPORT RULE is the point of the module. `qctx statusline` builds the parser on every
assistant message, so the heavy `stack` modules (installer, lifecycle, runtimes, process) must
not be imported at the top of this file or of `cli/qctx.py`: they are imported inside the
functions that use them. `state` is lazy too, because it reaches `process` through `compose`
and `backends`. Only the light `catalog` and `health` are at the top. A fresh-interpreter test
in `tests/test_stack_cli.py` builds the parser and asserts the heavy modules are not in
`sys.modules` -- that test is load-bearing, not style.
"""
import json
import os
import sys
from typing import Callable

import core
import core.config as core_config

from . import StackError, catalog, health


# ---- the terminal implementations of the use-case protocols -----------------


class TerminalPrompter:
    """The `Prompter` the use cases run against on a real terminal: `input()`, with
    end-of-input read as the default (a closed stdin is not an error, it is the answer
    "keep what is there" -- the same rule the wizard's own `_ask` follows)."""

    def _ask(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            print()
            return ""

    def ask(self, prompt: str) -> str:
        return self._ask(prompt)

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        """A yes/no. On a terminal `input()`; with no terminal (the agent and the script
        paths) the `default` is used, so a prompt never blocks a caller with no stdin."""
        if not sys.stdin.isatty():
            return default
        answer = self._ask(prompt + " ").strip().lower()
        if not answer:
            return default
        return answer in ("y", "yes")

    def choose(self, title: str, lines: list, default: int) -> int:
        """A pick from a numbered menu. With no terminal the default is chosen, so the
        menu never blocks; the options are still printed, for the log."""
        print(title)
        for index, line in enumerate(lines, 1):
            print(f"  {index}. {line}")
        if not sys.stdin.isatty():
            return default
        while True:
            answer = self._ask(f"[{default + 1}]: ").strip()
            if not answer:
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(lines):
                return int(answer) - 1


class TerminalReporter:
    """The `Reporter` the use cases run against on a real terminal: the wizard's marks.
    `bar` hands the download a progress line that rewrites itself on a TTY and prints one
    line per slice off a TTY (the `ProgressBar` owns the two shapes)."""

    def step(self, text: str) -> None:
        print(f"\n{text}")

    def ok(self, text: str) -> None:
        print(f"  ok    {text}")

    def info(self, text: str) -> None:
        print(f"  ..    {text}")

    def warn(self, text: str) -> None:
        print(f"  warn  {text}")

    def fail(self, text: str) -> None:
        print(f"  FAIL  {text}")

    def bar(self, total: int, label: str, start: int):
        import stack.fetch as fetch
        return fetch.ProgressBar(total, label, tty=sys.stdout.isatty(),
                                 write=_stdout_write, start=start)


def _stdout_write(text: str) -> None:
    """The `ProgressBar`'s writer: a bare write (its return is a count, the bar wants
    `None`)."""
    sys.stdout.write(text)


class CoreConfigSink:
    """The `ConfigSink` the installer and the lifecycle read and write through: the plugin's
    real config file. `current_file` is what is on disk, `save` writes only the fields the
    step owns, and `effective` resolves the config the way every other command does
    (environment over the file over the defaults)."""

    def current_file(self) -> dict:
        return core_config.read_file()

    def save(self, patch: dict) -> None:
        core.save(patch)

    def effective(self):
        return core.load()


# ---- the `qctx stack` group --------------------------------------------------


def register(sub, *, ask: Callable[[str], str]) -> None:
    """Add the `stack` subcommand group to the top-level parser's subparsers. `ask` is
    the one question `remove --purge-data` needs (type the project to confirm); it is
    passed in so the group does not own the input itself."""
    stack = sub.add_parser("stack", help="manage the local stack (qdrant and the two "
                                         "llama-servers, in containers)")
    verb = stack.add_subparsers(dest="stack_cmd", required=True)

    verb.add_parser("status",
                    help="report the managed stack: phase, health, pins, config, boot"
                    ).set_defaults(fn=lambda args, cfg: _cmd_status(args))

    up = verb.add_parser("up", help="start the stack again from stack.json")
    up.add_argument("--upgrade", action="store_true",
                    help="pull the catalogue's images (and the --image overrides)")
    up.add_argument("--image", action="append", default=[],
                    help="ROLE=REF, to override one image (llama or qdrant)")
    up.set_defaults(fn=lambda args, cfg: _cmd_up(args))

    verb.add_parser("down", help="stop the stack, keeping its volume and models"
                    ).set_defaults(fn=lambda args, cfg: _cmd_down(args))

    remove = verb.add_parser("remove", help="delete the stack (and, when asked, its "
                                            "models and its Qdrant volume)")
    remove.add_argument("--purge-models", action="store_true",
                        help="also delete the downloaded models")
    remove.add_argument("--purge-data", action="store_true",
                        help="also delete the Qdrant volume (the archive)")
    remove.add_argument("--yes", action="store_true",
                        help="answer yes to the confirmation")
    remove.set_defaults(fn=lambda args, cfg: _cmd_remove(args, ask))


def _life_deps(args, ask):
    """The `LifeDeps` for a lifecycle verb: the discovery is the CLI's job (the lifecycle
    never re-probes the host), the runner reads `systemctl`/`loginctl` for the boot line,
    and the status probe is the real one. Every heavy import is here, so the group's
    registration never loads them. `ask` is the group's question, routed into the
    prompter (the `remove --purge-data` confirmation uses it)."""
    import stack.lifecycle as lifecycle
    import stack.process as process
    import stack.state as state
    return lifecycle.LifeDeps(
        runtimes=_discover(), reporter=TerminalReporter(),
        prompter=_prompter(ask), config=CoreConfigSink(),
        stack_dir=state.stack_dir(os.environ), runner=process.SubprocessRunner())


def _cmd_status(args) -> int:
    import stack.lifecycle as lifecycle
    section, code = lifecycle.status(_life_deps(args, None))
    if getattr(args, "json", False):
        print(json.dumps(section, ensure_ascii=False, indent=2, default=str))
        return code
    for line in section_lines(section):
        print(line)
    return code


def _cmd_up(args) -> None:
    import stack.lifecycle as lifecycle
    images = catalog.parse_image_flags(list(args.image or []))
    lifecycle.up(_life_deps(args, None), upgrade=args.upgrade, images=images or None)


def _cmd_down(args) -> None:
    import stack.lifecycle as lifecycle
    lifecycle.down(_life_deps(args, None))


def _cmd_remove(args, ask) -> None:
    import stack.lifecycle as lifecycle
    lifecycle.remove(_life_deps(args, ask), purge_models=args.purge_models,
                     purge_data=args.purge_data, yes=args.yes)


def _prompter(ask):
    """The `Prompter` for a lifecycle verb: a `TerminalPrompter`, but `ask` (the
    one question the caller wants to route through its own input) wins when given."""
    prompter = TerminalPrompter()
    if ask is not None:
        prompter.ask = ask
    return prompter


# ---- `qctx install`: the flags and the step ----------------------------------


def add_install_flags(parser) -> None:
    """The three flags the `qctx install` step reads: the profile, the runtime and the
    image overrides. `--image` appends, so several may be given."""
    parser.add_argument("--stack", choices=("auto", "cpu", "amd", "intel", "nvidia",
                                            "apple"), default=None,
                        help="stand up the local stack with this profile (auto picks the "
                             "best one the host serves)")
    parser.add_argument("--runtime", choices=("docker", "podman"), default=None,
                        help="the runtime that serves the stack, when both answer")
    parser.add_argument("--image", action="append", default=[],
                        help="ROLE=REF, to override one image (llama or qdrant)")


#: The blockers of `diagnose` that the local stack fixes: the two endpoints it stands up
#: beside Qdrant. The rerank check is a warning, not a blocker, so it is not here.
_STACK_BLOCKERS = ("Qdrant", "Embedding")


def _blocker_names(report: dict) -> set:
    return {c["name"] for c in report.get("blockers", ())}


def _confirm(ask: Callable[[str], str], prompt: str) -> bool:
    """A y/N answer from a plain `ask` (string in, string out): only `y`/`yes` is a yes,
    everything else (an empty Enter included, the default) is no. A raw `if ask(...)`
    would read the string "n" as truthy and proceed, which is the wrong way round."""
    return ask(prompt).strip().lower() in ("y", "yes")


def install_step(args, report: dict, *, budgets: list,
                 ask: Callable[[str], str], interactive: bool) -> None:
    """The step of `qctx install`, in the order the spec's "Quando aparece" fixes. It is a
    small dispatcher over the use cases, not a reimplementation: when it provisions, the
    whole twelve-step flow runs through `installer.provision`.

    - a managed stack: `running` prints the status summary and goes on; `stopped` offers to
      restart it (yes with `--yes`); an other phase (interrupted) resumes by provisioning
      again. A catalogue pin that moved past the recorded one points at `stack up --upgrade`.
    - no stack: `--stack` provisions; `--yes` alone prints the line that would provision and
      returns (downloading gigabytes is not an implied yes); a blocker the stack fixes
      explains what it would do and asks `y/N`; nothing to do is one line.
    A `StackError` rises to `main()`, which prints it with its step and fix.
    """
    import stack.facts as facts
    import stack.installer as installer
    import stack.state as state

    # The plan's first branch: on Windows (WSL included) the step is not offered in
    # phase 1 -- it says the one line and returns. The gate is cheap (host system +
    # kernel release only, no hardware probe) so it runs on every call, even the ones
    # that would end in a no-op.
    if facts.is_windows_host(facts.Probe()):
        _windows_line()
        return

    stack_dir = state.stack_dir(os.environ)
    st = state.load(stack_dir)  # a corrupt stack.json raises; it rises to main()

    if st is not None:
        _step_managed(st, stack_dir, args, report, budgets, ask, installer, facts)
        return

    # No managed stack. `--stack` is the consent to provision, in any case.
    if args.stack:
        _provision(args, stack_dir, budgets, ask, installer, facts)
        return
    if getattr(args, "yes", False):
        # `--yes` alone does NOT provision: it says the one flag that would.
        print("  ..    the stack is not required unless you pass --stack; to stand it up "
              "run: qctx install --yes --stack auto")
        return
    if _blocker_names(report) & set(_STACK_BLOCKERS):
        if not interactive:
            print("  ..    Qdrant or the embedding endpoint is not answering; a local stack "
                  "would stand them up (qctx install --stack auto)")
            return
        # The spec's "Quando aparece": before it asks, the offer explains what it would
        # do -- what it downloads and how much, the ports, where the data lives. The
        # numbers come from the catalogue so they cannot drift from the download
        # (review round R7, item 3).
        print("  ..    Qdrant or the embedding endpoint is not answering; a local stack "
              "would stand them up:")
        print(f"        downloads {catalog.MODELS_BYTES / 2 ** 20:.0f} MiB of models "
              f"into {stack_dir / state.MODELS_DIR}")
        print("        ports: " + ", ".join(f"{name} on {port}"
                                            for name, port in catalog.PORTS.items()))
        print(f"        data: {stack_dir}")
        if _confirm(ask, "stand up the local stack now? [y/N] "):
            _provision(args, stack_dir, budgets, ask, installer, facts)
            return
        print("  ..    the stack was not stood up; its endpoints stay as they are")
        return
    print("  ok    the endpoints are answering: a local stack is not needed")


def _step_managed(st, stack_dir, args, report, budgets, ask, installer, facts) -> None:
    """The managed-stack half of `install_step`: report, and offer the one action the
    state calls for. The pins are checked against the catalogue in either case.

    A `running` phase is not proof the stack is up: only `qctx stack down` writes
    `stopped`, so a reboot leaves the phase `running` while every endpoint answers
    nothing. For a `running` stack the section's health decides: healthy prints the
    status and goes on; dead endpoints (the reboot case) offer the cheap restart. A
    `stopped` stack is known-dead and offers the restart without a probe; the
    interrupted `compose` phase resumes by provisioning again."""
    import stack.state as state
    outdated = [role for role in catalog.IMAGES
                if st.images.get(role) != catalog.IMAGES[role]]
    if st.phase == state.PHASE_STOPPED:
        print("  ..    the stack is stopped; restart it with: qctx stack up")
        if args.yes or _confirm(ask, "restart the stack now? [y/N] "):
            _restart(args, stack_dir, ask)
            return
    elif st.phase == state.PHASE_RUNNING:
        section, _code = _status_of(st, stack_dir)
        if section.get("healthy") is False:
            print("  ..    the stack is not answering (a reboot leaves the phase "
                  "running); restart it with: qctx stack up")
            if args.yes or _confirm(ask, "restart the stack now? [y/N] "):
                _restart(args, stack_dir, ask)
                return
        for line in section_lines(section):
            print(line)
    else:
        # An other phase (compose): the last install was interrupted. Resume by
        # provisioning again -- the flow is idempotent in every step that can be.
        print("  ..    the last install was interrupted; it resumes by provisioning again")
        _provision(args, stack_dir, budgets, ask, installer, facts)
        return
    if outdated:
        print("  ..    the catalogue has newer pins for: " + ", ".join(outdated) +
              "; upgrade with: qctx stack up --upgrade")


def _status_of(st, stack_dir):
    """A `lifecycle.status` of the recorded stack, through the CLI's discovery.
    `lifecycle.status` reads state and the endpoints, never a runtime, so the runtimes
    list is left empty; the boot line is read from the runner. Used to report a running
    stack inside `install` (the report carries the live picture)."""
    import stack.lifecycle as lifecycle
    import stack.process as process
    deps = lifecycle.LifeDeps(runtimes=[], reporter=TerminalReporter(),
                              prompter=_prompter(None), config=CoreConfigSink(),
                              stack_dir=stack_dir, runner=process.SubprocessRunner())
    return lifecycle.status(deps)


def _restart(args, stack_dir, ask) -> None:
    """Restart a STOPPED stack the cheap way: `lifecycle.up` re-renders the compose from
    the recorded state and does `compose up -d`, repeating exactly the images `stack.json`
    holds. It is not `_provision`, which would re-detect the runtimes, re-prove the GPUs,
    re-pull the images and re-verify for what a reboot left merely stopped. The deps are
    the same a `qctx stack up` builds (review round R7, item 1)."""
    import stack.lifecycle as lifecycle
    lifecycle.up(_life_deps(args, ask), upgrade=False, images=None)


def _windows_line() -> None:
    """The one line and return the plan's first `install_step` branch requires on Windows
    (WSL included): phase 1 does not offer the step there, so it points at the README's
    manual path instead of offering and dying in `installer._check_platform` (review
    round R7, item 2)."""
    print("  ..    the local stack is not available on this platform in this version "
          "(windows, incl. wsl); set it up by hand: see '## Local models' in the README")


def _provision(args, stack_dir, budgets, ask, installer, facts) -> None:
    """The provisioning half: gather the runtimes and the host facts, build the
    `Deps`, and run the twelve-step flow. The heavy imports arrive with the modules
    themselves, so this is where they load."""
    import stack.fetch as fetch
    runtimes = _discover()
    host = facts.collect(facts.Probe(), stack_dir)
    images = list(getattr(args, "image", None) or [])
    deps = installer.Deps(
        runtimes=runtimes, facts=host, prompter=_prompter(ask),
        reporter=TerminalReporter(), config=CoreConfigSink(),
        transport=fetch.UrllibTransport(), stack_dir=stack_dir, budgets=budgets,
        env=dict(os.environ), port_free=facts.port_free)
    request = installer.Request(
        profile=args.stack, runtime=args.runtime, yes=args.yes,
        images=catalog.parse_image_flags(images) if images else {})
    installer.provision(request, deps)


def _discover():
    """The runtimes that answer, through the CLI's own discovery: a real subprocess runner
    and the host's PATH. The `host_system` is the host's, so a mac answers macos."""
    import platform
    import stack.facts as facts
    import stack.process as process
    import stack.runtimes as runtimes_mod
    return runtimes_mod.discover(process.SubprocessRunner(),
                                 host_system=facts.normalize_system(platform.system()))


# ---- the report section: state and health, never a runtime --------------------


def check_section(env=os.environ) -> dict:
    """The `stack` section of the `qctx install` report and `--json`: the state of a
    managed stack and the health of its endpoints, read by PROBING the URLs only. It never
    talks to a runtime, so `--check` and `--json` stay read-only and cheap -- a `status`
    would need the discovery the report mode must not run. `state` is imported inside
    (not at the top): it reaches `process`, the one heavy module the parser must not load,
    and this is only called by `cmd_install`, never by the parser build."""
    import stack.state as state
    directory = state.stack_dir(env)
    try:
        st = state.load(directory)
    except StackError as exc:
        # A corrupt `stack.json`: the section names it and the one fix, and the report
        # keeps going (a corrupt file is a fact, not a crash of `--check`).
        return {"managed": True, "error": str(exc), "fix": exc.fix}
    if st is None:
        return {"managed": False}
    endpoints = health.endpoints(st.ports)
    services = {name: health.http_status(url) for name, url in endpoints.items()}
    healthy = all(answer == 200 for answer in services.values())
    outdated = [role for role in catalog.IMAGES
                if st.images.get(role) != catalog.IMAGES[role]]
    return {
        "managed": True,
        "phase": st.phase,
        "runtime": st.runtime,
        "profile": st.profile,
        "ports": dict(st.ports),
        "services": services,
        "healthy": healthy,
        "outdated_pins": outdated,
    }


def _service_status(value):
    """The live status of one service in a section's `services`, from either producer.

    `check_section` (the install report) maps a service to its int status, but
    `lifecycle.status` (the `stack status` verb) maps it to `{"url": …, "status":
    …}`; both feed `section_lines`, so the up/down render reads the int out of
    whichever shape arrived. A dict with no status (or a `None`) is not 200, so
    it renders down."""
    return value.get("status") if isinstance(value, dict) else value


def section_lines(section: dict) -> list:
    """The report lines of a stack section dict, in the wizard's style. A dict, not a
    `Config`: the section is what `check_section` and `lifecycle.status` produce, and this
    is the one place its shape is turned into terminal lines."""
    if not section.get("managed"):
        return ["  ok    no managed stack (the endpoints are external)"]
    if section.get("error"):
        return [f"  FAIL  {section['error']}",
                f"        -> {section.get('fix', '')}".rstrip()]
    services = section.get("services", {})
    healthy = section.get("healthy")
    lines = [f"  ..    managed stack: {section.get('phase')} "
             f"({section.get('runtime')}, profile {section.get('profile')})"]
    if healthy is not None:
        lines.append(f"  {'ok   ' if healthy else 'FAIL '} endpoints: "
                     + ", ".join(f"{name} {'up' if _service_status(status) == 200 else 'down'}"
                                 for name, status in services.items()))
    ports = section.get("ports")
    if ports:
        lines.append("        ports: " + ", ".join(f"{name} {port}"
                                                   for name, port in ports.items()))
    # The two fields only the `lifecycle.status` shape carries: the boot line
    # (how the stack comes back after a reboot) and the config verdict. The
    # install-report shape has neither, so both are rendered only when present.
    if section.get("boot"):
        lines.append(f"        boot: {section['boot']}")
    if section.get("config_points_here") is not None:
        points = ("points at this stack" if section["config_points_here"]
                  else "does not point at this stack")
        lines.append(f"        config: your plugin {points}")
    outdated = section.get("outdated_pins") or []
    if outdated:
        lines.append("  ..    newer catalogue pins for: " + ", ".join(outdated) +
                     " (qctx stack up --upgrade)")
    # In EVERY case the verb must accuse a stopped stack and point at the fix
    # (spec: `stack status` says it is down and points at `qctx stack up`).
    # Not healthy covers both the `stopped` phase and the post-reboot case
    # where the phase still says running but the endpoints answer nothing.
    if section.get("healthy") is False:
        lines.append("  ..    the stack is down; start it with: qctx stack up")
    return lines
