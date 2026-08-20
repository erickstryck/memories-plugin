"""`qctx install` — the wizard.

`--check` is the mode everything else is measured against: it must report the same picture
and write NOTHING. A wizard that repairs while you are looking cannot be used to find out
what state a machine is in.
"""
import contextlib
import http.server
import io
import json
import os
import shutil
import subprocess
import threading
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
CLI = REPO / "cli" / "qctx.py"

from tests.isolation import assert_hermetic, hermetic_env  # noqa: E402


class CheckMode(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

    def env(self):
        return hermetic_env(self.home, QCTX_CONFIG=self.config)

    def run_cli(self, *argv, **kwargs):
        return subprocess.run([sys.executable, str(CLI), "install", *argv],
                              capture_output=True, text=True, env=self.env(),
                              timeout=180, stdin=subprocess.DEVNULL, **kwargs)

    def test_the_environment_it_runs_in_is_assembled_not_inherited(self):
        """This module used to start from `dict(os.environ)`. The legacy aliases
        (`QDRANT_URL`, `SERVER_BASE_URL`, …) survived that scrubbing, so five of these
        check-only tests dialled the developer's real Qdrant and waited out its
        timeouts."""
        assert_hermetic(self, self.env(), allowed=("QCTX_CONFIG",))

    def test_check_writes_nothing(self):
        before = self.config.read_text()
        self.run_cli("--check")
        self.assertEqual(self.config.read_text(), before)
        self.assertFalse((self.home / ".local").exists())

    def test_check_reports_the_plumbing_section(self):
        done = self.run_cli("--check")
        self.assertIn("launcher", done.stdout)
        self.assertIn("no-shell config", done.stdout)

    def test_json_verdict_covers_the_plumbing_too(self):
        """`ready` and `blockers` came from `diagnose` alone, so a machine with a healthy
        Qdrant and no `qctx` on PATH was reported ready. The plumbing checks were merged
        into `checks` and nowhere else, which is the half that an agent does not read."""
        payload = json.loads(self.run_cli("--check", "--json").stdout)
        named = {c["name"] for c in payload["blockers"]}
        self.assertIn("launcher", named)
        self.assertIn("no-shell config", named)
        self.assertFalse(payload["ready"])

    def test_json_is_parseable_and_carries_both_shapes(self):
        done = self.run_cli("--check", "--json")
        payload = json.loads(done.stdout)
        self.assertIn("checks", payload)
        self.assertIn("hosts", payload)
        for section in payload["hosts"]:
            self.assertEqual({"host", "exit_code", "text"}, set(section))

    def test_absent_host_is_skipped_not_failed(self):
        """A machine with only one of the two hosts is the normal case."""
        done = self.run_cli("--check")
        self.assertNotIn("Traceback", done.stderr)
        self.assertEqual(done.returncode, 0)

    def test_check_reports_each_key_and_every_spelling_it_accepts(self):
        """The design's own words: reporting fewer spellings than the core accepts
        would be worse than not reporting. Before this, the whole report was silent
        about the credentials on a machine with no hermes."""
        from core.config import ENV_ALIASES
        done = self.run_cli("--check")
        for field in ("qdrant_api_key", "api_key"):
            self.assertIn(field, done.stdout)
            for alias in ENV_ALIASES[field]:
                self.assertIn(alias, done.stdout)

    def test_a_key_that_lives_only_in_a_file_is_found_and_never_printed(self):
        """Both halves at once, because they are one promise: the report says WHERE the
        key is and in which spelling, and the value itself never appears anywhere."""
        secrets = self.home / ".secrets"
        secrets.write_text("export QDRANT_SERVICE_API_KEY=s3cr3t-value\n")
        hermes_env = self.home / ".hermes" / ".env"
        hermes_env.parent.mkdir()
        hermes_env.write_text("SERVER_API_KEY=another-s3cr3t\n")
        done = self.run_cli("--check")
        self.assertIn(str(secrets), done.stdout)
        self.assertIn(str(hermes_env), done.stdout)
        self.assertNotIn("s3cr3t-value", done.stdout + done.stderr)
        self.assertNotIn("another-s3cr3t", done.stdout + done.stderr)

    def test_no_tty_never_blocks(self):
        """stdin closed, no --yes: it diagnoses and exits, like `qctx setup`."""
        done = self.run_cli()
        self.assertEqual(done.returncode, 0)
        self.assertIn("no interactive terminal", done.stdout)


class SkipSuiteVariable(unittest.TestCase):
    def test_cutover_refuses_to_apply_with_the_suite_skipped(self):
        """Copied from the hermes script, including the refusal — the flag exists to make
        the PLAN cheap, never to make an apply cheap.

        HOME is pointed at a scratch directory, not the real one: `--apply` rewrites
        ~/.claude/settings.json, .mcp.json and .claude.json, and MOVES ~/.claude/skills.
        The refusal below fires before any of that (it is unconditional on `--apply` with
        the suite skipped), so a real HOME was never touched even without this — but a
        test that depends on that ordering never changing is not a test worth trusting the
        developer's machine to.
        """
        with TemporaryDirectory() as home:
            env = hermetic_env(home, CUTOVER_SKIP_SUITE="1")
            done = subprocess.run(["bash", str(REPO / "scripts" / "cutover.sh"), "--apply"],
                                  capture_output=True, text=True, env=env, timeout=120)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("suite unverified", done.stdout + done.stderr)


class WritingPass(unittest.TestCase):
    """The interactive pass, driven through stdin with --yes off."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

    def env(self):
        return hermetic_env(self.home, QCTX_CONFIG=self.config,
                            QCTX_INSTALL_FORCE_TTY="1")

    def run_with_input(self, keystrokes: str, *argv):
        return subprocess.run([sys.executable, str(CLI), "install", "--config-only",
                               *argv],
                              input=keystrokes, capture_output=True, text=True,
                              env=self.env(), timeout=180)

    def test_the_environment_it_runs_in_is_assembled_not_inherited(self):
        """With `HERMES_HOME` exported, the inherited form rewrote the developer's live
        `~/.hermes/.env` with the `qkey`/`skey` this class types."""
        assert_hermetic(self, self.env(),
                        allowed=("QCTX_CONFIG", "QCTX_INSTALL_FORCE_TTY"))

    def test_writes_the_urls_and_keeps_the_defaults_on_enter(self):
        answers = "\n".join([
            "https://q.example",        # qdrant_url
            "https://e.example/v1",     # api_base_url
            "qkey", "skey",             # the two keys
            "mem",                      # memory_collection
        ] + [""] * 9) + "\n"            # pass 2: Enter all the way
        self.run_with_input(answers)
        written = json.loads(self.config.read_text())
        self.assertEqual(written["qdrant_url"], "https://q.example")
        self.assertEqual(written["memory_collection"], "mem")

    def test_no_key_reaches_the_config_file_in_any_spelling(self):
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "skey", "mem",
        ] + [""] * 9) + "\n"
        self.run_with_input(answers)
        body = self.config.read_text()
        for forbidden in ("qkey", "skey", "qdrant_api_key", "api_key"):
            self.assertNotIn(forbidden, body)

    def test_an_empty_hermes_home_does_not_drop_a_key_in_the_current_directory(self):
        """`Path("")` is `Path(".")`, and `Path(".").is_dir()` is True, so the wizard
        wrote a plaintext `.env` wherever it happened to be run — very likely a git
        repository. The sibling `hermes_install_path` already used the `or` idiom."""
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "skey", "mem",
        ] + [""] * 9) + "\n"
        here = Path(self.tmp.name) / "somebodys-repo"
        here.mkdir()
        subprocess.run([sys.executable, str(CLI), "install", "--config-only"],
                       input=answers, capture_output=True, text=True, timeout=180,
                       cwd=here, env=hermetic_env(self.home, QCTX_CONFIG=self.config,
                                                  QCTX_INSTALL_FORCE_TTY="1",
                                                  HERMES_HOME=""))
        self.assertFalse((here / ".env").exists(),
                         "a credential file was written into the working directory")

    def test_a_non_numeric_context_window_is_re_asked_not_a_crash(self):
        """`int(entry)` raised an uncaught ValueError on the LAST question, after
        fourteen answers had been typed and before `core.save` had written any of
        them. The whole pass was lost to one typo."""
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "skey", "mem",
        ] + [""] * 8 + ["200k", "200000"]) + "\n"
        done = self.run_with_input(answers)
        self.assertNotIn("Traceback", done.stderr)
        self.assertEqual(done.returncode, 0, done.stderr)
        written = json.loads(self.config.read_text())
        self.assertEqual(written["context_window"], 200000)
        self.assertEqual(written["qdrant_url"], "https://q.example")

    def test_the_key_value_is_never_echoed(self):
        answers = "\n".join([
            "https://q.example", "https://e.example/v1", "qkey", "s3cr3t-value", "mem",
        ] + [""] * 9) + "\n"
        done = self.run_with_input(answers)
        self.assertNotIn("s3cr3t-value", done.stdout + done.stderr)


class YesWithNoTerminal(unittest.TestCase):
    """`--yes` exists FOR the machine with no terminal, and that is where it crashed.

    The guard was `not _interactive and not args.yes`, so `--yes` with a closed stdin
    fell straight into the prompts and the first `input()` raised an uncaught EOFError.
    On the full path the launcher had already been copied into ~/.local/bin by then, so
    the run died half-done.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text('{"qdrant_url": "https://kept.example"}')
        self.addCleanup(self.tmp.cleanup)

    def run_cli(self, *argv):
        return subprocess.run([sys.executable, str(CLI), "install", *argv],
                              capture_output=True, text=True, timeout=180,
                              env=hermetic_env(self.home, QCTX_CONFIG=self.config),
                              stdin=subprocess.DEVNULL)

    def test_config_only_yes_with_no_stdin_exits_clean(self):
        done = self.run_cli("--config-only", "--yes")
        self.assertNotIn("Traceback", done.stderr)
        self.assertNotIn("EOFError", done.stderr)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_the_launcher_is_actually_installed_by_the_full_path(self):
        """Deleting the `install_launcher` call from `cmd_install` used to break no
        test at all — step 2 of the wizard's own order had no witness. It runs before
        the configuration pass, so a blocked machine still proves it."""
        self.run_cli("--yes")
        target = self.home / ".local" / "bin" / "qctx"
        self.assertTrue(target.exists(), "the wizard never put qctx on PATH")
        self.assertEqual(target.read_bytes(), (REPO / "bin" / "qctx").read_bytes())
        self.assertTrue(os.access(target, os.X_OK))

    def test_the_full_path_with_yes_and_no_stdin_exits_clean(self):
        """The worse case: `install_launcher` has already written before the crash."""
        done = self.run_cli("--yes")
        self.assertNotIn("Traceback", done.stderr)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_it_keeps_every_current_value(self):
        """Nothing was answered, so nothing may change. A `--yes` that wrote defaults
        over a working configuration would be worse than the crash."""
        before = self.config.read_text()
        self.run_cli("--config-only", "--yes")
        self.assertEqual(self.config.read_text(), before)


class StubEmbeddings(http.server.BaseHTTPRequestHandler):
    """An OpenAI-shaped /embeddings that answers in DIM dimensions and nothing else.

    `vector_size` is the one field the wizard must not ask for — it is written from what
    the endpoint answered. Proving it is written therefore needs an endpoint, and a
    loopback stub is the pattern this suite already uses for the big-file hooks.
    """

    DIM = 7

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"data": [{"index": 0,
                                     "embedding": [0.1] * self.DIM}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class AuthenticatedStubEmbeddings(StubEmbeddings):
    """The same endpoint, but it REFUSES a request that carries no key.

    `StubEmbeddings` answers anybody, which is why 1265 tests could not see that a key
    typed into the wizard never reached the process that dials the endpoint two lines
    later. A real endpoint with authentication turned on answers 401, and that is the
    only shape in which the defect is visible from outside.
    """

    TOKEN = "typed-just-now"

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {self.TOKEN}":
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = b'{"error":"no auth"}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

            return
        super().do_POST()


class AKeyIsUsableInTheSameRun(unittest.TestCase):
    """A key the user just typed has to be usable by the rest of THIS run.

    `_store_secrets` wrote the key to a credential file and stopped there. Everything
    afterwards re-reads configuration through `core.load()`, which sees the config file
    plus the PROCESS environment — and the key is in neither, because a secret never
    enters the config file and nothing had put it in the environment. So `vector_size`
    detection, and then `diagnose`, ran unauthenticated on exactly the fresh-machine
    flow the wizard exists for.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.hermes = Path(self.tmp.name) / "hermes"
        self.hermes.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                 AuthenticatedStubEmbeddings)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def test_the_endpoint_that_needs_the_key_is_reached_with_it(self):
        answers = "\n".join([
            "https://q.example",                          # qdrant_url
            f"{self.base}/v1",                            # api_base_url
            "qkey",                                       # qdrant_api_key
            AuthenticatedStubEmbeddings.TOKEN,            # api_key
            "mem",                                        # memory_collection
            f"{self.base}/embeddings",                    # embed_url
        ] + [""] * 8) + "\n"
        done = subprocess.run(
            [sys.executable, str(CLI), "install", "--config-only"],
            input=answers, capture_output=True, text=True, timeout=180,
            env=hermetic_env(self.home, QCTX_CONFIG=self.config,
                             HERMES_HOME=self.hermes, QCTX_INSTALL_FORCE_TTY="1"))
        self.assertNotIn("Traceback", done.stderr)
        self.assertNotIn("401", done.stdout + done.stderr)
        written = json.loads(self.config.read_text())
        self.assertEqual(written.get("vector_size"), StubEmbeddings.DIM,
                         "the key never reached the process that dialled the endpoint")

    def test_the_value_is_still_kept_out_of_the_config_file_and_the_output(self):
        """Putting the key in the environment is not a licence to persist or print it."""
        answers = "\n".join([
            "https://q.example", f"{self.base}/v1", "qkey",
            AuthenticatedStubEmbeddings.TOKEN, "mem", f"{self.base}/embeddings",
        ] + [""] * 8) + "\n"
        done = subprocess.run(
            [sys.executable, str(CLI), "install", "--config-only"],
            input=answers, capture_output=True, text=True, timeout=180,
            env=hermetic_env(self.home, QCTX_CONFIG=self.config,
                             HERMES_HOME=self.hermes, QCTX_INSTALL_FORCE_TTY="1"))
        self.assertNotIn(AuthenticatedStubEmbeddings.TOKEN, self.config.read_text())
        self.assertNotIn(AuthenticatedStubEmbeddings.TOKEN, done.stdout + done.stderr)


class EveryFieldIsSet(unittest.TestCase):
    """The proof the design demanded, done properly.

    It used to be a tuple compared against `config.DEFAULTS`. That is a declaration, not
    a proof: `vector_size` sat in `DETECTED_FIELDS` and nothing in the wizard ever wrote
    it, so 15 fields were claimed and 14 were walked. This drives the wizard end to end
    and reads back where each of the 15 actually landed.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.hermes = Path(self.tmp.name) / "hermes"
        self.hermes.mkdir()
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text("{}")
        self.addCleanup(self.tmp.cleanup)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubEmbeddings)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def value_for(self, field: str) -> str:
        if field == "embed_url":
            return f"{self.base}/embeddings"
        if field == "context_window":
            return "200000"
        if field.endswith("_url"):
            return f"https://{field.replace('_', '-')}.example"
        if field.endswith("_collection"):
            return f"coll-{field}"

        return f"value-{field}"

    def test_the_wizard_sets_all_fifteen(self):
        from core import config, install
        asked = install.REQUIRED_FIELDS + install.OPTIONAL_FIELDS
        answers = "\n".join(self.value_for(f) for f in asked) + "\n"
        done = subprocess.run(
            [sys.executable, str(CLI), "install", "--config-only"],
            input=answers, capture_output=True, text=True, timeout=180,
            env=hermetic_env(self.home, QCTX_CONFIG=self.config,
                             HERMES_HOME=self.hermes, QCTX_INSTALL_FORCE_TTY="1"))
        self.assertNotIn("Traceback", done.stderr)
        written = json.loads(self.config.read_text())
        credentials = (self.hermes / ".env").read_text()
        for field in config.DEFAULTS:
            with self.subTest(field=field):
                if field in config.SECRET_FIELDS:
                    canonical = config.ENV_ALIASES[field][0]
                    self.assertIn(f"{canonical}={self.value_for(field)}", credentials)
                    self.assertNotIn(field, self.config.read_text())
                elif field in install.DETECTED_FIELDS:
                    self.assertEqual(written[field], StubEmbeddings.DIM,
                                     "the wizard never wrote what the endpoint answered")
                else:
                    self.assertEqual(str(written[field]), self.value_for(field))


def load_cli():
    """Imports cli/qctx.py as a module. It is a script, not a package member.

    Copied verbatim from `tests/test_cli_render.py` — `import cli.qctx` does NOT work:
    there is no `cli/__init__.py`, deliberately, because the CLI is an entry point and
    not something the core imports.
    """
    import importlib.util
    path = REPO / "cli" / "qctx.py"
    spec = importlib.util.spec_from_file_location("qctx_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


class HostDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()

    def test_claude_absent_reads_as_none(self):
        self.assertIsNone(self.qctx.claude_install_path(self.home))

    def test_claude_present_returns_the_live_path(self):
        registry = self.home / ".claude" / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"plugins": {"memories-plugin@memories-plugin": [
            {"installPath": "/somewhere/b8008f7dac88"}]}}))
        self.assertEqual(self.qctx.claude_install_path(self.home),
                         "/somewhere/b8008f7dac88")

    def test_a_registry_that_is_not_the_expected_shape_reads_as_none(self):
        """`OSError` and `JSONDecodeError` were guarded; the shape was not.

        A registry holding a JSON ARRAY parses fine and then raises `AttributeError` on
        `data.get`, and one holding a list of strings raises on `entry.get`. Neither is
        exotic — it is what a truncated, hand-edited or half-migrated file looks like,
        and the bash sibling in `bin/qctx` already survives all of them because its
        reader wraps everything in one `except Exception`. Here it took the wizard down
        with a traceback in the middle of `--check`.
        """
        registry = self.home / ".claude" / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True)
        for malformed in ("[]", '["memories-plugin@memories-plugin"]', '"a string"',
                          '{"plugins": []}',
                          '{"plugins": {"memories-plugin@memories-plugin": ["x"]}}',
                          "not json at all", ""):
            with self.subTest(registry=malformed):
                registry.write_text(malformed)
                self.assertIsNone(self.qctx.claude_install_path(self.home))

    def test_the_hermes_command_carries_force_and_the_provider_switch(self):
        joined = " ".join(self.qctx.HOST_INSTALL_COMMANDS["hermes"])
        self.assertIn("--force", joined)
        self.assertIn("memory.provider memories", joined)


class LauncherInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()

    def test_copies_and_makes_it_executable(self):
        target = self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertTrue(os.access(target, os.X_OK))
        self.assertEqual(target.read_bytes(), (REPO / "bin" / "qctx").read_bytes())

    def bin_dir(self) -> Path:
        d = self.home / ".local" / "bin"
        d.mkdir(parents=True, exist_ok=True)

        return d

    def test_it_leaves_a_symlink_that_already_points_at_this_tree(self):
        """The DOCUMENTED development install is exactly this symlink, and
        `launcher_check` accepts it as current — `test_a_symlink_to_the_tree_counts_as
        _current` in `tests/test_install_core.py` pins that half.

        Copying over it raised `shutil.SameFileError` once, so the wizard learned to
        unlink first — unconditionally, which silently undid a link somebody made on
        purpose. From that run on, every `git pull` left `~/.local/bin/qctx` stale and
        `--check` reported "differs": a state the symlink itself could never reach.
        """
        link = self.bin_dir() / "qctx"
        link.symlink_to(REPO / "bin" / "qctx")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            target = self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertTrue(target.is_symlink(), "a link that already works was replaced")
        self.assertEqual(target.resolve(), (REPO / "bin" / "qctx").resolve())
        self.assertIn("symlink", out.getvalue().lower())

    def test_a_symlink_into_ANOTHER_checkout_is_still_replaced(self):
        """"Already works" means pointing at THIS tree. A link into somebody else's
        checkout is a stale launcher wearing the same name."""
        other = Path(self.tmp.name) / "another-checkout" / "bin"
        other.mkdir(parents=True)
        stand_in = other / "qctx"
        stand_in.write_bytes((REPO / "bin" / "qctx").read_bytes())
        (self.bin_dir() / "qctx").symlink_to(stand_in)
        with contextlib.redirect_stdout(io.StringIO()):
            target = self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertFalse(target.is_symlink(), "a link into another tree was kept")
        self.assertEqual(target.read_bytes(), (REPO / "bin" / "qctx").read_bytes())

    def test_a_stale_plain_copy_is_still_refreshed(self):
        copy = self.bin_dir() / "qctx"
        copy.write_text("#!/usr/bin/env bash\necho old\n")
        with contextlib.redirect_stdout(io.StringIO()):
            target = self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertEqual(target.read_bytes(), (REPO / "bin" / "qctx").read_bytes())

    def test_it_never_writes_through_a_symlink_into_another_checkout(self):
        """Worse than the crash: pointing at a DIFFERENT checkout, the copy went
        through the link and silently rewrote that checkout's `bin/qctx`."""
        other = Path(self.tmp.name) / "other-checkout" / "bin"
        other.mkdir(parents=True)
        stand_in = other / "qctx"
        stand_in.write_text("#!/usr/bin/env bash\necho the other checkout\n")
        (self.bin_dir() / "qctx").symlink_to(stand_in)
        self.qctx.install_launcher(REPO, {"HOME": str(self.home)})
        self.assertEqual(stand_in.read_text(),
                         "#!/usr/bin/env bash\necho the other checkout\n")


class ConfigPass(unittest.TestCase):
    """The two passes, driven directly.

    `_ask` and `_read_secret` are replaced by scripted answers rather than a pipe, so
    each assertion is about ONE prompt instead of about a whole transcript.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()
        self.saved = {}

    def cfg(self, **over):
        from core import config
        values = dict(config.DEFAULTS)
        values.update(over)

        return config.Config(**values)

    def ask_config(self, answers, secrets=("", ""), suggestions=(), env=None):
        """Runs `_ask_config` and returns (printed, prompts seen by the key questions)."""
        prompts = []

        def read_secret(prompt):
            prompts.append(prompt)

            return secrets[len(prompts) - 1]

        with mock.patch.dict(os.environ, env or {"HOME": str(self.home)}, clear=True), \
                mock.patch.object(self.qctx.core, "save", self.saved.update), \
                mock.patch.object(self.qctx, "_read_secret", read_secret), \
                mock.patch.object(self.qctx, "_ask", side_effect=list(answers)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.qctx._ask_config(self.cfg(), True, suggestions)

        return out.getvalue(), prompts

    def test_the_memory_collection_is_offered_with_its_suggestions(self):
        """`report["memory_suggestions"]` was computed and shown nowhere, while
        `cmd_setup` printed the same list twenty lines away. The design's pass 1 says
        this collection is asked "with the suggestions of `suggest_collections`"."""
        printed, _ = self.ask_config(
            ["", "", "1"] + [""] * 9,
            suggestions=[{"collection": "claude_memory", "points": 812},
                         {"collection": "old_archive", "points": 12}])
        self.assertIn("claude_memory", printed)
        self.assertIn("812", printed)
        self.assertEqual(self.saved.get("memory_collection"), "claude_memory",
                         "choosing by index did not resolve to the suggestion")

    def test_a_key_that_is_already_in_a_credential_file_is_not_asked_as_missing(self):
        """It ran on every verification, so re-asking is not a small annoyance: it is
        the paste-a-secret-every-run reflex the design forbids, taught by the tool."""
        secrets_file = self.home / ".secrets"
        secrets_file.write_text("export QDRANT_SERVICE_API_KEY=abcdef\n")
        _, prompts = self.ask_config([""] * 12)
        self.assertIn("already set as QDRANT_SERVICE_API_KEY", prompts[0])
        self.assertIn(str(secrets_file), prompts[0])
        self.assertIn("MISSING", prompts[1])          # the other key really is missing

    def test_a_key_in_the_environment_is_recognised_too(self):
        _, prompts = self.ask_config(
            [""] * 12, env={"HOME": str(self.home), "SERVER_API_KEY": "abc"})
        self.assertIn("already set as SERVER_API_KEY in the environment", prompts[1])


class ShellCredentialFile(unittest.TestCase):
    """`~/.secrets` is a DEFAULT, not a destination the wizard picks for you.

    The design: "an file the user names; default `~/.secrets` IF IT ALREADY EXISTS. It
    creates nothing on its own." The path was hard-coded, so a user who keeps
    credentials anywhere else was told to paste `export` lines instead.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()

    def store(self, answer: str):
        with mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=True), \
                mock.patch.object(self.qctx, "_ask", return_value=answer), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.qctx._store_secrets({"QCTX_QDRANT_API_KEY": "abcdef"})

        return out.getvalue()

    def test_it_writes_to_the_file_the_user_names(self):
        named = Path(self.tmp.name) / "elsewhere" / "keys.env"
        printed = self.store(str(named))
        self.assertIn("QCTX_QDRANT_API_KEY=abcdef", named.read_text())
        self.assertIn(str(named), printed)

    def test_the_default_is_offered_only_when_it_already_exists(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=True), \
                mock.patch.object(self.qctx, "_ask", return_value="") as asked, \
                contextlib.redirect_stdout(io.StringIO()):
            self.qctx._store_secrets({"QCTX_QDRANT_API_KEY": "abcdef"})
        self.assertIn("Enter to skip", asked.call_args[0][0])
        self.assertFalse((self.home / ".secrets").exists(),
                         "the wizard created a credential file nobody asked for")

        (self.home / ".secrets").write_text("OTHER=keep\n")
        with mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=True), \
                mock.patch.object(self.qctx, "_ask", return_value="") as asked, \
                contextlib.redirect_stdout(io.StringIO()):
            self.qctx._store_secrets({"QCTX_QDRANT_API_KEY": "abcdef"})
        self.assertIn(str(self.home / ".secrets"), asked.call_args[0][0])
        self.assertIn("QCTX_QDRANT_API_KEY=abcdef",
                      (self.home / ".secrets").read_text())

    def test_naming_nothing_leaves_the_key_pending_and_prints_the_exports(self):
        """Never "done". A key that only lives in this shell is gone in the next one,
        and saying otherwise is the lie that started this whole piece of work."""
        printed = self.store("")
        self.assertIn("export QCTX_QDRANT_API_KEY", printed)
        self.assertIn("PENDING", printed)
        self.assertNotIn("abcdef", printed)


class HostGroups(unittest.TestCase):
    """The four host combinations, and the two groups that actually run things.

    `_host_sections`, `_host_dry_run`, `_host_install_group` and `_host_cutover_group`
    had no direct test at all, and they are the surface that calls
    `subprocess(shell=True)` and `cutover.sh --apply`. The design's own test list asks
    for none / claude only / hermes only / both; only "none" was covered, and only
    because this machine happens to hide both binaries behind a reduced PATH.

    Nothing real is executed: `root` is a fake tree whose two cutover scripts are stubs
    that record their arguments, and the host binaries are stubs on a temporary PATH.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.home = self.dir / "home"
        self.home.mkdir()
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        # The ONLY PATH these tests run with. Two shells are linked in because the
        # cutovers are run as `bash <script>` and the host install commands go through
        # `sh -c`, and `mkdir` is what the stubs use to stand in for a plugin that got
        # installed. Nothing else is linked, so a real `claude` or `hermes` on this
        # machine cannot be reached from here by accident.
        for tool in ("bash", "sh", "mkdir"):
            (self.bin / tool).symlink_to(shutil.which(tool))
        self.witness = self.dir / "witness"
        self.addCleanup(self.tmp.cleanup)
        self.qctx = load_cli()
        self.root = self.fake_root()

    def fake_root(self) -> Path:
        """A tree with the two cutover names, and stubs behind them.

        The real scripts are the host-state layer and are not this test's subject; what
        is under test is WHICH of them the wizard runs, with which arguments.
        """
        root = self.dir / "tree"
        (root / "scripts").mkdir(parents=True)
        for host, script, skip in self.qctx.HOST_SECTIONS:
            body = (f'#!/usr/bin/env bash\n'
                    f'printf "%s plan skip=%s args=%s\\n" {host} "${{{skip}:-}}" "$*"\n'
                    f'printf "%s %s\\n" {host} "$*" >> {self.witness}\n')
            (root / script).write_text(body)
            (root / script).chmod(0o755)

        return root

    def stub_host(self, name: str, exit_code: int = 0, installs: Path = None,
                  provider: str = "") -> None:
        """A host binary on PATH that records every call instead of making one.

        `installs` is the directory the stub creates, standing in for a plugin install
        that worked — the wizard re-probes for it rather than believing the exit code.
        `provider` is what it answers to `config get memory.provider`, which the wizard
        reads to say what it is about to replace.
        """
        path = self.bin / name
        creates = f'mkdir -p {installs}\n' if installs else ""
        answer = (f'if [ "$*" = "config get memory.provider" ]; then\n'
                  f'  printf "%s\\n" {provider!r}\n  exit 0\nfi\n') if provider else ""
        path.write_text(f'#!/usr/bin/env bash\n'
                        f'printf "%s %s\\n" {name} "$*" >> {self.witness}\n'
                        f'{answer}'
                        f'{creates}'
                        f'exit {exit_code}\n')
        path.chmod(0o755)

    def env(self, **extra):
        return {"PATH": str(self.bin), "HOME": str(self.home), **extra}

    def calls(self) -> str:
        return self.witness.read_text() if self.witness.exists() else ""

    def sections(self, *hosts):
        for host in hosts:
            self.stub_host(self.qctx.HOST_BINARIES[host])
        with mock.patch.dict(os.environ, self.env(), clear=True):
            return self.qctx._host_sections(self.root)

    # ---- the four combinations ----

    def test_no_host_on_this_machine_skips_both_without_failing(self):
        sections = self.sections()
        self.assertEqual([s["host"] for s in sections], ["claude-code", "hermes"])
        for section in sections:
            self.assertEqual(section["exit_code"], 0)
            self.assertIn("is not on PATH", section["text"])
        self.assertEqual(self.calls(), "", "a cutover ran for a host that is not here")

    def test_claude_only_runs_the_claude_plan_and_skips_hermes(self):
        sections = {s["host"]: s["text"] for s in self.sections("claude-code")}
        self.assertIn("claude-code plan", sections["claude-code"])
        self.assertIn("is not on PATH", sections["hermes"])
        self.assertNotIn("hermes ", self.calls())

    def test_hermes_only_runs_the_hermes_plan_and_skips_claude(self):
        sections = {s["host"]: s["text"] for s in self.sections("hermes")}
        self.assertIn("hermes plan", sections["hermes"])
        self.assertIn("is not on PATH", sections["claude-code"])
        self.assertNotIn("claude-code ", self.calls())

    def test_both_hosts_each_get_their_own_section(self):
        sections = {s["host"]: s["text"] for s in self.sections("claude-code", "hermes")}
        self.assertIn("claude-code plan", sections["claude-code"])
        self.assertIn("hermes plan", sections["hermes"])

    def test_the_plan_asks_each_script_to_skip_its_suite(self):
        """41s per host to draw a list. The apply runs it once; the plan never does."""
        sections = {s["host"]: s["text"] for s in self.sections("claude-code", "hermes")}
        for text in sections.values():
            self.assertIn("skip=1", text)
            self.assertNotIn("--apply", text)

    # ---- the group that installs into the host ----

    def test_an_installed_plugin_is_recognised_and_not_reinstalled(self):
        self.stub_host("hermes")
        (self.home / ".hermes" / "plugins" / "memories").mkdir(parents=True)
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            present = self.qctx._host_install_group(Args(), self.root)
        self.assertTrue(present["hermes"])
        self.assertIn("installed at", out.getvalue())
        self.assertEqual(self.calls(), "", "it reinstalled a plugin that was there")

    def test_a_missing_plugin_is_installed_with_the_commands_it_showed(self):
        self.stub_host("hermes", installs=self.home / ".hermes" / "plugins" / "memories")
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            present = self.qctx._host_install_group(Args(), self.root)
        shown = out.getvalue()
        for command in self.qctx.HOST_INSTALL_COMMANDS["hermes"]:
            self.assertIn(command, shown)
        self.assertIn("plugins install", self.calls())
        self.assertIn("config set memory.provider memories", self.calls())
        self.assertTrue(present["hermes"])

    def test_the_install_commands_go_through_no_shell_and_carry_a_timeout(self):
        """Two risks removed at once, and neither costs anything.

        The commands are static module constants, so nothing needs a shell to build
        them: `shell=True` hands the string to `/bin/sh`, which is a whole layer of
        quoting and word-splitting between the constant and the process, for no gain.
        And the call had NO timeout — a `claude plugin install` that waits on a network
        it cannot reach hung the wizard with no way out but Ctrl-C.
        """
        seen = []

        def run(command, **kwargs):
            seen.append((command, kwargs))

            return SimpleNamespace(returncode=0, stdout="", stderr="")

        recorder = SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired,
                                   SubprocessError=subprocess.SubprocessError)
        self.stub_host("hermes")
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "subprocess", recorder), \
                contextlib.redirect_stdout(io.StringIO()):
            self.qctx._host_install_group(Args(), self.root)
        self.assertTrue(seen, "no host command was run at all")
        for command, kwargs in seen:
            with self.subTest(command=command):
                self.assertIsInstance(command, list,
                                      "a string command is a command run through a shell")
                self.assertNotIn("shell", kwargs)
                self.assertTrue(kwargs.get("timeout"), "no timeout on a host command")

    def test_a_hung_host_install_is_reported_instead_of_hanging_the_wizard(self):
        wedged = self.wedged_subprocess(lambda command: "plugins" in " ".join(command))
        self.stub_host("hermes")
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "subprocess", wedged), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            present = self.qctx._host_install_group(Args(), self.root)
        self.assertFalse(present["hermes"])
        self.assertIn("timed out", out.getvalue())

    def test_a_failed_host_install_is_not_recorded_as_present(self):
        """`present_now[host] = True` was set regardless of what the commands did, so a
        `hermes plugins install` that exited non-zero still led to an `--apply`
        cutover on a host with no plugin in it. The state is RE-PROBED instead."""
        self.stub_host("hermes", exit_code=1)
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            present = self.qctx._host_install_group(Args(), self.root)
        self.assertFalse(present["hermes"])
        self.assertIn("still not installed", out.getvalue())

    def test_it_explains_the_force_and_the_provider_switch_before_asking(self):
        """The design requires both to be shown, because both are the user's decision:
        `--force` is agreement with what the scanner flagged, and pointing
        `memory.provider` here throws away whatever provider was set. They lived only
        as a code comment, and the wizard printed the bare commands."""
        self.stub_host("hermes", provider="some-other-provider",
                       installs=self.home / ".hermes" / "plugins" / "memories")
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.qctx._host_install_group(Args(), self.root)
        shown = out.getvalue()
        self.assertIn("caution", shown)
        self.assertIn("REPLACES", shown)
        self.assertIn("some-other-provider", shown)
        self.assertLess(shown.index("caution"), shown.index("hermes plugins install"),
                        "the reason has to come before the command, not after it")

    def test_declining_leaves_the_host_alone(self):
        self.stub_host("hermes")
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "_ask", return_value="n"), \
                contextlib.redirect_stdout(io.StringIO()):
            present = self.qctx._host_install_group(Args(yes=False), self.root)
        self.assertFalse(present["hermes"])
        self.assertNotIn("plugins install", self.calls())
        self.assertNotIn("config set", self.calls())

    # ---- the group that applies the cutover ----

    def test_only_a_host_with_a_plugin_gets_a_cutover(self):
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.qctx._host_cutover_group(Args(), self.root,
                                          {"claude-code": True, "hermes": False})
        self.assertIn("claude-code — cutover plan", out.getvalue())
        self.assertNotIn("hermes — cutover plan", out.getvalue())
        self.assertIn("claude-code --apply", self.calls())
        self.assertNotIn("hermes --apply", self.calls())

    def test_the_apply_is_given_more_time_than_the_suite_costs(self):
        """The suite measured 852 s ON 2026-08-20 and the timeout was 900. The number it
        is compared against grows with every test added, so the margin has to be a
        multiple, not 48 seconds.

        A LITERAL FLOOR, deliberately, and it is the weaker half of the guarantee: the
        only test that could notice the suite actually outgrowing the timeout is one
        that runs the suite, which is exactly the cost the constant exists to avoid
        paying on every `--check`. So this pins the floor and the constant's docstring
        carries the date of the measurement it was chosen against.
        """
        self.assertGreaterEqual(self.qctx.CUTOVER_APPLY_TIMEOUT, 3600)

    def wedged_subprocess(self, when):
        """A `subprocess` for the CLI module alone, whose `run` times out when `when`
        says so and is the real one otherwise.

        The attribute is replaced ON THE PRIVATE MODULE `load_cli()` handed us, not on
        the shared `subprocess` module: `mock.patch.object(self.qctx.subprocess, "run")`
        mutates the one object every other test, and `unittest` itself, is holding.
        """
        real_run = subprocess.run

        def run(command, **kwargs):
            if when(command):
                raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 0))

            return real_run(command, **kwargs)

        return SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired,
                               SubprocessError=subprocess.SubprocessError)

    def test_a_timed_out_apply_is_reported_and_not_raised(self):
        """A TimeoutExpired fires while the script is rewriting settings.json. It used
        to propagate out of `cmd_install` uncaught, so the user saw a traceback where
        they needed to be told to go and look at the backup."""
        wedged = self.wedged_subprocess(lambda command: "--apply" in command)
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "subprocess", wedged), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.qctx._host_cutover_group(Args(), self.root, {"hermes": True})
        self.assertIn("was killed after", out.getvalue())
        self.assertIn("backup", out.getvalue())

    def test_a_wedged_dry_run_is_a_report_and_not_a_traceback(self):
        """`--check` is documented as "reports; writes nothing", and a wedged `claude`,
        `hermes`, `bash` or `jq` turned it into a traceback: the apply learned to catch
        its own timeout and the dry run, which runs first and on every single `--check`,
        did not."""
        self.stub_host("hermes")
        wedged = self.wedged_subprocess(lambda command: True)
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "subprocess", wedged):
            sections = {s["host"]: s for s in self.qctx._host_sections(self.root)}
        self.assertNotEqual(sections["hermes"]["exit_code"], 0,
                            "a plan that never answered was reported as a clean plan")
        self.assertIn("timed out", sections["hermes"]["text"])

    def test_a_wedged_dry_run_does_not_take_the_whole_check_down(self):
        """The report has to survive it end to end — the section is data the JSON mode
        hands to a program, so it has to be a section and not an exception."""
        self.stub_host("hermes")
        wedged = self.wedged_subprocess(lambda command: True)
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "subprocess", wedged):
            sections = self.qctx._host_sections(self.root)
        for section in sections:
            self.assertEqual({"host", "exit_code", "text"}, set(section))

    def test_the_cutover_group_reports_which_hosts_it_actually_applied(self):
        """The closing "what is left, and only you can do it" printed BOTH hosts' steps
        at the end of every full run — on a machine with only one host, and on a run
        where every cutover was declined. A one-time manual step for a host that was
        never touched is an instruction to go and do nothing, printed beside real ones."""
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(io.StringIO()):
            applied = self.qctx._host_cutover_group(
                Args(), self.root, {"claude-code": True, "hermes": False})
        self.assertEqual(list(applied), ["claude-code"])

    def test_a_declined_cutover_is_not_reported_as_applied(self):
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "_ask", return_value="n"), \
                contextlib.redirect_stdout(io.StringIO()):
            applied = self.qctx._host_cutover_group(Args(yes=False), self.root,
                                                    {"hermes": True})
        self.assertEqual(list(applied), [])

    def test_a_timed_out_cutover_is_not_reported_as_applied(self):
        wedged = self.wedged_subprocess(lambda command: "--apply" in command)
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                mock.patch.object(self.qctx, "subprocess", wedged), \
                contextlib.redirect_stdout(io.StringIO()):
            applied = self.qctx._host_cutover_group(Args(), self.root, {"hermes": True})
        self.assertEqual(list(applied), [])

    def test_the_apply_is_the_same_script_with_the_suite_not_skipped(self):
        with mock.patch.dict(os.environ, self.env(), clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.qctx._host_cutover_group(Args(), self.root, {"hermes": True})
        printed = out.getvalue()
        self.assertIn("skip=1", printed)          # the plan
        self.assertIn("skip= args=--apply", printed)   # the apply, suite not skipped
        self.assertIn("hermes --apply", self.calls())


class Args:
    """The two attributes the host groups read off argparse."""

    def __init__(self, yes=True, config_only=False):
        self.yes = yes
        self.config_only = config_only
        self.check = False
        self.json = False


class ClosingBehaviour(unittest.TestCase):
    """Two things the spec asks for that are easy to leave out, and both are about NOT
    pretending: stop when the thing underneath is still broken, and say out loud what the
    wizard cannot do for you."""

    def setUp(self):
        self.qctx = load_cli()

    def test_the_verdict_is_recomputed_over_the_merged_checks(self):
        """The discriminating case, which no offline end-to-end run can produce: a
        diagnose that says ready, over plumbing that does not."""
        healthy = {"ready": True, "checks": [], "blockers": [], "warnings": [],
                   "detected_dim": 1024, "memory_suggestions": []}
        broken = [{"name": "launcher", "ok": False, "detail": "not on PATH",
                   "fix_hint": None, "warning": False},
                  {"name": "PATH", "ok": False, "detail": "not on PATH",
                   "fix_hint": None, "warning": True}]
        merged = self.qctx.merged_report(healthy, broken)
        self.assertFalse(merged["ready"])
        self.assertEqual([c["name"] for c in merged["blockers"]], ["launcher"])
        self.assertEqual([c["name"] for c in merged["warnings"]], ["PATH"])
        self.assertEqual([c["name"] for c in merged["checks"]], ["launcher", "PATH"])

    def test_blockers_stop_the_run_before_the_host_steps(self):
        """No Qdrant, no point installing into a host. It names what did not answer and
        stops — instead of asking fifteen questions that cannot work."""
        blocked = {"ready": False, "blockers": [{"name": "Qdrant", "detail": "no answer",
                                                 "ok": False, "fix_hint": None,
                                                 "warning": False}]}
        self.assertTrue(self.qctx.should_stop_before_hosts(blocked))
        self.assertFalse(self.qctx.should_stop_before_hosts({"ready": True,
                                                             "blockers": []}))

    def test_the_manual_steps_are_named_per_host(self):
        """One entry per host, keyed by host, so a run can print only the steps that
        belong to what it actually touched."""
        hosts = [host for host, _script, _skip in self.qctx.HOST_SECTIONS]
        self.assertEqual(set(self.qctx.MANUAL_STEPS), set(hosts))
        self.assertIn("hermes hooks list", self.qctx.MANUAL_STEPS["hermes"])
        self.assertIn("restart", self.qctx.MANUAL_STEPS["claude-code"].lower())

    def test_the_report_phase_skips_the_host_plans_on_a_run_that_will_act(self):
        """On the acting path each host's dry run used to execute TWICE — once for the
        report block at the top and once inside `_host_cutover_group` — so the user saw
        the same plan printed twice, and the FIRST one was stale: it ran before the
        launcher and the configuration were written, which is exactly what the second
        run exists to reflect.
        """
        acting = Args(yes=True)
        self.assertFalse(self.qctx.report_hosts(acting))
        for reporting in (Args(yes=False), Args(yes=True)):
            reporting.check = True
            self.assertTrue(self.qctx.report_hosts(reporting))
        for as_json in (Args(yes=False), Args(yes=True)):
            as_json.json = True
            self.assertTrue(self.qctx.report_hosts(as_json),
                            "--json returns before acting, so it must carry the hosts")
        no_terminal = Args(yes=False)
        self.assertTrue(self.qctx.report_hosts(no_terminal),
                        "a run that only reports has to show what it found")
        self.assertFalse(self.qctx.report_hosts(Args(yes=True, config_only=True)))


if __name__ == "__main__":
    unittest.main()
