"""Quem ainda está usando o daemon.

Um lease é um bilhete de "estou vivo": o pid do processo do HOST e o instante em que ele
começou. O daemon confere os bilhetes a cada ciclo e encerra quando nenhum resta.

POR QUE `(pid, starttime)` E NÃO SÓ O PID: o sistema recicla números de processo. Um pid
sozinho faz um processo qualquer segurar o daemon para sempre, e a falha seria invisível —
o daemon apenas nunca encerraria.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import lease  # noqa: E402


def a_state_dir() -> str:
    d = tempfile.mkdtemp()
    os.environ["QCTX_STATE_DIR"] = d

    return d


class TestReadingAProcess(unittest.TestCase):
    def test_our_own_start_time_is_readable(self):
        self.assertTrue(lease.process_start(os.getpid()))

    def test_a_pid_that_does_not_exist_reads_as_None(self):
        """Sem levantar: o daemon chama isto a cada ciclo, para pids que ele ESPERA ver morrer."""
        self.assertIsNone(lease.process_start(4_000_000))

    def test_the_start_time_is_STABLE_across_reads(self):
        """Se variasse, todo lease pareceria reciclado e o daemon encerraria com o host vivo."""
        self.assertEqual(lease.process_start(os.getpid()), lease.process_start(os.getpid()))


class TestWritingAndCheckingALease(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_a_lease_for_a_live_process_is_alive(self):
        entry = lease.write("s1", "claude", pid=os.getpid())
        self.assertTrue(lease.alive(entry))

    def test_a_lease_for_a_DEAD_process_is_not_alive(self):
        done = subprocess.Popen([sys.executable, "-c", "pass"])
        done.wait(timeout=60)
        entry = lease.write("s1", "claude", pid=done.pid)
        self.assertFalse(lease.alive(entry))

    def test_a_RECYCLED_pid_is_not_alive(self):
        """O caso que o pid sozinho não pega: o número existe, mas é outro processo."""
        entry = lease.write("s1", "claude", pid=os.getpid())
        entry["starttime"] = "1"          # como se o processo original tivesse começado antes
        self.assertFalse(lease.alive(entry))

    def test_a_lease_missing_its_fields_is_not_alive(self):
        """Arquivo truncado ou de uma versão anterior: não segura o daemon."""
        for broken in ({}, {"pid": os.getpid()}, {"starttime": "1"}, {"pid": "x"}):
            with self.subTest(broken=broken):
                self.assertFalse(lease.alive(broken))


class TestSweeping(unittest.TestCase):
    def setUp(self):
        a_state_dir()

    def test_live_returns_the_living_and_REMOVES_the_dead(self):
        done = subprocess.Popen([sys.executable, "-c", "pass"])
        done.wait(timeout=60)
        lease.write("alive", "claude", pid=os.getpid())
        lease.write("dead", "hermes", pid=done.pid)
        live = lease.live()
        self.assertEqual([e["session_id"] for e in live], ["alive"])
        self.assertEqual(sorted(p.name for p in lease.dir().glob("*.json")), ["alive.json"])

    def test_no_leases_at_all_is_an_empty_list_and_not_an_error(self):
        """O estado normal antes da primeira sessão, e o estado que faz o daemon encerrar."""
        self.assertEqual(lease.live(), [])

    def test_a_corrupt_lease_file_is_removed_rather_than_crashing_the_sweep(self):
        lease.dir().mkdir(parents=True, exist_ok=True)
        (lease.dir() / "junk.json").write_text("{not json")
        self.assertEqual(lease.live(), [])
        self.assertFalse((lease.dir() / "junk.json").exists())


class TestFindingTheHOST(unittest.TestCase):
    def test_it_walks_up_and_finds_a_named_ancestor(self):
        """No claude-code o hook é subprocesso: python3 -> bash -> claude. Medido nesta
        máquina em 2026-08-18."""
        found = lease.find_host_pid(names=(os.path.basename(sys.executable),))
        self.assertIsNotNone(found, "não achou o próprio interpretador na árvore")
        pid, start = found
        self.assertEqual(lease.process_start(pid), start)

    def test_an_ancestor_that_is_not_there_yields_None(self):
        self.assertIsNone(lease.find_host_pid(names=("nao-existe-este-processo",)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
