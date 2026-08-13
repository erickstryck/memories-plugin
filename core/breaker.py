"""Disjuntor de arquivo, para dependência que fica indisponível por MINUTOS.

O caso concreto que motivou: o cross-encoder roda numa GPU compartilhada. Quando
outro processo satura a placa, a indisponibilidade dura minutos — e sem disjuntor
TODA invocação paga o timeout inteiro para redescobrir o que a anterior já sabia.
Num caminho disparado a cada interação do usuário, isso é um imposto direto no
tempo de resposta.

O estado é um ARQUIVO, não memória de processo, por duas razões: cada invocação de
hook é um processo novo (memória não sobrevive), e a dependência é compartilhada
entre sessões (a GPU é uma só, então o que uma sessão descobriu vale para as
outras).
"""
import time
from pathlib import Path


class Breaker:
    def __init__(self, path: str | Path, cooldown_seconds: float = 300.0):
        self.path = Path(path)
        self.cooldown = cooldown_seconds

    def aberto(self) -> float | None:
        """Segundos desde a última falha, se ainda estamos no período de espera.

        None significa "pode tentar". Qualquer problema de leitura resolve para
        None de propósito: disjuntor com defeito não pode virar o motivo de a
        funcionalidade parar.
        """
        if self.cooldown <= 0:
            return None
        try:
            ultima = float(self.path.read_text().strip())
        except Exception:
            return None
        ocioso = time.time() - ultima
        if ocioso < self.cooldown:
            return ocioso

        return None

    def armar(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(time.time()))
        except Exception:
            pass

    def limpar(self) -> None:
        try:
            self.path.unlink()
        except Exception:
            pass
