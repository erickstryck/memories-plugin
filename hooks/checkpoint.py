#!/usr/bin/env python3
"""Hook de CHECKPOINT: a cada N interações, injeta o procedimento de gravação.

Contraparte de escrita do `recall.py`. Este hook não grava nada — ele entrega ao
modelo o procedimento completo, no momento em que há conversa acumulada para
destilar.

O texto é deliberadamente AUTOSSUFICIENTE. Um lembrete de uma linha ("salve o que
for durável") produz memória vaga, duplicada e sem metadata, e o custo aparece
meses depois, quando a busca devolve três versões contraditórias do mesmo fato e
ninguém sabe qual vale. Quem lê o bloco tem de conseguir agir sem abrir mais nada.

Configuração:
    QCTX_CHECKPOINT_INTERVAL   interações entre checkpoints (default 5)
    QCTX_CHECKPOINT_DISABLED   "1" desliga
    QCTX_STATE_DIR             onde guardar o contador
"""
import json
import os
import sys
from pathlib import Path

INTERVALO = int(os.environ.get("QCTX_CHECKPOINT_INTERVAL")
                or os.environ.get("REMEMBER_INTERVAL") or "5")
STATE_DIR = Path(os.environ.get("QCTX_STATE_DIR") or (Path.home() / ".memories-plugin" / "state"))

PROCEDIMENTO = """[checkpoint de memória — escrita no acervo de longo prazo]
Interação {count} desta conversa (a cada {intervalo}). Faça o checkpoint AGORA, em uma
passada curta, sem desviar da tarefa em andamento. Se nada durável surgiu desde o
último checkpoint, não salve nada e diga isso em uma linha — memória vazia é melhor
que memória de enchimento.

1. VARRA a conversa desde o último checkpoint e liste os candidatos. O que qualifica,
   por tipo:
   - `user` — quem o usuário é: papel, expertise, preferências estáveis.
   - `feedback` — como ele quer que você trabalhe (correções E abordagens
     confirmadas). SEMPRE inclua o porquê; sem o motivo, a orientação é reaplicada
     fora de contexto na próxima sessão.
   - `project` — objetivos, restrições e decisões em curso que NÃO se deduzem do
     código nem do histórico do git. Converta data relativa em absoluta.
   - `reference` — ponteiros externos (URL, dashboard, ticket) e comportamento
     MEDIDO de plataforma, SDK ou biblioteca.
   VALE MAIS QUE TUDO: comportamento que você teve de MEDIR — uma probe, um grep, um
   branch que você rodou. É o conhecimento caro, e é o que impede a próxima sessão de
   re-medir. Registre COMO foi medido, para dar para refazer.
   DESCARTE: conversa passageira, detalhe de uma vez só, estado volátil ("estamos na
   linha 42"), e o que já está no repositório, no git ou nas instruções do projeto.

2. DEDUPLIQUE antes de escrever. Para cada candidato, uma busca curta. Match próximo
   (score alto, mesmo fato) é ATUALIZAÇÃO naquele id, NÃO registro novo.

3. CORRIJA o que está errado, na mesma passada. Se uma memória se revelou errada ou
   obsoleta — inclusive uma que VOCÊ escreveu hoje, se uma medição ou review a
   desmentiu — atualize dizendo o que a versão antiga afirmava, o que foi medido e
   quando. Nunca deixe a errada de pé com uma nova ao lado: duas memórias
   contraditórias são piores que uma corrigida, porque quem lê depois não tem como
   saber qual ganha.

4. ESCREVA um FATO ATÔMICO por registro. Parágrafo inteiro como uma memória só
   arruína a busca semântica, porque o vetor fica a média de vários assuntos.
   Metadata obrigatória: {{"type": "user|feedback|project|reference",
   "date": "YYYY-MM-DD", "source": "conversation"}}. Acrescente `project`, `area`,
   `corrected` ou `supersedes` quando ajudarem a filtrar depois.

5. CONFIRME em uma lista curta: o que foi salvo ou atualizado, cada item com seu id,
   no idioma do usuário.

Os comandos estão na skill de memória; o essencial do procedimento está aqui."""


def main() -> None:
    if os.environ.get("QCTX_CHECKPOINT_DISABLED") == "1":
        return

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    session = "".join(c if c.isalnum() or c in "-_" else "_"
                     for c in str(data.get("session_id") or "default"))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    counter = STATE_DIR / f"checkpoint-{session}.count"

    try:
        n = int(counter.read_text().strip())
    except Exception:
        n = 0
    n += 1
    counter.write_text(str(n))

    if INTERVALO <= 0 or n % INTERVALO != 0:
        return  # silencioso nas interações intermediárias

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": PROCEDIMENTO.format(count=n, intervalo=INTERVALO),
        }
    }))


if __name__ == "__main__":
    main()
