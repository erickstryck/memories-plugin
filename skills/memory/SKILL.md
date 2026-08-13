---
name: memory
description: Memória de longo prazo em Qdrant, via `qctx memory` — AS DUAS DIREÇÕES. Buscar primeiro, depois salvar. Use quando o usuário pedir para salvar/persistir/lembrar algo; E — obrigatório — antes de responder pergunta não trivial, iniciar investigação, afirmar fato sobre o codebase, SDK, plataforma ou decisão passada, propor desenho, ou reverter conclusão anterior. Se pode existir precedente, busque antes de responder.
---

# memory

Memória de longo prazo, entre sessões, no acervo semântico configurado. Duas
direções, e a de leitura é a que se esquece:

- **BUSCAR** — antes de afirmar. Barato, e é o que impede re-decidir o que já foi
  decidido.
- **SALVAR** — persistir fato durável, um fato atômico por registro, deduplicado.

**Princípio:** *busque antes de afirmar, salve só o que ainda vai importar depois.*

## Um hook já garante o piso — a profundidade é sua

Um hook de `UserPromptSubmit` roda uma busca a **cada** prompt do usuário, antes de
você ver o texto, e injeta as memórias relevantes. Você não precisa buscar para
responder o prompt em si, e repetir aquela mesma busca genérica é desperdício.

Leia o estado do bloco injetado, porque os três significam coisas diferentes:

| bloco injetado | o que significa |
|---|---|
| memórias listadas | o acervo foi consultado; trate cada uma pela tabela abaixo |
| "nenhuma memória acima do corte" | o acervo **foi** consultado e não tem nada. Não repita a mesma busca ampla. |
| "recall automático — INDISPONÍVEL" | a busca **não rodou** (infra fora). Isto NÃO é evidência de ausência de precedente. Nunca afirme que algo é inédito apoiado nesse turno; tente `qctx memory recall` você mesmo. |

**O que o hook não cobre, e continua inteiramente seu:**

- **Meio de turno.** Nenhum prompt novo chega quando você fecha uma etapa, abre um
  sub-assunto ou chega numa decisão de desenho 20 minutos dentro da tarefa. Busque
  de novo a cada etapa. Memória escrita por sessão paralela só aparece para quem
  relê.
- **Abrir ponteiros.** O hook entrega o hit; abrir os ids que ele cita é seu. Um
  índice é ponteiro, não resposta.
- **Ângulos que o texto do prompt não contém.** O hook monta as consultas com as
  palavras do prompt. Faceta que o usuário não nomeou não foi buscada.
- **Toda a escrita.**
- **Outros acervos.** Coleções de outros sistemas só sob pedido.

## 1. BUSCAR — antes de responder

```bash
qctx memory recall "<tema, em linguagem natural>"     # dois estágios, com re-rank
qctx memory find "<tema>" --limit 8                   # denso puro, mais barato
```

**Gatilhos** — busque quando qualquer um for verdade:

- O usuário pergunta **como algo funciona** no codebase, SDK, plataforma ou infra.
- Você vai **afirmar um fato** sobre comportamento de terceiro (biblioteca, API).
- Você vai **propor desenho ou decisão** em área com histórico.
- O usuário diz algo na forma *"já não discutimos isso?"*, *"não tem isso já?"* —
  é instrução de busca, não pergunta retórica.
- Você está **começando uma investigação** — busque antes de ler código.
- Você vai **reverter ou contradizer** algo que disse antes.
- Um **review ou subagente relata um achado** em área com histórico.

Busque por **tema**, não por símbolo exato: o acervo é semântico. Dois ou três
ângulos diferentes batem uma consulta longa.

| situação | ação |
|---|---|
| um precedente resolve | aplique, diga que é precedente e cite o id. Não re-derive. |
| a memória registra **decisão ou veto** do usuário | vale. Não re-proponha o vetado; se acha que deve mudar, diga explicitamente que é reversão, com a evidência nova. |
| a memória cita arquivo, linha, flag ou versão | **verifique na árvore atual** antes de agir. |
| a memória contradiz o que você mediu | a medição ganha — e então **corrija a memória** (§2.4). Nunca deixe memória sabidamente errada de pé. |
| nada relevante | siga, e considere se a resposta que você vai produzir merece ser salva. |

### Os modos de falha que isto existe para evitar

Observados, repetidamente:

- **Re-propor um desenho vetado.** Uma decisão que o usuário já tomara duas vezes
  foi reintroduzida porque a memória que a guardava nunca foi aberta.
- **Afirmar comportamento de plataforma que uma memória já mediu.** O usuário teve
  de trazer a correção.
- **Citar um índice e nunca abri-lo.** O índice foi citado várias vezes enquanto os
  ids que ele aponta seguiam sem leitura.

Nos três, o custo não foi o erro — foi o usuário ter de ser quem percebeu.

## 2. SALVAR

### 2.1 O que salvar

- `user` — quem o usuário é: papel, expertise, preferências estáveis.
- `feedback` — como ele quer que você trabalhe (correções e abordagens
  confirmadas); **inclua o porquê**, senão a regra é reaplicada fora de contexto.
- `project` — objetivos, restrições e decisões em curso que não se deduzem do
  código nem do git. Converta data relativa em absoluta.
- `reference` — ponteiros externos (URL, dashboard, ticket) e comportamento
  **medido** de plataforma ou SDK.

**Especialmente:** comportamento que você teve de **medir** — uma probe, um grep,
um branch que você rodou. É o conhecimento caro, e o que impede a próxima sessão de
re-medir. Registre **como** foi medido.

**Descarte:** conversa passageira, detalhe de uma vez só, estado volátil, e o que
já está no repositório, no git ou nas instruções do projeto.

### 2.2 Deduplique antes de escrever

```bash
qctx memory find "<consulta curta do fato>"
```

Match próximo (score alto, mesmo fato) é **atualização** naquele id, não registro
novo.

### 2.3 Corrija o que está errado

Quando uma memória se revela errada ou obsoleta, **atualize na mesma passada** —
não escreva uma nova competindo. Diga o que a versão antiga afirmava, o que foi
medido e quando. Acervo com duas memórias contraditórias é pior que um com memória
corrigida, porque quem lê depois não sabe qual vale. Vale também para memória que
**você** escreveu hoje.

### 2.4 Escreva

```bash
qctx memory store "<fato atômico>" --type reference --project X --area Y
qctx memory update <id> --text "<versão corrigida>"
qctx memory delete <id>
```

**Um fato atômico por registro.** Parágrafo inteiro como memória única arruína a
busca, porque o vetor vira a média de vários assuntos. Metadata sempre com `type` e
data.

### 2.5 Confirme

Lista curta do que foi salvo ou atualizado, cada item com seu id, no idioma do
usuário.

## Comandos

| ação | comando |
|---|---|
| buscar com re-rank | `qctx memory recall "<tema>"` |
| buscar denso | `qctx memory find "<tema>" --limit N` |
| ler por id | `qctx memory get <id>` |
| criar | `qctx memory store "<texto>" --type T` |
| atualizar | `qctx memory update <id> --text "..."` |
| remover | `qctx memory delete <id>` |
| listar | `qctx memory list --limit N` |
| ver config | `qctx config show` · `qctx collections list` |

`--json` em qualquer comando devolve saída estruturada.

## Erros comuns

**Na leitura:**
- Responder do que você acha que sabe, quando o acervo tem a medição.
- Tratar *"já não discutimos…"* como retórica em vez de instrução de busca.
- Ler uma memória-índice e nunca abrir os ids que ela aponta.
- Confiar num `arquivo:linha` de memória sem conferir a árvore atual.

**Na escrita:**
- Guardar um parágrafo inteiro como um registro → quebre em fatos atômicos.
- Criar quase-duplicata → busque primeiro, depois atualize.
- Salvar estado volátil ("estamos na linha 42").
- Esquecer metadata → sempre `type` e data.
- Deixar memória desmentida de pé com uma nova ao lado.
