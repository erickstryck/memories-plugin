# Resolução da janela de contexto — desenho

Sub-projeto próprio, nascido de uma pergunta do usuário em 2026-08-17: *"não tem como mesmo ele
pegar o contexto atual mostrado no hermes/claude-code?"*. A resposta curta é que o **contexto
usado** já é lido nos dois hosts; o que faltava era o **denominador** — o tamanho da janela.

## O problema, medido

A guarda de arquivo grande decide por percentual do que **resta**, então precisa da janela. Hoje
ela vem de `core/windows.py::MODEL_WINDOWS`, uma tabela de **tetos** por nome de modelo nu, e
`context_window` na config sobrescreve. Consequência medida em 2026-08-17:

| situação | janela | efeito |
|---|---|---|
| `claude-opus-5` (variante de 1M) | 1.000.000 | correto, guarda funciona |
| `claude-opus-5` de 200k, sem declarar | 1.000.000 | guarda quase **inerte** |
| `MiniMax-M2.7`, `Qwen3.8-27B` (fora da tabela) | **0** | guarda **libera tudo** |

O segundo caso é o custo aceito da emenda de tetos, e está documentado. O terceiro é o que este
sub-projeto ataca: **no hermes a guarda hoje não bloqueia nada** sem declaração manual.

## O que cada host de fato fornece — quatro caminhos verificados

Isto foi medido, e duas conclusões anteriores minhas estavam erradas antes de chegar aqui. O
registro das duas está em `progress.md` e na memória `000300fe`; o resumo:

**claude-code — a janela NÃO é obtenível por um plugin.**

| caminho | resultado |
|---|---|
| payload de hook | os 20 eventos saem do mesmo construtor `my()`; só `SessionStart` acrescenta `model`. Nenhum traz janela |
| transcript `.jsonl` | nenhum campo de janela; as ocorrências encontradas eram texto de conversa |
| estado em disco (`.claude.json`) | só flags de feature |
| `statusLine` | **tem** `context_window.context_window_size` — entregue a um comando de status, não a hooks |

**hermes — a janela É obtenível, do endpoint que serve o modelo.** Medido contra o servidor do
usuário: `GET <base_url>/models` devolve `max_model_len: 524288` para `Qwen3.8-27B`, **idêntico ao
`524.3K` que a interface do hermes exibe**.

**E o `models_dev_cache.json` do hermes NÃO serve**, embora tenha o dado. `get_model_context_length`
resolve em dez passos e, para provedores que sondam ao vivo, o passo 1 **ignora o cache de
propósito**; o cache é o passo 5f. Ele dizia 262.144 onde o real é 524.288 — errado por 2×, e na
direção que faz a guarda bloquear demais.

> **Lição que vale além desta spec:** achar um arquivo com o dado não é achar a **fonte** do dado.
> O passo que faltava era ler o resolvedor e a ordem de precedência dele, não procurar mais
> arquivos. Dado certo no lugar errado é pior que dado ausente, porque parece resposta.

## O que se decidiu

| Decisão | Escolha | Alternativas recusadas |
|---|---|---|
| Forma | **Mecanismo uniforme, fonte por host** | fonte única para os dois; nada |
| Onde a cascata mora | **`core`**, uma só | uma por adaptador |
| Quem sonda | **Os hooks que já pagam rede** | a guarda; um daemon |
| Endpoint do hermes | **Lido do config do próprio hermes** | nova chave de config declarada |
| claude-code | **Segue na tabela**, com o limite escrito | sequestrar o `statusLine` |

**Por que não uma fonte única:** os dois hosts não a fornecem, e forçar simetria seria inventar. O
que é uniforme é o **mecanismo** — uma decisão no núcleo, coleta diferente por host — que é como
este plugin já trata recall, checkpoint e a própria guarda.

**Por que o `statusLine` foi recusado:** o número existe lá, e um plugin poderia registrar um
comando que o captura. Mas isso ocupa uma configuração do usuário que costuma já estar em uso, e um
plugin de memória mexer no status bar para espiar um número é invasivo demais para o ganho — ainda
mais onde a tabela já acerta o caso real dele.

## Arquitetura

### A cascata, e ela é a única regra

```
context_window declarado  ->  valor sondado em cache  ->  tabela de tetos  ->  0 (libera)
```

Cada degrau só é consultado se o anterior não respondeu. **Declarado vence sempre**: quem declarou
não quer adivinhação, e um upgrade de host não pode mudar o que ele fixou.

### Quem preenche o cache nunca é a guarda

A guarda precisa da janela **para decidir**, então não pode ser quem a busca: seria circular, e
seria rede no caminho que roda antes de **cada** leitura de arquivo — o defeito que esta base já
mediu em 2026-08-16, quando seis idas sequenciais somaram 7,2 s contra um limite de hook de 5 s.

Quem preenche são os pontos que **já pagam rede por outro motivo**: o hook de recall no claude-code
e o `prefetch()` no hermes. Eles atualizam quando o valor está velho; a guarda apenas **lê o
cache**, e um cache ausente é simplesmente o degrau seguinte da cascata.

O cache vive em `state_dir()/model-windows.json`, chaveado por `(endpoint, modelo)` — porque o
mesmo nome de modelo em servidores diferentes pode ter janelas diferentes, e chavear só pelo nome
faria dois setups se contaminarem.

### A sonda do hermes

Lê `base_url` e o **nome** da variável de chave (`key_env`) do config do próprio hermes, e lê essa
variável do ambiente — que o processo do hermes já tem, porque é ele quem carrega o plugin. Chama
`GET <base_url>/models` e procura, na entrada cujo `id` casa com o modelo da sessão, um destes
campos, nesta ordem:

```
max_model_len      (vLLM — é o caso medido)
context_length     (Ollama e outros)
context_window
```

**A lista é assumidamente incompleta, e isso vai no docstring.** Campo ausente, entrada ausente,
endpoint fora do ar ou config ilegível: todos caem para o degrau seguinte. Nada é inventado.

Acoplamento aceito e declarado: ler o config de outro projeto é dependência de um formato que pode
mudar. É o mesmo acoplamento que o adaptador já tem com o `state.db` do hermes, e a defesa é a
mesma — teste que morde quando o formato muda, e falha que desce a cascata em vez de quebrar.

### O claude-code

Segue na tabela de tetos. Isso **não é uma regressão** — é o comportamento atual, e ele acerta a
variante de 1M. O que muda é que o limite passa a estar escrito com o motivo: os quatro caminhos
foram medidos e nenhum entrega a janela a um plugin.

## Modos de falha

**Toda falha desce a cascata, nunca sobe.** A assimetria que governa: janela grande demais só faz a
guarda **dormir**; pequena demais a **inverte** e ela bloqueia leitura legítima. Nenhum caminho novo
pode produzir uma janela menor que a verdade sem passar pelo usuário.

| Falha | Destino |
|---|---|
| config do hermes ilegível ou sem `base_url` | degrau seguinte (tabela) |
| variável de chave ausente do ambiente | degrau seguinte |
| endpoint fora do ar ou lento | **mantém o último valor conhecido**; se não há, degrau seguinte |
| modelo não está no `/models` | degrau seguinte |
| campo de janela com nome desconhecido | degrau seguinte |
| valor não numérico ou <= 0 | degrau seguinte |
| cache velho | **usado assim mesmo**, e a atualização é tentada; valor velho é melhor que a guarda dormir |
| cache corrompido | lido como vazio, nunca levanta |

## Testes

- **A cascata, degrau por degrau**, com cada um desligado isoladamente — quatro fixtures, cada uma
  satisfazendo **um** degrau, porque fixture que satisfaz dois não prova nenhum.
- **A guarda NUNCA sonda**: prova por execução, patchando a rede para levantar e afirmando que o
  caminho comum da guarda decide mesmo assim.
- **Os três nomes de campo**, e um quarto nome desconhecido que tem de cair para a tabela.
- **Falha que desce, nunca sobe**: para cada linha da tabela acima, afirmar que a janela resultante
  é **maior ou igual** à que o degrau seguinte daria — nunca menor.
- **Chaveamento por `(endpoint, modelo)`**: mesmo modelo em dois endpoints não compartilha valor.
- **Integração guardada** (`QCTX_INTEGRATION=1`) contra o endpoint real, afirmando o contrato da
  resposta — a metade que um fake não prova.
- **Sonda de mutação em toda guarda**, com contagem de ocorrências verificada, escopo declarado, e
  confirmação de que a mutação aterrou e a suíte rodou.

## SOLID

- **Responsabilidade única** — o `core` decide a cascata; cada adaptador só sabe **buscar** no seu
  host; o cache só persiste.
- **Aberto/fechado** — um host novo entra fornecendo uma função de busca, sem tocar a cascata.
- **Inversão de dependência** — a cascata recebe a função de busca; não conhece hermes nem HTTP.
- **Reusar, não copiar** — `state_dir` de `core/knobs.py`, e a leitura de env pelos helpers que já
  existem.

## Fora de escopo, deliberadamente

- **Sondar a API da Anthropic** no claude-code. Só valeria com chave de API, e o usuário usa
  assinatura; e a tabela já acerta o caso dele.
- **Registrar um `statusLine`** para capturar a janela. Recusado acima, com motivo.
- **Detectar a variante `[1m]`** pelo nome do modelo. Medido: o transcript grava o nome nu, e o
  sufixo não aparece em lugar nenhum.
- **Contexto usado** — já é lido nos dois hosts e não muda aqui.
