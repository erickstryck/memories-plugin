---
name: doc-index
description: Indexa documento longo no Qdrant e busca só os trechos relevantes, via `qctx docs` — em vez de ler o arquivo inteiro para o contexto. Use ANTES de ler arquivo grande (log, dump, transcrição, código extenso, relatório) quando a pergunta é sobre uma parte dele; e quando o usuário pedir para guardar um documento para consulta futura. Também para buscar em documentação já guardada.
---

# doc-index

Responder uma pergunta sobre 40 linhas de um arquivo de 8.000 não deve custar o
arquivo inteiro em contexto. O `qctx` lê o arquivo do disco **fora** do seu
contexto, fatia, embeda e devolve só os trechos que respondem.

## Dois acervos, e a escolha é sua

| comando | acervo | expira | quando |
|---|---|---|---|
| `qctx docs index <arquivo>` | temporário | 24h (`--ttl 2h`, `7d`…) | arquivo aberto para a tarefa de agora |
| `qctx docs keep <arquivo>` | biblioteca | nunca | documento que vale consultar depois |

Use `keep` quando o usuário disser algo como *"esse doc é importante guardar para
consulta"*. Use `index` para o resto.

## Quando indexar em vez de ler

**Indexe** quando as duas coisas valem: o arquivo é grande (acima de ~2.000 linhas
ou ~100 KB) **e** a pergunta é sobre uma parte dele. Casos típicos: log, dump,
transcrição, CSV grande, arquivo de código extenso, relatório longo.

**NÃO indexe** — leia inteiro:

- Spec, plano, README, design doc. Documentos que se leem do começo ao fim; buscar
  trechos perde o fio do argumento, que é justamente o conteúdo.
- Qualquer arquivo abaixo do limite. Indexar custa uma ida à rede e um índice para
  limpar depois; ler é imediato.
- Quando o usuário pediu explicitamente para você **ler** o arquivo.

Avise em uma linha quando indexar em vez de ler, para o usuário saber que você não
tem o arquivo inteiro em contexto.

## Buscar

```bash
qctx docs search "<pergunta>"                     # os dois acervos
qctx docs search "<pergunta>" --scope library     # só a biblioteca
qctx docs search "<pergunta>" --doc-id <id>       # só um documento
```

### Leia o que a saída está dizendo

**Arquivo de texto devolve LOCALIZAÇÃO, não conteúdo:**

```
1. [temporário] /caminho/arquivo.py:317-354  (CE 0.526)
   # DISJUNTOR. Numa GPU compartilhada a saturação dura minutos…
   -> ler linhas 317-354 do arquivo para o conteúdo atual
```

O trecho mostrado é uma **prévia**. Para trabalhar sobre o conteúdo, leia aquelas
linhas no arquivo: o índice é uma foto do momento da indexação, e o arquivo pode ter
mudado. Isto importa especialmente para editar código — você precisa do número de
linha e do texto atual.

**Origem não relegível por região** (PDF convertido, transcrição, dump colado) vem
marcada como `[FOTO de <data>]` com o texto completo, porque não há região para
reler.

**Marcas que exigem ação:**

| marca | significa |
|---|---|
| `⚠ arquivo mudou desde a indexação` | o trecho é de uma versão antiga. Releia o arquivo, e rode `qctx docs refresh` se for da biblioteca. |
| `⚠ arquivo não existe mais` | o índice sobreviveu ao arquivo. Remova com `drop`. |
| `(CE 0.xxx)` | o cross-encoder julgou a relevância. |
| `(CE? 0.xxx)` | abaixo do corte de confiança — pode não responder. |
| `(denso 0.xxx)` | o cross-encoder não foi usado. Não é veredito de relevância, só proximidade de vetor. |
| `re-rank colapsou … línguas diferentes` | pergunta e documento em idiomas diferentes derrubam o cross-encoder; a ordem passou a ser densa, que é indiferente à língua. Os resultados seguem úteis. Se quiser ordem melhor, **repita a pergunta no idioma do documento**. |

Aquele último caso é medido e vale saber: a mesma pergunta sobre o mesmo documento
em inglês deu score 0.2073 em inglês e 0.0004 em português. O estágio denso não se
abala com isso, o cross-encoder sim.

## Manter e limpar

```bash
qctx docs list                        # o que está indexado, e quando expira
qctx docs refresh --scope library     # reindexa o que mudou no disco
qctx docs drop <doc-id>               # remove um documento
qctx docs drop --purge-tmp            # apaga o temporário inteiro; biblioteca intacta
```

Reindexar o mesmo arquivo **substitui** o índice anterior — não duplica.

Ao terminar uma tarefa que usou índice temporário, um `drop <doc-id>` é cortesia,
mas não é obrigatório: o TTL limpa sozinho.

## Limites

- Arquivo binário é recusado. Converta para texto antes.
- Um trecho tem ~2.400 chars por alvo, cortado em fronteira estrutural (título,
  parágrafo, definição de topo) para o vetor não virar a média de dois assuntos.
- O acervo de documentos **nunca** é o acervo de memória: são coleções distintas, e
  a configuração recusa apontá-las para o mesmo lugar. Documento longo vira dezenas
  de trechos verbosos que venceriam por volume em toda busca de memória.
