---
name: doc-index
description: Indexes a long document in Qdrant and searches only the relevant chunks, via `qctx docs` — instead of reading the whole file into context. Use it BEFORE reading a large file (log, dump, transcript, extensive code, report) when the question is about one part of it; and when the user asks you to keep a document for later reference. Also for searching documentation already kept.
---

# doc-index

Answering a question about 40 lines of an 8,000-line file should not cost the whole
file in context. `qctx` reads the file from disk **outside** your context, slices it,
embeds it and returns only the chunks that answer.

## Two archives, and the choice is yours

| command | archive | expires | when |
|---|---|---|---|
| `qctx docs index <file>` | temporary | 24h (`--ttl 2h`, `7d`…) | a file opened for the task at hand |
| `qctx docs keep <file>` | library | never | a document worth consulting later |

Use `keep` when the user says something like *"this doc is important, keep it for
reference"*. Use `index` for everything else.

## When to index instead of reading

**Index** when both hold: the file is large (above ~2,000 lines or ~100 KB) **and** the
question is about one part of it. Typical cases: a log, a dump, a transcript, a large
CSV, an extensive code file, a long report.

**Do NOT index** — read it whole:

- A spec, a plan, a README, a design doc. Documents read from start to finish;
  searching for chunks loses the thread of the argument, which is precisely the
  content.
- Any file below the threshold. Indexing costs a network round trip and an index to
  clean up later; reading is immediate.
- When the user explicitly asked you to **read** the file.

Say so in one line when you index instead of reading, so the user knows you do not have
the whole file in context.

## When the read comes back refused

A `Read` (`read_file` on hermes) can be **refused** with a message about what it would cost
in context. That is this plugin's big-file guard — `hooks/bigfile.py` on claude-code,
`hosts/hermes/bigfile.py` on hermes — and not a broken tool, a missing file or a permission
problem. Retrying the same call returns the same refusal.

The message names the one thing to do instead: index the file and search it, or search it
because it is already indexed. Do that, and say in one line that you did.

Three things worth knowing so you do not work around it by accident:

- It prices **one read**, not the file — so a bounded read (`offset`/`limit`) of a region a
  search pointed you at is charged for what it actually loads, and normally passes.
- Paging through the whole file in slices is not a way around it: every slice you load adds
  to the context in use, and the guard starts refusing as it fills.
- `--full` is the **user's** escape word. It is read from the last user turn only, so
  writing it in your own output does nothing at all. If the file really has to be read
  whole, say why and let the user decide. The word is configurable
  (`QCTX_BIGFILE_ESCAPE`), and the refusal message always names the one in force — quote it
  from there rather than from here.

## Searching

```bash
qctx docs search "<question>"                     # both archives
qctx docs search "<question>" --scope library     # library only
qctx docs search "<question>" --doc-id <id>       # a single document
```

### Read what the output is telling you

**A text file returns a LOCATION, not content:**

```
1. [temporary] /path/file.py:317-354  (CE 0.526)
   # BREAKER. On a shared GPU, saturation lasts minutes…
   -> read lines 317-354 of the file for the current content
```

The chunk shown is a **preview**. To work on the content, read those lines in the file:
the index is a snapshot from the moment of indexing, and the file may have changed.
This matters especially for editing code — you need the line number and the current
text.

**A source that cannot be re-read by region** (a converted PDF, a transcript, a pasted
dump) comes back marked `[SNAPSHOT from <date>]` with the full text, because there is
no region to re-read.

**Marks that call for action:**

| mark | means |
|---|---|
| `⚠ file changed since indexing` | the chunk is from an old version. Re-read the file, and run `qctx docs refresh` if it is from the library. |
| `⚠ file no longer exists` | the index outlived the file. Remove it with `drop`. |
| `(CE 0.xxx)` | the cross-encoder judged the relevance. |
| `(CE? 0.xxx)` | below the confidence cutoff — it may not answer. |
| `(dense 0.xxx)` | the cross-encoder was not used. This is not a relevance verdict, only vector proximity. |
| `re-rank collapsed … different languages` | a question and a document in different languages knock the cross-encoder down; the order became dense, which is language-agnostic. The results are still useful. If you want a better order, **repeat the question in the document's language**. |

That last case is measured and worth knowing: the same question about the same English
document scored 0.2073 in English and 0.0004 in Portuguese. The dense stage is unmoved
by this; the cross-encoder is not.

## Maintaining and cleaning up

```bash
qctx docs list                        # what is indexed, and when it expires
qctx docs refresh --scope library     # reindexes what changed on disk
qctx docs drop <doc-id>               # removes one document
qctx docs drop --purge-tmp            # deletes the whole temporary archive; library untouched
```

Reindexing the same file **replaces** the previous index — it does not duplicate.

When finishing a task that used a temporary index, a `drop <doc-id>` is a courtesy but
not required: the TTL cleans up on its own.

## The same operations, whatever the host gives you

The commands above are the CLI surface. Where the host exposes document **tools**
(`docs_index`, `docs_keep`, `docs_search`, `docs_list`, `docs_refresh`, `docs_drop`, as
hermes-agent does), they are the same operations over the same two archives — the choice of
surface changes nothing in this skill, including which archive a command writes to.

## Limits

- A binary file is refused. Convert it to text first.
- A chunk targets ~2,400 chars, cut on a structural boundary (a heading, a paragraph, a
  top-level definition) so the vector does not become the average of two subjects.
- The document archive is **never** the memory archive: they are distinct collections,
  and the configuration refuses to point them at the same place. A long document becomes
  dozens of verbose chunks that would win on volume in every memory search.
