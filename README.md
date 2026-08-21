# ECHO

A minimal experimental conversational agent, extended one capability at a time.

No chatbot UI, no autonomous loop, no vector DB, no RAG, no web search, no
self-modification, no multi-agent orchestration.

## What has been built, in order

Each capability is deliberately narrow, and each report says plainly what the
capability is *not*.

| # | Capability | Code | Report |
| --- | --- | --- | --- |
| 1 | Conversation, LLM call, save/reload | `conversation.py`, `llm.py`, `agent.py`, `storage.py` | this file |
| 2 | Memory — extraction, not transcript-saving | `memory.py`, `memory_store.py`, `extraction.py`, `context.py` | this file |
| 1.5 | Evaluation harness for extraction quality | `evaluation/` | [`docs/ECHO_1_5_MEMORY_EVALUATION.md`](docs/ECHO_1_5_MEMORY_EVALUATION.md) |
| 2 | Belief revision — evidence-weighted, never overwriting | `belief.py`, `belief_store.py` | [`docs/ECHO_2_BELIEF_REVISION.md`](docs/ECHO_2_BELIEF_REVISION.md) |
| 3 | Prediction — sealed, temporally clean, properly scored | `prediction.py`, `calibration.py`, `prediction_ledger.py` | [`docs/ECHO_3_PREDICTION.md`](docs/ECHO_3_PREDICTION.md) |
| 4 | Learning from prediction errors — strategy revision | `learning.py`, `learning_ledger.py`, `strategy.py`, `predictors.py` | [`docs/ECHO_4_LEARNING.md`](docs/ECHO_4_LEARNING.md) |
| 5 | Discovery — bounded hypothesis search over a closed language | `expressions.py`, `hypothesis.py`, `observations.py`, `discovery.py`, `discovery_ledger.py` | [`docs/ECHO_5_DISCOVERY.md`](docs/ECHO_5_DISCOVERY.md) |
| 6 | Transfer — reusing a discovered structure in a different world | `pattern.py`, `transfer.py`, `transfer_ledger.py` | [`docs/ECHO_6_TRANSFER.md`](docs/ECHO_6_TRANSFER.md) |
| 7 | Experimentation — designing an intervention when watching cannot settle it | `causal.py`, `experiment.py` | [`docs/ECHO_7_EXPERIMENTATION.md`](docs/ECHO_7_EXPERIMENTATION.md) |
| 8 | Causal abstraction — seeing vs doing, and counterfactuals | `causal.py`, `counterfactual.py` | [`docs/ECHO_8_CAUSAL.md`](docs/ECHO_8_CAUSAL.md) |
| 9 | Metacognition — a measured record of its own reliability | `metacognition.py` | [`docs/ECHO_9_METACOGNITION.md`](docs/ECHO_9_METACOGNITION.md) |
| 10 | Social learning — weighing testimony by earned reliability | `social.py` | [`docs/ECHO_10_SOCIAL_LEARNING.md`](docs/ECHO_10_SOCIAL_LEARNING.md) |
| 11 | Integration — all of it as one loop, with provenance | `integration.py` | [`docs/ECHO_11_INTEGRATION.md`](docs/ECHO_11_INTEGRATION.md) |
| — | Consolidated summary and capability table | — | [`docs/ECHO_7_11_SUMMARY.md`](docs/ECHO_7_11_SUMMARY.md) |

**These are different things and the reports keep them apart.** Memory stores;
belief revision moves a number by a fixed rule; prediction seals a probability
before an outcome exists; learning changes the *method* because measured errors
said so; discovery finds a relationship nobody encoded; transfer reuses the
*shape* of one discovery elsewhere; experimentation acts when watching cannot
settle a question; causal inference distinguishes seeing from doing;
metacognition scores ECHO's own track record; social learning weighs testimony
by earned reliability; integration runs all of it in one cycle.

A value changing is not learning. Predicting something is not understanding it.
Reusing a structure is not knowing what it means. A scoreboard of past accuracy
is not self-awareness, and a fixed sequence of stages is not autonomy. This is
an experimental adaptive reasoning architecture; nothing in it establishes
consciousness, sentience, self-awareness, or general intelligence, and no
measurement in any report was designed to.

The sections below describe capabilities 1 and 2; the reports cover the rest.

## Three kinds of state, kept separate

The distinction is the point of the design, so it is enforced structurally
rather than by convention.

| | What it holds | Scope | Survives restart? |
| --- | --- | --- | --- |
| **Conversation history** (`conversation.py`) | The verbatim transcript — what was *said* | One conversation | Yes, as `<root>/conversations/<id>.json` |
| **Persistent memory** (`memory.py`, `memory_store.py`) | Discrete extracted statements — what is *known* | Every conversation | Yes, as `<root>/memories.json` |
| **Temporary context** (`context.py`) | Scratch for the turn being taken now | One turn | **No — by design** |

`WorkingContext` has no `to_dict` and no `save`. That is not an oversight: if
temporary state could be persisted it eventually would be, and the line between
what ECHO knows and what it happens to be holding would blur. A test asserts
those methods do not exist.

Recalled memories reach the model by being folded into the **system prompt for a
single call**. They are never appended as messages, so a memory that informs a
reply does not thereby become conversation history.

## Nothing becomes a memory by accident

`Agent.send()` creates no memories. Ever. The only path from conversation to
persistent memory is `Agent.remember()`, and two gates stand in the way:

1. **Extraction** — a *separate* model call with its own prompt and a strict JSON
   schema reads the transcript and proposes discrete durable statements. The
   prompt says plainly that most conversations contain nothing worth recording
   and an empty list is the correct, common answer. It extracts statements; it
   does not summarize.
2. **Thresholds** — proposals below `MIN_CONFIDENCE` (0.6) or `MIN_IMPORTANCE`
   (0.3) are discarded in `consolidate()`, in ordinary Python, regardless of what
   the extractor thought. The judgment is ECHO's, not the extractor's.

Malformed proposals are dropped rather than stored badly, and identical content
is not stored twice — an exact-match guard so re-running extraction doesn't pile
up copies. That guard is **not** belief updating: nothing is merged, reconciled,
superseded, or revised.

### The memory record

```python
Memory(
    id,                    # unique, generated
    content,               # one self-contained sentence
    memory_type,           # identity | preference | fact | goal | decision
    source_conversation,   # the conversation id it was extracted from
    confidence,            # 0.0–1.0 — how sure we are it's true
    importance,            # 0.0–1.0 — how much it should matter later
    created_at,
    last_accessed,         # None until first retrieved
    access_count,          # incremented on every retrieval
)
```

`confidence` and `importance` are separate on purpose: a throwaway remark can be
certainly true and not worth keeping. Retrieval genuinely mutates
`last_accessed` and `access_count` — those fields are used, not decorative.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # or see .env.example
```

## Use it as a library

```python
from echo import Agent, AnthropicLLM, Conversation, LLMMemoryExtractor

llm = AnthropicLLM()
agent = Agent(llm, Conversation(system="Be brief."),
              directory="echo_data", extractor=LLMMemoryExtractor(llm))

agent.send("Hi, I live in Dhaka and I prefer short answers.")
agent.remember()      # -> [Memory(...), Memory(...)]  — the deliberate step
agent.save()

# later, in a different process, in a different conversation
later = Agent(llm, Conversation(), directory="echo_data")
later.recall("where does the user live")   # loads hits into temporary context
later.send("Remind me where I live?")      # the model sees the memory
```

## Use it from the terminal

```bash
python -m echo                    # new conversation
python -m echo --resume <id>      # continue one
python -m echo --list             # saved conversation ids
```

In the loop: `/remember`, `/recall <text>`, `/memories`, `/context`, `/forget`,
`/help`, `/quit`.

## Architecture

```
cli.py ──> agent.py ──> llm.py           (the only file that knows a provider exists)
              │
              ├──> conversation.py       history   — durable, per-conversation
              ├──> storage.py              └─ <root>/conversations/<id>.json
              │
              ├──> extraction.py         the deliberate step
              │       └──> memory.py     memory    — durable, cross-conversation
              │            memory_store.py  └─ <root>/memories.json
              │
              └──> context.py            temporary — never written anywhere
```

Conversation history is resent in full on every turn — the API is stateless, and
that is the whole short-term memory model. There is no truncation or compaction
yet; long conversations will eventually hit the context window.

Transcripts live in their own subdirectory so listing conversations can never
pick up the memory store. (An earlier layout put both at the data root; `--list`
reported `memories` as a conversation and `--resume memories` would have tried to
parse the store as a transcript. The subdirectory removes the collision by
construction rather than by filtering filenames.)

## Model configuration

`AnthropicLLM` defaults to `claude-opus-5`, `max_tokens=16000`, thinking left at
the model's default. It implements two protocols: `LLMClient.complete()` for
conversational replies and `StructuredLLMClient.complete_structured()` for
schema-constrained JSON, which extraction uses.

It opts into server-side refusal fallbacks (`fallbacks="default"`) so a request
the safety classifiers decline is re-run on Anthropic's recommended fallback
model rather than coming back empty. That is the only non-obvious request
parameter; it is two entries in `_request()` and is safe to delete.

## Tests

```bash
python -m pytest
```

**97 tests, all offline** — no API key, no network.

### The four required proofs

`tests/test_memory_requirements.py` is written to be read as evidence:

| # | Requirement | Test |
| --- | --- | --- |
| 1 | A conversation can create a memory | `test_1_a_conversation_can_create_a_memory` |
| 2 | The memory survives restarting the program | `test_2_memory_survives_restarting_the_program` — every in-memory object is deleted; only files remain |
| 3 | The memory can be retrieved later | `test_3_the_memory_can_be_retrieved_later` — new process, new conversation, found by search |
| 4 | Ordinary conversation text does not automatically become permanent memory | `test_4_*` — four separate angles (see below) |

Requirement 4 is proved four ways, because it is the easy one to fake:

- **`test_4`** — three exchanges produce six transcript messages and zero
  memories; the extractor is never even called by `send()`.
- **`test_4b`** — running extraction over small talk returns nothing. The
  extractor ran and declined; running it is not the same as agreeing to remember.
- **`test_4c`** — an *eager* extractor proposing low-value candidates still yields
  nothing, because the threshold gate is ECHO's own code.
- **`test_4d`** — `remember()` refuses without an extractor, so there is no
  default path that quietly turns conversation into memory.

### What else is covered

| Area | What is checked |
| --- | --- |
| `conversation.py` | ordering, role validation, dict round trip, schema-version rejection, system prompt kept out of the `messages` payload |
| `storage.py` | save/load equality, JSON readability, directory creation, overwrite leaves no temp file, sorted listing, missing-id error, path-traversal rejection, no collision with the memory store |
| `memory.py` | every required field present, unique ids, enum coercion, score range validation, empty content rejected, `touch()` accounting, dict round trip |
| `memory_store.py` | missing file is empty not an error, save/load round trip, schema versioning, atomic write, exact-duplicate suppression, keyword search + ranking + limits, stopword handling, access tracking survives reload, listing does not count as access |
| `extraction.py` | candidate validation, separate call with its own prompt and schema, empty extraction is normal, malformed rows dropped, empty conversation not sent, confidence/importance gates, thresholds applied by ECHO, re-running does not duplicate |
| `context.py` | rendering, dedup, clearing, **no serialization methods exist**, saving writes no context, context does not survive a restart, the three kinds of state stay separate |
| `agent.py` | both turns recorded, full history resent, system prompt passed separately, rollback on failure, save → resume → continue across a simulated restart |
| `llm.py` | request shape, fallback opt-in, text-block joining, non-text blocks ignored, refusal raises, structured JSON parsed, non-JSON and wrong-type responses raise |

### What is *simulated* rather than genuinely implemented

Being precise about this, since it is the thing worth knowing:

- **The model, in tests.** `tests/fakes.py` defines `FakeLLM`,
  `FakeStructuredLLM`, `FakeExtractor`, and `ExplodingLLM`. Every agent and
  extraction test runs against these. They exist only in `tests/` — nothing in
  the `echo` package fakes a model response.
- **The Anthropic SDK, in `test_llm.py`.** `StubSDK` mimics the small slice of
  `anthropic.Anthropic` that `AnthropicLLM` touches and returns hand-built
  response objects. This genuinely exercises ECHO's request-building and
  response-parsing code, but it does **not** verify the real API accepts that
  request shape or returns that response shape.
- **Extraction *quality*, everywhere.** The tests prove the extraction
  *mechanism* — that it is a separate call, that its output is filtered, that
  nothing bypasses it. They cannot prove the prompt makes good judgments about
  what is worth remembering. That requires a live model and real conversations,
  and has not been measured.

Everything else is real: file I/O is real file I/O in a `tmp_path`, the
serialization is the same code the CLI uses, the threshold gate is real code, and
`AnthropicLLM` is the real code path against a live API.

### What is not covered

- No live API call in the test suite.
- The extraction prompt has never run against a real model. **This is the most
  likely place for the system to disappoint in practice** — the plumbing is
  tested, the judgment is not.
- `cli.py` has no automated test; it was smoke-tested by hand.
- Retrieval is keyword overlap. It will miss paraphrases (`"Where's home?"` won't
  match `"Sam lives in Dhaka."`). Adequate for proving the shape; a real system
  needs embeddings, which are explicitly out of scope for now.
- Consolidation re-reads the whole transcript each run; the exact-duplicate guard
  is what keeps that from producing copies. There is no per-message watermark.
- Concurrency: two processes writing the same store will have a last-writer-wins
  race. The atomic rename prevents corruption, not lost updates.

## Deliberately not built yet

**In the memory system specifically**, there is still no revision of any kind.
Memories are never revised, merged, contradicted, decayed, or re-scored after
creation. `confidence` and `importance` are set once at extraction and never
move. When two memories conflict, both simply sit in the store. The belief,
prediction and learning systems added later are separate modules with their own
records — none of them reaches back into `MemoryStore`.

## Where this goes next

The seams are deliberate. Adding a capability should mean touching one file:

- **Semantic retrieval** → `memory_store.search()` keeps its signature.
- **Belief updating / contradiction handling** → a new module over `MemoryStore`.
- **Memory decay or re-scoring** → uses `last_accessed` / `access_count`, already
  recorded.
- **Automatic consolidation** → call `remember()` on a trigger; the gate stays.
- **A database instead of JSON** → `MemoryStore` keeps its method names.
- **Streaming, tool use, a different provider** → `llm.py`.
