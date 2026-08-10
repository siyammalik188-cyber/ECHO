# ECHO

A minimal experimental conversational agent. It does five things and nothing else:

1. sends a conversation to an LLM,
2. receives the response,
3. maintains the current conversation in memory,
4. saves the conversation to disk,
5. reloads it later and keeps going.

No chatbot UI, no autonomous loop, no vector DB, no RAG, no web search, no
self-modification, no multi-agent orchestration. This is a foundation to extend
one capability at a time.

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # or see .env.example
```

## Use it as a library

```python
from echo import Agent, AnthropicLLM, Conversation

agent = Agent(AnthropicLLM(), Conversation(system="Be brief."))
print(agent.send("Hi, I'm Sam."))
agent.save()                             # -> conversations/<id>.json

# later, in a different process
resumed = Agent.resume(agent.id, AnthropicLLM())
print(resumed.send("What's my name?"))   # the model sees the earlier turns
```

## Use it from the terminal

```bash
python -m echo                           # new conversation
python -m echo --system "Be brief."      # with a system prompt
python -m echo --list                    # saved conversation ids
python -m echo --resume <id>             # continue one
```

## Architecture

Four modules, each with one job, wired in one direction:

```
cli.py  ──>  agent.py  ──>  llm.py          (talks to the model)
                 │
                 ├──────>  conversation.py  (in-memory state)
                 └──────>  storage.py       (JSON files on disk)
```

- **`conversation.py`** — `Message` and `Conversation`. Pure state. Never touches
  the network or the disk. Serializes to a dict with a `schema_version`.
- **`storage.py`** — `save` / `load` / `list_ids` / `exists`. One JSON file per
  conversation, named by id. Writes go to a temp file and are renamed, so an
  interrupted save can't leave a half-written file. Ids are validated against
  `[A-Za-z0-9_-]+` so they can't escape the storage directory.
- **`llm.py`** — the only file that knows a provider exists. `LLMClient` is a
  one-method protocol (`complete(messages, system) -> str`); `AnthropicLLM`
  implements it against the Claude Messages API.
- **`agent.py`** — `send()` is one round trip: append the user turn, call the
  model with the full history, append the reply. If the call fails, the user
  turn is rolled back so the conversation never holds an unanswered question.

Conversation history is resent in full on every turn — the API is stateless, and
that is the whole memory model for now. There is no truncation or compaction
yet; long conversations will eventually hit the context window.

## Model configuration

`AnthropicLLM` defaults to `claude-opus-5` with `max_tokens=16000` and thinking
left at the model's default. It opts into server-side refusal fallbacks
(`fallbacks="default"`), so a request the safety classifiers decline is re-run on
Anthropic's recommended fallback model inside the same call instead of coming
back empty. That is the only non-obvious request parameter; it is three lines at
the top of `complete()` and is safe to delete.

## Tests

```bash
python -m pytest
```

29 tests, all offline — no API key needed, no network calls.

### What the tests cover

| Area | What is checked |
| --- | --- |
| `conversation.py` | ordering, role validation, dict round trip, schema-version rejection, system prompt kept out of the `messages` payload |
| `storage.py` | save/load equality, JSON readability, directory creation, overwrite leaves no temp file, sorted listing, missing-id error, path-traversal rejection |
| `agent.py` | both turns recorded, full history resent each turn, system prompt passed separately, rollback on failure, save → resume → continue across a simulated restart |
| `llm.py` | request shape (model, messages, system, fallback opt-in), text-block joining, non-text blocks ignored, refusal raises, empty response raises |

### What is *simulated* rather than genuinely implemented

Being precise about this, since it's the thing worth knowing:

- **The model itself, in the tests.** `tests/fakes.py` defines `FakeLLM`, which
  returns canned strings, and `ExplodingLLM`, which always raises. Every
  `agent.py` test runs against these. They exist only in `tests/` — nothing in
  the `echo` package fakes a model response.
- **The Anthropic SDK, in `test_llm.py`.** `StubSDK` mimics the small slice of
  `anthropic.Anthropic` that `AnthropicLLM` touches, and returns hand-built
  response objects. This genuinely exercises ECHO's request-building and
  response-parsing code, but it does **not** verify that the real API accepts
  that request shape or returns that response shape. The first real call is the
  first time that's tested.

Everything else is real: the file I/O is real file I/O in a `tmp_path`, the
serialization is the same code the CLI uses, and `AnthropicLLM.complete()` is
the real code path against a live API when you run it.

### What is not covered

- No live API call is made in the test suite. Sending a real request requires a
  key and a network, and is not part of `pytest`.
- The CLI loop (`cli.py`) has no automated test; it was smoke-tested by hand.
- Concurrency: two processes writing the same conversation id will have a
  last-writer-wins race. The atomic rename prevents corruption, not lost updates.

## Where this goes next

The seams are deliberate. Adding a capability should mean touching one file:

- **Streaming** → `llm.py` (add a `stream()` method to the protocol).
- **A different provider** → a second class in `llm.py`.
- **Tool use** → `agent.py` grows a loop around `send()`; `Message` grows a
  content-block type.
- **A database instead of JSON** → `storage.py` keeps its four function names.
- **Summarization / context management** → a function over `Conversation`.
