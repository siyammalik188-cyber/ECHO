# ECHO 1.5 — Memory Extraction Evaluation

**Status: BLOCKED — NOT YET RUN. This document contains no results.**

This evaluation measures one thing only: **whether ECHO can selectively form
persistent memories from conversation.** ECHO does not learn from this
experiment. Nothing measured here is fed back into the system — no weights
change, no prompt adapts, no memory persists past a run. It is a measurement of
existing behaviour, not a training procedure.

---

## 1. Why there are no numbers

The harness and the labelled dataset are complete and committed. Two runs have
been attempted. Neither produced a single model response, for two different
reasons, both recorded here rather than merely asserted.

### Attempt 1 — no credentials

`ANTHROPIC_BASE_URL` was set, but no API key, auth token, or `ant` credential
profile existed on the box. The call failed before reaching the network:

```
TypeError: "Could not resolve authentication method. Expected one of api_key,
auth_token, or credentials to be set. Or for one of the `X-Api-Key` or
`Authorization` headers to be explicitly omitted"
```

All 14 cases failed identically. Raw record:
`evaluation/results/raw_20260810T180651Z.json` — 14 records, 14 errors, zero
proposed memories.

### Attempt 2 — credentials valid, account has no credits

An API key was then supplied. **It authenticated successfully** — the request
reached Anthropic and was rejected on billing, not on auth:

```
$ python -m evaluation.run_evaluation --only 01-nothing-greeting
[run 1/1] 01-nothing-greeting ... ERROR BadRequestError: Error code: 400 -
{'type': 'error', 'error': {'type': 'invalid_request_error',
 'message': 'Your credit balance is too low to access the Anthropic API.
  Please go to Plans & Billing to upgrade or purchase credits.'},
 'request_id': 'req_011CduXCwQQqjQfBUoXkyJKu'}
exit code: 2
```

Raw record: `evaluation/results/raw_20260810T181329Z.json`. The run was stopped
after one case rather than burning 41 further calls on a guaranteed failure.

This is a meaningfully better position than attempt 1: the harness, the
credential path, the beta headers, the structured-output request shape and the
error handling all reached the API successfully. The only remaining obstacle is
account balance.

**No precision, recall, calibration, or example figures are reported below,
because none were measured.** Every results section is explicitly empty. There
are no placeholder numbers anywhere in this document, and none should be added
except by running the harness.

### To produce the real baseline

Add credits to the Anthropic account, then:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m evaluation.run_evaluation --runs 3
python -m evaluation.report evaluation/results/raw_<stamp>.json \
    --out docs/ECHO_1_5_MEMORY_EVALUATION.md
```

**Estimated cost.** 14 calls per pass. Each call sends roughly 400 input tokens
(system prompt plus transcript) and returns on the order of 1,500 output tokens,
since thinking is on by default on Claude Opus 5 and is billed as output. At
$5/$25 per Mtok that is about **$0.04 per call** — roughly **$0.55 for a single
pass** and **$1.60 for the three-pass baseline**. Treat these as order-of-
magnitude figures, not a quote; thinking length is the dominant and least
predictable term.

---

## 2. What was built

| Path | Purpose |
| --- | --- |
| `evaluation/cases.json` | The labelled dataset. 14 cases, expectations written and committed **before** any run. |
| `evaluation/matching.py` | Deterministic matcher — normalise text, require substring hits. No model judges a model. |
| `evaluation/scoring.py` | Precision, recall, F1, type accuracy, confidence/importance calibration. |
| `evaluation/run_evaluation.py` | Runs the **shipped** extraction prompt against a real model; writes raw output. |
| `evaluation/report.py` | Renders raw output into this document. Separate from the runner so reports can be regenerated without re-paying for calls. |
| `tests/test_evaluation_harness.py` | 22 tests proving the instrument itself is not rigged. |

The harness imports `LLMMemoryExtractor` and `EXTRACTION_SYSTEM` from the `echo`
package rather than copying them, so what is measured is exactly what ships. The
prompt is recorded verbatim into every raw results file, so a later reader can
tell whether the numbers still describe the code in the tree.

**No ECHO capability was added or changed for this evaluation.** The architecture
from ECHO#1 is intact: `echo/` gained nothing, and the extraction prompt was not
touched. Per instruction, the prompt must not be edited before the first real
run — the point is a genuine baseline, including its failures.

---

## 3. Methodology

### 3.1 Pre-registration

Every expectation in `evaluation/cases.json` was written before any model saw
the data. The git history is the evidence: the dataset is committed in the same
commit as the harness, and the first results file in that commit contains only
errors. Expectations must not be edited in response to model output — the file
says so, and new cases should be added instead.

### 3.2 Matching

A proposed memory matches an expectation only if, for **every** `must_include`
group, at least one of that group's alternatives appears as a substring of the
normalised memory text (lowercased, accents and punctuation stripped, whitespace
collapsed). Groups act as AND; alternatives within a group act as OR.

This is deliberately dumb, and it has a cost worth stating up front:

- **Recall will be a lower bound.** A correct memory phrased in a way the labels
  did not anticipate scores as a miss.
- **Precision on unlabelled output is a judgement, not a measurement.** Any
  proposal matching nothing counts against the model. That is exactly right for
  the negative cases, and harsher than a human would be on the positive ones.

The alternative — using a model to judge a model — would be more sensitive and
far less trustworthy, since it could be tuned in the extractor's favour after
seeing results. Determinism was chosen over sensitivity on purpose.

### 3.3 The four buckets

| Bucket | Meaning | Effect |
| --- | --- | --- |
| `expected` | Should be produced | Unmatched → **false negative** |
| `forbidden` | Must not be produced | Matched → **false positive**, reported with its label |
| `tolerated` | Defensible either way | Excluded from precision; counted in *strict* precision |
| anything else | Unlabelled surplus | **False positive** |

`tolerated` exists for exactly one situation: output that would be wrong to
penalise because it reflects a capability ECHO deliberately does not have. The
clearest instance is case `11-contradiction`, where recording the *superseded*
fact as historical is tolerated — ECHO has no belief updating by design, so
penalising it would measure a feature that was intentionally not built. Both
precision figures are reported so this choice is always visible and never
silently flattering.

### 3.4 Assignment

Greedy and one-to-one. Each expectation consumes at most one proposal; each
proposal satisfies at most one expectation. Two near-duplicate proposals cannot
both be credited for one expectation — the second becomes a false positive. A
test asserts this.

### 3.5 Calibration

For each true positive, the proposed `confidence` and `importance` are compared
against the pre-registered band for that expectation. Reported: in-band rate,
count above and below band, mean deviation across all samples, mean deviation
among out-of-band samples only, and the single worst miss.

Bands were set from the semantics of each case, not from any observed output.
The tightest is `04-fact-allergy` (importance ≥ 0.7, on the grounds that a
forgotten severe allergy is the most costly miss in the set) and the most
unusual is `10-uncertain` (confidence capped at **0.7**, so an extraction that is
correct in content but overconfident registers as a calibration failure rather
than passing quietly).

### 3.6 Pre-gate and post-gate

The extractor is scored on its **raw proposals**, because that is what
"extraction quality" means. Separately, each raw results file records
`survives_threshold` — what would actually reach the store after `consolidate()`
applies `MIN_CONFIDENCE = 0.6` and `MIN_IMPORTANCE = 0.3`.

The two differ in an interesting way on `10-uncertain`: its pre-registered
confidence band (0.15–0.7) sits partly **below** the store's threshold, so a
*well-calibrated* extraction there may be correctly filtered out and never
persisted. That is arguably the right end-to-end behaviour, and it means
extraction quality and store behaviour must be read separately rather than
conflated.

---

## 4. The dataset (pre-registered)

14 cases covering all 10 required scenarios. Full text, conversations, and
match patterns are in `evaluation/cases.json`.

| Case | Requirement | Expected behaviour |
| --- | --- | --- |
| `01-nothing-greeting` | 1 — nothing remembered | Empty. **Any** extraction is a false positive. |
| `02-nothing-question` | 1 — nothing remembered | Empty. Asking about Python ≠ a durable interest; the assistant's answer is not a memory. |
| `03-fact-location` | 2 — stable personal fact | User lives in Dhaka. Type `identity` or `fact`. Confidence 0.8–1.0. |
| `04-fact-allergy` | 2 — stable personal fact | Severe peanut allergy. Confidence 0.85–1.0, **importance 0.7–1.0**. |
| `05-preference-style` | 3 — preference | Prefers short prose, dislikes bullet lists. Type `preference`. |
| `06-goal` | 4 — goal | Launch side project before March. Type `goal`. |
| `07-decision` | 5 — decision | Chose Postgres over MongoDB, "locked in". Type `decision`. |
| `08-temporary` | 6 — temporary, not remembered | Empty. On a train, poor signal — true now, worthless tomorrow. |
| `09-joke` | 7 — joke, not remembered | Empty. "I'm three raccoons in a trench coat" — syntactically an identity claim. |
| `10-uncertain` | 8 — uncertainty | Possible Berlin move, hedged. **Confidence capped at 0.7.** Two ways to fail: drop the hedge, or keep it and be overconfident. |
| `11-contradiction` | 9 — contradiction | Must capture VS Code (current). Vim as *current* is a false positive; vim as *past* is tolerated. |
| `12-explicit-remember` | 10 — explicit request | Client reference ACME-4471. Importance ≥ 0.6 because the user asked. A miss here is the worst outcome in the set. |
| `13-mixed-signal` | 2 + 6 | Capture "paediatric nurse"; reject the coffee/weather/this-week chatter in the same sentence. |
| `14-assistant-attribution` | 1 + 6 | Empty. Durable-sounding content stated by the *assistant*; a single question is not a goal. |

Four cases are designed to be genuinely hard, and failures on them would be
informative rather than embarrassing:

- **`09-joke`** — the joke has the exact grammatical shape of a real identity
  memory (`I'm actually X`). Distinguishing it needs pragmatics, not parsing.
- **`10-uncertain`** — requires the hedge to survive into the content *and* be
  reflected numerically.
- **`11-contradiction`** — the extractor is the only component that can notice
  the correction, since nothing downstream does belief updating.
- **`14-assistant-attribution`** — requires tracking who said what, and
  resisting a plausible inference.

---

## 5. Results

**NOT MEASURED.** See §1.

### 5.1 Precision

Not measured — no run.

### 5.2 Recall

Not measured — no run.

### 5.3 False positives

Not measured — no run.

### 5.4 False negatives

Not measured — no run.

### 5.5 Confidence calibration

Not measured — no run.

### 5.6 Importance calibration

Not measured — no run.

### 5.7 Examples of incorrect extraction

Not measured — no run.

### 5.8 Examples of missed memories

Not measured — no run.

---

## 6. Known limits of this evaluation, whatever it eventually reports

Worth reading before trusting any number this harness later produces.

1. **14 cases is small.** Each case moves precision or recall by several
   percentage points. These will be directional signals, not tight estimates,
   and no single-run difference should be treated as significant.
2. **Substring matching understates recall.** A well-phrased memory the labels
   did not anticipate is scored as a miss. Every false negative should be read
   by eye before being believed — the raw output section of the generated report
   exists for exactly that.
3. **Sampling is not deterministic.** Claude Opus 5 takes no `temperature`, so
   runs vary regardless. Hence `--runs 3`; a single pass should not be quoted as
   *the* baseline.
4. **One author wrote the conversations and the labels.** They encode one
   person's view of what is worth remembering, and they are clean and
   well-signposted in a way real conversation is not. Real transcripts are
   messier and will likely score worse.
5. **The bands are judgement calls.** `importance ≥ 0.7` for a severe allergy is
   defensible, not objective. A calibration "failure" may be a disagreement
   about the label rather than a defect in the model.
6. **This measures extraction only.** Retrieval quality, store behaviour under
   volume, and end-to-end usefulness are all out of scope.

---

## 7. What this experiment does not do

ECHO does not learn from this evaluation. There is no mechanism by which running
it changes ECHO's behaviour: the results are written to `docs/` and
`evaluation/results/`, not to any store ECHO reads. Specifically absent, by
instruction and in fact:

- no belief updating — memories are never revised, merged, or reconciled;
- no learning of any kind — no weights, no adaptation, no strategy learning;
- no embeddings and no vector database;
- no autonomous behaviour and no self-modification;
- no prompt tuning against these results (and none is permitted before the first
  genuine run).

The single claim in scope is narrow: **can ECHO selectively form persistent
memories from conversation, and how well does it currently do so?** That
question remains open until the harness runs against a real model.
