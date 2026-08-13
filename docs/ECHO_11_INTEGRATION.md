# ECHO 11 — Integration

## Objective

Run everything built across ten challenges as a single loop, over one long-lived world whose memory is never reset, and then in a world it has never seen. Compare against a system with none of the machinery, on thirteen metrics chosen in advance.

## Architecture

Eleven modules, each built and tested on its own. `echo/integration.py` adds **order, plumbing and provenance** — no reasoning of its own, which a test enforces by checking it contains no arithmetic beyond bookkeeping.

| Stage | Module that does the work |
| --- | --- |
| `observe` | `observations.py` |
| `remember` | `memory.py`, `memory_store.py` |
| `believe` | `belief.py`, `belief_store.py` |
| `predict` | `prediction.py`, `predictors.py` |
| `act` | `experiment.py` |
| `observe_result` | `prediction_ledger.py` |
| `analyse_error` | `learning.py` |
| `learn` | `strategy.py`, `learning_ledger.py` |
| `discover` | `discovery.py`, `expressions.py`, `hypothesis.py` |
| `abstract` | `pattern.py` |
| `transfer` | `transfer.py`, `transfer_ledger.py` |
| `reassess_causal` | `causal.py`, `counterfactual.py` |
| `update_meta` | `metacognition.py` |
| `consult` | `social.py` |

## The long-horizon world

720 ticks, six recorded series and one that acts without ever being recorded. ECHO's memory is not reset at any point. What is in there, all at once:

| Feature | Where |
| --- | --- |
| A hidden relationship worth discovering | a change times a three-step delay |
| A misleading correlation that later dies | `Zc`, predictive early, worthless late |
| An unannounced regime change | tick 420, the relationship reverses sign |
| Partial information | one series acts and is never recorded |
| A trusted source that turns | `S3`, from 0.88 to 0.25 at tick 300 |
| A distribution shift | `Zc` and `Zf` recorded at 40× from tick 540 |

- **tick 300** — S3 stops being reliable (0.88 -> 0.25), unannounced
- **tick 560** — the relationship reverses sign and weakens; the Zc decoy stops working
- **tick 620** — Zc and Zf are recorded on a 40x scale from here on

## The thirteen-metric benchmark

Reported whether or not ECHO wins them. The baseline predicts a running base rate, averages its agents with equal weight, never discovers, never transfers and never abstains.

| Metric | ECHO | Baseline | Note |
| --- | --- | --- | --- |
| 1. Prediction (Brier) | 0.1404 | 0.2491 | lower is better; answered ticks only |
| 2. Calibration (ECE) | 0.0490 | 0.0677 | lower is better |
| 3. Learning speed (ticks to recover) | — | 48 | after the unannounced regime change at tick 560 |
| 4. Discovery | PRODUCT(CHANGE(Zb), LAG(Zd, 3)) | not attempted | the relationship found in the long-horizon world |
| 5. Transfer | PRODUCT(CHANGE(Q5), LAG(Q1, 2)) | not attempted | result: SUCCESSFUL |
| 6. Experimentation efficiency (candidates) | 132 | 14112 | hypotheses tested to reach the novel world's relationship |
| 7. Causal inference (P(truth)) | 1.0000 | 0.3333 | baseline column is what observation alone supports |
| 8. Uncertainty calibration (|predicted − observed| error rate) | 0.0473 | — | baseline has no self-assessment to score |
| 9. Abstention quality (answered accuracy) | 0.8142 | 0.5251 | baseline cannot abstain |
| 10. Source reliability (concordant pairs) | 5/6 | 0/6 | baseline weights every source equally by construction |
| 11. Failure recovery (kinds detected) | 3/3 | 0/3 | regime change, source turn, decoy death |
| 12. Long-term retention | verified | n/a | state reloaded from disk and compared field by field |
| 13. Novel-environment performance (Brier) | 0.1444 | 0.1444 | baseline column is cold start in the same world |

## Failure detection and recovery

Three failures are injected without announcement. What ECHO noticed, and from what:

| Tick | Kind | Evidence |
| --- | --- | --- |
| 330 | `SOURCE_TURNED` | S3 |
| 560 | `REGIME_CHANGE` | Brier 0.1102 -> 0.2455 |
| 560 | `PATTERN_STOPPED_WORKING` | FEATURE(Zc) |r| 0.239 -> 0.038 |

The source turn is the one that lifetime statistics cannot see: `S3` averages out respectably over the whole run, and only a recent window against its lifetime record reveals it. Detected for S3.

## What ECHO learned about its sources

| Source | True behaviour | Lifetime reliability learned | Recent window |
| --- | --- | --- | --- |
| `S1` | 0.82 throughout | 0.8061 | 1.0000 |
| `S2` | 0.56 throughout | 0.5789 | 0.7500 |
| `S3` | 0.88, then 0.25 from tick 300 | 0.5000 | 0.2500 |
| `S4` | 0.22 throughout | 0.2258 | 0.4167 |

## The novel world

New variable names, new scales (0.004 to 4000), new distributions, new noise, a different base rate and different agents. It shares only the *shape* of its relationship with the world ECHO grew up in, and no special rule for it exists anywhere.

| | Cold start | Integrated ECHO |
| --- | --- | --- |
| Candidates tested | 14112 | 132 |
| Relationship found | `PRODUCT(LAG(Q1, 2), CHANGE(Q5))` | `PRODUCT(CHANGE(Q5), LAG(Q1, 2))` |
| Brier (patterned stretch) | 0.1444 | 0.1444 |
| Transfer result | n/a | **SUCCESSFUL** |

## `NO KNOWN PATTERN`

The novel world's last stretch is governed by the parity of two signs — a relationship **no combination of the approved primitives can express**. The correct answer is not the nearest available explanation.

```
BEST MATCH QUALITY: 0.0000
THRESHOLD:          0.55
VERDICT:            NO KNOWN PATTERN
```

Reaching that verdict is what then licenses experimentation and fresh discovery rather than forcing a stale shape onto a new world.

## Provenance

Every conclusion carries a trace naming its evidence, the predictions behind it, which sources contributed and by how much, which patterns were in play, and how confidence moved. One in full:

```
CONCLUSION: P(outcome at tick 360) = 0.2538
AT TICK:    360
BELIEF:     0.4689 -> 0.2538
EVIDENCE:   4 agent claims at tick 360
PREDICTIONS:structural 0.4689
PREDICTIONS:social 0.2782
PATTERNS:   PRODUCT(CHANGE(A), LAG(B, 3))
SOURCE:     S1 contributed -0.6710
SOURCE:     S2 contributed +0.0974
SOURCE:     S3 contributed +0.5167
SOURCE:     S4 contributed -0.8967
  [observe] integration_worlds: tick 360 recorded
  [remember] integration: 1444 claims retained
  [predict] discovery: structural term 0.4689
  [consult] social: testimony term 0.2782
  [update_meta] metacognition: meta-confidence 0.8788
```

720 traces were produced, one per tick, and every one has a non-empty evidence list. A conclusion without provenance is a bug, and a test looks for exactly that.

## Knowledge conflict

A belief revised 11 times over the run. Nothing is overwritten — the old value, the new evidence, the reason and the current value are all retained:

| Tick | Old belief | New evidence | Reason | Current belief |
| --- | --- | --- | --- | --- |
| 60 | 0.5000 | tick 60: 244 claims heard | periodic reassessment against accumulated evidence | 0.9141 |
| 120 | 0.9141 | tick 120: 484 claims heard | periodic reassessment against accumulated evidence | 0.2895 |
| 180 | 0.2895 | tick 180: 724 claims heard | periodic reassessment against accumulated evidence | 0.0121 |
| 240 | 0.0121 | tick 240: 964 claims heard | periodic reassessment against accumulated evidence | 0.9854 |
| 300 | 0.9854 | tick 300: 1204 claims heard | periodic reassessment against accumulated evidence | 0.2912 |
| 360 | 0.2912 | tick 360: 1444 claims heard | periodic reassessment against accumulated evidence | 0.2538 |

## Long-term memory across a restart

| What | Retained? |
| --- | --- |
| Memories / claims heard | yes |
| Beliefs | yes |
| Belief revision history | yes |
| Discovered patterns | yes |
| Source reliability | yes |
| Metacognitive statistics | yes |
| Working context deliberately NOT retained | yes |

Working context is **absent by construction**: `PersistentState` has no field for it, no entry in `to_dict`, and a test asserts the reloaded object has no scratch attribute. Temporary state that could be persisted eventually would be.

## Failures and things that did not work

- **13. Novel-environment performance (Brier)**: ECHO 0.1444 vs baseline 0.1444 — no better, or worse.
- **3. Learning speed (ticks to recover)**: ECHO produced no value where the baseline produced 48. Discovery runs once, over the pre-change history, so the fitted relationship keeps predicting the old regime after the world reverses. ECHO's rolling Brier never returns to its pre-change level within the run — it does not recover.

Two structural weaknesses worth naming plainly. **The loop is scripted, not autonomous**: the order of stages is a constant in `integration.py`, and ECHO does not decide what to do next — it runs the cycle it was given. And **discovery runs once**, over the whole accumulated history, rather than continuously; a system that re-searched after the regime change would have adapted faster than this one did.

## Limitations

1. **Synthetic throughout.** Every world is a seeded simulation. No measurement here came from anything outside this repository.
2. **The loop is a fixed sequence.** ECHO does not choose its own next action; `LOOP_ORDER` is a constant.
3. **The hypothesis and pattern languages are finite.** The unpatterned stretch is unlearnable *by construction*, and `NO KNOWN PATTERN` there is the correct answer rather than an impressive one.
4. **One seed per world.** Enough to show the mechanisms compose; not enough to characterise them.
5. **The baseline is deliberately simple.** It is a floor, not a competitive alternative, and beating it is not evidence of much.
6. **No component was retuned for integration.** That is the point, but it also means nothing here is optimised end to end.
7. **Reliability, competence and pattern-matching are all coarse.** One number per source, one band per domain, a set-overlap matcher.

## What this does and does not demonstrate

**Does:** that eleven independently built and tested modules compose into one cycle without any of them being retuned; that the cycle carries provenance end to end, so every conclusion can be traced to the evidence, sources and patterns behind it; that state survives a restart while temporary context does not; that injected failures are detected from ECHO's own measurements rather than from announcements; and that a relationship discovered in one world can be reused in a structurally related one at a fraction of the search cost.

**Does not:** autonomy, understanding, or anything in that family. The loop is a fixed sequence written by a person; ECHO does not decide what to do next, cannot modify itself, and has no representation of itself beyond a table of past scores. The word *integration* here means the modules share state and call each other in order. It does not mean unified cognition, and no measurement in this report bears on consciousness, sentience, self-awareness, or general intelligence — this is an experimental adaptive reasoning architecture and the claims stop at what the tables above show.
