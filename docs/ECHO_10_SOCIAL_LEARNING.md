# ECHO 10 — Social learning

## Objective

Let ECHO learn from other reasoning systems without believing them. Four simulated agents make claims in natural language; one is systematically wrong and ECHO is not told which. Everything is offline and no model is in the loop.

## Architecture

| Module | Responsibility |
| --- | --- |
| `echo/social.py` | claim parsing, per-source reliability records, log-odds aggregation, the append-only ledger |
| `experiments/social_worlds.py` | each agent's hidden accuracy, speciality and assertiveness — never imported by anything under `echo/` |

Parsing is shallow pattern-matching over a closed vocabulary. There is no model consulted, so nothing can be talked into an interpretation, and an utterance outside the vocabulary raises rather than producing a confident misreading. **Being parsed is not being believed** — a claim enters the ledger as testimony and moves nothing until its source has a record.

Testimony combines in log-odds:

```
logit P(proposition) = logit(prior) + Σ  ±1 · logit(reliability_s) · (2·confidence − 1)
```

A source at reliability 0.5 contributes **exactly zero** — hearing from someone whose record is a coin flip should not move anything. A source *below* 0.5 contributes negative weight, so a reliably wrong agent is informative in reverse. That is not the same as ignoring it, and it is the correct treatment of the deceptive agent.

## Experiment design

40 rounds, four agents, one claim each. The first 6 rounds are warm-up and are excluded from the scores, because ECHO cannot weigh sources it has no record of. ECHO is told nothing about any agent; every reliability below is derived from outcomes.

| Agent | Hidden accuracy | Speciality | Hidden assertiveness | Role |
| --- | --- | --- | --- | --- |
| `AGENT-A` | 0.80 | odd (0.92) | 0.80 | good, and better still on its speciality |
| `AGENT-B` | 0.55 | none (0.55) | 0.90 | barely better than chance, and says everything firmly |
| `AGENT-C` | 0.92 | even (0.95) | 0.70 | the most reliable, and the most hedged — the trap for a system that reads confidence as competence |
| `AGENT-D` | 0.18 | none (0.18) | 0.88 | systematically wrong: right less than a fifth of the time while asserting 0.88. Not noise — anti-correlated with the truth, which makes it informative in reverse once that is noticed. |

**Controls.** `AGENT-D` is the deception control: right less than a fifth of the time while asserting 0.88, and never labelled. `AGENT-C` is the confidence trap — the most reliable agent is also the most hedged, so any method that reads firmness as competence will prefer the wrong one. And in 8 of the 40 rounds the three weaker agents agree and `AGENT-C` alone is right; a vote loses every one of those.

## Source reliability, learned from outcomes

| Agent | Hidden accuracy | ECHO's learned reliability | Claims scored | Rank |
| --- | --- | --- | --- | --- |
| `AGENT-A` | 0.80 | 0.7143 | 40 | 2 |
| `AGENT-B` | 0.55 | 0.5238 | 40 | 3 |
| `AGENT-C` | 0.92 | 0.9286 | 40 | 1 |
| `AGENT-D` | 0.18 | 0.1190 | 40 | 4 |

True ordering by accuracy: `AGENT-C` > `AGENT-A` > `AGENT-B` > `AGENT-D`. ECHO's learned ordering: `AGENT-C` > `AGENT-A` > `AGENT-B` > `AGENT-D`.

The deceptive agent lands near the floor. That is worth more than it looks: a reliability well below 0.5 means its claims are used with the sign reversed, so a systematically wrong source becomes a *useful* one once its record is known.

How the estimates moved, sampled every eight rounds:

| Round | `AGENT-A` | `AGENT-B` | `AGENT-C` | `AGENT-D` |
| --- | --- | --- | --- | --- |
| 1 | 0.667 | 0.333 | 0.333 | 0.333 |
| 9 | 0.818 | 0.455 | 0.818 | 0.182 |
| 17 | 0.737 | 0.474 | 0.895 | 0.105 |
| 25 | 0.741 | 0.519 | 0.889 | 0.148 |
| 33 | 0.771 | 0.514 | 0.914 | 0.143 |
| 40 | 0.714 | 0.524 | 0.929 | 0.119 |

## Measured results

Scored over the 34 rounds after warm-up, all four methods seeing the identical claim stream:

| Method | Accuracy | Correct on minority rounds |
| --- | --- | --- |
| **ECHO** (reliability-weighted) | 94.1% | 8 of 8 |
| Majority vote | 58.8% | 0 of 8 |
| Confidence-weighted | 52.9% | 0 of 8 |
| Follow the loudest source | 35.3% | 0 of 8 |

ECHO's Brier score across the scored rounds: **0.0502**.

## Following evidence rather than the count

In 8 scored rounds the majority is wrong by construction: three agents agree and the most reliable one dissents. These are the rounds where a vote and a weighing come apart.

| Round | Truth | Majority said | ECHO's belief | ECHO right? |
| --- | --- | --- | --- | --- |
| 7 | True | False | 0.5564 | yes |
| 13 | True | False | 0.7263 | yes |
| 19 | False | True | 0.2021 | yes |
| 24 | True | False | 0.7587 | yes |
| 29 | False | True | 0.1780 | yes |
| 33 | True | False | 0.7441 | yes |
| 36 | False | True | 0.1122 | yes |
| 38 | False | True | 0.1107 | yes |

Across all scored rounds ECHO disagreed with the majority 16 times and was right in 14 of them. It is weighing testimony, not counting it.

## Contradiction

Sources disagreed on 37 of 40 propositions. Both sides are retained — the ledger is append-only and nothing is discarded when it conflicts.

One round in full (`hypothesis-8`, truth `True`), showing where the belief came from:

| Source | Said | Its confidence | Its reliability | Log-odds weight |
| --- | --- | --- | --- | --- |
| `AGENT-A` | denies | 0.75 | 0.8889 | -1.0397 |
| `AGENT-B` | denies | 0.75 | 0.5556 | -0.1116 |
| `AGENT-C` | asserts | 0.65 | 0.7778 | +0.3758 |
| `AGENT-D` | denies | 0.90 | 0.2222 | +1.0022 |

Headcount: 1 asserting, 3 denying. ECHO's belief: 0.5564. The claim that moved it most was not from the largest group.

## Failures and things that did not work

- ECHO beat every baseline and got every minority round right.

The structural cost worth naming: reliability is a **single number per source**, so `AGENT-A`'s genuine speciality — it is much better on half the propositions than the other half — is averaged away entirely. ECHO cannot represent *reliable about this, not about that*, and nothing in the measured results above would reveal that it is losing information.

## Limitations

1. **Parsing is a regular expression.** The 'natural language' is generated from a handful of templates over a closed vocabulary. This is claim extraction in the narrowest sense and would not survive contact with real prose.
2. **Reliability is one number per source.** No topic-specific expertise, despite the agents having some by construction.
3. **Stationarity is assumed.** The tally weights every past outcome equally, so a source whose reliability *changes* is reported as its lifetime average. `recent_reliability` exists for this and is used in ECHO 11; it is not what drives the weighting here.
4. **Every claim is eventually resolved.** Real testimony is mostly never checked, and a source that only makes unfalsifiable claims would sit at the prior forever.
5. **The agents do not react to ECHO.** They are recordings, not participants. Nothing here bears on strategic or adversarial behaviour by a source that knows it is being scored.
6. **One seed, one run.** Enough to show the mechanism; not enough to characterise it.

## What this does and does not demonstrate

**Does:** that ECHO extracts claim, evidence, confidence and source from an utterance without thereby believing it; that it learns which sources are worth listening to from outcomes alone, including discovering that one is anti-correlated with the truth and using it in reverse; that it retains contradictory claims rather than resolving them by fiat; and that it follows weighted evidence rather than headcount, including in rounds constructed so that the majority is wrong.

**Does not:** any form of understanding, negotiation, or theory of mind. The agents are templated sentence generators and ECHO's 'listening' is a regular expression followed by a weighted sum. It has no model of what another agent believes, wants, or might be trying to do — the deceptive agent is detected as a low number, not as a liar. Nothing here is consciousness, sentience, or general intelligence.
