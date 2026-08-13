# ECHO 9 — Metacognition

## Objective

Give ECHO a measurable account of how reliable its own answers have been, per kind of question, and test whether that account is any good: whether it can predict its own error rate, tell a strong area from a weak one without being told, and decline to answer where declining is correct.

**Nothing here is introspection and nothing simulates it.** There is no method that returns *I feel uncertain*. Every quantity below is a count, a frequency, or a proper scoring rule over recorded outcomes, and each traces back to the claims that produced it.

## Architecture

| Module | Responsibility |
| --- | --- |
| `echo/metacognition.py` | immutable `Claim` records, per-domain tallies, competence bands, abstention, error-prediction calibration |
| `experiments/metacognition_worlds.py` | the hidden skill and assertiveness of each domain — never imported by anything under `echo/` |

Two quantities are kept strictly apart:

| | Question it answers | Evidence behind it |
| --- | --- | --- |
| **belief** | how likely is this proposition? | the evidence for that proposition |
| **meta-confidence** | how often have claims *like this* been right? | a tally of past claims and their outcomes |

Meta-confidence is Laplace-smoothed accuracy, so one lucky answer is not a track record. Below 8 scored claims a domain is `UNTESTED` — a third state, distinct from weak. Collapsing *known to be bad* into *no idea* would throw away the difference.

## Experiment design

Five domains, 60 calibration tasks and 60 evaluation tasks each. Every domain has a hidden **skill** (how often ECHO is right) and a hidden **assertiveness** (how strongly it says so), set independently. ECHO is told neither.

| Domain | Hidden skill | Hidden assertiveness | Overconfident by design? |
| --- | --- | --- | --- |
| `prediction` | 0.86 | 0.84 | no |
| `discovery` | 0.74 | 0.78 | no |
| `transfer` | 0.66 | 0.88 | **yes** |
| `experimentation` | 0.92 | 0.88 | no |
| `causal_inference` | 0.53 | 0.85 | **yes** |

The interesting failure is not being bad, it is being bad *and* confident. `causal_inference` answers barely better than chance while asserting 0.85, and `transfer` delivers 0.66 while asserting 0.88. Those two are what the abstention mechanism exists for.

**Controls.** A strong-and-calibrated domain (`experimentation`) so the mechanism cannot pass by distrusting everything; a middling one (`discovery`) so the bands are not merely a two-way split; and the `UNTESTED` state, which is checked separately because a system that abstained on everything unmeasured could never build a record at all.

## Capability monitoring

After the calibration phase, derived from outcomes alone — no labels about which domain is easy reach ECHO:

| Domain | Answered | Accuracy | Brier | Meta-confidence | Band | Confidently wrong | Hidden skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prediction` | 120 | 85.8% | 0.1256 | 0.8525 | **STRONG** | 16 | 0.86 |
| `discovery` | 120 | 79.2% | 0.1661 | 0.7869 | **UNCERTAIN** | 8 | 0.74 |
| `transfer` | 120 | 68.3% | 0.2611 | 0.6803 | **WEAK** | 38 | 0.66 |
| `experimentation` | 120 | 94.2% | 0.0633 | 0.9344 | **STRONG** | 7 | 0.92 |
| `causal_inference` | 60 | 50.0% | 0.3700 | 0.5000 | **WEAK** | 23 | 0.53 |

Bands are fixed thresholds on the Brier score: `STRONG` at or below 0.15, `WEAK` at or above 0.25 — which is the score of always saying 0.5, so a `WEAK` domain is doing no better than shrugging — and `UNCERTAIN` between them.

## Belief and meta-confidence are different numbers

| Domain | A typical belief | Meta-confidence | Gap |
| --- | --- | --- | --- |
| `prediction` | 0.8399 | 0.8525 | -0.0126 |
| `discovery` | 0.7784 | 0.7869 | -0.0085 |
| `transfer` | 0.8797 | 0.6803 | +0.1994 |
| `experimentation` | 0.8807 | 0.9344 | -0.0537 |
| `causal_inference` | 0.8493 | 0.5000 | +0.3493 |

The gap is the point. In `causal_inference` ECHO states its answers at around 0.85 and has been right about half the time; the belief and the warrant for it are not the same quantity, and only tracking both makes the difference visible.

## Error prediction

Before each outcome, ECHO states how likely it thinks it is to be wrong — read off its own record so far, which starts empty and says 0.5. Whether that estimate tracks reality is measurable:

| Domain | Predicted error rate | Observed error rate | Gap |
| --- | --- | --- | --- |
| `prediction` | 0.2013 | 0.1417 | 0.0596 |
| `discovery` | 0.2349 | 0.2083 | 0.0265 |
| `transfer` | 0.3691 | 0.3167 | 0.0524 |
| `experimentation` | 0.1082 | 0.0583 | 0.0499 |
| `causal_inference` | 0.5129 | 0.5000 | 0.0129 |

Bucketed across all domains:

| Predicted-error band | Claims | Mean predicted | Observed | Gap |
| --- | --- | --- | --- | --- |
| 0.0–0.2 | 257 | 0.1355 | 0.1401 | -0.0046 |
| 0.2–0.4 | 182 | 0.3021 | 0.2473 | +0.0548 |
| 0.4–0.6 | 93 | 0.4902 | 0.3441 | +0.1461 |
| 0.6–0.8 | 8 | 0.6258 | 0.5000 | +0.1258 |

Expected calibration error of ECHO's self-assessment: **0.0477**.

## Self-diagnosis

With the record built but before seeing any new outcome, ECHO ranks the domains by how likely it expects to be right. The right-hand column is what then actually happened on the fresh tasks.

| ECHO's rank | Domain | Expected success | Actual next-phase accuracy |
| --- | --- | --- | --- |
| 1 | `experimentation` | 0.9355 | 93.3% |
| 2 | `prediction` | 0.8065 | 90.0% |
| 3 | `discovery` | 0.7903 | 78.3% |
| 4 | `transfer` | 0.6452 | 71.7% |
| 5 | `causal_inference` | 0.5000 | 43.3% |

ECHO's ordering agrees with the outcome on **10 of 10** pairs. Its top pick was `experimentation`; the domain that actually scored highest next was `experimentation`.

The self-diagnostic is the whole of `rank_by_expected_success`: each domain's own measured record, ordered. No labels about difficulty, no hand-written notion of which capability is hard.

## Abstention

ECHO declines a domain when its meta-confidence falls below 0.55. An `UNTESTED` domain is **not** an automatic abstention — refusing everything it has not already been scored on would make the record unfillable.

| Domain | Abstained? | Accuracy when answering | Would have scored on declined | Confidently wrong | Unnecessary abstentions |
| --- | --- | --- | --- | --- | --- |
| `prediction` | no | 90.0% | — | 6 | 0 of 0 |
| `discovery` | no | 78.3% | — | 6 | 0 of 0 |
| `transfer` | no | 71.7% | — | 17 | 0 of 0 |
| `experimentation` | no | 93.3% | — | 4 | 0 of 0 |
| `causal_inference` | **yes** | — | 43.3% | 0 | 26 of 60 |

Answering everything would have scored **75.3%**. Answering only where the record supports it scored **83.3%** on 240 questions, while declining 60 on which it would have scored 43.3% had it tried.

The cost is explicit: 26 of the 60 declined questions would have been answered correctly. Abstention is not free, and a system that abstained on everything would score 100% on the first column and lose every one of those.

## Failures and things that did not work

- `transfer` is banded **WEAK** (Brier 0.2611) but did **not** abstain, because meta-confidence 0.6803 is above the 0.55 gate. It went on to be confidently wrong 17 times — the most of any answering domain. The two mechanisms disagree, and the band is the one that was right.

The structural weakness worth naming: ECHO's abstention decision is per **domain**, not per question. It cannot notice that one particular causal question is easy while the domain as a whole is weak, so every unnecessary abstention in the table above is a direct consequence of the granularity, not of the thresholds.

## Limitations

1. **Domain-level granularity.** Reliability is tracked per capability area, not per question. This is the main cost, and it is visible in the unnecessary-abstention column.
2. **Stationarity assumed.** The tally weights every past claim equally. A domain that improved or degraded would be reported as its lifetime average, and nothing here detects the change.
3. **Thresholds are hand-set.** 0.15, 0.25, 0.55 and 8 were fixed before the run and not adjusted, but they were still chosen by a person.
4. **The tasks are synthetic.** A domain is a Bernoulli draw against a hidden skill. Real capability is not one number.
5. **The record is ECHO's only source, which is also a ceiling.** It cannot recognise a domain as hard before it has failed at it, and nothing here transfers a lesson from one domain to another.
6. **One seed, one run.** Enough to show the mechanism works; not enough to characterise it.

## What this does and does not demonstrate

**Does:** that ECHO maintains a per-domain record of its own accuracy and derives competence bands from it without being told which areas are hard; that it separates belief in a proposition from the historical reliability of that kind of belief, and reports both; that its stated error rate can be scored against its actual error rate; and that it declines to answer where its record does not support answering, at a measured and reported cost.

**Does not:** anything about self-awareness. This is a scoreboard with thresholds on it. It has no access to its own reasoning process, no representation of itself, and no state that could be called noticing anything — when it abstains, that is a comparison between two floats. The word *metacognition* is used here in the narrow measurement sense and nothing in this report bears on consciousness, sentience, or general intelligence.
