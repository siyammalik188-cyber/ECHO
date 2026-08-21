# ECHO 3 — Prediction

## PREDICTION ≠ LEARNING

This experiment establishes whether ECHO can state explicit probabilities about future events and later score them against what actually happened. **It records experience. It does not generalise from it.**

- No prediction, outcome, or `Experience` record is read back to change a belief, a probability, or a strategy. Nothing consumes them.
- The predictor is a fixed formula — Laplace-smoothed frequency — identical on trial 1 and trial 60. It does not improve; it only accumulates observations, which is not the same thing.
- Environment B changes half way through and ECHO's calibration collapses there. **It does not notice, adapt, or recover.** A learning system would. That failure is left in the numbers below rather than designed around.

What exists now is the loop that learning would later require: a prediction, sealed; an outcome, recorded; an error, computed; an experience, filed. The loop is empty by design.

## Totals

- Total predictions issued: **180**
- Evaluated against reality: **180**
- Environments: **3**

## Scores by environment

Brier is mean squared error (0 perfect, 0.25 for always saying 0.5). Log loss is mean surprisal in nats — lower is better. ECE is the count-weighted average gap between stated and observed frequency. **Accuracy is shown last and on purpose**: it scores `P(A)=0.51` and `P(A)=0.99` identically, which is precisely the distinction this experiment exists to preserve.

| Environment | N | Brier | vs always-0.5 | vs base rate | Log loss | ECE | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Environment A — Stable | 60 | **0.2086** | 0.2500 | 0.1956 | 0.6125 | 0.0952 | 0.717 |
| Environment B — Changing | 60 | **0.2533** | 0.2500 | 0.2431 | 0.7100 | 0.1685 | 0.567 |
| Environment C — Noisy | 60 | **0.2577** | 0.2500 | 0.1875 | 0.7103 | 0.2591 | 0.433 |

A Brier score means little alone, so the two trivial strategies it must beat are shown beside it. Beating *always predict the base rate* is the real bar, and it is a far harder one than beating 0.5 — a predictor can look respectable against 0.5 while adding nothing over simply knowing how often A tends to happen.

### How to read these numbers

**The base-rate column is a hindsight oracle, and it is the floor.** It is computed from the base rate that actually occurred, which no predictor could have known in advance. For a coin that lands heads with probability *p*, the best achievable Brier score is *p*(1−*p*) — irreducible noise that no amount of intelligence removes. Matching that column is therefore **the ceiling on performance, not a passing grade**, and slightly exceeding it is expected rather than a failure.

So Environment A's Brier of 0.2086 against a floor of 0.1956 is close to the best any predictor could do on that sequence — the residual gap is mostly the early trials, where ECHO correctly had no idea and said so.

### Environment A — Stable

**What ECHO could tell.** A stationary process. Nothing about it changes for the duration of the run.

**Predictions:** 60 · **Observed base rate:** 0.733

| Stated probability | N | Mean stated | Observed frequency | Gap |
| --- | --- | --- | --- | --- |
| 0.5–0.6 | 1 | 0.500 | 1.000 | -0.500 |
| 0.6–0.7 | 31 | 0.661 | 0.774 | -0.113 |
| 0.7–0.8 | 24 | 0.725 | 0.708 | +0.017 |
| 0.8–0.9 | 4 | 0.823 | 0.500 | +0.323 |

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.70, constant for all 60 trials.

### Environment B — Changing

**What ECHO could tell.** A process that is stationary for a while and then is not. No signal is given when it changes.

**Predictions:** 60 · **Observed base rate:** 0.583

| Stated probability | N | Mean stated | Observed frequency | Gap |
| --- | --- | --- | --- | --- |
| 0.5–0.6 | 9 | 0.572 | 0.556 | +0.016 |
| 0.6–0.7 | 12 | 0.654 | 0.250 | +0.405 |
| 0.7–0.8 | 11 | 0.747 | 0.545 | +0.202 |
| 0.8–0.9 | 27 | 0.852 | 0.778 | +0.074 |
| 0.9–1.0 | 1 | 0.900 | 0.000 | +0.900 |

**Before and after the change:**

| Segment | N | Brier | Base rate | Mean stated |
| --- | --- | --- | --- | --- |
| Trials 0–29 | 30 | 0.1108 | 0.900 | 0.826 |
| Trials 30–59 | 30 | 0.3957 | 0.267 | 0.678 |

This is the honest failure of the experiment. The estimator averages all history equally, so after the process flips it keeps predicting the old regime and is confidently wrong for a long stretch. It never recovers within the run. **Nothing in ECHO detects the change**, because nothing in ECHO is looking.

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.85 for trials 0–29, then drops to 0.20 for trials 30–59. The change is abrupt and unannounced.

### Environment C — Noisy

**What ECHO could tell.** A stationary process whose observations are unreliable: a substantial fraction of what ECHO is told is simply wrong.

**Predictions:** 60 · **Observed base rate:** 0.750

| Stated probability | N | Mean stated | Observed frequency | Gap |
| --- | --- | --- | --- | --- |
| 0.2–0.3 | 1 | 0.250 | 1.000 | -0.750 |
| 0.3–0.4 | 2 | 0.333 | 1.000 | -0.667 |
| 0.4–0.5 | 18 | 0.461 | 0.778 | -0.317 |
| 0.5–0.6 | 39 | 0.519 | 0.718 | -0.199 |

**Why this one is miscalibrated.** 23 of 60 observations shown to ECHO were flipped, so the apparent rate in its evidence was 0.500 while the true rate was 0.750. ECHO converged on what it was told, which is the correct response to its evidence and the wrong answer about the world. Its error here is inherited from its information, not produced by its reasoning — and nothing in the scores can separate those two, which is a limitation of the experiment rather than a finding about ECHO.

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.65, constant. Each observation shown to ECHO has a 30% chance of being reported as the opposite of what happened. Predictions are scored against the truth, not the report.

## Prediction history examples

The ledger answers, for any prediction: what did I predict, what probability did I assign, what information did I have, what actually happened, and how wrong was I.

| # | t | P(A) | Information | Actual | Error |
| --- | --- | --- | --- | --- | --- |
| `d7e2f2` | 0 | 0.5000 | 0 obs | OUTCOME_A | 0.5000 |
| `74607e` | 1 | 0.6667 | 1 obs | OUTCOME_A | 0.3333 |
| `6b1b9e` | 2 | 0.7500 | 2 obs | OUTCOME_A | 0.2500 |
| `d228ca` | 3 | 0.8000 | 3 obs | OUTCOME_A | 0.2000 |
| `adde4a` | 4 | 0.8333 | 4 obs | OUTCOME_A | 0.1667 |
| `eeeda2` | 5 | 0.8571 | 5 obs | OUTCOME_B | 0.8571 |
| `7882c8` | 6 | 0.7500 | 6 obs | OUTCOME_A | 0.2500 |
| `ddff40` | 7 | 0.7778 | 7 obs | OUTCOME_A | 0.2222 |

**Correct predictions.** The single best-scored prediction across all runs:

- `PRED-07c61bfac7` at t=7: stated **0.8889**, outcome **OUTCOME_A**, Brier **0.0123**

**Failed predictions.** The single worst-scored prediction across all runs:

- `PRED-e4a1fb02a3` at t=8: stated **0.9000**, outcome **OUTCOME_B**, Brier **0.8100**

- Environment A — Stable: 43 anticipated correctly, 17 not (a coarse count that ignores how strongly each was claimed).
- Environment B — Changing: 34 anticipated correctly, 26 not (a coarse count that ignores how strongly each was claimed).
- Environment C — Noisy: 26 anticipated correctly, 34 not (a coarse count that ignores how strongly each was claimed).

## The critical sequence

Prediction 1 at 0.80 → **B**. Prediction 2 at 0.65 → **B**. Prediction 3 at 0.40, left pending. After both failures, prediction 1 must still read 0.80. Reloaded from disk:

| # | P(A) stated | Status | Actual | Error |
| --- | --- | --- | --- | --- |
| 1 | **0.80** | evaluated | OUTCOME_B | 0.800 |
| 2 | **0.65** | evaluated | OUTCOME_B | 0.650 |
| 3 | **0.40** | pending | — | n/a |

Sequence read back from disk: `[0.8, 0.65, 0.4]` — **unchanged**.

## TEMPORAL_LEAKAGE_TEST

Each probe below deliberately tries to smuggle information from after the prediction timestamp into the prediction. All must be refused. The last two are controls: one checks the view cannot reach the future, the other checks a legitimate prediction is still accepted — a system that rejects everything would pass the first four and be useless.

| Probe | Outcome |
| --- | --- |
| `cite_the_outcome_being_predicted` | ✅ correct |
| `cite_far_future_information` | ✅ correct |
| `smuggle_future_through_evidence_ids` | ✅ correct |
| `cite_an_unknown_observation` | ✅ correct |
| `view_cannot_reach_past_its_horizon` | ✅ correct |
| `legitimate_prediction_still_accepted` | ✅ correct |

- **`cite_the_outcome_being_predicted`** — Predict at t=5 while citing the observation stamped t=5 — the outcome itself.
  - prediction rejected — information from at or after the prediction time: OBS-1f9a86e0d1 (t=5, prediction at t=5)
- **`cite_far_future_information`** — Predict at t=1 while citing an observation from t=99.
  - prediction rejected — information from at or after the prediction time: OBS-941edc93e6 (t=99, prediction at t=1)
- **`smuggle_future_through_evidence_ids`** — Keep `information_available` clean but hide the future in `evidence_ids_used`.
  - prediction rejected — information from at or after the prediction time: OBS-941edc93e6 (t=99, prediction at t=1)
- **`cite_an_unknown_observation`** — Cite an id the timeline has never heard of — it cannot be shown to predate anything.
  - prediction rejected — unknown observation ids: OBS-invented
- **`view_cannot_reach_past_its_horizon`** — Ask a t=5 view for observations stamped t=5 and t=99.
  - view exposed only observations strictly before t=5
- **`legitimate_prediction_still_accepted`** — Guard against a system that rejects everything and calls it safety.
  - a prediction citing only the past was accepted

**Result: 6/6 probes behaved correctly.**

## Persistence

Run as part of this report, not asserted: the ledgers were discarded and rebuilt from their JSON files, then compared field by field.

| After restart | Survives |
| --- | --- |
| Every prediction | yes |
| The exact stated probabilities | yes |
| What information each prediction had | yes |
| The recorded outcomes and errors | yes |
| The filed experience records | yes |

Predictions checked: 180; experiences checked: 180.

## Limitations

1. **The predictor is deliberately naive.** Laplace-smoothed frequency over all history. It cannot represent trend, regime, or recency, and Environment B punishes it accordingly. A better predictor would score better; that would not tell us anything more about whether the prediction *loop* works, which is what is being tested.
2. **Environment C conflates two things.** ECHO is scored against truth while seeing corrupted observations, so its error there mixes bad prediction with bad information. The scores cannot separate them, and no attempt is made to.
3. **60 trials per environment is small.** Calibration buckets hold few predictions each, so observed frequencies are noisy. Bucket gaps here are suggestive, not solid.
4. **Predictions within a run are not independent.** Each is built from the history that produced the ones before it, so the scores should not be read as 60 independent trials.
5. **Only binary outcomes.** No multi-class, no continuous quantities, no structured predictions.
6. **Time is a logical tick, not a clock.** This is the right choice for reproducibility and for reasoning about ordering, but it means the temporal guarantee is about sequence, not about wall-clock time. A real deployment would have to defend against clock skew, which this does not.
7. **The proposition is fixed and supplied.** ECHO does not decide what is worth predicting, only what probability to assign.

## Conclusion

ECHO issued **180** explicit probabilistic predictions across three environments, sealed each one before the outcome existed, scored them with proper scoring rules, and filed an immutable experience record for each. Every temporal-leakage probe was refused. The critical sequence survived two failed predictions without a single stated probability moving.

It is well calibrated on a stationary process, roughly base-rate on a noisy one, and **badly wrong on a changing one — where it fails to notice anything has changed at all.** That last result is the most informative thing in this report and the clearest demonstration of what is missing.

**PREDICTION ≠ LEARNING.** The experience records exist and nothing reads them. ECHO does not currently generalise from any of this.
