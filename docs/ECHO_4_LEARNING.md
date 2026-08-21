# ECHO 4 — Learning From Prediction Errors

## What the words mean here

These four are different things, and the whole point of this series is that they are not interchangeable.

| Term | Definition | Where it lives |
| --- | --- | --- |
| **MEMORY** | Stored information. A statement that was true and was kept. | `echo/memory.py` |
| **BELIEF REVISION** | Changing confidence in a proposition when evidence arrives. One number, one sentence, a fixed rule. | `echo/belief.py` |
| **PREDICTION** | A probabilistic statement about a future outcome, sealed before the outcome exists. | `echo/prediction.py` |
| **LEARNING** | A persistent change in *strategy or behaviour*, caused by experience, that improves performance on **future** situations. | `echo/learning.py` |

**A value changing is not learning.** Belief revision changes a number by a rule that never itself changes; ECHO is exactly the same afterwards. What is claimed in this document is narrower and more specific: ECHO's *method of predicting* changed, the change was caused by measured errors, it persists, and it is tested on situations that came after it.

This is not AGI, not consciousness, and not autonomous intelligence. It is a change-detector wired to a two-item menu of estimators. The mechanism is a few hundred lines of arithmetic and it is described in full below.

## Condition A (no learning) vs Condition B (learning)

Identical deterministic environments, identical starting strategy. The only difference is whether error analysis and strategy revision are enabled.

| Environment | Metric | No learning | Learning | Change |
| --- | --- | --- | --- | --- |
| `ENV-B-changing` | Brier (whole run) | 0.2533 | 0.2120 | -0.0412 (better) |
| `ENV-B-changing` | Log loss | 0.7100 | 0.6270 | -0.0830 (better) |
| `ENV-B-changing` | ECE | 0.1685 | 0.1425 | -0.0259 (better) |
| `ENV-B-changing` | Brier after change | 0.3957 | 0.3133 | -0.0824 (better) |
| `ENV-B-changing` | Brier, final 20 | 0.3328 | 0.2250 | -0.1078 (better) |
| `ENV-B-changing` | High-confidence errors | 7 | 8 | +1 |
| `ENV-B-changing` | Recovery trial | never | 52 | — |
| `ENV-C-noisy` | Brier (whole run) | 0.2577 | 0.2513 | -0.0065 (better) |
| `ENV-C-noisy` | Log loss | 0.7103 | 0.6962 | -0.0140 (better) |
| `ENV-C-noisy` | ECE | 0.2591 | 0.2312 | -0.0280 (better) |
| `ENV-C-noisy` | Brier, final 20 | 0.2457 | 0.2263 | -0.0195 (better) |
| `ENV-C-noisy` | High-confidence errors | 0 | 0 | +0 |
| `ENV-B-LONG` | Brier (whole run) | 0.2429 | 0.1897 | -0.0532 (better) |
| `ENV-B-LONG` | Log loss | 0.6840 | 0.5759 | -0.1081 (better) |
| `ENV-B-LONG` | ECE | 0.2174 | 0.0935 | -0.1239 (better) |
| `ENV-B-LONG` | Brier after change | 0.2870 | 0.2160 | -0.0710 (better) |
| `ENV-B-LONG` | Brier, final 20 | 0.2112 | 0.1823 | -0.0290 (better) |
| `ENV-B-LONG` | High-confidence errors | 7 | 12 | +5 |
| `ENV-B-LONG` | Recovery trial | never | 52 | — |
| `ENV-D-late-change` | Brier (whole run) | 0.2532 | 0.1890 | -0.0642 (better) |
| `ENV-D-late-change` | Log loss | 0.7024 | 0.5651 | -0.1373 (better) |
| `ENV-D-late-change` | ECE | 0.2099 | 0.1129 | -0.0970 (better) |
| `ENV-D-late-change` | Brier after change | 0.3051 | 0.1723 | -0.1329 (better) |
| `ENV-D-late-change` | Brier, final 20 | 0.2263 | 0.1427 | -0.0836 (better) |
| `ENV-D-late-change` | High-confidence errors | 2 | 7 | +5 |
| `ENV-D-late-change` | Recovery trial | 103 | 76 | — |
| `ENV-F-false-alarm` | Brier (whole run) | 0.2188 | 0.2213 | +0.0025 (worse) |
| `ENV-F-false-alarm` | Log loss | 0.6297 | 0.6354 | +0.0057 (worse) |
| `ENV-F-false-alarm` | ECE | 0.1014 | 0.1065 | +0.0051 (worse) |
| `ENV-F-false-alarm` | Brier, final 20 | 0.1743 | 0.1743 | +0.0000 (same) |
| `ENV-F-false-alarm` | High-confidence errors | 0 | 0 | +0 |

Lower is better for Brier, log loss and ECE. 'Recovery trial' is the first post-change trial at which a rolling 10-prediction Brier returns to within 0.05 of the pre-change level; **never** means it did not, within the run.

## `ENV-B-changing` — Environment B — Changing

**What ECHO could tell.** A process that is stationary for a while and then is not. No signal is given when it changes.

### Timeline (learning condition)

```
TRIAL 30 — environment changes (hidden from ECHO)
TRIAL 0 — strategy STRAT-c8fe10f4 ACTIVE: Laplace-smoothed frequency over the entire observation history.
TRIAL 32 — deterioration flagged (p=7.8e-03), not yet confirmed; no action taken
TRIAL 33 — deterioration flagged (p=7.1e-04), not yet confirmed; no action taken
TRIAL 34 — deterioration flagged (p=4.1e-05), not yet confirmed; no action taken
TRIAL 35 — deterioration flagged (p=1.6e-06), not yet confirmed; no action taken
TRIAL 36 — deterioration CONFIRMED (60% recent miss rate vs 11% historical, p=2.7e-04)
TRIAL 36 — hypothesis: REGIME_CHANGE (confidence 0.95); alternatives considered and rejected
TRIAL 36 — strategy STRAT-f6affc70 proposed (window=10) → TESTING for 8 trials
TRIAL 37 — deterioration flagged (p=2.2e-04), not yet confirmed; no action taken
TRIAL 38 — deterioration flagged (p=1.8e-04), not yet confirmed; no action taken
TRIAL 39 — deterioration flagged (p=9.1e-06), not yet confirmed; no action taken
TRIAL 40 — deterioration flagged (p=6.1e-04), not yet confirmed; no action taken
TRIAL 44 — test concluded: STRAT-f6affc70 Brier 0.220 vs incumbent 0.399 → STRAT-f6affc70 ACTIVE, STRAT-c8fe10f4 WEAKENED
TRIAL 52 — rolling performance back to pre-change form
```

**Failure explanations reached:**

| Explanation | Count |
| --- | --- |
| `RANDOM_VARIATION` | 172 |
| `EVIDENCE_ERROR` | 0 |
| `REGIME_CHANGE` | 4 |
| `STRATEGY_FAILURE` | 4 |
| `INSUFFICIENT_INFORMATION` | 28 |
| `UNKNOWN` | 32 |

**What did I learn, and did it help?**

- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-02e140ac', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-4c351ae9', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-02e140ac if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-0f8d8d00', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-a360d44e', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-0f8d8d00 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-829b64cb', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-7d6d06c4', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-829b64cb if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-c8fe10f4', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-f6affc70', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-c8fe10f4 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.

**Strategies, including retired ones:**

| Strategy | v | Kind | Params | Status | Confidence |
| --- | --- | --- | --- | --- | --- |
| `STRAT-02e140ac` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-4c351ae9` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-0f8d8d00` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-a360d44e` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-829b64cb` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-7d6d06c4` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-c8fe10f4` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-f6affc70` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.85 for trials 0–29, then drops to 0.20 for trials 30–59. The change is abrupt and unannounced.

## `ENV-C-noisy` — Environment C — Noisy

**What ECHO could tell.** A stationary process whose observations are unreliable: a substantial fraction of what ECHO is told is simply wrong.

### Timeline (learning condition)

```
TRIAL 0 — strategy STRAT-bcba5c98 ACTIVE: Laplace-smoothed frequency over the entire observation history.
TRIAL 41 — deterioration flagged (p=7.2e-03), not yet confirmed; no action taken
TRIAL 42 — deterioration flagged (p=7.9e-04), not yet confirmed; no action taken
TRIAL 43 — deterioration flagged (p=4.9e-05), not yet confirmed; no action taken
TRIAL 44 — deterioration flagged (p=3.8e-05), not yet confirmed; no action taken
TRIAL 45 — deterioration CONFIRMED (90% recent miss rate vs 28% historical, p=7.4e-05)
TRIAL 45 — hypothesis: REGIME_CHANGE (confidence 0.95); alternatives considered and rejected
TRIAL 45 — strategy STRAT-5bea045b proposed (window=10) → TESTING for 8 trials
TRIAL 46 — deterioration flagged (p=1.5e-03), not yet confirmed; no action taken
TRIAL 53 — test concluded: STRAT-5bea045b Brier 0.151 vs incumbent 0.230 → STRAT-5bea045b ACTIVE, STRAT-bcba5c98 WEAKENED
```

**Failure explanations reached:**

| Explanation | Count |
| --- | --- |
| `RANDOM_VARIATION` | 100 |
| `EVIDENCE_ERROR` | 108 |
| `REGIME_CHANGE` | 4 |
| `STRATEGY_FAILURE` | 0 |
| `INSUFFICIENT_INFORMATION` | 28 |
| `UNKNOWN` | 0 |

**What did I learn, and did it help?**

- **At trial 45**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=7.38e-05 <= 0.01 and effect=0.62 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-e8e05320', 'mean_brier': 0.229704, 'window': 'trials 46-53'}
  - Measured after: {'strategy': 'STRAT-e00ef771', 'mean_brier': 0.151042, 'window': 'trials 46-53'}
  - Would reverse if: Revert to STRAT-e8e05320 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 45**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=7.38e-05 <= 0.01 and effect=0.62 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-9274e537', 'mean_brier': 0.229704, 'window': 'trials 46-53'}
  - Measured after: {'strategy': 'STRAT-132b5c4d', 'mean_brier': 0.151042, 'window': 'trials 46-53'}
  - Would reverse if: Revert to STRAT-9274e537 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 45**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=7.38e-05 <= 0.01 and effect=0.62 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-d4cbf939', 'mean_brier': 0.229704, 'window': 'trials 46-53'}
  - Measured after: {'strategy': 'STRAT-6ab35a9f', 'mean_brier': 0.151042, 'window': 'trials 46-53'}
  - Would reverse if: Revert to STRAT-d4cbf939 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 45**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=7.38e-05 <= 0.01 and effect=0.62 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-bcba5c98', 'mean_brier': 0.229704, 'window': 'trials 46-53'}
  - Measured after: {'strategy': 'STRAT-5bea045b', 'mean_brier': 0.151042, 'window': 'trials 46-53'}
  - Would reverse if: Revert to STRAT-bcba5c98 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.

**Strategies, including retired ones:**

| Strategy | v | Kind | Params | Status | Confidence |
| --- | --- | --- | --- | --- | --- |
| `STRAT-e8e05320` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-e00ef771` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-9274e537` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-132b5c4d` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-d4cbf939` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-6ab35a9f` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-bcba5c98` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-5bea045b` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.65, constant. Each observation shown to ECHO has a 30% chance of being reported as the opposite of what happened. Predictions are scored against the truth, not the report.

## `ENV-B-LONG` — Environment B (extended) — Changing, with room to recover

**What ECHO could tell.** The same regime structure as Environment B, run for longer so that post-adaptation performance can actually be measured rather than inferred from a handful of trials.

### Timeline (learning condition)

```
TRIAL 30 — environment changes (hidden from ECHO)
TRIAL 0 — strategy STRAT-6a5e1f78 ACTIVE: Laplace-smoothed frequency over the entire observation history.
TRIAL 32 — deterioration flagged (p=7.8e-03), not yet confirmed; no action taken
TRIAL 33 — deterioration flagged (p=7.1e-04), not yet confirmed; no action taken
TRIAL 34 — deterioration flagged (p=4.1e-05), not yet confirmed; no action taken
TRIAL 35 — deterioration flagged (p=1.6e-06), not yet confirmed; no action taken
TRIAL 36 — deterioration CONFIRMED (60% recent miss rate vs 11% historical, p=2.7e-04)
TRIAL 36 — hypothesis: REGIME_CHANGE (confidence 0.95); alternatives considered and rejected
TRIAL 36 — strategy STRAT-dd612060 proposed (window=10) → TESTING for 8 trials
TRIAL 37 — deterioration flagged (p=2.2e-04), not yet confirmed; no action taken
TRIAL 38 — deterioration flagged (p=1.8e-04), not yet confirmed; no action taken
TRIAL 39 — deterioration flagged (p=9.1e-06), not yet confirmed; no action taken
TRIAL 40 — deterioration flagged (p=6.1e-04), not yet confirmed; no action taken
TRIAL 44 — test concluded: STRAT-dd612060 Brier 0.220 vs incumbent 0.399 → STRAT-dd612060 ACTIVE, STRAT-6a5e1f78 WEAKENED
TRIAL 52 — rolling performance back to pre-change form
```

**Failure explanations reached:**

| Explanation | Count |
| --- | --- |
| `RANDOM_VARIATION` | 412 |
| `EVIDENCE_ERROR` | 0 |
| `REGIME_CHANGE` | 4 |
| `STRATEGY_FAILURE` | 4 |
| `INSUFFICIENT_INFORMATION` | 28 |
| `UNKNOWN` | 32 |

**What did I learn, and did it help?**

- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-3862bcc4', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-ea8dfc46', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-3862bcc4 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-3ec41978', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-cf5e988f', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-3ec41978 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-ba5337c9', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-e78139f2', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-ba5337c9 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 36**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=2.65e-04 <= 0.01 and effect=0.49 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-6a5e1f78', 'mean_brier': 0.398892, 'window': 'trials 37-44'}
  - Measured after: {'strategy': 'STRAT-dd612060', 'mean_brier': 0.219618, 'window': 'trials 37-44'}
  - Would reverse if: Revert to STRAT-6a5e1f78 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.

**Strategies, including retired ones:**

| Strategy | v | Kind | Params | Status | Confidence |
| --- | --- | --- | --- | --- | --- |
| `STRAT-3862bcc4` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-ea8dfc46` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-3ec41978` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-cf5e988f` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-ba5337c9` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-e78139f2` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-6a5e1f78` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-dd612060` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.85 for trials 0-29, then 0.20 for trials 30-119.

## `ENV-D-late-change` — Environment D — Generalisation: a different change, in a different place

**What ECHO could tell.** A process that changes once. Nothing about where or in which direction is available to ECHO.

### Timeline (learning condition)

```
TRIAL 62 — environment changes (hidden from ECHO)
TRIAL 0 — strategy STRAT-32d13b3a ACTIVE: Laplace-smoothed frequency over the entire observation history.
TRIAL 68 — deterioration flagged (p=5.8e-03), not yet confirmed; no action taken
TRIAL 69 — deterioration flagged (p=6.7e-04), not yet confirmed; no action taken
TRIAL 70 — deterioration flagged (p=4.5e-05), not yet confirmed; no action taken
TRIAL 71 — deterioration flagged (p=1.3e-06), not yet confirmed; no action taken
TRIAL 72 — deterioration CONFIRMED (90% recent miss rate vs 27% historical, p=5.7e-05)
TRIAL 72 — hypothesis: REGIME_CHANGE (confidence 0.95); alternatives considered and rejected
TRIAL 72 — strategy STRAT-f5561eec proposed (window=10) → TESTING for 8 trials
TRIAL 73 — deterioration flagged (p=9.9e-04), not yet confirmed; no action taken
TRIAL 74 — deterioration flagged (p=9.1e-03), not yet confirmed; no action taken
TRIAL 80 — test concluded: STRAT-f5561eec Brier 0.111 vs incumbent 0.354 → STRAT-f5561eec ACTIVE, STRAT-32d13b3a WEAKENED
TRIAL 76 — rolling performance back to pre-change form
```

**Failure explanations reached:**

| Explanation | Count |
| --- | --- |
| `RANDOM_VARIATION` | 424 |
| `EVIDENCE_ERROR` | 0 |
| `REGIME_CHANGE` | 4 |
| `STRATEGY_FAILURE` | 0 |
| `INSUFFICIENT_INFORMATION` | 28 |
| `UNKNOWN` | 24 |

**What did I learn, and did it help?**

- **At trial 72**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=5.74e-05 <= 0.01 and effect=0.63 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-4bba09eb', 'mean_brier': 0.353506, 'window': 'trials 73-80'}
  - Measured after: {'strategy': 'STRAT-ca9d6254', 'mean_brier': 0.111111, 'window': 'trials 73-80'}
  - Would reverse if: Revert to STRAT-4bba09eb if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 72**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=5.74e-05 <= 0.01 and effect=0.63 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-6e4c07dc', 'mean_brier': 0.353506, 'window': 'trials 73-80'}
  - Measured after: {'strategy': 'STRAT-7d44d024', 'mean_brier': 0.111111, 'window': 'trials 73-80'}
  - Would reverse if: Revert to STRAT-6e4c07dc if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 72**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=5.74e-05 <= 0.01 and effect=0.63 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-ea668e24', 'mean_brier': 0.353506, 'window': 'trials 73-80'}
  - Measured after: {'strategy': 'STRAT-955bed1c', 'mean_brier': 0.111111, 'window': 'trials 73-80'}
  - Would reverse if: Revert to STRAT-ea668e24 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 72**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=5.74e-05 <= 0.01 and effect=0.63 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 1.000
  - Did it help: **improved**
  - Measured before: {'strategy': 'STRAT-32d13b3a', 'mean_brier': 0.353506, 'window': 'trials 73-80'}
  - Measured after: {'strategy': 'STRAT-f5561eec', 'mean_brier': 0.111111, 'window': 'trials 73-80'}
  - Would reverse if: Revert to STRAT-32d13b3a if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.

**Strategies, including retired ones:**

| Strategy | v | Kind | Params | Status | Confidence |
| --- | --- | --- | --- | --- | --- |
| `STRAT-4bba09eb` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-ca9d6254` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-6e4c07dc` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-7d44d024` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-ea668e24` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-955bed1c` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |
| `STRAT-32d13b3a` | 1 | laplace_all_history | {'alpha': 1.0} | **weakened** | 0.20 |
| `STRAT-f5561eec` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **active** | 0.85 |

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.25 for trials 0-61, then 0.85 for trials 62-119. The change is 32 trials later than Environment B's and runs the opposite way, so a system that memorised 'trial 30' or 'A becomes rarer' cannot score well here.

## `ENV-F-false-alarm` — Environment F — Unusual outcomes, unchanged process

**What ECHO could tell.** A stationary process. Nothing about it changes at any point in the run.

### Timeline (learning condition)

```
TRIAL 0 — strategy STRAT-303f3290 ACTIVE: Laplace-smoothed frequency over the entire observation history.
TRIAL 46 — deterioration flagged (p=2.8e-03), not yet confirmed; no action taken
TRIAL 47 — deterioration flagged (p=2.3e-03), not yet confirmed; no action taken
TRIAL 48 — deterioration flagged (p=1.8e-04), not yet confirmed; no action taken
TRIAL 49 — deterioration flagged (p=2.8e-03), not yet confirmed; no action taken
TRIAL 50 — deterioration CONFIRMED (80% recent miss rate vs 34% historical, p=4.0e-03)
TRIAL 50 — hypothesis: REGIME_CHANGE (confidence 0.95); alternatives considered and rejected
TRIAL 50 — strategy STRAT-fbeafb19 proposed (window=10) → TESTING for 8 trials
TRIAL 51 — deterioration flagged (p=5.6e-03), not yet confirmed; no action taken
TRIAL 52 — deterioration flagged (p=7.4e-03), not yet confirmed; no action taken
TRIAL 58 — test concluded: STRAT-fbeafb19 Brier 0.296 did NOT beat incumbent 0.258 → STRAT-fbeafb19 RETIRED, reverting to STRAT-303f3290
```

**Failure explanations reached:**

| Explanation | Count |
| --- | --- |
| `RANDOM_VARIATION` | 424 |
| `EVIDENCE_ERROR` | 0 |
| `REGIME_CHANGE` | 4 |
| `STRATEGY_FAILURE` | 0 |
| `INSUFFICIENT_INFORMATION` | 28 |
| `UNKNOWN` | 24 |

**What did I learn, and did it help?**

- **At trial 50**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=4.04e-03 <= 0.01 and effect=0.46 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 0.996
  - Did it help: **no_improvement**
  - Measured before: {'strategy': 'STRAT-0b875773', 'mean_brier': 0.258277, 'window': 'trials 51-58'}
  - Measured after: {'strategy': 'STRAT-a9a8da66', 'mean_brier': 0.296007, 'window': 'trials 51-58'}
  - Would reverse if: Revert to STRAT-0b875773 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 50**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=4.04e-03 <= 0.01 and effect=0.46 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 0.996
  - Did it help: **no_improvement**
  - Measured before: {'strategy': 'STRAT-aa6ef4aa', 'mean_brier': 0.258277, 'window': 'trials 51-58'}
  - Measured after: {'strategy': 'STRAT-0bd6e633', 'mean_brier': 0.296007, 'window': 'trials 51-58'}
  - Would reverse if: Revert to STRAT-aa6ef4aa if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 50**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=4.04e-03 <= 0.01 and effect=0.46 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 0.996
  - Did it help: **no_improvement**
  - Measured before: {'strategy': 'STRAT-dc8ad20c', 'mean_brier': 0.258277, 'window': 'trials 51-58'}
  - Measured after: {'strategy': 'STRAT-b3ea5c7f', 'mean_brier': 0.296007, 'window': 'trials 51-58'}
  - Would reverse if: Revert to STRAT-dc8ad20c if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.
- **At trial 50**
  - From: Laplace-smoothed frequency over the entire observation history. (v1)
  - To: Laplace-smoothed frequency over the most recent 10 observations, discarding older evidence. (v2)
  - Why: Sustained, statistically significant deterioration against historical performance (p=4.04e-03 <= 0.01 and effect=0.46 >= 0.25, sustained for 4 trials (confirmed)). The active strategy assumes a stationary process; that assumption is what the evidence contradicts.
  - Confidence: 0.996
  - Did it help: **no_improvement**
  - Measured before: {'strategy': 'STRAT-303f3290', 'mean_brier': 0.258277, 'window': 'trials 51-58'}
  - Measured after: {'strategy': 'STRAT-fbeafb19', 'mean_brier': 0.296007, 'window': 'trials 51-58'}
  - Would reverse if: Revert to STRAT-303f3290 if the successor's mean Brier over the 8-trial test window is not at least 0.02 better than the incumbent's over the same trials, or if it later loses that margin over any subsequent 10-trial window.

**Strategies, including retired ones:**

| Strategy | v | Kind | Params | Status | Confidence |
| --- | --- | --- | --- | --- | --- |
| `STRAT-0b875773` | 1 | laplace_all_history | {'alpha': 1.0} | **active** | 0.50 |
| `STRAT-a9a8da66` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **retired** | 0.10 |
| `STRAT-aa6ef4aa` | 1 | laplace_all_history | {'alpha': 1.0} | **active** | 0.50 |
| `STRAT-0bd6e633` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **retired** | 0.10 |
| `STRAT-dc8ad20c` | 1 | laplace_all_history | {'alpha': 1.0} | **active** | 0.50 |
| `STRAT-b3ea5c7f` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **retired** | 0.10 |
| `STRAT-303f3290` | 1 | laplace_all_history | {'alpha': 1.0} | **active** | 0.50 |
| `STRAT-fbeafb19` | 2 | laplace_recent_window | {'alpha': 1.0, 'window': 10} | **retired** | 0.10 |

**Hidden process (never shown to ECHO):** P(OUTCOME_A) = 0.75 for all 120 trials. The underlying probability never changes. Trials 40-46 are forced to OUTCOME_B, producing a run of seven unusual results that a naive detector should mistake for a regime change. Any confirmed detection here is a false alarm.

## Generalisation, and whether this is just memorisation

`ENV-D-late-change` exists to answer one question: has ECHO learned *'regime changes happen at trial 30'*, or *'sustained deterioration means the process may have changed'*? Its change is **32 trials later** than Environment B's and runs in the **opposite direction** — a rarer outcome becoming common rather than the reverse.

- Detection in ENV-D: TRIAL 72 — deterioration CONFIRMED (90% recent miss rate vs 27% historical, p=5.7e-05)
- Brier after the change: 0.3051 without learning, 0.1723 with (-0.1329 (better))
- **Nothing in the detector references a trial number.** It compares a recent window to the history preceding it, wherever that window happens to fall.

## False alarms

`ENV-F-false-alarm` is stationary throughout — P(A) = 0.75 from first trial to last — but trials 40–46 are forced to OUTCOME_B, producing a run of seven unusual results. A detector that fires on surprise alone will call this a regime change. **Any confirmed detection here is a false alarm.**

| Environment | Real changes | Flagged | Confirmed | Verdict |
| --- | --- | --- | --- | --- |
| `ENV-B-changing` | 1 | 9 | 1 | detected |
| `ENV-C-noisy` | 0 | 6 | 1 | **1 FALSE ALARM(S)** |
| `ENV-B-LONG` | 1 | 9 | 1 | detected |
| `ENV-D-late-change` | 1 | 7 | 1 | detected |
| `ENV-F-false-alarm` | 0 | 7 | 1 | **1 FALSE ALARM(S)** |

The gap between *flagged* and *confirmed* is where overreaction is prevented. A single significant window is not enough — the deterioration must persist across a gap of 4 trials. A streak of bad luck stops; a changed process does not.

## Learning history survives restart

| After restart | Survives |
| --- | --- |
| Learning experiences | yes |
| Strategy revisions | yes |
| All strategies, including retired ones | yes |
| Whether each revision helped | yes |
| What would reverse each change | yes |

## Limitations

1. **The menu has two items.** A strategy picks between all-history and recent-window frequency estimation. ECHO cannot invent an estimator, and calling a choice between two options 'learning' is only fair because the choice is caused by evidence, persists, and is tested — not because the space is impressive.
2. **The window size is not learned.** The successor's window is the detector's own window. A system that learned *how much* history to keep would be doing something substantially harder.
3. **One adaptation per run, effectively.** After a promotion there is no mechanism for proposing a third strategy on new evidence beyond the monitored reversal condition.
4. **The detector only notices deterioration.** A process that changed in ECHO's favour would go unremarked, because nothing looks for improvement.
5. **Thresholds are hand-set.** α, the effect size, the confirmation gap, the test window and the promotion margin were all chosen by a human before the run and not tuned afterwards — but they were still chosen, and different values would give different results.
6. **Four environments, one author.** They test the failure modes their author anticipated. Environment F is the only adversarial case, and it is adversarial in exactly the way its author thought to be adversarial.
7. **Deterministic and small.** A single seed per environment. Results are reproducible but not statistically robust; a different seed could tell a different story.

## Conclusion

- `ENV-B-changing`: post-change Brier improved from 0.3957 to 0.3133.
- `ENV-B-LONG`: post-change Brier improved from 0.2870 to 0.2160.
- `ENV-D-late-change`: post-change Brier improved from 0.3051 to 0.1723.

**And what it cost.** In the environments where nothing changed, there was nothing to gain — only accuracy to lose by chasing noise:

- `ENV-C-noisy`: 1 confirmed detection(s) where the process never changed — **1 false alarm(s)**. Whole-run Brier 0.2577 → 0.2513 (no cost — 0.0065 better despite the false alarm).
- `ENV-F-false-alarm`: 1 confirmed detection(s) where the process never changed — **1 false alarm(s)**. Whole-run Brier 0.2188 → 0.2213 (cost 0.0025 Brier).

Both stationary environments produced a false alarm. In `ENV-F-false-alarm` the candidate strategy was tested and lost to the incumbent, so it was retired and the incumbent restored — the mechanism recovered, but the trials spent testing it were paid for in accuracy. Detection here is not free, and these numbers are the price.

**LEARNING ≠ MEMORY ≠ BELIEF REVISION ≠ PREDICTION.** What changed here is the method, not a stored value: ECHO detected that its own predictions had deteriorated beyond what chance explains, considered several reasons why, proposed a different estimator whose assumptions matched the evidence, tested it against the incumbent on trials neither had seen, and kept it only because it measurably won. The change persisted, and it transferred to a regime change in a different place and direction.

That is a narrow, mechanical form of learning. It is not general intelligence, not autonomy, and not understanding — ECHO does not know what an orchid or an outcome is. It noticed a number getting worse and changed a formula, for stated reasons, and can show its work.
