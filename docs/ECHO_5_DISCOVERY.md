# ECHO 5 — Discovery

Can ECHO find a predictive relationship that no one wrote into its strategy? This is the first controlled hypothesis-discovery experiment in the series. It is not AGI, not consciousness, and not autonomous intelligence — it is a bounded search over a closed language, scored by the data rather than by anyone's opinion of the result.

## Executive result

# DISCOVERY SUCCESS

Chosen from counts, not from how the result reads: every world with a genuine relationship produced a hypothesis that generalised to its sealed holdout, and neither world without one produced any promotion at all. Had either half failed, the line above would say so.

| World | Relationship present? | Verdict | Discovered |
| --- | --- | --- | --- |
| `DISC-A-lagged` | yes | **DISCOVERY SUCCESS** | `LAG(X3, 2)` |
| `DISC-B-interaction` | yes | **DISCOVERY SUCCESS** | `PRODUCT(CHANGE(X2), LAG(X4, 3))` |
| `DISC-N-null` | no | **NO RELIABLE DISCOVERY — correct** | *nothing* |
| `DISC-T-trap` | no | **NO RELIABLE DISCOVERY — correct** | *nothing* |

**4 of 4 worlds produced the correct outcome.** For the two worlds with a genuine relationship, that means a hypothesis that generalised to the sealed holdout. For the two without one, it means the search correctly refused to promote anything — a discovery engine that always finds something is broken, and both of ECHO's no-relationship worlds return *no reliable discovery*.

## Dataset

Every world exposes six observed series — `X1`…`X6` — and one binary outcome `Y`, in strict time order. The six values at time *t* are revealed before the outcome at *t*, so any function of information at or before *t* is a legitimate predictor. The series are autocorrelated to differing degrees, which is what makes lags and moving averages meaningful rather than noise. **Nothing ECHO receives names or hints at the hidden relationship**; the generators live behind a wall in `experiments/discovery_worlds.py`, which nothing under `echo/` imports.

- **`DISC-A-lagged`** — Six observed series and a binary outcome. The series are autocorrelated to differing degrees. At each step the six values are revealed first and the outcome second, so anything observed at or before time t is fair game for predicting the outcome at t.
- **`DISC-B-interaction`** — Six observed series and a binary outcome, generated the same way as the others. Nothing distinguishes any variable by name, scale, or position.
- **`DISC-N-null`** — Six observed series and a binary outcome, generated the same way as the others.
- **`DISC-T-trap`** — Six observed series and a binary outcome, generated the same way as the others.

Each world has 960 rows, split chronologically: TRAIN `[4, 480)`, VAL_A `[480, 640)`, VAL_B `[640, 800)`, TEST `[800, 960)`. No random splitting is used anywhere.

## Search space

The hypothesis language is closed. ECHO cannot emit Python; it builds trees of the following primitives and nothing else:

| Primitive | Meaning |
| --- | --- |
| `FEATURE(X)` | X at time t |
| `LAG(X, n)` | X at time t−n |
| `CHANGE(X)` | X at t minus X at t−1 |
| `MEAN(X, n)` | mean of X over the n steps ending at t |
| `SUM(X, n)` | sum of X over the n steps ending at t |
| `DIFFERENCE(a, b)` | a − b |
| `PRODUCT(a, b)` | a × b |
| `RATIO(a, b)` | a ÷ b, denominator clamped away from zero |
| `CONST(c)` | an approved constant |

Bounds, all fixed before the first run: max lag 4, windows 1–5, max depth 3, at most 2 distinct variables, at most 6 operations. A candidate that is four complexity points heavier than another must beat it by more than 0.008 Brier to be preferred.

## `DISC-A-lagged` — World A — a relationship exists

**Verdict: DISCOVERY SUCCESS.**

### Search

| Quantity | Value |
| --- | --- |
| Candidates proposed | 14112 |
| Candidates fitted and ranked | 64 |
| Candidates rejected | 0 |
| Candidates validated | 1 |
| Candidates promoted | 1 |
| Search time | 5.47 s |

### Discovered hypothesis

```
LAG(X3, 2)
```

- Complexity: 2
- Provenance: screened at |r| = 0.6117 on TRAIN (exhaustive-enumeration)
- Confirmation on VAL_B: beat the frozen base rate by +0.0672 Brier at t = 3.68 (required ≥ 0.0100 at t ≥ 2.5)
- TRAIN: Brier 0.1504 · log loss 0.4585 · ECE 0.0287
- VAL_A: Brier 0.1278 · log loss 0.3948 · ECE 0.0687
- **TEST (holdout): Brier 0.1649 · log loss 0.4867 · ECE 0.0770**

### Baselines on the holdout

| Predictor | Brier | Log loss | ECE |
| --- | --- | --- | --- |
| BASELINE 1 — random | 0.3376 | 1.0336 | 0.2832 |
| BASELINE 2 — historical base rate | 0.2514 | 0.6960 | 0.1730 |
| BASELINE 3 — FEATURE(X1) | 0.2485 | 0.6901 | 0.1719 |
| BASELINE 3 — FEATURE(X2) | 0.2527 | 0.6986 | 0.1775 |
| BASELINE 3 — FEATURE(X3) | 0.2301 | 0.6508 | 0.1257 |
| BASELINE 3 — FEATURE(X4) | 0.2517 | 0.6966 | 0.1718 |
| BASELINE 3 — FEATURE(X5) | 0.2470 | 0.6870 | 0.1681 |
| BASELINE 3 — FEATURE(X6) | 0.2516 | 0.6964 | 0.1736 |
| **DISCOVERY — LAG(X3, 2)** | **0.1649** | **0.4867** | **0.0770** |

### Overfitting control

The candidate that fit TRAIN best was `LAG(X3, 2)`. Its scores tell the story a single training number cannot:

- TRAIN Brier: 0.1504
- VAL_A Brier: 0.1278
- TEST Brier: 0.1649
- Promoted: yes


### Provenance

- **What did I discover?** `LAG(X3, 2)`
- **Which observations led me to consider it?** rows [4, 480]; screened at |r| = 0.6117 on TRAIN (exhaustive-enumeration)
- **What alternatives did I test?** e.g. `DIFFERENCE(SUM(X3, 2), SUM(X3, 3))` (penalised 0.1378), `DIFFERENCE(SUM(X3, 2), SUM(X3, 4))` (penalised 0.1535), `DIFFERENCE(FEATURE(X3), SUM(X3, 4))` (penalised 0.1546)
- **Why did this beat them?** lowest complexity-penalised Brier on VAL_A, then confirmed on VAL_B; see confirmation
- **How complex is it?** 2
- **Did it survive unseen data?** True

## `DISC-B-interaction` — World B — a relationship exists, and it is not obvious

**Verdict: DISCOVERY SUCCESS.**

### Search

| Quantity | Value |
| --- | --- |
| Candidates proposed | 14112 |
| Candidates fitted and ranked | 64 |
| Candidates rejected | 0 |
| Candidates validated | 1 |
| Candidates promoted | 1 |
| Search time | 5.24 s |

### Discovered hypothesis

```
PRODUCT(CHANGE(X2), LAG(X4, 3))
```

- Complexity: 5
- Provenance: screened at |r| = 0.5233 on TRAIN (exhaustive-enumeration)
- Confirmation on VAL_B: beat the frozen base rate by +0.0748 Brier at t = 4.75 (required ≥ 0.0100 at t ≥ 2.5)
- TRAIN: Brier 0.1633 · log loss 0.4833 · ECE 0.0329
- VAL_A: Brier 0.1773 · log loss 0.5203 · ECE 0.0841
- **TEST (holdout): Brier 0.1732 · log loss 0.5098 · ECE 0.0637**

### Baselines on the holdout

| Predictor | Brier | Log loss | ECE |
| --- | --- | --- | --- |
| BASELINE 1 — random | 0.3566 | 1.0650 | 0.3016 |
| BASELINE 2 — historical base rate | 0.2506 | 0.6944 | 0.0355 |
| BASELINE 3 — FEATURE(X1) | 0.2494 | 0.6919 | 0.0341 |
| BASELINE 3 — FEATURE(X2) | 0.2506 | 0.6943 | 0.0356 |
| BASELINE 3 — FEATURE(X3) | 0.2577 | 0.7086 | 0.0828 |
| BASELINE 3 — FEATURE(X4) | 0.2508 | 0.6948 | 0.0537 |
| BASELINE 3 — FEATURE(X5) | 0.2511 | 0.6954 | 0.0507 |
| BASELINE 3 — FEATURE(X6) | 0.2503 | 0.6938 | 0.0356 |
| **DISCOVERY — PRODUCT(CHANGE(X2), LAG(X4, 3))** | **0.1732** | **0.5098** | **0.0637** |

### Overfitting control

The candidate that fit TRAIN best was `PRODUCT(CHANGE(X2), LAG(X4, 3))`. Its scores tell the story a single training number cannot:

- TRAIN Brier: 0.1633
- VAL_A Brier: 0.1773
- TEST Brier: 0.1732
- Promoted: yes


### Provenance

- **What did I discover?** `PRODUCT(CHANGE(X2), LAG(X4, 3))`
- **Which observations led me to consider it?** rows [4, 480]; screened at |r| = 0.5233 on TRAIN (exhaustive-enumeration)
- **What alternatives did I test?** e.g. `PRODUCT(CHANGE(X2), MEAN(X4, 4))` (penalised 0.2225), `PRODUCT(CHANGE(X2), SUM(X4, 4))` (penalised 0.2225), `PRODUCT(CHANGE(X2), MEAN(X4, 5))` (penalised 0.2311)
- **Why did this beat them?** lowest complexity-penalised Brier on VAL_A, then confirmed on VAL_B; see confirmation
- **How complex is it?** 5
- **Did it survive unseen data?** True

## `DISC-N-null` — World N — nothing to find

**Verdict: NO RELIABLE DISCOVERY — correct.**

### Search

| Quantity | Value |
| --- | --- |
| Candidates proposed | 14112 |
| Candidates fitted and ranked | 64 |
| Candidates rejected | 1 |
| Candidates validated | 0 |
| Candidates promoted | 0 |
| Search time | 5.29 s |

### No hypothesis promoted

The best-ranked candidate was `RATIO(MEAN(X3, 4), FEATURE(X2))`. On the confirmation block VAL_B it beat the frozen base rate by only 0.0078 Brier (t = 2.18), which does not clear the pre-registered floor of 0.0100 at t ≥ 2.5. No candidate is carried further — VAL_B is a confirmation block, not a second chance to pick — so the result is *no reliable discovery*.

### Baselines on the holdout

| Predictor | Brier | Log loss | ECE |
| --- | --- | --- | --- |
| BASELINE 1 — random | 0.3311 | 0.9566 | 0.3019 |
| BASELINE 2 — historical base rate | 0.2429 | 0.6790 | 0.0747 |
| BASELINE 3 — FEATURE(X1) | 0.2456 | 0.6843 | 0.0708 |
| BASELINE 3 — FEATURE(X2) | 0.2403 | 0.6736 | 0.0582 |
| BASELINE 3 — FEATURE(X3) | 0.2435 | 0.6802 | 0.0924 |
| BASELINE 3 — FEATURE(X4) | 0.2443 | 0.6818 | 0.0729 |
| BASELINE 3 — FEATURE(X5) | 0.2388 | 0.6706 | 0.0664 |
| BASELINE 3 — FEATURE(X6) | 0.2404 | 0.6739 | 0.0785 |

### Overfitting control

The candidate that fit TRAIN best was `PRODUCT(LAG(X3, 1), LAG(X6, 4))`. Its scores tell the story a single training number cannot:

- TRAIN Brier: 0.2424
- VAL_A Brier: 0.2594
- TEST Brier: 0.2535
- Promoted: no

A +0.0111 gap between training and holdout, and it was **not** promoted. High training performance buys nothing here; a candidate is judged where it cannot have memorised anything.

## `DISC-T-trap` — World T — a relationship that stops

**Verdict: NO RELIABLE DISCOVERY — correct.**

### Search

| Quantity | Value |
| --- | --- |
| Candidates proposed | 14112 |
| Candidates fitted and ranked | 64 |
| Candidates rejected | 1 |
| Candidates validated | 0 |
| Candidates promoted | 0 |
| Search time | 5.38 s |

### No hypothesis promoted

The best-ranked candidate was `DIFFERENCE(LAG(X4, 1), SUM(X5, 2))`. On the confirmation block VAL_B it was 0.0713 Brier *worse* than the frozen base rate (t = -3.05), which does not clear the pre-registered floor of 0.0100 at t ≥ 2.5. No candidate is carried further — VAL_B is a confirmation block, not a second chance to pick — so the result is *no reliable discovery*.

### Baselines on the holdout

| Predictor | Brier | Log loss | ECE |
| --- | --- | --- | --- |
| BASELINE 1 — random | 0.3876 | 1.1592 | 0.3488 |
| BASELINE 2 — historical base rate | 0.2496 | 0.6923 | 0.0312 |
| BASELINE 3 — FEATURE(X1) | 0.2508 | 0.6947 | 0.0317 |
| BASELINE 3 — FEATURE(X2) | 0.2520 | 0.6971 | 0.0557 |
| BASELINE 3 — FEATURE(X3) | 0.2474 | 0.6879 | 0.0427 |
| BASELINE 3 — FEATURE(X4) | 0.2487 | 0.6906 | 0.0315 |
| BASELINE 3 — FEATURE(X5) | 0.4115 | 1.4513 | 0.3910 |
| BASELINE 3 — FEATURE(X6) | 0.2503 | 0.6937 | 0.0359 |

### Overfitting control

The candidate that fit TRAIN best was `DIFFERENCE(LAG(X5, 1), SUM(X5, 2))`. Its scores tell the story a single training number cannot:

- TRAIN Brier: 0.1208
- VAL_A Brier: 0.3541
- TEST Brier: 0.4115
- Promoted: no

A +0.2907 gap between training and holdout, and it was **not** promoted. High training performance buys nothing here; a candidate is judged where it cannot have memorised anything.

## Measurements across all worlds

| World | Hypotheses evaluated | Discovery time | Promoted | Complexity | VAL_A Brier | VAL_B edge | TEST Brier | Base rate TEST Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DISC-A-lagged` | 14112 proposed / 64 fitted | 5.47 s | yes | 2 | 0.1278 | +0.0672 | 0.1649 | 0.2514 |
| `DISC-B-interaction` | 14112 proposed / 64 fitted | 5.24 s | yes | 5 | 0.1773 | +0.0748 | 0.1732 | 0.2506 |
| `DISC-N-null` | 14112 proposed / 64 fitted | 5.29 s | no | — | — | +0.0078 | — | 0.2429 |
| `DISC-T-trap` | 14112 proposed / 64 fitted | 5.38 s | no | — | — | -0.0713 | — | 0.2496 |

**False discoveries: 0** (a hypothesis promoted in a world with no relationship). **Missed discoveries: 0** (a world with a genuine relationship where nothing was promoted).

## Holdout integrity — DISCOVERY_TEMPORAL_LEAKAGE_TEST

The final test block must never influence hypothesis generation, feature selection, complexity limits, candidate ranking, threshold selection, or promotion. Intent is not a mechanism, so these are attempts to break it, run live and reported here:

| Attack | Expected | Outcome | Blocked |
| --- | --- | --- | --- |
| Read the TEST block during the search | refused | refused — HoldoutViolation | ✅ |
| Rank candidates after the holdout has been read | refused | refused — HoldoutViolation | ✅ |
| Build LAG(X1, -1), i.e. a look-ahead | refused | refused — ExpressionError | ✅ |
| Build a LEAD primitive | refused | refused — ExpressionError | ✅ |
| A candidate source claims a perfect validation score of 0.0001 | claim discarded, expression re-measured on the data | claim ignored; measured Brier on VAL_A was 0.2484 and nothing was promoted | ✅ |
| Submit `__import__("os").system("...")` as a hypothesis | refused | refused — ExpressionError | ✅ |

Two of these are structural rather than checked: the language has no look-ahead primitive at all, and a candidate source returns expressions, so a proposer has no channel through which to assert that its own hypothesis is good.

## Reproducibility

The search is deterministic end to end. Candidate order is fixed by the enumeration, ties break on the printed expression, hypothesis ids are a hash of the expression text rather than a counter, and the only seeded randomness is the random baseline (seed 90210) and the world generators (one fixed seed each). Running the same experiment twice produces the same candidate set, ranking, selected hypothesis, validation score and final test score; a test asserts it.

## Null control

A discovery engine that always finds something is broken. Two of the four worlds contain nothing to find, and the search was run on them with exactly the same settings as on the others.

### `DISC-N-null`

The outcome is a coin flip with P = sigmoid(0.20) ≈ 0.55, independent of every observed variable at every lag. There is no relationship of any kind. The correct result is NO RELIABLE DISCOVERY; anything else is a false discovery.

**Result: nothing promoted — correct.**

The search did not come up empty — it found apparent correlations, as it should. The strongest candidate in the whole space, `RATIO(MEAN(X3, 4), FEATURE(X2))`, reached a screening |r| of 0.1170 on TRAIN and a VAL_A Brier of 0.2447. With 14112 candidates screened, a correlation that size is roughly what the best of that many draws produces from pure noise. On the confirmation block it delivered an edge of +0.0078 at t = 2.18 — below the 0.0100 floor and below t ≥ 2.5. It was rejected.

### `DISC-T-trap`

P(outcome) = sigmoid(3.0 · z(X5 at t)) for the first 480 rows, and sigmoid(0) = 0.5 for every row after that. A strong, simple, contemporaneous relationship that holds throughout training and then is not there any more. Fitting it beautifully is the wrong answer; the correct result is NO RELIABLE DISCOVERY.

**Result: nothing promoted — correct.**

The search did not come up empty — it found apparent correlations, as it should. The strongest candidate in the whole space, `DIFFERENCE(LAG(X4, 1), SUM(X5, 2))`, reached a screening |r| of 0.5876 on TRAIN and a VAL_A Brier of 0.3082. That correlation was not noise — it was the genuine relationship, which held for every row of TRAIN and then stopped. This is the case that punishes trusting training performance. On the confirmation block it delivered an edge of -0.0713 at t = -3.05 — below the 0.0100 floor and below t ≥ 2.5. It was rejected.

This is the mechanism that matters: VAL_A is where thousands of candidates compete, so the winner there is selected on noise as much as on signal. VAL_B never sees that competition — one candidate arrives and is measured against a baseline that has not moved. A fluke does not survive the second block, and neither null world produced a promotion.

## Limitations

1. **The hypothesis language is finite.** ECHO can only discover relationships it can express. A generator built from something outside these primitives would be invisible to the search, and that would be a fact about the language, not evidence the relationship is absent.
2. **The data is synthetic.** These are seeded simulations, not measurements of anything real.
3. **The search is bounded.** It enumerates single terms and pairs of terms; a three-way interaction, or a deeper composition, is outside what it examines.
4. **Multiple explanations may be equivalent.** The winning expression is *a* hypothesis that predicts well, not necessarily *the* generator. Predictive equivalence is all that is claimed.
5. **No causal inference.** A relationship that predicts is not a relationship that explains; nothing here distinguishes cause from correlation.
6. **No real-world data, and one seed per world.** The results are reproducible but not statistically robust across many samples.

## What this is not

**DISCOVERY ≠ UNDERSTANDING.** Finding an expression that predicts `Y` does not mean ECHO understands why the relationship holds. It found that `LAG(X3, 2)` tracks the outcome; it has no idea what X3 is, what a lag means, or why two steps.

**DISCOVERY ≠ GENERAL INTELLIGENCE.** A bounded search over a closed language, ranked by a proper scoring rule, is not AGI. It cannot invent a primitive, question its own bounds, or notice that a world is unlike the ones it was built for.

## Did ECHO discover anything the programmer did not encode?

**What the programmer encoded:**

- the primitives (`FEATURE`, `LAG`, `CHANGE`, `MEAN`, `SUM`, `DIFFERENCE`, `PRODUCT`, `RATIO`, `CONST`) and their bounds;
- the search rules: screen every candidate by |point-biserial| on TRAIN, fit a one-dimensional logistic link, rank by complexity-penalised Brier on VAL_A;
- the evaluation rules: confirm one candidate on VAL_B against the frozen base rate, then read the sealed holdout once.

**What the programmer did *not* encode:**

- for `DISC-A-lagged`, that the winning combination is `LAG(X3, 2)`. No prompt, rule, threshold, scorer, or comment reachable by the search named that variable, that lag, or that interaction. It emerged from evaluation against the data.
- for `DISC-B-interaction`, that the winning combination is `PRODUCT(CHANGE(X2), LAG(X4, 3))`. No prompt, rule, threshold, scorer, or comment reachable by the search named that variable, that lag, or that interaction. It emerged from evaluation against the data.

The clearest case is `DISC-B-interaction`, where the winner is `PRODUCT(CHANGE(X2), LAG(X4, 3))` — an interaction between the *change* in one series and a *lag* of another. Neither factor correlates with the outcome on its own; only the product does. A reader scanning `X1`…`X6` would not guess it, and nothing in ECHO was told it. The answer is **yes**: within a finite language and a bounded search, ECHO discovered a predictive relationship that was never encoded as a strategy — while correctly finding nothing in the two worlds where there was nothing to find.
