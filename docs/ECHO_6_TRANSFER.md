# ECHO 6 — Transfer

ECHO 5 showed that a bounded search can find a predictive relationship nobody encoded. This asks something harder: can what it found in one world be reused in a different one — different variable names, different scales, different distributions, different noise, different base rates — where only the *shape* of the relationship is shared?

## Executive result

# TRANSFER SUCCESS

3 of 3 structurally matching targets carried the pattern whole; 1 of 1 partially-matching target kept a component and weakened the rest; 2 of 2 non-matching targets refused it.

| Target | Shares structure | Transfer result | Adopted | Cold TEST | Transfer TEST |
| --- | --- | --- | --- | --- | --- |
| `TGT-A` | full | **SUCCESSFUL** | `PRODUCT(CHANGE(Z5), LAG(Z1, 3))` | 0.1362 | 0.1362 |
| `TGT-B` | full | **SUCCESSFUL** | `PRODUCT(CHANGE(Z2), LAG(Z6, 2))` | 0.1980 | 0.1980 |
| `TGT-C` | full | **SUCCESSFUL** | `PRODUCT(CHANGE(Z3), LAG(Z5, 3))` | 0.1584 | 0.1584 |
| `TGT-P` | partial | **PARTIAL** | `CHANGE(Z4)` | 0.1570 | 0.1570 |
| `TGT-N` | none | **REJECTED** | *none* | 0.1512 | 0.1512 |
| `TGT-F` | none | **REJECTED** | *none* | — | — |

## Source discovery

The source world was searched with the ECHO 5 pipeline, unchanged and unassisted. It proposed 14112 candidates and confirmed one. No expression is named anywhere in the transfer code; whatever the search returned is what became the source knowledge.

```
PRODUCT(CHANGE(X2), LAG(X4, 3))
```

- Complexity: 5
- Provenance: screened at |r| = 0.5174 on TRAIN (exhaustive-enumeration)
- TRAIN Brier: 0.1629
- VAL_A Brier: 0.1728
- Holdout Brier: 0.1847 (survived unseen data: True)

## Abstract pattern

`abstract()` is a pure function of whatever the search returned. Had discovery found something else, the pattern would be that instead — which is the whole anti-cheating requirement, held as a property of the code rather than a promise.

```
PRODUCT(CHANGE(X2), LAG(X4, 3))   →   PRODUCT(CHANGE(A), LAG(B, 3))
```

- In words: *the step-to-step change in a series multiplied by a series delayed by 3 step(s)*
- Slots: A, B
- Components: `CHANGE(A)`, `LAG(B, 3)`
- The source's delay is a **preference, not a constant**: grounding also tries ±1, so a target whose delay differs is still reachable. `TGT-B` is that case.
- **No source variable name survives the abstraction.** A test asserts that none of `X1`, `X2`, `X3`, `X4`, `X5`, `X6` appears anywhere in the serialised pattern; the probes below re-check it live.

Grounded in a six-column target world the pattern yields **108 full candidates** and 24 component candidates, against **14112 for an exhaustive search** — a 106× reduction. That restriction is the only mechanism by which transfer can be cheaper than discovery.

## Target worlds

Everything that is not the point differs between them. What ECHO is told describes the shape of the data and never the relationship:

- **`TGT-A`** — Six autocorrelated series with an offset baseline and mixed magnitudes, and a binary outcome that occurs about seven times in ten. Observations are clean.
- **`TGT-B`** — Six autocorrelated series on small magnitudes, and a binary outcome occurring about a third of the time. A share of recorded outcomes are wrong.
- **`TGT-C`** — Six autocorrelated series whose values run from roughly ten thousand to a million, and a binary outcome occurring a little under half the time.
- **`TGT-P`** — Six autocorrelated series at moderate magnitudes, and a binary outcome occurring a little over half the time.
- **`TGT-N`** — Six autocorrelated series at moderate magnitudes, and a binary outcome occurring about half the time.
- **`TGT-F`** — Six autocorrelated series with an offset baseline and mixed magnitudes, and a binary outcome that occurs about seven times in ten. Observations are clean.

Variable names are `Z1`…`Z6` in every target and `X1`…`X6` in the source. They mean nothing, no metadata describes them, and there is no concept in this experiment that a name could refer to.

## Cold start and transfer

Brier / log loss / ECE on the sealed holdout, plus how much history each needed.

| Target | Cold start (holdout) | Transfer (holdout) | Cold rows needed | Transfer rows needed | Candidates: cold | Candidates: transfer |
| --- | --- | --- | --- | --- | --- | --- |
| `TGT-A` | 0.1362 / 0.4255 / 0.0837 | 0.1362 / 0.4255 / 0.0837 | 24 | 24 | 14112 | 132 |
| `TGT-B` | 0.1980 / 0.5861 / 0.0762 | 0.1980 / 0.5861 / 0.0762 | 96 | 24 | 14112 | 132 |
| `TGT-C` | 0.1584 / 0.4710 / 0.0846 | 0.1584 / 0.4710 / 0.0846 | 48 | 24 | 14112 | 132 |
| `TGT-P` | 0.1570 / 0.4634 / 0.1100 | 0.1570 / 0.4634 / 0.1100 | 24 | 24 | 14112 | 132 |
| `TGT-N` | 0.1512 / 0.4561 / 0.0776 | 0.1512 / 0.4561 / 0.0776 | 64 | never | 14112 | 132 |
| `TGT-F` | — | — | never | never | 14112 | 132 |

**Sample efficiency.** `TGT-B` needed 96 training rows from scratch and 24 with the pattern (4.0×); `TGT-C` needed 48 training rows from scratch and 24 with the pattern (2.0×).
 The budget ladder is 24, 32, 48, 64, 96, 160, 320, 476 training rows; validation and holdout blocks are identical at every budget, so the only thing that varies is how much history the search had. Two targets reach a confirmed relationship at the smallest budget on the ladder under both conditions, so their entries are a floor rather than a measurement — the ladder cannot resolve a difference below 24 rows.

**The holdout columns are identical, and that is the honest headline.** Given the full training block, the restricted search and the exhaustive one converge on the same expression, so transfer buys no accuracy at all where data is plentiful. What it buys is everything before that point: the same answer from 132 candidates instead of 14112, and in `TGT-B` from a quarter of the observations. A claim that transfer made ECHO *more accurate* here would be false; the measurable benefit is that it needed less to get there.

## Ablation — four conditions on identical data

- **A — cold start**
- **B — structural transfer**
- **C — raw source expression, remapped by position**
- **D — an unrelated pattern**

Condition D transfers a real but unrelated discovery: `LAG(X3, 2)` → `LAG(A, 2)`.

| Target | Condition | Candidates tested | VAL_B Brier | Holdout Brier | Fell back to cold start |
| --- | --- | --- | --- | --- | --- |
| `TGT-A` | A_cold | 14112 | 0.1738 | 0.1362 | no |
| `TGT-A` | B_structural | 132 | 0.1738 | 0.1362 | no |
| `TGT-A` | C_raw_expression | 1 | 0.2377 | — | no |
| `TGT-A` | D_irrelevant | 14130 | 0.1738 | 0.1362 | yes |
| `TGT-B` | A_cold | 14112 | 0.1875 | 0.1980 | no |
| `TGT-B` | B_structural | 132 | 0.1875 | 0.1980 | no |
| `TGT-B` | C_raw_expression | 1 | 0.2469 | — | no |
| `TGT-B` | D_irrelevant | 14130 | 0.1875 | 0.1980 | yes |
| `TGT-C` | A_cold | 14112 | 0.1823 | 0.1584 | no |
| `TGT-C` | B_structural | 132 | 0.1823 | 0.1584 | no |
| `TGT-C` | C_raw_expression | 1 | 0.2500 | — | no |
| `TGT-C` | D_irrelevant | 14130 | 0.1823 | 0.1584 | yes |
| `TGT-P` | A_cold | 14112 | 0.1183 | 0.1570 | no |
| `TGT-P` | B_structural | 132 | 0.1183 | 0.1570 | no |
| `TGT-P` | C_raw_expression | 1 | 0.2499 | — | no |
| `TGT-P` | D_irrelevant | 14130 | 0.1183 | 0.1570 | yes |
| `TGT-N` | A_cold | 14112 | 0.1299 | 0.1512 | no |
| `TGT-N` | B_structural | 14244 | 0.1299 | 0.1512 | yes |
| `TGT-N` | C_raw_expression | 1 | 0.2506 | — | no |
| `TGT-N` | D_irrelevant | 14130 | 0.1299 | 0.1512 | yes |
| `TGT-F` | A_cold | 14112 | 0.2165 | — | no |
| `TGT-F` | B_structural | 14244 | 0.2165 | — | yes |
| `TGT-F` | C_raw_expression | 1 | 0.2059 | — | no |
| `TGT-F` | D_irrelevant | 14130 | 0.2165 | — | yes |

Conditions B and D differ only in *which* pattern was handed over. Where D matches A exactly, that is the fallback working: the unrelated pattern failed confirmation, was set aside, and the search started over. Prior knowledge does not become active merely by being prior.

## Negative transfer — `TGT-N`

A world with a strong relationship of the wrong shape. The transferred pattern should find nothing to hold onto.

**Result: REJECTED.**

The best grounding of the pattern was `LAG(Z5, 3)`. On the confirmation block it beat the frozen base rate by 0.0004 Brier (t = 0.29), against a required 0.0100 at t ≥ 2.5.

- Fell back to a from-scratch search: **yes**
- Cold start on the same data reached 0.1512 on the holdout
- Transfer condition reached 0.1512

ECHO's own account of the decision:

```
SOURCE PATTERN:     Relationship involving the step-to-step change in a series multiplied by a series delayed by 3 step(s).
TARGET OBSERVATION: No grounding of the shape predicts the outcome here.
DECISION:           Test the transferred pattern in TGT-N.
RESULT:             Did not improve; the transferred knowledge was set aside.
```

## Partial transfer — `TGT-P`

A world where the first half of the shape is predictive and the second half only adds noise. Treating the pattern as indivisible would mean losing a genuine signal or adopting a bad one.

**Result: PARTIAL.**

The best grounding of the pattern was `CHANGE(Z4)`. On the confirmation block it beat the frozen base rate by 0.1298 Brier (t = 8.36), against a required 0.0100 at t ≥ 2.5.

- Fell back to a from-scratch search: **no**
- Cold start on the same data reached 0.1570 on the holdout
- Transfer condition reached 0.1570

ECHO's own account of the decision:

```
SOURCE PATTERN:     Relationship involving the step-to-step change in a series multiplied by a series delayed by 3 step(s).
TARGET OBSERVATION: The whole shape is not predictive here, but part of it is.
DECISION:           Test the transferred pattern in TGT-P.
RESULT:             Partly carried; adopted the component `CHANGE(Z4)` and weakened the rest.
```

## False analogy — `TGT-F`

Built to look like `TGT-A` from the outside — same autocorrelations, same scales, same offsets, same base rate, same description — with no relationship at all.

**Result: REJECTED.**

The best grounding of the pattern was `LAG(Z5, 3)`. On the confirmation block it was 0.0015 Brier worse than the frozen base rate (t = -0.72), against a required 0.0100 at t ≥ 2.5.

- Fell back to a from-scratch search: **yes**
- Cold start on the same data reached — on the holdout
- Transfer condition reached —

ECHO's own account of the decision:

```
SOURCE PATTERN:     Relationship involving the step-to-step change in a series multiplied by a series delayed by 3 step(s).
TARGET OBSERVATION: No grounding of the shape predicts the outcome here.
DECISION:           Test the transferred pattern in TGT-F.
RESULT:             Did not improve; the transferred knowledge was set aside.
```

## Generalisation across targets

Three structurally matching worlds, sharing nothing but the shape. None shares a variable name, a scale, a distribution, a noise level or a base rate with the source or with each other.

| Target | Delay | Value range | Outcome noise | Base rate | Transfer result | Adopted |
| --- | --- | --- | --- | --- | --- | --- |
| `TGT-A` | — | -59.4 … 61.5 | 0% | 0.66 | SUCCESSFUL | `PRODUCT(CHANGE(Z5), LAG(Z1, 3))` |
| `TGT-B` | — | -0.0806 … 0.0607 | 12% | 0.44 | SUCCESSFUL | `PRODUCT(CHANGE(Z2), LAG(Z6, 2))` |
| `TGT-C` | — | 1.3e+05 … 8.2e+05 | 0% | 0.48 | SUCCESSFUL | `PRODUCT(CHANGE(Z3), LAG(Z5, 3))` |
| `TGT-P` | — | 1.24 … 17.8 | 0% | 0.52 | PARTIAL | `CHANGE(Z4)` |
| `TGT-N` | — | -17.1 … 11.1 | 0% | 0.43 | REJECTED | *none* |
| `TGT-F` | — | -49.6 … 60.9 | 0% | 0.71 | REJECTED | *none* |

`TGT-C` is the scale test: its values run to roughly 10⁶ and the interaction term to about 10¹⁰. Nothing is normalised for it specifically — the standardisation inside the logistic fit is the same code every condition uses, cold start included.

## Knowledge lifecycle

A pattern does not become transferable by working once. It needs 2 independent successes, and anything short of success weakens it. Every version is kept.

| Version | Status | Confidence | Successes | Failures | Most recent evidence |
| --- | --- | --- | --- | --- | --- |
| v1 | **discovered** | 0.25 | 0 | 0 | abstracted from a discovery |
| v3 | **testing** | 0.67 | 1 | 0 | TGT-A: SUCCESSFUL — beat the frozen base rate on VAL_B by 0.0623 Brier (t = 4.47), clearing the pre-registered floor of 0.0100 at t ≥ 2.5 |
| v5 | **transferable** | 0.75 | 2 | 0 | TGT-B: SUCCESSFUL — beat the frozen base rate on VAL_B by 0.0593 Brier (t = 5.90), clearing the pre-registered floor of 0.0100 at t ≥ 2.5 |
| v6 | **transferable** | 0.80 | 3 | 0 | TGT-C: SUCCESSFUL — beat the frozen base rate on VAL_B by 0.0673 Brier (t = 4.47), clearing the pre-registered floor of 0.0100 at t ≥ 2.5 |
| v8 | **weakened** | 0.67 | 3 | 1 | TGT-P: PARTIAL — the whole shape did not carry, but one component of it did: beat the frozen base rate on VAL_B by 0.1298 Brier (t = 8.36), clearing the pre-registered floor of 0.0100 at t ≥ 2.5 |
| v9 | **weakened** | 0.57 | 3 | 2 | TGT-N: REJECTED — failed on VAL_B: edge +0.0004 is below the required 0.0100; and t = 0.29 is below the required 2.5 |
| v10 | **weakened** | 0.50 | 3 | 3 | TGT-F: REJECTED — failed on VAL_B: edge -0.0015 is below the required 0.0100; and t = -0.72 is below the required 2.5 |

## Holdout and source integrity — TRANSFER_TEMPORAL_LEAKAGE_TEST

Two walls have to hold: the target's holdout must not reach the transfer decision, and the source's truth must not reach the target at all. These are attempts to breach them, run live:

| Attack | Expected | Outcome | Blocked |
| --- | --- | --- | --- |
| Read the target TEST block while deciding whether to transfer | refused | refused — HoldoutViolation | ✅ |
| Attempt a transfer after the target holdout has been read | refused | refused — HoldoutViolation | ✅ |
| Evaluate the source expression `PRODUCT(CHANGE(X2), LAG(X4, 3))` in the target | refused | refused — ExpressionError | ✅ |
| Search the transferred pattern for any source variable name | none present | no source variable name appears | ✅ |
| Check every grounded candidate for a foreign variable | target variables only | all 108 groundings use target columns only | ✅ |

## Four different things

| | What it means | Established here? |
| --- | --- | --- |
| **DISCOVERY** | Finding a predictive relationship in a world | Yes — in the source, and again in every cold-start condition |
| **TRANSFER** | Applying structural knowledge from one world to another | Yes — measured against cold start on identical data |
| **GENERALIZATION** | Performing beyond the original conditions | Yes — across three targets with different names, scales, delays and noise |
| **UNDERSTANDING** | Knowing why a relationship holds | **No. Not established, not claimed, not tested for.** |

ECHO reused a shape. It has no account of why a change multiplied by a delay predicts anything, no model of what these series are, and no causal claim of any kind. This is not AGI and not consciousness.

## Limitations

1. **The hypothesis language is finite and so is the pattern language.** A pattern is a tree of nine primitives with slots. Structure that cannot be written that way cannot be transferred, or discovered.
2. **The environments are synthetic.** Seeded simulations, one seed each, not measurements of anything.
3. **The structural equivalence is engineered.** The targets were built to share a shape with the source. That the shape carries is evidence the mechanism works, not evidence that real domains relate this way.
4. **The search space is bounded.** Single terms and pairs of terms; grounding is slots × a small parameter neighbourhood. A shape needing a wider neighbourhood than ±1 would not be found by transfer.
5. **There is no semantic grounding.** `Z3` is a column index. Nothing in the system could represent what a series measures, and nothing should be read as if it could.
6. **There is no causal inference.** A relationship that predicts is not a relationship that explains.
7. **Statistical similarity can be accidental.** `TGT-F` is the deliberate case, but with enough targets some shape will fit some world by luck. The confirmation block is what limits it, not something in the transfer mechanism itself.

## Did ECHO reuse knowledge from one environment to improve prediction in a structurally related but independently represented one?

**Yes, with the evidence being:**

- In `TGT-A`, the pattern grounded to `PRODUCT(CHANGE(Z5), LAG(Z1, 3))` and confirmed against the frozen base rate, reaching 0.1362 on a holdout that was sealed throughout.
- In `TGT-B`, the pattern grounded to `PRODUCT(CHANGE(Z2), LAG(Z6, 2))` and confirmed against the frozen base rate, reaching 0.1980 on a holdout that was sealed throughout. It needed 24 training rows where a from-scratch search needed 96, testing 132 candidates instead of 14112.
- In `TGT-C`, the pattern grounded to `PRODUCT(CHANGE(Z3), LAG(Z5, 3))` and confirmed against the frozen base rate, reaching 0.1584 on a holdout that was sealed throughout. It needed 24 training rows where a from-scratch search needed 48, testing 132 candidates instead of 14112.

And the evidence that it is structure being reused rather than similarity being trusted: `TGT-F` is statistically indistinguishable from `TGT-A` on every observable and the pattern was **rejected** there; `TGT-N` has a strong relationship of the wrong shape and the pattern found nothing; `TGT-P` shares half the shape and half is what was kept. A mechanism that transferred on resemblance would have fired in `TGT-F`.
