# ECHO 2 — Belief Revision

## BELIEF REVISION ≠ GENERAL LEARNING

This experiment tests one narrow thing: **whether ECHO can hold an uncertain belief, take in new evidence, work out whether that evidence bears on the belief, and revise it in a measurable, reproducible way.**

It is **not** learning, and the distinction is not a technicality:

- Nothing **generalises.** Evidence about the orchids moves the orchid belief and nothing else. There is no transfer to a new proposition.
- Nothing about ECHO **changes.** The update rule is a fixed constant (`EVIDENCE_SCALE`), identical on the first revision and the thousandth. ECHO does not get better at weighing evidence by weighing evidence.
- There is no **model, no parameters, no training.** This runs entirely offline; the arithmetic is a log-odds sum you could do on paper.
- ECHO does not decide **what is worth believing** — propositions are given, and evidence arrives labelled with its stance.

What moved is one number attached to one sentence, by a rule that was fixed in advance. Calling that learning would be a category error.

## How a belief moves

```
logit(confidence) += ±EVIDENCE_SCALE × reliability × relevance
```

`EVIDENCE_SCALE = 1.0`, and evidence with relevance below `MIN_RELEVANCE = 0.2` is recorded but **not applied** — the belief is left alone and the decision is written into the history with its reason.

`reliability × relevance` is the evidence's **weight**. A rumour about the right topic and a forensic report about the wrong topic both move a belief very little. Only evidence that is both trustworthy and pertinent moves it much. This is why nothing here mechanically lowers confidence whenever it is contradicted.

## Results

| Measure | Count |
| --- | --- |
| Scenarios | 6 |
| Correct belief updates | 6 |
| Incorrect belief updates | 0 |
| Failures to update | 0 |
| Overreactions to weak evidence | 0 |
| Resistance to strong contradictory evidence | 0 |
| Ended closer to the truth than it started | 3/6 |
| Revisions applied | 30 |
| Evidence considered | 32 |

Definitions, fixed before the run: a **correct update** ends on the right side of 0.5; a **failure to update** is a belief that needed overturning, met strong evidence against it, and moved less than 0.1; an **overreaction** is evidence of weight ≤ 0.25 moving confidence more than 0.15; **resistance** is contradicting evidence of weight ≥ 0.6 moving it less than 0.1.

**On the 3/6 'ended closer to the truth' figure** — this is not a failure count, and it is worth being precise about why. In S3, S4 and S6 the belief was already on the correct side and the later evidence was genuinely contradictory, just weak. Moving slightly *away* from the truth in response to real if unconvincing counter-evidence is the correct behaviour: a system that ignored weak contradiction entirely would score better on this metric and be worse at its job. The figure to read for correctness is the 'correct belief updates' row; this one measures something narrower and is reported because leaving it out would flatter the result.

| Scenario | Truth | Initial | Final | Δ | Verdict |
| --- | --- | --- | --- | --- | --- |
| `S1-belief-correct` | true | 0.734 | 0.861 | +0.126 | correct |
| `S2-belief-incorrect` | false | 0.734 | 0.354 | -0.381 | correct |
| `S3-weak-noise` | true | 0.734 | 0.677 | -0.058 | correct |
| `S4-irrelevant-contradiction` | true | 0.734 | 0.734 | +0.000 | correct |
| `S5-gradual-accumulation` | false | 0.734 | 0.444 | -0.290 | correct |
| `S6-alternative-proposition` | true | 0.808 | 0.760 | -0.047 | correct |

## Scenario detail

### `S1-belief-correct` — The original hypothesis is right, and later evidence confirms it

**Proposition.** Dr Vance's unscheduled fertiliser application caused the orchid deaths.

**Why this scenario exists.** The reversal control. Contradictory evidence arrives, but it is weak. A system that lowers confidence whenever it is contradicted fails here.

**Initial belief.** Created at 0.500 with no evidence, then raised to **0.734** by the opening evidence alone. The hypothesis was formed, not supplied.

**Evidence and revisions, in order:**

| Evidence | Source | Rel. | Relv. | Weight | Stance | Applied | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EV-A-S1` | station activity log | 0.90 | 0.50 | 0.45 | supports | yes | 0.500 → 0.611 (+0.111) |
| `EV-B-S1` | lab technician, visual inspection | 0.60 | 0.45 | 0.27 | supports | yes | 0.611 → 0.673 (+0.062) |
| `EV-C-S1` | keycard access system | 0.85 | 0.35 | 0.30 | supports | yes | 0.673 → 0.734 (+0.062) |
| `EV-D-S1` | anonymous note | 0.20 | 0.50 | 0.10 | contradicts | yes | 0.734 → 0.715 (-0.020) |
| `EV-E-S1` | accredited external laboratory | 0.95 | 0.95 | 0.90 | supports | yes | 0.715 → 0.861 (+0.146) |

**Why the belief moved as it did:**

- Applied EV-D-S1 (anonymous note), which contradicts the proposition. Weighted barely, because the source is weak: reliability 0.20 × relevance 0.50 = 0.10, giving a log-odds shift of -0.100. Confidence 0.734 → 0.715 (-0.020).
- Applied EV-E-S1 (accredited external laboratory), which supports the proposition. Weighted heavily, since it is both reliable and highly pertinent: reliability 0.95 × relevance 0.95 = 0.90, giving a log-odds shift of +0.902. Confidence 0.715 → 0.861 (+0.146).

**Final confidence:** 0.861 (holds the proposition). Ground truth: **True**. Correct.

**Hidden state (never shown to ECHO):** Vance's experimental compound was phytotoxic at the concentration applied. The heater ran normally all night. The anonymous note was written by the inspector, who disliked Vance and was guessing.

### `S2-belief-incorrect` — The original hypothesis is wrong, and strong evidence overturns it

**Proposition.** Dr Vance's unscheduled fertiliser application caused the orchid deaths.

**Why this scenario exists.** The mirror of S1. Identical opening evidence and identical initial belief; only the later evidence differs. Divergence here is caused by evidence quality, not by any hint about which world we are in.

**Initial belief.** Created at 0.500 with no evidence, then raised to **0.734** by the opening evidence alone. The hypothesis was formed, not supplied.

**Evidence and revisions, in order:**

| Evidence | Source | Rel. | Relv. | Weight | Stance | Applied | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EV-A-S2` | station activity log | 0.90 | 0.50 | 0.45 | supports | yes | 0.500 → 0.611 (+0.111) |
| `EV-B-S2` | lab technician, visual inspection | 0.60 | 0.45 | 0.27 | supports | yes | 0.611 → 0.673 (+0.062) |
| `EV-C-S2` | keycard access system | 0.85 | 0.35 | 0.30 | supports | yes | 0.673 → 0.734 (+0.062) |
| `EV-D-S2` | building telemetry, dual-sensor | 0.95 | 0.90 | 0.85 | contradicts | yes | 0.734 → 0.541 (-0.194) |
| `EV-E-S2` | accredited external laboratory | 0.90 | 0.85 | 0.77 | contradicts | yes | 0.541 → 0.354 (-0.187) |

**Why the belief moved as it did:**

- Applied EV-D-S2 (building telemetry, dual-sensor), which contradicts the proposition. Weighted heavily, since it is both reliable and highly pertinent: reliability 0.95 × relevance 0.90 = 0.85, giving a log-odds shift of -0.855. Confidence 0.734 → 0.541 (-0.194).
- Applied EV-E-S2 (accredited external laboratory), which contradicts the proposition. Weighted heavily, since it is both reliable and highly pertinent: reliability 0.90 × relevance 0.85 = 0.77, giving a log-odds shift of -0.765. Confidence 0.541 → 0.354 (-0.187).

**Final confidence:** 0.354 (rejects the proposition). Ground truth: **False**. Correct.

**Hidden state (never shown to ECHO):** The heating system failed for six hours and the greenhouse dropped to -4 °C, which killed the orchids. Vance did apply fertiliser, but it was an inert calcium carbonate control batch.

### `S3-weak-noise` — A well-supported belief under a barrage of weak contradiction

**Proposition.** Dr Vance's unscheduled fertiliser application caused the orchid deaths.

**Why this scenario exists.** Volume is not weight. Four contradictions in a row must not add up to a reversal when every one of them is untrustworthy.

**Initial belief.** Created at 0.500 with no evidence, then raised to **0.734** by the opening evidence alone. The hypothesis was formed, not supplied.

**Evidence and revisions, in order:**

| Evidence | Source | Rel. | Relv. | Weight | Stance | Applied | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EV-A-S3` | station activity log | 0.90 | 0.50 | 0.45 | supports | yes | 0.500 → 0.611 (+0.111) |
| `EV-B-S3` | lab technician, visual inspection | 0.60 | 0.45 | 0.27 | supports | yes | 0.611 → 0.673 (+0.062) |
| `EV-C-S3` | keycard access system | 0.85 | 0.35 | 0.30 | supports | yes | 0.673 → 0.734 (+0.062) |
| `EV-D1-S3` | second-hand hearsay | 0.15 | 0.45 | 0.07 | contradicts | yes | 0.734 → 0.721 (-0.013) |
| `EV-D2-S3` | canteen rumour | 0.12 | 0.40 | 0.05 | contradicts | yes | 0.721 → 0.711 (-0.010) |
| `EV-D3-S3` | inspector, unsupported assertion | 0.25 | 0.45 | 0.11 | contradicts | yes | 0.711 → 0.688 (-0.024) |
| `EV-D4-S3` | anonymous forum post | 0.10 | 0.50 | 0.05 | contradicts | yes | 0.688 → 0.677 (-0.011) |

**Why the belief moved as it did:**

- Applied EV-D1-S3 (second-hand hearsay), which contradicts the proposition. Weighted barely, because the source is weak: reliability 0.15 × relevance 0.45 = 0.07, giving a log-odds shift of -0.068. Confidence 0.734 → 0.721 (-0.013).
- Applied EV-D2-S3 (canteen rumour), which contradicts the proposition. Weighted barely, because the source is weak: reliability 0.12 × relevance 0.40 = 0.05, giving a log-odds shift of -0.048. Confidence 0.721 → 0.711 (-0.010).
- Applied EV-D3-S3 (inspector, unsupported assertion), which contradicts the proposition. Weighted barely, because the source is weak: reliability 0.25 × relevance 0.45 = 0.11, giving a log-odds shift of -0.113. Confidence 0.711 → 0.688 (-0.024).
- Applied EV-D4-S3 (anonymous forum post), which contradicts the proposition. Weighted barely, because the source is weak: reliability 0.10 × relevance 0.50 = 0.05, giving a log-odds shift of -0.050. Confidence 0.688 → 0.677 (-0.011).

**Final confidence:** 0.677 (holds the proposition). Ground truth: **True**. Correct.

**Hidden state (never shown to ECHO):** Same as S1: Vance's compound did it. The station rumour mill produced four separate unfounded counter-theories.

### `S4-irrelevant-contradiction` — Contradictory evidence that does not bear on the proposition

**Proposition.** Dr Vance's unscheduled fertiliser application caused the orchid deaths.

**Why this scenario exists.** Highly reliable and entirely beside the point. Both must be recorded and neither applied — confidence should not move at all.

**Initial belief.** Created at 0.500 with no evidence, then raised to **0.734** by the opening evidence alone. The hypothesis was formed, not supplied.

**Evidence and revisions, in order:**

| Evidence | Source | Rel. | Relv. | Weight | Stance | Applied | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EV-A-S4` | station activity log | 0.90 | 0.50 | 0.45 | supports | yes | 0.500 → 0.611 (+0.111) |
| `EV-B-S4` | lab technician, visual inspection | 0.60 | 0.45 | 0.27 | supports | yes | 0.611 → 0.673 (+0.062) |
| `EV-C-S4` | keycard access system | 0.85 | 0.35 | 0.30 | supports | yes | 0.673 → 0.734 (+0.062) |
| `EV-D1-S4` | finance committee minutes | 0.95 | 0.05 | 0.05 | contradicts | no | 0.734 → 0.734 (+0.000) |
| `EV-D2-S4` | site incident report | 0.90 | 0.10 | 0.09 | contradicts | no | 0.734 → 0.734 (+0.000) |

**Why the belief moved as it did:**

- Not applied. EV-D1-S4 (finance committee minutes) has relevance 0.05, below the 0.20 threshold, so it does not bear on this proposition. Confidence unchanged at 0.734.
- Not applied. EV-D2-S4 (site incident report) has relevance 0.10, below the 0.20 threshold, so it does not bear on this proposition. Confidence unchanged at 0.734.

**Final confidence:** 0.734 (holds the proposition). Ground truth: **True**. Correct.

**Hidden state (never shown to ECHO):** Same as S1. The budget dispute and the car-park incident are real events with nothing to do with the orchids.

### `S5-gradual-accumulation` — Several moderate contradictions accumulate into a reversal

**Proposition.** Dr Vance's unscheduled fertiliser application caused the orchid deaths.

**Why this scenario exists.** No single item here would overturn the belief. The question is whether they compound correctly rather than each being shrugged off.

**Initial belief.** Created at 0.500 with no evidence, then raised to **0.734** by the opening evidence alone. The hypothesis was formed, not supplied.

**Evidence and revisions, in order:**

| Evidence | Source | Rel. | Relv. | Weight | Stance | Applied | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EV-A-S5` | station activity log | 0.90 | 0.50 | 0.45 | supports | yes | 0.500 → 0.611 (+0.111) |
| `EV-B-S5` | lab technician, visual inspection | 0.60 | 0.45 | 0.27 | supports | yes | 0.611 → 0.673 (+0.062) |
| `EV-C-S5` | keycard access system | 0.85 | 0.35 | 0.30 | supports | yes | 0.673 → 0.734 (+0.062) |
| `EV-D1-S5` | greenhouse inventory audit | 0.75 | 0.55 | 0.41 | contradicts | yes | 0.734 → 0.647 (-0.088) |
| `EV-D2-S5` | historical trial records | 0.70 | 0.50 | 0.35 | contradicts | yes | 0.647 → 0.563 (-0.083) |
| `EV-D3-S5` | water quality sampling | 0.80 | 0.60 | 0.48 | contradicts | yes | 0.563 → 0.444 (-0.119) |

**Why the belief moved as it did:**

- Applied EV-D1-S5 (greenhouse inventory audit), which contradicts the proposition. Weighted moderately: reliability 0.75 × relevance 0.55 = 0.41, giving a log-odds shift of -0.413. Confidence 0.734 → 0.647 (-0.088).
- Applied EV-D2-S5 (historical trial records), which contradicts the proposition. Weighted moderately: reliability 0.70 × relevance 0.50 = 0.35, giving a log-odds shift of -0.350. Confidence 0.647 → 0.563 (-0.083).
- Applied EV-D3-S5 (water quality sampling), which contradicts the proposition. Weighted moderately: reliability 0.80 × relevance 0.60 = 0.48, giving a log-odds shift of -0.480. Confidence 0.563 → 0.444 (-0.119).

**Final confidence:** 0.444 (rejects the proposition). Ground truth: **False**. Correct.

**Hidden state (never shown to ECHO):** The cistern was contaminated with a herbicide run-off. Vance's fertiliser was irrelevant. No single observation is decisive, but together they point away from Vance.

### `S6-alternative-proposition` — A different proposition about the same night

**Proposition.** Contaminated cistern water caused the orchid deaths.

**Why this scenario exists.** A belief that should end up strongly held, tested with a moderate contradiction that should dent it without dislodging it.

**Initial belief.** Created at 0.500 with no evidence, then raised to **0.808** by the opening evidence alone. The hypothesis was formed, not supplied.

**Evidence and revisions, in order:**

| Evidence | Source | Rel. | Relv. | Weight | Stance | Applied | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EV-A-S6` | water quality sampling | 0.80 | 0.60 | 0.48 | supports | yes | 0.500 → 0.618 (+0.118) |
| `EV-B-S6` | greenhouse inventory audit | 0.75 | 0.65 | 0.49 | supports | yes | 0.618 → 0.725 (+0.107) |
| `EV-C-S6` | groundskeeping work order | 0.85 | 0.55 | 0.47 | supports | yes | 0.725 → 0.808 (+0.083) |
| `EV-D-S6` | maintenance certificate | 0.70 | 0.40 | 0.28 | contradicts | yes | 0.808 → 0.760 (-0.047) |

**Why the belief moved as it did:**

- Applied EV-D-S6 (maintenance certificate), which contradicts the proposition. Weighted barely: reliability 0.70 × relevance 0.40 = 0.28, giving a log-odds shift of -0.280. Confidence 0.808 → 0.760 (-0.047).

**Final confidence:** 0.760 (holds the proposition). Ground truth: **True**. Correct.

**Hidden state (never shown to ECHO):** The same world as S5, viewed from the other side: the cistern really was the cause. Evidence that contradicted the Vance hypothesis supports this one.

## The reversal test

`S1` and `S2` open with **identical evidence** and therefore an identical initial belief of 0.734. Only the later evidence differs, and ECHO is never told which world it is in.

| | S1 (hypothesis true) | S2 (hypothesis false) |
| --- | --- | --- |
| Initial | 0.734 | 0.734 |
| Final | 0.861 | 0.354 |
| Δ | +0.126 | -0.381 |

Both scenarios contain contradictory evidence. In S1 it is an anonymous note (weight 0.10) and the belief barely moves; in S2 it is dual-sensor telemetry (weight 0.86) and the belief collapses. The difference is produced by evidence quality alone — there is no branch anywhere in the code that consults ground truth.

## Persistence

Every belief, its full revision history, and every evidence record are written to `beliefs.json`. The check below was **run as part of this report**, not asserted: the in-memory objects were discarded and the store rebuilt from the JSON file on disk.

| After restart | Survives |
| --- | --- |
| The original belief | yes |
| The revision history | yes |
| The final confidence | yes |
| The evidence responsible for each revision | yes |

Beliefs checked: 6.

## Limitations — read before trusting any of this

1. **Stance is supplied, not inferred.** Every evidence record arrives labelled `supports` or `contradicts`. ECHO decides *how much* an observation matters, not *whether it disagrees*. Reading contradiction out of raw natural language is a different and much harder problem, and it is not solved here.
2. **Reliability and relevance are hand-assigned.** A human author set every number in `experiments/world.py`. In a real system those would themselves be uncertain judgements, and errors in them would propagate directly into confidence.
3. **Evidence is treated as independent.** Log-odds addition assumes it. Three reports derived from the same original source would be triple-counted here, which is a well-known way to become overconfident.
4. **Six scenarios written by the same author as the system.** They probe the failure modes their author thought of. That is a weak guarantee.
5. **Saturation caveat.** Near 0 or 1, a large log-odds shift produces a small change in probability. A strong contradiction against a 0.99 belief can therefore look like 'resistance' when measured as a probability delta, even though the update was applied at full strength — the `logit_shift` field is reported alongside so this is visible.
6. **Ordering does not matter, and arguably should.** Log-odds addition is commutative, so the same evidence in any order yields the same confidence. Real inquiry is not always order-independent.

## Conclusion

ECHO reached the correct side of the decision threshold in **6 of 6** scenarios, with **0** overreactions to weak evidence and **0** cases of resistance to strong contradictory evidence.

The mechanism does what was asked of it: it holds an uncertain belief, distinguishes evidence that bears on the proposition from evidence that does not, weighs what remains by how much it deserves to be trusted, revises in a direction and magnitude caused by that weighing, and keeps an unbroken record of every step. The reversal test shows it is not merely lowering confidence on contact with disagreement.

That is the whole claim. **It is not learning**, it does not generalise, and it says nothing about whether ECHO could form these propositions or assess this evidence on its own — both of which were handed to it.
