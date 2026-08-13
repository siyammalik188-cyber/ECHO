# ECHO 8 — Causal abstraction

## Objective

Test whether ECHO can tell **correlation** from **causal structure** — and, where the data cannot tell them apart, whether it declines to pretend otherwise.

## Architecture

| Module | Responsibility |
| --- | --- |
| `echo/causal.py` | structures, `do()`, a posterior held over several models at once |
| `echo/counterfactual.py` | abduction → action → prediction, with confidence and stated assumptions |
| `echo/experiment.py` | ECHO 7's selection machinery, reused unchanged |
| `experiments/causal_worlds.py` | the hidden structures — never imported by anything under `echo/` |

## The worlds, and why observation is not enough

| Model | Structure | corr(V1,V2) | corr(V2,V3) | corr(V1,V3) |
| --- | --- | --- | --- | --- |
| `M1-chain` | V1 -> V2, V2 -> V3 | +0.700 | +0.600 | +0.420 |
| `M2-fork` | V2 -> V1, V2 -> V3 | +0.700 | +0.600 | +0.420 |
| `M3-reverse-chain` | V2 -> V1, V3 -> V2 | +0.700 | +0.600 | +0.420 |
| `M4-collider` | V1 -> V2, V3 -> V2 | +0.450 | +0.450 | +0.000 |

The first three are a **Markov equivalence class**: same skeleton, no v-structure, and therefore the *identical* joint distribution. Their parameters are derived from one another by Bayes' rule rather than fitted, so the equality is exact rather than close. No observational method can order them — not correlation ranking, not conditional independence testing, not more data.

The collider is different, and deliberately so: it makes `V1` and `V3` **marginally independent** (correlation exactly 0.000). Observation can and should rule it out. Including it keeps the claim honest — observational data is not useless in general, it is insufficient for a specific and identifiable reason.

Under a flat prior over four models (2.0000 bits), observation is worth 0.2670 bits — roughly the one bit needed to eliminate the collider, and nothing more. `do(V2)` is worth 1.2645 bits, because setting the middle variable is what separates a chain from a fork.

## Measured results

| Scenario | Truth | P(truth) after watching | P(truth) after intervening | Identified | Cost | Experiments chosen |
| --- | --- | --- | --- | --- | --- | --- |
| `CAUSAL-chain` | `M1-chain` | 0.3333 | 1.0000 | yes | 9 | `DO-V2`, `DO-V1`, `DO-V1` |
| `CAUSAL-fork` | `M2-fork` | 0.3333 | 0.9991 | yes | 9 | `DO-V2`, `DO-V2`, `DO-V2` |
| `CAUSAL-reverse` | `M3-reverse-chain` | 0.3333 | 0.9999 | yes | 9 | `DO-V2`, `DO-V3`, `DO-V3` |
| `CAUSAL-collider` | `M4-collider` | 1.0000 | 1.0000 | yes | 9 | `DO-V2`, `DO-V2`, `DO-V2` |
| `CAUSAL-starved` | `M1-chain` | 0.3333 | 0.8401 | yes | 3 | `DO-V2` |

Watching alone leaves the truth at roughly 1/3 in every scenario whose answer lies inside the equivalence class — which is exactly right, because a third is all the observational data can support. The collider scenario is the exception: there, watching does the job on its own.

### `CAUSAL-chain` — truth is `M1-chain`

- Prior: `M1-chain` 0.250, `M2-fork` 0.250, `M3-reverse-chain` 0.250, `M4-collider` 0.250
- After 200 observational rows: `M2-fork` 0.333, `M1-chain` 0.333, `M3-reverse-chain` 0.333, `M4-collider` 0.000
- After 3 intervention(s): `M1-chain` 1.000, `M2-fork` 0.000, `M3-reverse-chain` 0.000, `M4-collider` 0.000
  - step 0: `DO-V2` (INTERVENE(V2=1)), expected 1.0581 bits, actually removed +1.5847 bits — contradicted the leader
  - step 1: `DO-V1` (INTERVENE(V1=1)), expected 0.0001 bits, actually removed +0.0003 bits
  - step 2: `DO-V1` (INTERVENE(V1=1)), expected 0.0000 bits, actually removed +0.0000 bits

### `CAUSAL-fork` — truth is `M2-fork`

- Prior: `M1-chain` 0.250, `M2-fork` 0.250, `M3-reverse-chain` 0.250, `M4-collider` 0.250
- After 200 observational rows: `M2-fork` 0.333, `M1-chain` 0.333, `M3-reverse-chain` 0.333, `M4-collider` 0.000
- After 3 intervention(s): `M2-fork` 0.999, `M3-reverse-chain` 0.001, `M1-chain` 0.000, `M4-collider` 0.000
  - step 0: `DO-V2` (INTERVENE(V2=1)), expected 1.0581 bits, actually removed +1.0131 bits
  - step 1: `DO-V2` (INTERVENE(V2=1)), expected 0.3284 bits, actually removed +0.4442 bits
  - step 2: `DO-V2` (INTERVENE(V2=1)), expected 0.0584 bits, actually removed +0.1166 bits

### `CAUSAL-reverse` — truth is `M3-reverse-chain`

- Prior: `M1-chain` 0.250, `M2-fork` 0.250, `M3-reverse-chain` 0.250, `M4-collider` 0.250
- After 200 observational rows: `M2-fork` 0.333, `M1-chain` 0.333, `M3-reverse-chain` 0.333, `M4-collider` 0.000
- After 3 intervention(s): `M3-reverse-chain` 1.000, `M2-fork` 0.000, `M1-chain` 0.000, `M4-collider` 0.000
  - step 0: `DO-V2` (INTERVENE(V2=1)), expected 1.0581 bits, actually removed +1.5420 bits — contradicted the leader
  - step 1: `DO-V3` (INTERVENE(V3=1)), expected 0.0163 bits, actually removed +0.0395 bits
  - step 2: `DO-V3` (INTERVENE(V3=1)), expected 0.0010 bits, actually removed +0.0026 bits

### `CAUSAL-collider` — truth is `M4-collider`

- Prior: `M1-chain` 0.250, `M2-fork` 0.250, `M3-reverse-chain` 0.250, `M4-collider` 0.250
- After 200 observational rows: `M4-collider` 1.000, `M2-fork` 0.000, `M3-reverse-chain` 0.000, `M1-chain` 0.000
- After 3 intervention(s): `M4-collider` 1.000, `M1-chain` 0.000, `M3-reverse-chain` 0.000, `M2-fork` 0.000
  - step 0: `DO-V2` (INTERVENE(V2=1)), expected 0.0000 bits, actually removed -0.0000 bits
  - step 1: `DO-V2` (INTERVENE(V2=1)), expected 0.0000 bits, actually removed +0.0000 bits
  - step 2: `DO-V2` (INTERVENE(V2=1)), expected 0.0000 bits, actually removed +0.0000 bits

### `CAUSAL-starved` — truth is `M1-chain`

- Prior: `M1-chain` 0.250, `M2-fork` 0.250, `M3-reverse-chain` 0.250, `M4-collider` 0.250
- After 200 observational rows: `M2-fork` 0.333, `M1-chain` 0.333, `M3-reverse-chain` 0.333, `M4-collider` 0.000
- After 1 intervention(s): `M1-chain` 0.840, `M2-fork` 0.129, `M3-reverse-chain` 0.031, `M4-collider` 0.000
  - step 0: `DO-V2` (INTERVENE(V2=1)), expected 0.4195 bits, actually removed +0.8366 bits — contradicted the leader

## Causal uncertainty is represented, not resolved away

The sharpest case is the observational phase. After 200 rows, ECHO's posterior over the four structures is:

| Model | Posterior after watching |
| --- | --- |
| `M2-fork` | 0.3333 |
| `M1-chain` | 0.3333 |
| `M3-reverse-chain` | 0.3333 |
| `M4-collider` | 0.0000 |

That is the correct answer and the only defensible one. The collider is eliminated because it predicts something visibly false — `V1` and `V3` independent. The other three are left at **exactly one third each**, 1.5850 bits, because the observational data contains nothing that could order them. ECHO reports a distribution and declines to name a winner; nothing in the pipeline forces one.

`CAUSAL-starved` is the partial case: one intervention on three samples, which is real evidence but not much of it.

| Model | Posterior |
| --- | --- |
| `M1-chain` | 0.8401 |
| `M2-fork` | 0.1285 |
| `M3-reverse-chain` | 0.0314 |
| `M4-collider` | 0.0000 |

It leans toward `M1-chain` at 0.8401 — which happens to be correct — but keeps 0.7483 bits of a possible 2.0000 on the table. A system that reported this as settled would be overstating three samples.

## Counterfactual inference

Abduction, then action, then prediction: infer the background from what actually happened, change the antecedent with `do()`, and push the *same* background through the changed model. Holding the background fixed is what makes it a claim about this case rather than an average.

### `CAUSAL-chain`

| Question | ECHO's answer | Confidence | Model disagreement | True model's answer |
| --- | --- | --- | --- | --- |
| Given V1=1, V2=1, V3=1: had V1 been 0, what is P(V3 = 1)? | 0.2900 | 1.0000 | 0.0000 | 0.2900 |
| Given V1=1, V2=1, V3=1: had V2 been 0, what is P(V3 = 1)? | 0.2000 | 1.0000 | 0.0000 | 0.2000 |
| Given V1=0, V2=0, V3=0: had V3 been 1, what is P(V1 = 1)? | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

### `CAUSAL-fork`

| Question | ECHO's answer | Confidence | Model disagreement | True model's answer |
| --- | --- | --- | --- | --- |
| Given V1=1, V2=1, V3=1: had V1 been 0, what is P(V3 = 1)? | 0.9999 | 0.9940 | 0.0002 | 1.0000 |
| Given V1=1, V2=1, V3=1: had V2 been 0, what is P(V3 = 1)? | 0.2006 | 0.9921 | 0.0012 | 0.2000 |
| Given V1=0, V2=0, V3=0: had V3 been 1, what is P(V1 = 1)? | 0.0005 | 0.9924 | 0.0010 | 0.0000 |

### `CAUSAL-reverse`

| Question | ECHO's answer | Confidence | Model disagreement | True model's answer |
| --- | --- | --- | --- | --- |
| Given V1=1, V2=1, V3=1: had V1 been 0, what is P(V3 = 1)? | 1.0000 | 0.9995 | 0.0000 | 1.0000 |
| Given V1=1, V2=1, V3=1: had V2 been 0, what is P(V3 = 1)? | 1.0000 | 0.9994 | 0.0001 | 1.0000 |
| Given V1=0, V2=0, V3=0: had V3 been 1, what is P(V1 = 1)? | 0.7100 | 0.9994 | 0.0001 | 0.7100 |

### `CAUSAL-collider`

| Question | ECHO's answer | Confidence | Model disagreement | True model's answer |
| --- | --- | --- | --- | --- |
| Given V1=1, V2=1, V3=1: had V1 been 0, what is P(V3 = 1)? | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| Given V1=1, V2=1, V3=1: had V2 been 0, what is P(V3 = 1)? | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| Given V1=0, V2=0, V3=0: had V3 been 1, what is P(V1 = 1)? | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

### `CAUSAL-starved`

| Question | ECHO's answer | Confidence | Model disagreement | True model's answer |
| --- | --- | --- | --- | --- |
| Given V1=1, V2=1, V3=1: had V1 been 0, what is P(V3 = 1)? | 0.4035 | 0.3871 | 0.1908 | 0.2900 |
| Given V1=1, V2=1, V3=1: had V2 been 0, what is P(V3 = 1)? | 0.2251 | 0.5650 | 0.0486 | 0.2000 |
| Given V1=0, V2=0, V3=0: had V3 been 1, what is P(V1 = 1)? | 0.0223 | 0.5718 | 0.0432 | 0.0000 |

A single answer in full, to show what accompanies every number:

```
QUESTION:   Given V1=1, V2=1, V3=1: had V1 been 0, what is P(V3 = 1)?
PROBABILITY: 0.2900
CONFIDENCE:  1.0000
EVIDENCE:    200 observational rows
EVIDENCE:    3 intervention(s) costing 9
EVIDENCE:    belief over structures: M1-chain 1.00, M2-fork 0.00, M3-reverse-chain 0.00, M4-collider 0.00
EVIDENCE:    per-model answers: M1-chain 0.290, M2-fork 1.000, M3-reverse-chain 1.000, M4-collider 1.000
ASSUMPTION:  The true structure is one of the models under consideration.
ASSUMPTION:  The latent background is inferred from the observed facts and held fixed while the antecedent is changed.
ASSUMPTION:  The conditional probability tables are taken as given rather than estimated from data.
ASSUMPTION:  This is a probability, not a prediction about any individual case.
```

Confidence is not the precision of the average. It falls both with structural uncertainty and with how much the live models disagree: two models that both say 0.8 support a confident answer, while two that say 0.1 and 0.9 do not, however certain ECHO is that one of them is right. Under a flat prior the same question returns confidence 0.0000 — the mechanism refuses to launder a split posterior into a firm number.

## Five things this report keeps apart

| Term | What it means | Where it lives | Established here? |
| --- | --- | --- | --- |
| **Correlation** | two variables move together | `CausalModel.correlation` | Yes — and shown to be identical across three different structures |
| **Prediction** | P(Y \| X = x): what to expect on *seeing* X | `CausalModel.conditioned` | Yes, and it is the same for all three |
| **Causal evidence** | data generated under `do(X = x)` | `HypothesisSet.updated(..., intervention=...)` | Yes — this is what breaks the tie |
| **Causal belief** | a posterior over structures, held plural | `HypothesisSet.posterior` | Yes, including cases where it stays split |
| **Counterfactual inference** | P(Y_{X=x'} \| what actually happened) | `echo/counterfactual.py` | Yes, with confidence and assumptions attached |

The gap between the second and third rows is the entire point. In the confounded and forked worlds, *seeing* `V1 = 1` raises the probability of `V3`; *setting* `V1 = 1` does not move it at all. Prediction and causation come apart, and only intervention notices.

## Failures and limitations

- Every scenario with an adequate budget identified its true structure.
- `CAUSAL-starved` leads with `M1-chain` at 0.8401. It is included to check that a starved posterior stays spread rather than collapsing, and it does — but note that it landed on the true model, which was not guaranteed and should not be read as the mechanism getting it right on this little evidence.

1. **The model set is supplied.** ECHO weighs four given structures; it does not search the space of DAGs. Structure learning is a different and much harder problem, and nothing here attempts it.
2. **The conditional probability tables are given too.** Only the structure is uncertain. Real causal inference estimates both.
3. **Three binary variables.** Exactness is bought with size.
4. **Counterfactuals assume the truth is in the set.** If the real structure is not among the four, every answer is conditional on a false premise — stated in the assumptions of every answer, but worth repeating.
5. **No unmeasured-confounding search.** ECHO 7's worlds had a latent variable; these do not. Deciding *whether* a confounder exists is not tested here.
6. **Synthetic and offline.** An intervention is arithmetic.

## What this does and does not demonstrate

**Does:** that ECHO distinguishes seeing from doing, on worlds where the two genuinely diverge; that it holds several causal structures at once with calibrated weights instead of committing early; that it answers counterfactual questions with a probability, its evidence, its confidence and its assumptions; and that its confidence collapses when its models disagree rather than reporting a firm-looking average.

**Does not:** that ECHO understands causation in any sense a person would mean by the word. It manipulates conditional probability tables over three meaningless labels. It cannot discover a structure that was not handed to it, cannot tell whether an unmeasured confounder exists, and has no notion of mechanism, intervention-in-the-world, or why any of these arrows would point anywhere. Nothing here is consciousness, self-awareness, or general intelligence, and no measurement in this report bears on those questions.
