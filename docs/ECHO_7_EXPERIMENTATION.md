# ECHO 7 — Experimentation

## Objective

Move ECHO from *I can predict* to *I cannot tell these explanations apart, so I will design something that can*. Everything is synthetic and offline; an 'intervention' sets a variable in a simulation.

## Architecture

| Module | Responsibility |
| --- | --- |
| `echo/causal.py` | structural causal models, `do()`, latent variables, a posterior over competing models |
| `echo/experiment.py` | expected information gain, cost-aware selection policies, the immutable experiment record and ledger |
| `experiments/experiment_worlds.py` | the hidden truth — never imported by anything under `echo/` |

Expected information gain is the mutual information between the hypothesis and the experiment's result:

```
EIG(e) = H(P(H)) − Σ_c P(c | e) · H(P(H | c, e))
```

computed **exactly** — every count vector of an n-sample experiment is enumerated and weighted by its multinomial probability. There is no sampling in the appraisal, so the choice is reproducible to the last bit. Utility subtracts price: `EIG − 0.05 · cost − 0.1 · risk`, both weights fixed before the first run.

## Experiment design and controls

Three explanations of one visible fact: `V1` and `V2` move together. `V3` exists and acts but is **never recorded**, which is what makes the confounded story possible.

| Hypothesis | Structure |
| --- | --- |
| `H1-direct` | V1 -> V2 |
| `H2-confounded` | V3 -> V1, V3 -> V2 |
| `H3-reversed` | V2 -> V1 |

The three are constructed to produce the **identical** observed joint distribution over `(V1, V2)`:

| Model | P(0,0) | P(0,1) | P(1,0) | P(1,1) | corr(V1,V2) |
| --- | --- | --- | --- | --- | --- |
| `H1-direct` | 0.4250 | 0.0750 | 0.0750 | 0.4250 | +0.7000 |
| `H2-confounded` | 0.4250 | 0.0750 | 0.0750 | 0.4250 | +0.7000 |
| `H3-reversed` | 0.4250 | 0.0750 | 0.0750 | 0.4250 | +0.7000 |

**This is the misleading observational correlation, and it is exactly misleading.** No amount of watching separates these models, because there is nothing in the observational distribution to separate. The appraisal discovers this rather than being told it: observation scores 0.0000 bits.

Starting uncertainty is 1.5850 bits (three models, flat prior). The menu, appraised against that prior:

| Option | Intervention | Cost | Risk | EIG (bits) | Utility | Bits per unit cost | Control it provides |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `OBS` | OBSERVE() | 1.0 | 0.0 | 0.0000 | -0.0500 | 0.0000 | misleading observational correlation — provably 0 bits |
| `DO-V1` | INTERVENE(V1=1) | 2.0 | 0.1 | 0.5038 | +0.3938 | 0.2519 | cheap and informative |
| `DO-V2` | INTERVENE(V2=1) | 12.0 | 0.5 | 0.5038 | -0.1462 | 0.0420 | expensive, same information as DO-V1 at 6× the price |
| `DO-V3` | INTERVENE(V3=1) | 6.0 | 0.2 | 0.8150 | +0.4950 | 0.1358 | most informative single experiment |
| `DO-V1V2` | INTERVENE(V1=1, V2=1) | 9.0 | 0.6 | 0.0000 | -0.5100 | 0.0000 | expensive and **inconclusive by construction** — 0 bits at any budget |

Two options score exactly zero. `OBS` does because the models agree on everything visible; `DO-V1V2` does because fixing both recorded variables leaves nothing to observe. Neither zero is written down anywhere — both fall out of the formula.

Each campaign runs 4 experiments and is repeated 40 times per policy with matched seeds.

## Measured results

### `EXP-confounded` — truth is `H2-confounded`

*Three binary variables. V1 and V2 are strongly associated in the observational record. Three explanations are on the table and the observational data does not separate them.*

| Policy | P(truth) after | Final entropy | Identified | Total cost | Bits removed | Bits per unit cost | Cost to identify |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `information_gain` | 0.9827 | 0.1406 | 100% | 9.8 | 1.4444 | 0.1528 | 6.6 |
| `random` | 0.7939 | 0.3912 | 68% | 23.1 | 1.1938 | 0.0557 | 12.7 |
| `max_information` | 1.0000 | 0.0001 | 100% | 24.0 | 1.5849 | 0.0660 | 6.6 |
| `cheapest` | 0.3333 | 1.5850 | 0% | 4.0 | -0.0000 | -0.0000 | — |

### `EXP-direct` — truth is `H1-direct`

*The same three variables and the same three explanations. The observational record again shows V1 and V2 moving together.*

| Policy | P(truth) after | Final entropy | Identified | Total cost | Bits removed | Bits per unit cost | Cost to identify |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `information_gain` | 0.9297 | 0.1786 | 85% | 11.7 | 1.4064 | 0.1231 | 9.0 |
| `random` | 0.7138 | 0.6192 | 45% | 23.1 | 0.9658 | 0.0465 | 20.9 |
| `max_information` | 0.9636 | 0.1193 | 92% | 16.1 | 1.4657 | 0.1038 | 12.0 |
| `cheapest` | 0.3333 | 1.5850 | 0% | 4.0 | -0.0000 | -0.0000 | — |

### `EXP-reversed` — truth is `H3-reversed`

*The same three variables and the same three explanations. The observational record is, once more, an association between V1 and V2.*

| Policy | P(truth) after | Final entropy | Identified | Total cost | Bits removed | Bits per unit cost | Cost to identify |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `information_gain` | 0.9601 | 0.0882 | 92% | 11.3 | 1.4968 | 0.1351 | 9.3 |
| `random` | 0.7588 | 0.5964 | 50% | 23.1 | 0.9885 | 0.0454 | 19.1 |
| `max_information` | 0.9726 | 0.1010 | 92% | 30.9 | 1.4839 | 0.0501 | 13.6 |
| `cheapest` | 0.3333 | 1.5850 | 0% | 4.0 | -0.0000 | -0.0000 | — |

## Does information-gain selection beat random selection?

| Scenario | Metric | Random | Information gain | Difference |
| --- | --- | --- | --- | --- |
| `EXP-confounded` | P(truth) after | 0.7939 | 0.9827 | +0.1888 (better) |
| `EXP-confounded` | Bits removed | 1.1938 | 1.4444 | +0.2506 (better) |
| `EXP-confounded` | Bits per unit cost | 0.0557 | 0.1528 | +0.0971 (better) |
| `EXP-confounded` | Final entropy | 0.3912 | 0.1406 | -0.2506 (better) |
| `EXP-direct` | P(truth) after | 0.7138 | 0.9297 | +0.2159 (better) |
| `EXP-direct` | Bits removed | 0.9658 | 1.4064 | +0.4406 (better) |
| `EXP-direct` | Bits per unit cost | 0.0465 | 0.1231 | +0.0766 (better) |
| `EXP-direct` | Final entropy | 0.6192 | 0.1786 | -0.4406 (better) |
| `EXP-reversed` | P(truth) after | 0.7588 | 0.9601 | +0.2013 (better) |
| `EXP-reversed` | Bits removed | 0.9885 | 1.4968 | +0.5083 (better) |
| `EXP-reversed` | Bits per unit cost | 0.0454 | 0.1351 | +0.0897 (better) |
| `EXP-reversed` | Final entropy | 0.5964 | 0.0882 | -0.5083 (better) |

Information-gain selection is more cost-efficient than random in **3 of 3** scenarios.

## Failures and things that did not work

- `EXP-confounded`: 0.1406 bits of uncertainty remained at the end. The campaign did not fully resolve the question.
- `EXP-direct`: the information-gain policy identified the true model in only 85% of campaigns within 4 experiments.
- `EXP-direct`: 0.1786 bits of uncertainty remained at the end. The campaign did not fully resolve the question.
- `EXP-reversed`: the information-gain policy identified the true model in only 92% of campaigns within 4 experiments.

The `cheapest` policy is the clearest negative result in the table: it buys the observational option, which is provably worth nothing, and ends every campaign exactly as uncertain as it started. Choosing by price alone is not a weaker version of choosing well — it is a way of spending money to learn nothing.

## Limitations

1. **The hypothesis set is given.** ECHO chooses among three supplied models; it does not invent candidate structures. It is not told which is true, which is the part that matters here, but the space is not its own.
2. **Binary variables, tiny graphs.** Exactness is bought with size. Three variables and two observed; nothing here scales as written.
3. **The menu is fixed.** Costs, risks and sample budgets are given by the environment. ECHO selects, it does not design new instruments.
4. **Synthetic throughout.** An intervention is a line of arithmetic. Nothing was done to anything real.
5. **One truth per scenario.** Three scenarios, forty campaigns each; enough to compare policies, not enough to characterise the method.
6. **Utility weights are hand-set.** Different cost and risk weights would order the menu differently. They were fixed in advance and not adjusted afterwards, but they were still chosen.

## What this does and does not demonstrate

**Does:** that ECHO can compute, from its own current uncertainty, which available action would reduce it most per unit cost; that this beats random selection on measured campaigns; that it correctly assigns zero value to an experiment whose result cannot depend on which hypothesis is true, without being told which one that is.

**Does not:** that ECHO understands causation — ECHO 8 is where that is tested and it is a narrower claim than the word suggests. Nothing here involves the real world, autonomy, or any capacity to act outside a simulation. This is Bayesian experimental design over a small enumerable space. It is not consciousness, not self-awareness, and not general intelligence, and no measurement here bears on those questions at all.
