"""Counterfactuals, and the five different things people mean by "causes".

ECHO 8 asks a question no amount of prediction answers: *if that had not
happened, what would have followed?* Answering it needs more than a joint
distribution, because the question is about a world that did not occur.

The standard three-step recipe is implemented literally:

1. **Abduction** — take what actually happened and work out what the
   unobserved background must have been. Here that means a posterior over the
   latent variables given the observed facts.
2. **Action** — apply `do(...)` to the antecedent, severing it from its causes.
3. **Prediction** — push the same background through the modified model.

The background is held fixed between steps 1 and 3. That is what makes the
answer counterfactual rather than merely interventional: it is about *this*
case, with whatever unobserved conditions this case happened to have, not about
what setting a variable does on average.

**Five things this module keeps apart, because conflating them is the usual
error:**

| Term | What it is | How it is computed here |
| --- | --- | --- |
| Correlation | two variables move together | `CausalModel.correlation` |
| Prediction | P(Y \\| X = x), what to expect on seeing X | `conditioned` |
| Causal evidence | data from `do(X = x)` | `HypothesisSet.updated(..., intervention=...)` |
| Causal belief | a posterior over structures | `HypothesisSet.posterior` |
| Counterfactual inference | P(Y_{X=x'} \\| observed facts) | `counterfactual` below |

A counterfactual answer is always reported with its assumptions attached and
never with certainty it has not earned. When the models in the set disagree, the
answer is averaged over them weighted by belief, and the disagreement is
reported alongside the number — a confident answer from an uncertain posterior
would be a lie about where the confidence came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

from .causal import Assignment, CausalError, CausalModel, HypothesisSet, entropy


@dataclass(frozen=True)
class CounterfactualQuery:
    """*Given what we saw, what if the antecedent had been different?*"""

    #: What was actually observed, as variable -> value.
    facts: tuple[tuple[str, int], ...]
    #: The variable to change, and what to change it to.
    antecedent: tuple[str, int]
    #: The variable asked about, and the value whose probability is wanted.
    consequent: tuple[str, int]

    def label(self) -> str:
        seen = ", ".join(f"{v}={x}" for v, x in self.facts)
        a_var, a_val = self.antecedent
        c_var, c_val = self.consequent
        return (
            f"Given {seen}: had {a_var} been {a_val}, "
            f"what is P({c_var} = {c_val})?"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [list(f) for f in self.facts],
            "antecedent": list(self.antecedent),
            "consequent": list(self.consequent),
            "question": self.label(),
        }


@dataclass(frozen=True)
class CounterfactualAnswer:
    """A probability, what supports it, how much to trust it, and what it assumes."""

    query: CounterfactualQuery
    probability: float
    #: Per-model answers, so disagreement is visible rather than averaged away.
    per_model: tuple[tuple[str, float, float], ...]  # (model id, belief, answer)
    confidence: float
    supporting_evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    model_disagreement: float
    undefined: bool = False

    @property
    def spread(self) -> float:
        """Widest disagreement between any two models ECHO still takes seriously."""
        live = [answer for _, belief, answer in self.per_model if belief > 0.01]
        return max(live) - min(live) if len(live) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "probability": self.probability,
            "per_model": [list(row) for row in self.per_model],
            "confidence": self.confidence,
            "supporting_evidence": list(self.supporting_evidence),
            "assumptions": list(self.assumptions),
            "model_disagreement": self.model_disagreement,
            "spread": self.spread,
            "undefined": self.undefined,
        }


#: Assumptions every counterfactual here rests on. Stated every time, because
#: an answer without them would be claiming more than the method supports.
STANDING_ASSUMPTIONS = (
    "The true structure is one of the models under consideration.",
    "The latent background is inferred from the observed facts and held fixed "
    "while the antecedent is changed.",
    "The conditional probability tables are taken as given rather than estimated "
    "from data.",
    "This is a probability, not a prediction about any individual case.",
)


def _latent_posterior(
    model: CausalModel, facts: Mapping[str, int]
) -> list[tuple[Assignment, float]]:
    """Abduction: what the unobserved background must have been, given the facts."""
    weighted: list[tuple[Assignment, float]] = []
    total = 0.0
    for assignment in product((0, 1), repeat=len(model.variables)):
        if any(
            assignment[model.variables.index(name)] != value
            for name, value in facts.items()
        ):
            continue
        probability = model.probability_of(assignment)
        if probability > 0.0:
            weighted.append((assignment, probability))
            total += probability
    if total <= 0.0:
        return []
    return [(assignment, p / total) for assignment, p in weighted]


def counterfactual_for_model(
    model: CausalModel, query: CounterfactualQuery
) -> float | None:
    """P(consequent | facts, do(antecedent)) for one model. None if undefined.

    Abduction, then action, then prediction — with the background held fixed
    across the change, which is the whole difference between this and an
    ordinary interventional query.
    """
    facts = dict(query.facts)
    antecedent_variable, antecedent_value = query.antecedent
    consequent_variable, consequent_value = query.consequent

    for name in (*facts, antecedent_variable, consequent_variable):
        if name not in model.variables:
            raise CausalError(f"{name} is not a variable of {model.id}")

    background = _latent_posterior(model, facts)
    if not background:
        return None

    # Variables that are not descendants of the antecedent keep whatever value
    # the actual case had; the rest are recomputed under the change.
    modified = model.intervened({antecedent_variable: antecedent_value})
    exogenous = [
        name
        for name in model.variables
        if name != antecedent_variable
        and not _is_descendant(model, antecedent_variable, name)
    ]

    total = 0.0
    for assignment, weight in background:
        pinned = {
            name: assignment[model.variables.index(name)] for name in exogenous
        }
        pinned[antecedent_variable] = antecedent_value
        world = modified.intervened(
            {k: v for k, v in pinned.items() if k != antecedent_variable}
        )
        total += weight * world.marginal(consequent_variable) if consequent_value == 1 else (
            weight * (1.0 - world.marginal(consequent_variable))
        )
    return total


def _is_descendant(model: CausalModel, ancestor: str, node: str) -> bool:
    if node == ancestor:
        return True
    return any(_is_descendant(model, ancestor, parent) for parent in model.parents[node])


def counterfactual(
    hypotheses: HypothesisSet,
    query: CounterfactualQuery,
    *,
    evidence: Sequence[str] = (),
) -> CounterfactualAnswer:
    """Answer a counterfactual across every model ECHO still believes possible.

    The reported probability is the belief-weighted average. Confidence is
    deliberately *not* that average's precision: it is reduced both by how
    uncertain ECHO is about the structure and by how much the structures
    disagree about the answer. Two models that both say 0.8 support a confident
    answer; two that say 0.1 and 0.9 do not, however sure ECHO is that one of
    them is right.
    """
    per_model: list[tuple[str, float, float]] = []
    total = 0.0
    live_weight = 0.0
    undefined = False

    for model, belief in zip(hypotheses.models, hypotheses.posterior):
        answer = counterfactual_for_model(model, query)
        if answer is None:
            undefined = True
            continue
        per_model.append((model.id, belief, answer))
        total += belief * answer
        live_weight += belief

    if live_weight <= 0.0:
        return CounterfactualAnswer(
            query=query,
            probability=0.5,
            per_model=(),
            confidence=0.0,
            supporting_evidence=tuple(evidence),
            assumptions=STANDING_ASSUMPTIONS
            + ("The observed facts are impossible under every model considered.",),
            model_disagreement=0.0,
            undefined=True,
        )

    probability = total / live_weight

    # How much the live models disagree, as a belief-weighted spread.
    disagreement = sum(
        belief * abs(answer - probability) for _, belief, answer in per_model
    ) / live_weight

    # Structural uncertainty, normalised to [0, 1]: 0 when one model holds all
    # the mass, 1 when the posterior is flat.
    maximum_entropy = entropy([1 / len(hypotheses)] * len(hypotheses))
    structural = hypotheses.entropy() / maximum_entropy if maximum_entropy > 0 else 0.0

    confidence = max(0.0, min(1.0, (1.0 - structural) * (1.0 - 2.0 * disagreement)))

    supporting = list(evidence)
    supporting.append(
        "belief over structures: "
        + ", ".join(f"{mid} {belief:.2f}" for mid, belief, _ in per_model)
    )
    supporting.append(
        "per-model answers: "
        + ", ".join(f"{mid} {answer:.3f}" for mid, _, answer in per_model)
    )

    assumptions = list(STANDING_ASSUMPTIONS)
    if disagreement > 0.05:
        assumptions.append(
            f"The models disagree about this question by {disagreement:.3f} on "
            "average; the single number above hides that."
        )
    if undefined:
        assumptions.append(
            "At least one model calls the observed facts impossible and was "
            "excluded from the average."
        )

    return CounterfactualAnswer(
        query=query,
        probability=probability,
        per_model=tuple(per_model),
        confidence=confidence,
        supporting_evidence=tuple(supporting),
        assumptions=tuple(assumptions),
        model_disagreement=disagreement,
        undefined=undefined,
    )
