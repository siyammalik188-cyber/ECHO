"""Structural causal models, and what an intervention does to one.

A `CausalModel` is a directed acyclic graph over binary variables plus a
conditional probability table for each. Everything about it is enumerable: with
a handful of variables the full joint distribution is a few dozen numbers, so
every quantity in ECHO 7 and ECHO 8 — likelihoods, posteriors, expected
information gain, counterfactual probabilities — is computed exactly rather
than sampled. No Monte Carlo, no seeds, no tolerance thresholds.

The distinction the whole of ECHO 8 rests on lives in one method here.
Conditioning on `V1 = 1` asks *what else tends to be true when V1 is 1*.
Intervening with `do(V1 = 1)` asks *what happens when V1 is set to 1*, which
means severing V1 from its parents before computing anything. Two models can
agree perfectly on the first and disagree completely on the second, and that gap
is exactly what an experiment is for.

Variable names are `V1`, `V2`, … and mean nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

Assignment = tuple[int, ...]


class CausalError(ValueError):
    """Raised when a model is malformed or asked something it cannot answer."""


@dataclass(frozen=True)
class CausalModel:
    """A DAG over binary variables with a CPT per variable.

    `parents[v]` lists v's parents in the order their values index `cpts[v]`.
    `cpts[v][assignment]` is P(v = 1 | parents = assignment).
    """

    id: str
    variables: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    cpts: Mapping[str, Mapping[Assignment, float]]
    description: str = ""
    #: Which variables are actually recorded. Empty means all of them. A
    #: variable that is not observed still acts on the world — it just cannot
    #: be seen, which is what makes genuine confounding possible: two models
    #: can then agree on everything visible and disagree on what would happen
    #: if you intervened.
    observed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for variable in self.variables:
            if variable not in self.parents:
                raise CausalError(f"{variable} has no parent list")
            for parent in self.parents[variable]:
                if parent not in self.variables:
                    raise CausalError(f"{variable} has unknown parent {parent}")
            table = self.cpts.get(variable)
            if table is None:
                raise CausalError(f"{variable} has no conditional table")
            expected = 2 ** len(self.parents[variable])
            if len(table) != expected:
                raise CausalError(
                    f"{variable} needs {expected} rows for "
                    f"{len(self.parents[variable])} parent(s), has {len(table)}"
                )
            for probability in table.values():
                if not 0.0 <= probability <= 1.0:
                    raise CausalError(f"{variable} has a probability outside [0, 1]")
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        colour: dict[str, int] = {}

        def visit(node: str) -> None:
            state = colour.get(node, 0)
            if state == 1:
                raise CausalError(f"the graph has a cycle through {node}")
            if state == 2:
                return
            colour[node] = 1
            for parent in self.parents[node]:
                visit(parent)
            colour[node] = 2

        for variable in self.variables:
            visit(variable)

    # ----------------------------------------------------------------- edges

    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (parent, child)
            for child in self.variables
            for parent in self.parents[child]
        )

    def structure(self) -> str:
        """A readable form, e.g. `V1 -> V2, V3 -> V2`. Structure only."""
        edges = self.edges()
        return ", ".join(f"{a} -> {b}" for a, b in edges) if edges else "no edges"

    # ------------------------------------------------------------ the joint

    @property
    def observed_variables(self) -> tuple[str, ...]:
        return self.observed or self.variables

    def _index(self, variable: str) -> int:
        return self.variables.index(variable)

    def probability_of(self, assignment: Assignment) -> float:
        """P(assignment) under this model, by the chain rule over the DAG."""
        total = 1.0
        for variable in self.variables:
            index = self._index(variable)
            parent_values = tuple(
                assignment[self._index(parent)] for parent in self.parents[variable]
            )
            p_one = self.cpts[variable][parent_values]
            total *= p_one if assignment[index] == 1 else 1.0 - p_one
        return total

    def joint(self) -> dict[Assignment, float]:
        """The full joint distribution. Exact, enumerated."""
        return {
            assignment: self.probability_of(assignment)
            for assignment in product((0, 1), repeat=len(self.variables))
        }

    def observed_joint(self) -> dict[Assignment, float]:
        """The joint over recorded variables, latent ones summed out.

        This is everything observation can ever reveal. Two models with the
        same observed joint are indistinguishable no matter how long you watch.
        """
        indices = [self._index(v) for v in self.observed_variables]
        out: dict[Assignment, float] = {}
        for assignment, probability in self.joint().items():
            key = tuple(assignment[i] for i in indices)
            out[key] = out.get(key, 0.0) + probability
        return out

    def probability_of_observed(self, assignment: Assignment) -> float:
        return self.observed_joint().get(tuple(assignment), 0.0)

    def marginal(self, variable: str) -> float:
        index = self._index(variable)
        return sum(p for a, p in self.joint().items() if a[index] == 1)

    def correlation(self, first: str, second: str) -> float:
        """Pearson correlation between two binary variables under this model."""
        i, j = self._index(first), self._index(second)
        joint = self.joint()
        p_i = sum(p for a, p in joint.items() if a[i] == 1)
        p_j = sum(p for a, p in joint.items() if a[j] == 1)
        p_ij = sum(p for a, p in joint.items() if a[i] == 1 and a[j] == 1)
        covariance = p_ij - p_i * p_j
        spread = math.sqrt(max(p_i * (1 - p_i) * p_j * (1 - p_j), 1e-18))
        return covariance / spread

    # --------------------------------------------------------- intervention

    def intervened(self, assignments: Mapping[str, int]) -> "CausalModel":
        """`do(...)`: cut each named variable from its parents and fix its value.

        This is the operation that separates causation from correlation. A
        variable that has been intervened on no longer depends on anything;
        whatever used to explain it is severed, so any remaining association
        with it has to run *forwards*.
        """
        parents = dict(self.parents)
        cpts: dict[str, Mapping[Assignment, float]] = dict(self.cpts)
        for variable, value in assignments.items():
            if variable not in self.variables:
                raise CausalError(f"cannot intervene on unknown variable {variable!r}")
            if value not in (0, 1):
                raise CausalError(f"intervention value must be 0 or 1, got {value!r}")
            parents[variable] = ()
            cpts[variable] = {(): 1.0 if value == 1 else 0.0}
        label = ", ".join(f"do({v}={x})" for v, x in sorted(assignments.items()))
        return CausalModel(
            id=f"{self.id}|{label}",
            variables=self.variables,
            parents=parents,
            cpts=cpts,
            description=self.description,
            observed=self.observed,
        )

    def conditioned(self, variable: str, value: int) -> dict[Assignment, float]:
        """P(· | variable = value) — observation, *not* intervention."""
        index = self._index(variable)
        joint = self.joint()
        total = sum(p for a, p in joint.items() if a[index] == value)
        if total <= 0.0:
            raise CausalError(f"cannot condition on a zero-probability event")
        return {a: p / total for a, p in joint.items() if a[index] == value}

    # -------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "variables": list(self.variables),
            "parents": {v: list(p) for v, p in self.parents.items()},
            "cpts": {
                v: [[list(k), p] for k, p in table.items()]
                for v, table in self.cpts.items()
            },
            "description": self.description,
            "observed": list(self.observed_variables),
            "structure": self.structure(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalModel":
        return cls(
            id=str(payload["id"]),
            variables=tuple(payload["variables"]),
            parents={v: tuple(p) for v, p in payload["parents"].items()},
            cpts={
                v: {tuple(k): float(p) for k, p in rows}
                for v, rows in payload["cpts"].items()
            },
            description=str(payload.get("description", "")),
            observed=tuple(payload.get("observed", ())),
        )


# ------------------------------------------------------------ hypothesis sets


def entropy(distribution: Sequence[float]) -> float:
    """Shannon entropy in bits. The unit of 'how much I don't know'."""
    total = 0.0
    for p in distribution:
        if p > 0.0:
            total -= p * math.log2(p)
    return total


def normalise(weights: Sequence[float]) -> list[float]:
    total = sum(weights)
    if total <= 0.0:
        raise CausalError("cannot normalise a distribution summing to zero")
    return [w / total for w in weights]


@dataclass(frozen=True)
class HypothesisSet:
    """Several causal models held at once, with a probability over them.

    ECHO does not pick one and pretend. It carries all of them, and every piece
    of evidence moves the weights. A set whose posterior is flat is a set that
    knows it cannot yet tell the difference — which is the state an experiment
    is designed to escape.
    """

    models: tuple[CausalModel, ...]
    posterior: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.models) != len(self.posterior):
            raise CausalError("models and posterior disagree in length")
        if abs(sum(self.posterior) - 1.0) > 1e-9:
            raise CausalError(f"posterior sums to {sum(self.posterior)}, not 1")

    @classmethod
    def uniform(cls, models: Sequence[CausalModel]) -> "HypothesisSet":
        n = len(models)
        if n == 0:
            raise CausalError("a hypothesis set needs at least one model")
        return cls(tuple(models), tuple(1.0 / n for _ in range(n)))

    def __len__(self) -> int:
        return len(self.models)

    def entropy(self) -> float:
        return entropy(self.posterior)

    def probability(self, model_id: str) -> float:
        for model, p in zip(self.models, self.posterior):
            if model.id == model_id:
                return p
        raise KeyError(f"no model {model_id!r} in this set")

    def best(self) -> tuple[CausalModel, float]:
        index = max(range(len(self.models)), key=lambda i: self.posterior[i])
        return self.models[index], self.posterior[index]

    def ranked(self) -> list[tuple[str, float]]:
        pairs = [(m.id, p) for m, p in zip(self.models, self.posterior)]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        return pairs

    # ------------------------------------------------------------- updating

    def updated(
        self,
        observations: Iterable[Assignment],
        *,
        intervention: Mapping[str, int] | None = None,
    ) -> "HypothesisSet":
        """Bayes, over models. Returns a new set; nothing is mutated.

        Under an intervention the likelihood comes from the *intervened* model,
        which is the entire reason interventional data can separate hypotheses
        that observational data cannot.
        """
        observations = list(observations)
        weights: list[float] = []
        for model, prior in zip(self.models, self.posterior):
            effective = model.intervened(intervention) if intervention else model
            likelihood = 1.0
            for observation in observations:
                likelihood *= effective.probability_of_observed(observation)
            weights.append(prior * likelihood)
        if sum(weights) <= 0.0:
            # Every model calls the data impossible. Refusing to update is the
            # honest response: the truth is outside the set considered, and
            # pretending otherwise would manufacture a winner from rounding.
            return self
        return HypothesisSet(self.models, tuple(normalise(weights)))

    def with_posterior(self, posterior: Sequence[float]) -> "HypothesisSet":
        return HypothesisSet(self.models, tuple(normalise(posterior)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [m.to_dict() for m in self.models],
            "posterior": list(self.posterior),
            "entropy_bits": self.entropy(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisSet":
        return cls(
            tuple(CausalModel.from_dict(m) for m in payload["models"]),
            tuple(float(p) for p in payload["posterior"]),
        )
