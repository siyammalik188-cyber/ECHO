"""Choosing which experiment to run, by computing what each one would tell you.

When observational data cannot separate two explanations, more of it will not
help. The way out is to *act*: set a variable rather than watch it, and see
whether the rest of the world follows. Which variable to set is the question
this module answers, and it answers it by arithmetic rather than by a rule
someone wrote down.

**Expected information gain.** Before running experiment *e*, ECHO knows how
uncertain it is: `H(P(H))`, the entropy of its posterior over hypotheses. For
each possible result *c* it can compute how uncertain it *would* be afterwards,
`H(P(H | c, e))`, and how likely that result is, `P(c | e) = Σ_h P(h) P(c | h, e)`.
The expected gain is the difference:

```
EIG(e) = H(P(H)) − Σ_c P(c | e) · H(P(H | c, e))
```

This is the mutual information between the hypothesis and the experiment's
result. It is zero exactly when every hypothesis predicts the same distribution
of results — which is what makes an experiment useless, and is a fact the
formula discovers rather than one it is told.

Everything is enumerated exactly. With binary variables the result of an
n-sample experiment is a count vector over `K` possible observations, and both
the sum over count vectors and the multinomial probabilities are computed in
closed form. There is no sampling anywhere in the appraisal, so two runs cannot
disagree.

**Cost is real.** An experiment that would settle everything but costs a
fortune may still be the wrong one to run. Utility trades the two off:

```
utility(e) = EIG(e) − cost_weight · cost(e) − risk_weight · risk(e)
```

with both weights fixed in advance. Nowhere does anything say "choose the
intervention on V1".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .causal import Assignment, CausalError, CausalModel, HypothesisSet, entropy

# ------------------------------------------------------------------- settings
#
# Pre-registered, fixed before any experiment ran.

#: How many bits one unit of cost is worth. An experiment must gain more than
#: this per unit of cost to be worth its price.
COST_WEIGHT = 0.05

#: The same, for risk.
RISK_WEIGHT = 0.10

#: Above this many distinct count vectors, exact enumeration is abandoned and
#: the per-sample bound is used instead — recorded, never silent.
EXACT_LIMIT = 60_000


class ExperimentError(ValueError):
    pass


# ------------------------------------------------------------- the vocabulary


@dataclass(frozen=True)
class Intervention:
    """`do(V = v)` for one or more variables. An empty one is pure observation."""

    assignments: tuple[tuple[str, int], ...] = ()

    @classmethod
    def of(cls, **assignments: int) -> "Intervention":
        return cls(tuple(sorted(assignments.items())))

    @classmethod
    def observation(cls) -> "Intervention":
        return cls(())

    @property
    def is_observational(self) -> bool:
        return not self.assignments

    def as_mapping(self) -> dict[str, int]:
        return dict(self.assignments)

    def label(self) -> str:
        if self.is_observational:
            return "OBSERVE()"
        inner = ", ".join(f"{v}={x}" for v, x in self.assignments)
        return f"INTERVENE({inner})"

    def to_dict(self) -> dict[str, Any]:
        return {"assignments": [list(a) for a in self.assignments], "label": self.label()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Intervention":
        return cls(tuple((str(v), int(x)) for v, x in payload["assignments"]))


@dataclass(frozen=True)
class ExperimentOption:
    """One thing ECHO could do, and what doing it would take."""

    id: str
    intervention: Intervention
    cost: float
    risk: float
    time: float
    samples: int
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intervention": self.intervention.to_dict(),
            "cost": self.cost,
            "risk": self.risk,
            "time": self.time,
            "samples": self.samples,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentOption":
        return cls(
            id=str(payload["id"]),
            intervention=Intervention.from_dict(payload["intervention"]),
            cost=float(payload["cost"]),
            risk=float(payload["risk"]),
            time=float(payload["time"]),
            samples=int(payload["samples"]),
            description=str(payload.get("description", "")),
        )


# --------------------------------------------------------------- the appraisal


def outcome_space(model: CausalModel, intervention: Intervention) -> list[Assignment]:
    """Every observation the experiment could produce, in a fixed order.

    Intervened variables are held at their set value, so they contribute
    nothing; the outcome is what the *rest* of the world does.
    """
    fixed = intervention.as_mapping()
    ranges = [
        (fixed[variable],) if variable in fixed else (0, 1)
        for variable in model.observed_variables
    ]
    return [tuple(values) for values in product(*ranges)]


def outcome_distribution(
    model: CausalModel, intervention: Intervention, outcomes: Sequence[Assignment]
) -> list[float]:
    effective = (
        model.intervened(intervention.as_mapping())
        if not intervention.is_observational
        else model
    )
    return [effective.probability_of_observed(outcome) for outcome in outcomes]


def _count_vectors(total: int, slots: int) -> Iterable[tuple[int, ...]]:
    """Every way `total` samples can fall into `slots` categories."""
    if slots == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _count_vectors(total - first, slots - 1):
            yield (first,) + rest


def _multinomial_log_probability(counts: Sequence[int], probabilities: Sequence[float]) -> float:
    total = 0.0
    for count, p in zip(counts, probabilities):
        if count == 0:
            continue
        if p <= 0.0:
            return float("-inf")
        total += count * math.log(p)
    return total


@dataclass(frozen=True)
class ExperimentAppraisal:
    """What an experiment is expected to be worth, before it is run."""

    option: ExperimentOption
    prior_entropy: float
    expected_posterior_entropy: float
    expected_information_gain: float
    utility: float
    information_per_cost: float
    exact: bool
    outcomes_considered: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option.id,
            "intervention": self.option.intervention.label(),
            "prior_entropy_bits": self.prior_entropy,
            "expected_posterior_entropy_bits": self.expected_posterior_entropy,
            "expected_information_gain_bits": self.expected_information_gain,
            "cost": self.option.cost,
            "risk": self.option.risk,
            "time": self.option.time,
            "samples": self.option.samples,
            "utility": self.utility,
            "information_per_cost": self.information_per_cost,
            "exact": self.exact,
            "outcomes_considered": self.outcomes_considered,
        }


def appraise(
    option: ExperimentOption,
    hypotheses: HypothesisSet,
    *,
    cost_weight: float = COST_WEIGHT,
    risk_weight: float = RISK_WEIGHT,
    exact_limit: int = EXACT_LIMIT,
) -> ExperimentAppraisal:
    """Compute what this experiment is expected to tell ECHO, in bits."""
    if option.samples < 1:
        raise ExperimentError(f"{option.id} has a sample budget of {option.samples}")

    reference = hypotheses.models[0]
    outcomes = outcome_space(reference, option.intervention)
    per_model = [
        outcome_distribution(model, option.intervention, outcomes)
        for model in hypotheses.models
    ]

    prior_entropy = hypotheses.entropy()
    slots = len(outcomes)
    exact = math.comb(option.samples + slots - 1, slots - 1) <= exact_limit

    if exact:
        expected_posterior = 0.0
        considered = 0
        for counts in _count_vectors(option.samples, slots):
            considered += 1
            log_coefficient = (
                math.lgamma(option.samples + 1)
                - sum(math.lgamma(c + 1) for c in counts)
            )
            weights = []
            for prior, probabilities in zip(hypotheses.posterior, per_model):
                log_likelihood = _multinomial_log_probability(counts, probabilities)
                if log_likelihood == float("-inf") or prior <= 0.0:
                    weights.append(0.0)
                else:
                    weights.append(prior * math.exp(log_coefficient + log_likelihood))
            evidence = sum(weights)
            if evidence <= 0.0:
                continue
            posterior = [w / evidence for w in weights]
            expected_posterior += evidence * entropy(posterior)
        gain = prior_entropy - expected_posterior
    else:
        # One sample at a time, scaled. Information is subadditive across
        # samples, so this is an upper bound — flagged so a reader knows the
        # number is a bound rather than the quantity itself.
        single = _single_sample_gain(hypotheses, per_model, prior_entropy)
        gain = min(prior_entropy, single * option.samples)
        expected_posterior = prior_entropy - gain
        considered = slots

    gain = max(0.0, gain)
    utility = gain - cost_weight * option.cost - risk_weight * option.risk
    per_cost = gain / option.cost if option.cost > 0 else float("inf")
    return ExperimentAppraisal(
        option=option,
        prior_entropy=prior_entropy,
        expected_posterior_entropy=expected_posterior,
        expected_information_gain=gain,
        utility=utility,
        information_per_cost=per_cost,
        exact=exact,
        outcomes_considered=considered,
    )


def _single_sample_gain(
    hypotheses: HypothesisSet,
    per_model: Sequence[Sequence[float]],
    prior_entropy: float,
) -> float:
    expected = 0.0
    for index in range(len(per_model[0])):
        weights = [
            prior * probabilities[index]
            for prior, probabilities in zip(hypotheses.posterior, per_model)
        ]
        evidence = sum(weights)
        if evidence <= 0.0:
            continue
        expected += evidence * entropy([w / evidence for w in weights])
    return prior_entropy - expected


# ------------------------------------------------------------------ policies


class Policy:
    """How to pick among appraised experiments."""

    name = "policy"

    def choose(
        self, appraisals: Sequence[ExperimentAppraisal], rng: random.Random
    ) -> ExperimentAppraisal:
        raise NotImplementedError


class InformationGainPolicy(Policy):
    """Highest utility: expected bits, net of what they cost to buy."""

    name = "information_gain"

    def choose(
        self, appraisals: Sequence[ExperimentAppraisal], rng: random.Random
    ) -> ExperimentAppraisal:
        return max(appraisals, key=lambda a: (a.utility, -a.option.cost, a.option.id))


class MaxInformationPolicy(Policy):
    """Most bits, cost ignored. Included to show what ignoring cost buys."""

    name = "max_information"

    def choose(
        self, appraisals: Sequence[ExperimentAppraisal], rng: random.Random
    ) -> ExperimentAppraisal:
        return max(
            appraisals,
            key=lambda a: (a.expected_information_gain, -a.option.cost, a.option.id),
        )


class CheapestPolicy(Policy):
    """Lowest cost, information ignored. The other half of the trade-off."""

    name = "cheapest"

    def choose(
        self, appraisals: Sequence[ExperimentAppraisal], rng: random.Random
    ) -> ExperimentAppraisal:
        return min(appraisals, key=lambda a: (a.option.cost, a.option.id))


class RandomPolicy(Policy):
    """A uniformly random choice. The control the whole challenge turns on."""

    name = "random"

    def choose(
        self, appraisals: Sequence[ExperimentAppraisal], rng: random.Random
    ) -> ExperimentAppraisal:
        return appraisals[rng.randrange(len(appraisals))]


POLICIES: dict[str, Policy] = {
    p.name: p
    for p in (
        InformationGainPolicy(),
        MaxInformationPolicy(),
        CheapestPolicy(),
        RandomPolicy(),
    )
}


def choose_experiment(
    options: Sequence[ExperimentOption],
    hypotheses: HypothesisSet,
    *,
    policy: Policy | str = "information_gain",
    rng: random.Random | None = None,
    cost_weight: float = COST_WEIGHT,
    risk_weight: float = RISK_WEIGHT,
) -> tuple[ExperimentAppraisal, list[ExperimentAppraisal]]:
    """Appraise every option, then pick one. Returns the pick and the field."""
    if not options:
        raise ExperimentError("there are no experiments to choose from")
    resolved = POLICIES[policy] if isinstance(policy, str) else policy
    appraisals = [
        appraise(
            option, hypotheses, cost_weight=cost_weight, risk_weight=risk_weight
        )
        for option in options
    ]
    chosen = resolved.choose(appraisals, rng or random.Random(0))
    return chosen, appraisals


# -------------------------------------------------------------- the record


@dataclass(frozen=True)
class ExperimentRecord:
    """One experiment, frozen: what was chosen, why, what happened, what moved."""

    experiment_id: str
    step: int
    option_id: str
    intervention: Intervention
    policy: str
    appraisal: dict[str, Any]
    alternatives: tuple[dict[str, Any], ...]
    outcome_counts: tuple[tuple[Assignment, int], ...]
    posterior_before: tuple[tuple[str, float], ...]
    posterior_after: tuple[tuple[str, float], ...]
    entropy_before: float
    entropy_after: float
    cost: float
    created_at: int
    conclusive: bool
    note: str = ""

    @property
    def information_gained(self) -> float:
        """Bits actually removed. Can be negative if the result surprised it."""
        return self.entropy_before - self.entropy_after

    @staticmethod
    def make_id(option_id: str, step: int) -> str:
        digest = hashlib.sha256(f"{option_id}|{step}".encode("utf-8")).hexdigest()[:12]
        return f"EXP-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "step": self.step,
            "option_id": self.option_id,
            "intervention": self.intervention.to_dict(),
            "policy": self.policy,
            "appraisal": self.appraisal,
            "alternatives": list(self.alternatives),
            "outcome_counts": [[list(a), c] for a, c in self.outcome_counts],
            "posterior_before": [list(p) for p in self.posterior_before],
            "posterior_after": [list(p) for p in self.posterior_after],
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "information_gained": self.information_gained,
            "cost": self.cost,
            "created_at": self.created_at,
            "conclusive": self.conclusive,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=str(payload["experiment_id"]),
            step=int(payload["step"]),
            option_id=str(payload["option_id"]),
            intervention=Intervention.from_dict(payload["intervention"]),
            policy=str(payload["policy"]),
            appraisal=dict(payload["appraisal"]),
            alternatives=tuple(dict(a) for a in payload.get("alternatives", [])),
            outcome_counts=tuple(
                (tuple(a), int(c)) for a, c in payload.get("outcome_counts", [])
            ),
            posterior_before=tuple(
                (str(k), float(v)) for k, v in payload["posterior_before"]
            ),
            posterior_after=tuple(
                (str(k), float(v)) for k, v in payload["posterior_after"]
            ),
            entropy_before=float(payload["entropy_before"]),
            entropy_after=float(payload["entropy_after"]),
            cost=float(payload["cost"]),
            created_at=int(payload["created_at"]),
            conclusive=bool(payload["conclusive"]),
            note=str(payload.get("note", "")),
        )


class ExperimentLedger:
    """Append-only history of experiments. Nothing is ever rewritten."""

    SCHEMA_VERSION = 1
    DEFAULT_FILENAME = "experiments.json"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records: list[ExperimentRecord] = []

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentLedger":
        ledger = cls(path)
        if not ledger.path.is_file():
            return ledger
        with ledger.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported experiment schema_version {payload.get('schema_version')!r}"
            )
        ledger._records = [
            ExperimentRecord.from_dict(r) for r in payload.get("experiments", [])
        ]
        return ledger

    @classmethod
    def in_directory(cls, directory: Path | str) -> "ExperimentLedger":
        return cls.load(Path(directory) / cls.DEFAULT_FILENAME)

    def add(self, record: ExperimentRecord) -> ExperimentRecord:
        self._records.append(record)
        return record

    def records(self) -> tuple[ExperimentRecord, ...]:
        return tuple(self._records)

    def total_cost(self) -> float:
        return sum(r.cost for r in self._records)

    def total_information(self) -> float:
        if not self._records:
            return 0.0
        return self._records[0].entropy_before - self._records[-1].entropy_after

    def __len__(self) -> int:
        # Never use in a boolean context; an empty ledger is falsy.
        return len(self._records)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "experiments": [r.to_dict() for r in self._records],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)
        return self.path
