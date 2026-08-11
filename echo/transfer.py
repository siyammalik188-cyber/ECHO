"""Transfer: trying a structure discovered elsewhere, and being willing to drop it.

What crosses from one world to another is a `StructuralPattern` — a shape with
slots, not an expression with variables. In the target world that shape is
*grounded*: every assignment of the target's own columns to the slots, and every
parameter value in a small neighbourhood of the source's. The result is a
restricted candidate set, typically two orders of magnitude smaller than the
exhaustive space, which is where any sample-efficiency benefit has to come from.

Then the target's data decides. The gates are the ones ECHO 5 already uses —
rank on `VAL_A`, confirm exactly one candidate on `VAL_B` against a frozen base
rate, holdout sealed throughout — because a transferred hypothesis that got an
easier test than a discovered one would make the comparison meaningless.

Five things can happen, and four of them are not success:

- `SUCCESSFUL` — the whole shape grounded and confirmed.
- `PARTIAL` — the whole shape failed, but a component of it confirmed. The
  pattern is weakened, not adopted whole and not thrown away.
- `HARMFUL` — the shape confirmed, but following it lands somewhere materially
  worse than searching from scratch would have.
- `REJECTED` — nothing grounded from the shape survived confirmation.
- `INCONCLUSIVE` — the shape could not be grounded or fitted here at all.

A rejection is not the end of the run: transfer is a hint about where to look
first, so when the hint fails ECHO falls back to searching from scratch. A
system that could only ever act on prior knowledge would not be transferring
intelligently, it would be stuck.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from . import discovery as D
from . import expressions as ex
from .expressions import Expr
from .observations import VAL_B, ChronologicalSplit, ObservationSet
from .pattern import PatternError, PatternStatus, StructuralPattern

#: How much worse than cold start a confirmed transfer may be before it counts
#: as harmful rather than merely unhelpful, in absolute Brier on VAL_B.
HARMFUL_MARGIN = 0.02


class TransferResult(str, Enum):
    SUCCESSFUL = "SUCCESSFUL"
    PARTIAL = "PARTIAL"
    HARMFUL = "HARMFUL"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


# ------------------------------------------------------------ candidate source


class PatternSource:
    """Proposes only what a structural pattern can become in this world.

    Like every other `CandidateSource`, it returns expressions and no verdicts.
    Being handed a pattern that worked somewhere else buys a candidate exactly
    one thing: the chance to be measured first. It does not buy a lower bar.
    """

    def __init__(
        self,
        pattern: StructuralPattern,
        *,
        include_components: bool = True,
        name: str | None = None,
    ) -> None:
        self.pattern = pattern
        self.include_components = include_components
        self.name = name or f"pattern-grounding({pattern.text()})"
        self._full: list[Expr] = []
        self._component: list[Expr] = []

    def groundings(self, variable_names: Sequence[str]) -> tuple[list[Expr], list[Expr]]:
        full = self.pattern.ground(variable_names)
        components: list[Expr] = []
        if self.include_components:
            seen = {expr.text() for expr in full}
            for component in self.pattern.components():
                for expr in component.ground(variable_names):
                    if expr.text() not in seen:
                        seen.add(expr.text())
                        components.append(expr)
        return full, components

    def propose(
        self, observations: ObservationSet, split: ChronologicalSplit
    ) -> Iterable[Expr]:
        self._full, self._component = self.groundings(observations.variable_names)
        yield from self._full
        yield from self._component


class FixedSource:
    """Proposes a hand-supplied list of expressions. Used by the ablation.

    Condition C — transferring the raw source expression rather than its shape —
    needs exactly this: no search at all, just the old answer, re-pointed at the
    new world's columns by position.
    """

    def __init__(self, expressions: Sequence[Expr], *, name: str = "fixed") -> None:
        self.expressions = list(expressions)
        self.name = name

    def propose(
        self, observations: ObservationSet, split: ChronologicalSplit
    ) -> Iterable[Expr]:
        for expr in self.expressions:
            try:
                ex.check_schema(expr, observations.variable_names)
            except ex.ExpressionError:
                continue
            yield expr


def remap_positionally(expr: Expr, source: Sequence[str], target: Sequence[str]) -> Expr:
    """Rewrite an expression's variables by column position.

    The naive way to reuse a discovery: assume the new world's third column
    means what the old world's third column meant. It is included in the
    ablation precisely because it is naive — position is not structure, and
    nothing guarantees the columns line up.
    """
    mapping = {s: t for s, t in zip(source, target)}

    def rewrite(node: Expr) -> Expr:
        if node.op == "CONST":
            return node
        if node.children():
            left, right = (rewrite(child) for child in node.children())
            return type(node)(left, right)
        variable = mapping.get(getattr(node, "variable"), getattr(node, "variable"))
        if node.op == "FEATURE":
            return ex.Feature(variable)
        if node.op == "CHANGE":
            return ex.Change(variable)
        if node.op == "LAG":
            return ex.Lag(variable, getattr(node, "steps"))
        if node.op == "MEAN":
            return ex.Mean(variable, getattr(node, "window"))
        return ex.Sum(variable, getattr(node, "window"))

    return rewrite(expr)


# ------------------------------------------------------------ transfer record


@dataclass(frozen=True)
class TransferRecord:
    """One transfer decision, frozen at the moment it was made.

    `performance_before_transfer` is the frozen base rate — what the target
    world looks like to a system that knows nothing. `performance_after_transfer`
    is what the transferred structure achieved. Both are measured on the
    confirmation block, never on the holdout, because a decision informed by
    the holdout would not be a transfer decision, it would be hindsight.
    """

    transfer_id: str
    source_pattern_id: str
    target_environment_id: str
    pattern_version: int
    target_evidence: dict[str, Any]
    transfer_confidence: float
    result: TransferResult
    performance_before_transfer: dict[str, Any] | None
    performance_after_transfer: dict[str, Any] | None
    created_at: int
    explanation: dict[str, str] = field(default_factory=dict)
    grounded_candidates: int = 0
    hypotheses_tested: int = 0
    adopted_expression: str | None = None
    adopted_is_component: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def make_id(pattern_id: str, environment_id: str, created_at: int) -> str:
        digest = hashlib.sha256(
            f"{pattern_id}|{environment_id}|{created_at}".encode("utf-8")
        ).hexdigest()[:12]
        return f"TRF-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "source_pattern_id": self.source_pattern_id,
            "target_environment_id": self.target_environment_id,
            "pattern_version": self.pattern_version,
            "target_evidence": self.target_evidence,
            "transfer_confidence": self.transfer_confidence,
            "result": self.result.value,
            "performance_before_transfer": self.performance_before_transfer,
            "performance_after_transfer": self.performance_after_transfer,
            "created_at": self.created_at,
            "explanation": self.explanation,
            "grounded_candidates": self.grounded_candidates,
            "hypotheses_tested": self.hypotheses_tested,
            "adopted_expression": self.adopted_expression,
            "adopted_is_component": self.adopted_is_component,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransferRecord":
        return cls(
            transfer_id=str(payload["transfer_id"]),
            source_pattern_id=str(payload["source_pattern_id"]),
            target_environment_id=str(payload["target_environment_id"]),
            pattern_version=int(payload["pattern_version"]),
            target_evidence=dict(payload.get("target_evidence", {})),
            transfer_confidence=float(payload["transfer_confidence"]),
            result=TransferResult(payload["result"]),
            performance_before_transfer=payload.get("performance_before_transfer"),
            performance_after_transfer=payload.get("performance_after_transfer"),
            created_at=int(payload["created_at"]),
            explanation=dict(payload.get("explanation", {})),
            grounded_candidates=int(payload.get("grounded_candidates", 0)),
            hypotheses_tested=int(payload.get("hypotheses_tested", 0)),
            adopted_expression=payload.get("adopted_expression"),
            adopted_is_component=bool(payload.get("adopted_is_component", False)),
            notes=tuple(payload.get("notes", ())),
        )


# ----------------------------------------------------------------- assessment


@dataclass
class TransferAttempt:
    """Everything one transfer attempt produced, decided before the holdout."""

    pattern: StructuralPattern
    observations_id: str
    outcome: D.SearchOutcome
    full_groundings: int
    component_groundings: int
    result: TransferResult
    adopted: D.Ranked | None
    adopted_is_component: bool
    reason: str
    cold_reference: float | None = None

    @property
    def confirmed(self) -> bool:
        return self.result in (TransferResult.SUCCESSFUL, TransferResult.PARTIAL)


def _is_component_expression(expr: Expr, pattern: StructuralPattern, names: Sequence[str]) -> bool:
    component_texts = set()
    for component in pattern.components():
        component_texts.update(e.text() for e in component.ground(names))
    return expr.text() in component_texts


def attempt_transfer(
    pattern: StructuralPattern,
    observations: ObservationSet,
    split: ChronologicalSplit,
    *,
    config: D.SearchConfig | None = None,
    cold_reference: float | None = None,
) -> TransferAttempt:
    """Ground the pattern here, search only within it, and judge the outcome.

    `cold_reference` is what a from-scratch search achieved on the confirmation
    block. Supplying it lets a *confirmed* transfer still be called harmful:
    beating the do-nothing baseline is not the same as being the right thing to
    have done.
    """
    config = config or D.SearchConfig()
    source = PatternSource(pattern)
    try:
        full, components = source.groundings(observations.variable_names)
    except PatternError as exc:
        return TransferAttempt(
            pattern=pattern,
            observations_id=observations.id,
            outcome=D.SearchOutcome(
                observations_id=observations.id,
                config=config,
                source_name=source.name,
                proposed=0,
                screened=0,
                fitted=0,
                rejected_degenerate=0,
            ),
            full_groundings=0,
            component_groundings=0,
            result=TransferResult.INCONCLUSIVE,
            adopted=None,
            adopted_is_component=False,
            reason=f"the pattern could not be grounded here: {exc}",
        )

    outcome = D.DiscoverySearch(observations, split, config=config, source=source).run()

    if not outcome.ranking:
        return TransferAttempt(
            pattern=pattern,
            observations_id=observations.id,
            outcome=outcome,
            full_groundings=len(full),
            component_groundings=len(components),
            result=TransferResult.INCONCLUSIVE,
            adopted=None,
            adopted_is_component=False,
            reason="no grounding of the pattern could be fitted in this world",
        )

    winner = outcome.winner
    confirmation = outcome.confirmation
    if outcome.promoted is None or winner is None or confirmation is None:
        return TransferAttempt(
            pattern=pattern,
            observations_id=observations.id,
            outcome=outcome,
            full_groundings=len(full),
            component_groundings=len(components),
            result=TransferResult.REJECTED,
            adopted=None,
            adopted_is_component=False,
            reason=(
                confirmation.why()
                if confirmation is not None
                else "no grounding reached confirmation"
            ),
        )

    is_component = _is_component_expression(
        winner.hypothesis.expression, pattern, observations.variable_names
    )

    # A confirmed transfer can still be the wrong move. If searching from
    # scratch would have landed materially better, following the pattern cost
    # something, and that is what "harmful" means here.
    if (
        cold_reference is not None
        and confirmation.candidate_brier > cold_reference + HARMFUL_MARGIN
    ):
        return TransferAttempt(
            pattern=pattern,
            observations_id=observations.id,
            outcome=outcome,
            full_groundings=len(full),
            component_groundings=len(components),
            result=TransferResult.HARMFUL,
            adopted=winner,
            adopted_is_component=is_component,
            reason=(
                f"confirmed, but {confirmation.candidate_brier:.4f} on "
                f"{VAL_B} is {confirmation.candidate_brier - cold_reference:+.4f} "
                f"worse than searching from scratch ({cold_reference:.4f})"
            ),
            cold_reference=cold_reference,
        )

    if is_component:
        return TransferAttempt(
            pattern=pattern,
            observations_id=observations.id,
            outcome=outcome,
            full_groundings=len(full),
            component_groundings=len(components),
            result=TransferResult.PARTIAL,
            adopted=winner,
            adopted_is_component=True,
            reason=(
                f"the whole shape did not carry, but one component of it did: "
                f"{confirmation.why()}"
            ),
            cold_reference=cold_reference,
        )

    return TransferAttempt(
        pattern=pattern,
        observations_id=observations.id,
        outcome=outcome,
        full_groundings=len(full),
        component_groundings=len(components),
        result=TransferResult.SUCCESSFUL,
        adopted=winner,
        adopted_is_component=False,
        reason=confirmation.why(),
        cold_reference=cold_reference,
    )


# ---------------------------------------------------------------- explanation


NO_UNDERSTANDING = (
    "This describes a predictive relationship and nothing more. ECHO has no "
    "account of why the relationship holds, no model of what these series are, "
    "and makes no causal claim."
)


def explain(attempt: TransferAttempt, target_id: str) -> dict[str, str]:
    """The four-line, human-readable account of a transfer decision."""
    pattern = attempt.pattern
    adopted = (
        attempt.adopted.hypothesis.expression.text() if attempt.adopted else None
    )
    if attempt.result is TransferResult.SUCCESSFUL:
        observation = "An equivalent structural relationship appears predictive here."
        result = f"Performance improved; adopted `{adopted}`."
    elif attempt.result is TransferResult.PARTIAL:
        observation = (
            "The whole shape is not predictive here, but part of it is."
        )
        result = f"Partly carried; adopted the component `{adopted}` and weakened the rest."
    elif attempt.result is TransferResult.HARMFUL:
        observation = (
            "A grounding of the shape passed confirmation, but searching without "
            "it does materially better."
        )
        result = "Following the transferred shape cost accuracy; it was not kept."
    elif attempt.result is TransferResult.REJECTED:
        observation = "No grounding of the shape predicts the outcome here."
        result = "Did not improve; the transferred knowledge was set aside."
    else:
        observation = "The shape could not be evaluated in this world."
        result = "No conclusion drawn."

    return {
        "source_pattern": f"Relationship involving {pattern.description()}.",
        "target_observation": observation,
        "decision": f"Test the transferred pattern in {target_id}.",
        "result": result,
        "evidence": attempt.reason,
        "disclaimer": NO_UNDERSTANDING,
    }


# ------------------------------------------------------- lifecycle bookkeeping


def update_pattern(
    pattern: StructuralPattern, attempt: TransferAttempt, target_id: str
) -> StructuralPattern:
    """Move the pattern along its lifecycle, on evidence rather than on hope.

    Working once does not make a pattern transferable; `TRANSFERABLE_AFTER`
    independent successes do. Anything less than success weakens it.
    """
    success = attempt.result is TransferResult.SUCCESSFUL
    note = f"{target_id}: {attempt.result.value} — {attempt.reason}"
    updated = pattern.with_evidence(note, success=success)

    if updated.status is PatternStatus.DISCOVERED:
        updated = updated.with_status(
            PatternStatus.TESTING, f"first tried in {target_id}"
        )
    elif updated.status is PatternStatus.TRANSFERABLE and not success:
        updated = updated.with_status(
            PatternStatus.WEAKENED, f"{target_id} did not carry: {attempt.result.value}"
        )
        return updated

    if success and updated.earned_transferable():
        if updated.status is PatternStatus.TESTING:
            updated = updated.with_status(
                PatternStatus.TRANSFERABLE,
                f"carried in {updated.successes} independent worlds",
            )
        elif updated.status is PatternStatus.WEAKENED:
            updated = updated.with_status(
                PatternStatus.TESTING, f"re-entered testing after {target_id}"
            )
    elif attempt.result is TransferResult.PARTIAL and updated.status is PatternStatus.TESTING:
        updated = updated.with_status(
            PatternStatus.WEAKENED, f"only a component carried in {target_id}"
        )
    elif attempt.result is TransferResult.HARMFUL and updated.status is PatternStatus.TESTING:
        updated = updated.with_status(
            PatternStatus.WEAKENED, f"following it cost accuracy in {target_id}"
        )

    return updated
