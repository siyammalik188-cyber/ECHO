"""Structural patterns: the shape of a discovered relationship, minus the variables.

Challenge 5 ended with a concrete expression — say `PRODUCT(CHANGE(X2), LAG(X4,
3))`. That expression is useless in a world that has no `X2`. What might carry
over is its *shape*: a product of a first difference and a delayed second
series. This module turns the former into the latter.

```
concrete                              abstract
PRODUCT(CHANGE(X2), LAG(X4, 3))  -->  PRODUCT(CHANGE(A), LAG(B, 3))
```

`A` and `B` are slots. They are not `X2` and `X4` under another name — the
pattern keeps no record of which variable filled them, and a test asserts that
no source variable name survives serialisation. Grounding a pattern in a new
world means trying every assignment of that world's variables to the slots.

**The abstraction is mechanical.** `abstract()` is a pure function of whatever
expression discovery returned. Nowhere does a human write down the right shape;
if discovery had found something else, the pattern would be that instead. That
is the whole anti-cheating requirement, and it is a property of the code rather
than a promise.

**Parameters are hints, not constants.** A lag of 3 in the source becomes a
*preference* for 3 in the target, and grounding also tries the neighbouring
values. A pattern that could only ever fire at exactly lag 3 would have
memorised a number rather than learned a shape.

**Patterns decompose.** `PRODUCT(CHANGE(A), LAG(B, 3))` has two components, and
each is a pattern in its own right. A world where only the first difference
matters can accept half of what was transferred and weaken the rest, instead of
swallowing or rejecting the whole thing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import product as iter_product
from typing import Any, Iterable, Mapping, Sequence

from . import expressions as ex
from .expressions import Expr

#: How far from the source's parameter value grounding will look. Fixed before
#: any transfer experiment ran.
PARAMETER_TOLERANCE = 1

#: Slot names, in assignment order.
SLOT_NAMES = ("A", "B", "C")

#: A grounding that would produce more than this many candidates is refused
#: rather than silently truncated — a "restricted" search that quietly became
#: an exhaustive one would make every sample-efficiency number meaningless.
MAX_GROUNDINGS = 4096


class PatternError(ValueError):
    """Raised when a pattern cannot be built, grounded, or read."""


class PatternStatus(str, Enum):
    DISCOVERED = "discovered"  # abstracted from a discovery, never yet tried elsewhere
    TESTING = "testing"  # being tried in at least one target world
    TRANSFERABLE = "transferable"  # earned it: repeated successful transfer
    WEAKENED = "weakened"  # part of it failed, or a transfer went badly
    REJECTED = "rejected"  # tried and did not carry
    RETIRED = "retired"  # withdrawn from use; kept in history


_ALLOWED: dict[PatternStatus, tuple[PatternStatus, ...]] = {
    PatternStatus.DISCOVERED: (PatternStatus.TESTING, PatternStatus.REJECTED),
    PatternStatus.TESTING: (
        PatternStatus.TRANSFERABLE,
        PatternStatus.WEAKENED,
        PatternStatus.REJECTED,
    ),
    PatternStatus.TRANSFERABLE: (
        PatternStatus.TESTING,
        PatternStatus.WEAKENED,
        PatternStatus.RETIRED,
    ),
    PatternStatus.WEAKENED: (
        PatternStatus.TESTING,
        PatternStatus.REJECTED,
        PatternStatus.RETIRED,
    ),
    PatternStatus.REJECTED: (PatternStatus.RETIRED,),
    PatternStatus.RETIRED: (),
}

#: How many independent successful transfers a pattern needs before it is
#: called TRANSFERABLE. Two, not one: working once is an anecdote.
TRANSFERABLE_AFTER = 2


# ------------------------------------------------------------- abstract trees


@dataclass(frozen=True)
class AbstractNode:
    """One node of a pattern. Carries a slot name, never a variable name."""

    op: str
    slot: str | None = None
    parameter: float | None = None
    children: tuple["AbstractNode", ...] = ()

    def walk(self) -> tuple["AbstractNode", ...]:
        out: list[AbstractNode] = [self]
        for child in self.children:
            out.extend(child.walk())
        return tuple(out)

    def slots(self) -> tuple[str, ...]:
        seen: list[str] = []
        for node in self.walk():
            if node.slot is not None and node.slot not in seen:
                seen.append(node.slot)
        return tuple(seen)

    def parameterised(self) -> tuple["AbstractNode", ...]:
        return tuple(n for n in self.walk() if n.parameter is not None and n.op != "CONST")

    def text(self) -> str:
        if self.op in ("FEATURE", "CHANGE"):
            return f"{self.op}({self.slot})"
        if self.op in ("LAG", "MEAN", "SUM"):
            return f"{self.op}({self.slot}, {int(self.parameter)})"
        if self.op == "CONST":
            return f"CONST({self.parameter:g})"
        return f"{self.op}({', '.join(child.text() for child in self.children)})"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.op}
        if self.slot is not None:
            payload["slot"] = self.slot
        if self.parameter is not None:
            payload["parameter"] = self.parameter
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AbstractNode":
        op = payload.get("op")
        if op not in _ABSTRACTABLE:
            raise PatternError(f"{op!r} is not an abstractable primitive")
        return cls(
            op=str(op),
            slot=payload.get("slot"),
            parameter=payload.get("parameter"),
            children=tuple(
                cls.from_dict(child) for child in payload.get("children", ())
            ),
        )


_UNARY_PARAMETERISED = {"LAG": "steps", "MEAN": "window", "SUM": "window"}
_UNARY_PLAIN = {"FEATURE", "CHANGE"}
_BINARY = {"DIFFERENCE", "PRODUCT", "RATIO"}
_ABSTRACTABLE = set(_UNARY_PARAMETERISED) | _UNARY_PLAIN | _BINARY | {"CONST"}


def abstract_node(expr: Expr, assignment: dict[str, str]) -> AbstractNode:
    """Strip variable identities, keep structure. Slots are assigned in order."""
    op = expr.op
    if op == "CONST":
        return AbstractNode(op="CONST", parameter=float(getattr(expr, "value")))
    if op in _UNARY_PLAIN or op in _UNARY_PARAMETERISED:
        variable = getattr(expr, "variable")
        if variable not in assignment:
            index = len(assignment)
            if index >= len(SLOT_NAMES):
                raise PatternError("expression uses more variables than there are slots")
            assignment[variable] = SLOT_NAMES[index]
        slot = assignment[variable]
        parameter = None
        if op in _UNARY_PARAMETERISED:
            parameter = float(getattr(expr, _UNARY_PARAMETERISED[op]))
        return AbstractNode(op=op, slot=slot, parameter=parameter)
    if op in _BINARY:
        return AbstractNode(
            op=op,
            children=tuple(abstract_node(child, assignment) for child in expr.children()),
        )
    raise PatternError(f"{op!r} cannot be abstracted")


def describe(node: AbstractNode) -> str:
    """Plain prose for the shape. No variable names, and no causal claim."""
    if node.op == "FEATURE":
        return "a series"
    if node.op == "CHANGE":
        return "the step-to-step change in a series"
    if node.op == "LAG":
        return f"a series delayed by {int(node.parameter)} step(s)"
    if node.op == "MEAN":
        return f"the {int(node.parameter)}-step moving average of a series"
    if node.op == "SUM":
        return f"the {int(node.parameter)}-step running total of a series"
    if node.op == "CONST":
        return f"the constant {node.parameter:g}"
    joiner = {
        "PRODUCT": "multiplied by",
        "DIFFERENCE": "minus",
        "RATIO": "divided by",
    }[node.op]
    left, right = node.children
    return f"{describe(left)} {joiner} {describe(right)}"


# --------------------------------------------------------------- the pattern


@dataclass(frozen=True)
class StructuralPattern:
    pattern_id: str
    template: AbstractNode
    slots: tuple[str, ...]
    complexity: int
    created_at: int
    origin: str  # how it came to exist; never names a source variable
    status: PatternStatus = PatternStatus.DISCOVERED
    confidence: float = 0.25
    version: int = 1
    successes: int = 0
    failures: int = 0
    evidence: tuple[str, ...] = field(default_factory=tuple)
    status_reason: str = "abstracted from a discovery"
    parent_pattern_id: str | None = None

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_expression(
        cls, expr: Expr, *, created_at: int, origin: str = "abstracted from a discovery"
    ) -> "StructuralPattern":
        """The only way a pattern is built: mechanically, from a discovery."""
        template = abstract_node(expr, {})
        return cls._of(template, created_at=created_at, origin=origin)

    @classmethod
    def _of(
        cls,
        template: AbstractNode,
        *,
        created_at: int,
        origin: str,
        parent: str | None = None,
    ) -> "StructuralPattern":
        digest = hashlib.sha256(template.text().encode("utf-8")).hexdigest()[:12]
        return cls(
            pattern_id=f"PAT-{digest}",
            template=template,
            slots=template.slots(),
            complexity=len(template.walk()),
            created_at=created_at,
            origin=origin,
            parent_pattern_id=parent,
        )

    # ------------------------------------------------------------- structure

    def text(self) -> str:
        return self.template.text()

    def description(self) -> str:
        return describe(self.template)

    def components(self) -> tuple["StructuralPattern", ...]:
        """The pattern's parts, each usable on its own.

        This is what makes partial transfer possible: a world in which only
        half the structure holds can take that half rather than being forced
        to choose between the whole thing and nothing.
        """
        if not self.template.children:
            return ()
        return tuple(
            StructuralPattern._of(
                child,
                created_at=self.created_at,
                origin=f"component of {self.pattern_id}",
                parent=self.pattern_id,
            )
            for child in self.template.children
        )

    # ------------------------------------------------------------- lifecycle

    def _next(self, **changes: Any) -> "StructuralPattern":
        return replace(self, version=self.version + 1, **changes)

    def with_status(self, status: PatternStatus, reason: str) -> "StructuralPattern":
        if status not in _ALLOWED[self.status]:
            raise PatternError(
                f"{self.pattern_id} cannot move from {self.status.value} to {status.value}"
            )
        return self._next(status=status, status_reason=reason)

    def with_evidence(
        self, note: str, *, success: bool, confidence: float | None = None
    ) -> "StructuralPattern":
        """Record one transfer outcome. Confidence is earned, never assumed."""
        successes = self.successes + (1 if success else 0)
        failures = self.failures + (0 if success else 1)
        total = successes + failures
        # Laplace-smoothed success rate: one good result is not a track record.
        earned = (successes + 1) / (total + 2) if confidence is None else confidence
        return self._next(
            successes=successes,
            failures=failures,
            confidence=earned,
            evidence=self.evidence + (note,),
        )

    def earned_transferable(self) -> bool:
        return self.successes >= TRANSFERABLE_AFTER

    # --------------------------------------------------------------- grounding

    def ground(
        self,
        variable_names: Sequence[str],
        *,
        tolerance: int = PARAMETER_TOLERANCE,
        max_candidates: int = MAX_GROUNDINGS,
    ) -> list[Expr]:
        """Every concrete expression this shape can take in a given world.

        Slots range over the world's variables; parameters range over a small
        neighbourhood of the source's value. The result is a restricted
        candidate set — the point of transfer is that it is far smaller than
        the exhaustive space, and how much smaller is measured, not asserted.
        """
        slots = self.slots
        parameterised = self.template.parameterised()

        parameter_choices: list[list[int]] = []
        for node in parameterised:
            base = int(node.parameter or 0)
            if node.op == "LAG":
                low, high = 1, ex.MAX_LAG
            else:
                low, high = max(2, ex.MIN_WINDOW), ex.MAX_WINDOW
            values = sorted(
                {v for v in range(base - tolerance, base + tolerance + 1) if low <= v <= high}
            )
            if not values:
                values = [max(low, min(high, base))]
            parameter_choices.append(values)

        assignments = list(iter_product(variable_names, repeat=len(slots))) or [()]
        parameter_sets = list(iter_product(*parameter_choices)) if parameter_choices else [()]

        total = len(assignments) * len(parameter_sets)
        if total > max_candidates:
            raise PatternError(
                f"grounding {self.text()} in {len(variable_names)} variables would "
                f"produce {total} candidates, above the limit of {max_candidates}"
            )

        out: list[Expr] = []
        seen: set[str] = set()
        for assignment in assignments:
            mapping = dict(zip(slots, assignment))
            for parameters in parameter_sets:
                overrides = {id(node): value for node, value in zip(parameterised, parameters)}
                try:
                    expr = _instantiate(self.template, mapping, overrides)
                    ex.validate(expr)
                except (ex.ExpressionError, PatternError):
                    continue
                text = expr.text()
                if text in seen:
                    continue
                seen.add(text)
                out.append(expr)
        # Fixed order, so a grounded search is as reproducible as an exhaustive one.
        out.sort(key=lambda e: e.text())
        return out

    # ---------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "template": self.template.to_dict(),
            "template_text": self.template.text(),
            "description": self.description(),
            "slots": list(self.slots),
            "complexity": self.complexity,
            "created_at": self.created_at,
            "origin": self.origin,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "confidence": self.confidence,
            "version": self.version,
            "successes": self.successes,
            "failures": self.failures,
            "evidence": list(self.evidence),
            "parent_pattern_id": self.parent_pattern_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StructuralPattern":
        template = AbstractNode.from_dict(payload["template"])
        return cls(
            pattern_id=str(payload["pattern_id"]),
            template=template,
            slots=tuple(payload["slots"]),
            complexity=int(payload["complexity"]),
            created_at=int(payload["created_at"]),
            origin=str(payload["origin"]),
            status=PatternStatus(payload["status"]),
            status_reason=str(payload.get("status_reason", "")),
            confidence=float(payload.get("confidence", 0.25)),
            version=int(payload.get("version", 1)),
            successes=int(payload.get("successes", 0)),
            failures=int(payload.get("failures", 0)),
            evidence=tuple(payload.get("evidence", ())),
            parent_pattern_id=payload.get("parent_pattern_id"),
        )


def _instantiate(
    node: AbstractNode, mapping: Mapping[str, str], overrides: Mapping[int, int]
) -> Expr:
    op = node.op
    if op == "CONST":
        return ex.Const(float(node.parameter or 0.0))
    if op in _UNARY_PLAIN:
        variable = mapping[node.slot or ""]
        return ex.Feature(variable) if op == "FEATURE" else ex.Change(variable)
    if op in _UNARY_PARAMETERISED:
        variable = mapping[node.slot or ""]
        value = overrides.get(id(node), int(node.parameter or 1))
        if op == "LAG":
            return ex.Lag(variable, value)
        if op == "MEAN":
            return ex.Mean(variable, value)
        return ex.Sum(variable, value)
    if op in _BINARY:
        left, right = (_instantiate(child, mapping, overrides) for child in node.children)
        if op == "PRODUCT":
            return ex.Product(left, right)
        if op == "DIFFERENCE":
            return ex.Difference(left, right)
        return ex.Ratio(left, right)
    raise PatternError(f"cannot instantiate {op!r}")


def contains_variable_names(payload: Any, names: Iterable[str]) -> list[str]:
    """Audit helper: which of `names` appear anywhere in a serialised object.

    Used by the tests that assert a pattern carries no trace of the world it
    came from. It is here rather than in the test file so the check is part of
    the system, not just part of its examination.
    """
    import json

    text = json.dumps(payload, default=str)
    return [name for name in names if name in text]
