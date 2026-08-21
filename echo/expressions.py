"""A closed language for candidate hypotheses.

ECHO searches for predictive relationships by building expressions. It does
**not** build Python. There is no `eval`, no `exec`, no `compile`, no import
machinery and no string that is ever executed anywhere in this module or in the
search that uses it. A hypothesis is a tree of frozen dataclasses, and the only
thing that can happen to a tree is that a hand-written interpreter walks it and
returns numbers.

That restriction is the point. An unbounded generator that can emit arbitrary
code is not a hypothesis-search system; it is a remote shell with extra steps.

**Primitives.**

| Form | Meaning | Warm-up |
| --- | --- | --- |
| `FEATURE(X)` | the value of X at time t | 0 |
| `LAG(X, n)` | the value of X at time t-n | n |
| `CHANGE(X)` | X at t minus X at t-1 | 1 |
| `MEAN(X, n)` | mean of X over the n steps ending at t | n-1 |
| `SUM(X, n)` | sum of X over the n steps ending at t | n-1 |
| `CONST(c)` | an approved constant | 0 |
| `DIFFERENCE(a, b)` | a - b | max |
| `RATIO(a, b)` | a / b, denominator clamped away from zero | max |
| `PRODUCT(a, b)` | a * b | max |

The unary forms take a **variable**, not a sub-expression. That is a deliberate
bound on the space: it keeps every tree shallow enough to enumerate and to read,
and it matches the shapes the challenge names. Binary forms combine any two
sub-expressions.

**No primitive can see the future.** Every form reads time t or earlier. There
is no `LEAD`, and a negative lag is rejected at construction. This is not a
convention that the search is trusted to respect — the language cannot express
the violation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

#: The default schema, used by the ECHO 5 worlds and by anything that needs
#: a schema without being handed one. Worlds may use other names entirely —
#: transfer depends on it, since a target world shares no names with its
#: source. Membership is checked by `check_schema`, not at construction.
VARIABLE_NAMES: tuple[str, ...] = ("X1", "X2", "X3", "X4", "X5", "X6")

# --------------------------------------------------------------------- limits
#
# Chosen before any experiment was run and not changed afterwards. They bound
# the search space; they are not tuned against results.

MAX_LAG = 4
MIN_WINDOW = 1
MAX_WINDOW = 5
MAX_DEPTH = 3
MAX_VARIABLES = 2
MAX_OPERATIONS = 6

#: No expression in this language can require more history than this.
WARMUP_FLOOR = max(MAX_LAG, MAX_WINDOW - 1)

#: Denominators smaller than this are clamped, so RATIO is total: it returns a
#: number for every row rather than dropping rows and quietly changing which
#: observations a candidate is scored on.
RATIO_EPSILON = 1e-3

ALLOWED_CONSTANTS: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


class ExpressionError(ValueError):
    """Raised when a tree steps outside the language."""


# ----------------------------------------------------------------------- nodes


class Expr:
    """Base class. Every node is frozen; there is no mutation anywhere."""

    op: ClassVar[str] = ""
    #: What this node costs towards an expression's complexity. A parameterised
    #: node costs more than a bare one: choosing `LAG(X3, 2)` is two decisions
    #: (which variable, which lag) where `FEATURE(X3)` is one.
    cost: ClassVar[int] = 1

    # -- structure ----------------------------------------------------------

    def children(self) -> tuple["Expr", ...]:
        return ()

    def variables(self) -> tuple[str, ...]:
        seen: list[str] = []
        for node in self.walk():
            name = getattr(node, "variable", None)
            if name is not None and name not in seen:
                seen.append(name)
        return tuple(seen)

    def walk(self) -> tuple["Expr", ...]:
        out: list[Expr] = [self]
        for child in self.children():
            out.extend(child.walk())
        return tuple(out)

    def complexity(self) -> int:
        """Summed node cost. `PRODUCT(CHANGE(X2), LAG(X4, 3))` costs 1+2+2 = 5."""
        return sum(node.cost for node in self.walk())

    def depth(self) -> int:
        children = self.children()
        return 1 + (max(child.depth() for child in children) if children else 0)

    def operations(self) -> int:
        """Every node except a bare constant counts as an operation."""
        return sum(1 for node in self.walk() if node.op != "CONST")

    def warmup(self) -> int:
        """Rows at the start of a series for which this expression is undefined."""
        children = self.children()
        return max((child.warmup() for child in children), default=0)

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        raise NotImplementedError

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def text(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text()


def _column(columns: Mapping[str, Sequence[float]], name: str) -> Sequence[float]:
    try:
        return columns[name]
    except KeyError:
        raise ExpressionError(f"no observed variable named {name!r}") from None


#: A variable name is a plain identifier and nothing else. The name is only
#: ever used as a dictionary key against an observation set's columns, so this
#: is not the security boundary — that is the whitelist of operators. What it
#: does buy is that a serialised expression cannot carry a payload dressed up
#: as a variable name, and that malformed names fail loudly at construction.
_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


def _check_variable(name: str) -> str:
    if not isinstance(name, str) or not _VALID_NAME.match(name):
        raise ExpressionError(f"{name!r} is not a valid variable name")
    return name


def check_schema(expr: "Expr", allowed: Sequence[str]) -> "Expr":
    """Assert every variable in `expr` exists in a particular world.

    Kept separate from `validate` because which variables exist is a fact about
    an observation set, not about the language. A world named `X1…X6` and a
    world named `Z1…Z6` are both expressible; only one of them has an `X2`.
    """
    unknown = [name for name in expr.variables() if name not in allowed]
    if unknown:
        raise ExpressionError(
            f"{', '.join(unknown)} not present in this world "
            f"(has: {', '.join(allowed)})"
        )
    return expr


@dataclass(frozen=True)
class Feature(Expr):
    variable: str
    op: ClassVar[str] = "FEATURE"

    def __post_init__(self) -> None:
        _check_variable(self.variable)

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        return list(_column(columns, self.variable))

    def to_dict(self) -> dict[str, Any]:
        return {"op": "FEATURE", "variable": self.variable}

    def text(self) -> str:
        return f"FEATURE({self.variable})"


@dataclass(frozen=True)
class Lag(Expr):
    variable: str
    steps: int
    op: ClassVar[str] = "LAG"
    cost: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _check_variable(self.variable)
        if not isinstance(self.steps, int) or isinstance(self.steps, bool):
            raise ExpressionError("LAG steps must be a whole number")
        if self.steps < 1:
            # A lag of zero is FEATURE; a negative lag would read the future.
            # Neither is expressible here.
            raise ExpressionError(f"LAG steps must be at least 1, got {self.steps}")
        if self.steps > MAX_LAG:
            raise ExpressionError(f"LAG steps must be at most {MAX_LAG}, got {self.steps}")

    def warmup(self) -> int:
        return self.steps

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        series = _column(columns, self.variable)
        n = self.steps
        return [0.0] * n + list(series[: len(series) - n])

    def to_dict(self) -> dict[str, Any]:
        return {"op": "LAG", "variable": self.variable, "steps": self.steps}

    def text(self) -> str:
        return f"LAG({self.variable}, {self.steps})"


@dataclass(frozen=True)
class Change(Expr):
    variable: str
    op: ClassVar[str] = "CHANGE"
    cost: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _check_variable(self.variable)

    def warmup(self) -> int:
        return 1

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        series = _column(columns, self.variable)
        out = [0.0]
        out.extend(series[t] - series[t - 1] for t in range(1, len(series)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"op": "CHANGE", "variable": self.variable}

    def text(self) -> str:
        return f"CHANGE({self.variable})"


def _check_window(window: int) -> int:
    if not isinstance(window, int) or isinstance(window, bool):
        raise ExpressionError("window must be a whole number")
    if window < MIN_WINDOW or window > MAX_WINDOW:
        raise ExpressionError(
            f"window must be between {MIN_WINDOW} and {MAX_WINDOW}, got {window}"
        )
    return window


@dataclass(frozen=True)
class Mean(Expr):
    variable: str
    window: int
    op: ClassVar[str] = "MEAN"
    cost: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _check_variable(self.variable)
        _check_window(self.window)

    def warmup(self) -> int:
        return self.window - 1

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        series = _column(columns, self.variable)
        window = self.window
        out: list[float] = []
        running = 0.0
        for t, value in enumerate(series):
            running += value
            if t >= window:
                running -= series[t - window]
            out.append(running / window if t >= window - 1 else 0.0)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"op": "MEAN", "variable": self.variable, "window": self.window}

    def text(self) -> str:
        return f"MEAN({self.variable}, {self.window})"


@dataclass(frozen=True)
class Sum(Expr):
    variable: str
    window: int
    op: ClassVar[str] = "SUM"
    cost: ClassVar[int] = 2

    def __post_init__(self) -> None:
        _check_variable(self.variable)
        _check_window(self.window)

    def warmup(self) -> int:
        return self.window - 1

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        series = _column(columns, self.variable)
        window = self.window
        out: list[float] = []
        running = 0.0
        for t, value in enumerate(series):
            running += value
            if t >= window:
                running -= series[t - window]
            out.append(running if t >= window - 1 else 0.0)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"op": "SUM", "variable": self.variable, "window": self.window}

    def text(self) -> str:
        return f"SUM({self.variable}, {self.window})"


@dataclass(frozen=True)
class Const(Expr):
    value: float
    op: ClassVar[str] = "CONST"

    def __post_init__(self) -> None:
        if float(self.value) not in ALLOWED_CONSTANTS:
            raise ExpressionError(
                f"{self.value!r} is not an approved constant "
                f"(allowed: {', '.join(str(c) for c in ALLOWED_CONSTANTS)})"
            )

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        length = len(next(iter(columns.values())))
        return [float(self.value)] * length

    def to_dict(self) -> dict[str, Any]:
        return {"op": "CONST", "value": float(self.value)}

    def text(self) -> str:
        return f"CONST({float(self.value):g})"


@dataclass(frozen=True)
class _Binary(Expr):
    left: Expr
    right: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)

    def _combine(self, a: float, b: float) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def evaluate(self, columns: Mapping[str, Sequence[float]]) -> list[float]:
        left = self.left.evaluate(columns)
        right = self.right.evaluate(columns)
        combine = self._combine
        return [combine(a, b) for a, b in zip(left, right)]

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}

    def text(self) -> str:
        return f"{self.op}({self.left.text()}, {self.right.text()})"


@dataclass(frozen=True)
class Difference(_Binary):
    op: ClassVar[str] = "DIFFERENCE"

    def _combine(self, a: float, b: float) -> float:
        return a - b


@dataclass(frozen=True)
class Product(_Binary):
    op: ClassVar[str] = "PRODUCT"

    def _combine(self, a: float, b: float) -> float:
        return a * b


@dataclass(frozen=True)
class Ratio(_Binary):
    op: ClassVar[str] = "RATIO"

    def _combine(self, a: float, b: float) -> float:
        # Clamped rather than masked. Dropping rows where a denominator happens
        # to be small would mean different candidates were scored on different
        # observations, which would make their scores incomparable in a way
        # nothing downstream could detect.
        if b >= 0.0:
            return a / max(b, RATIO_EPSILON)
        return a / min(b, -RATIO_EPSILON)


# ------------------------------------------------------------------ validation


def validate(
    expr: Expr,
    *,
    max_depth: int = MAX_DEPTH,
    max_variables: int = MAX_VARIABLES,
    max_operations: int = MAX_OPERATIONS,
) -> Expr:
    """Reject anything outside the declared bounds. Returns the expression."""
    if not isinstance(expr, Expr):
        raise ExpressionError(f"{type(expr).__name__} is not an expression")
    depth = expr.depth()
    if depth > max_depth:
        raise ExpressionError(f"depth {depth} exceeds the limit of {max_depth}")
    variables = expr.variables()
    if len(variables) > max_variables:
        raise ExpressionError(
            f"{len(variables)} variables exceeds the limit of {max_variables}: {variables}"
        )
    operations = expr.operations()
    if operations > max_operations:
        raise ExpressionError(
            f"{operations} operations exceeds the limit of {max_operations}"
        )
    warmup = expr.warmup()
    if warmup > WARMUP_FLOOR:
        raise ExpressionError(f"warm-up {warmup} exceeds the floor of {WARMUP_FLOOR}")
    return expr


# --------------------------------------------------------------- serialisation

#: The whitelist. `from_dict` will build nothing that is not named here, so a
#: payload naming `__import__`, `system`, or any other attribute is simply an
#: unknown operator and is refused.
_BUILDERS: dict[str, Any] = {
    "FEATURE": lambda d: Feature(variable=str(d["variable"])),
    "LAG": lambda d: Lag(variable=str(d["variable"]), steps=int(d["steps"])),
    "CHANGE": lambda d: Change(variable=str(d["variable"])),
    "MEAN": lambda d: Mean(variable=str(d["variable"]), window=int(d["window"])),
    "SUM": lambda d: Sum(variable=str(d["variable"]), window=int(d["window"])),
    "CONST": lambda d: Const(value=float(d["value"])),
    "DIFFERENCE": lambda d: Difference(from_dict(d["left"]), from_dict(d["right"])),
    "PRODUCT": lambda d: Product(from_dict(d["left"]), from_dict(d["right"])),
    "RATIO": lambda d: Ratio(from_dict(d["left"]), from_dict(d["right"])),
}

PRIMITIVES: tuple[str, ...] = tuple(sorted(_BUILDERS))


def from_dict(payload: Any) -> Expr:
    """Rebuild an expression from its serialised form.

    Structural, not textual: nothing here is executed, and an operator that is
    not in the whitelist above is an error rather than an attribute lookup.
    """
    if not isinstance(payload, dict):
        raise ExpressionError(f"expression payload must be an object, got {type(payload).__name__}")
    op = payload.get("op")
    if not isinstance(op, str):
        raise ExpressionError("expression payload has no operator")
    builder = _BUILDERS.get(op)
    if builder is None:
        raise ExpressionError(f"{op!r} is not an approved primitive")
    try:
        return builder(payload)
    except ExpressionError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ExpressionError(f"malformed {op} payload: {exc}") from None


# ---------------------------------------------------------------------- parser
#
# A tokeniser and a recursive-descent reader for the printed form. It exists so
# that tests and provenance records can be written the way a person reads them.
# It builds the same whitelisted nodes as `from_dict` and shares its refusal to
# recognise anything else.

_PUNCTUATION = "(),"


def _tokenise(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for character in text:
        if character in _PUNCTUATION:
            if current.strip():
                tokens.append(current.strip())
            current = ""
            tokens.append(character)
        elif character.isspace():
            if current.strip():
                tokens.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        tokens.append(current.strip())
    return tokens


def _number(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ExpressionError(f"expected a number, got {token!r}") from None


def _parse_tokens(tokens: list[str], position: int) -> tuple[Expr, int]:
    if position >= len(tokens):
        raise ExpressionError("expression ended early")
    op = tokens[position]
    if op not in _BUILDERS:
        raise ExpressionError(f"{op!r} is not an approved primitive")
    position += 1
    if position >= len(tokens) or tokens[position] != "(":
        raise ExpressionError(f"expected '(' after {op}")
    position += 1

    arguments: list[Any] = []
    while True:
        if position >= len(tokens):
            raise ExpressionError("unbalanced parentheses")
        token = tokens[position]
        if token == ")":
            position += 1
            break
        if token == ",":
            position += 1
            continue
        if token in _BUILDERS:
            child, position = _parse_tokens(tokens, position)
            arguments.append(child)
        else:
            arguments.append(token)
            position += 1

    node = _build(op, arguments)
    return node, position


def _build(op: str, arguments: list[Any]) -> Expr:
    def expect(count: int) -> None:
        if len(arguments) != count:
            raise ExpressionError(f"{op} takes {count} argument(s), got {len(arguments)}")

    if op == "FEATURE":
        expect(1)
        return Feature(variable=str(arguments[0]))
    if op == "CHANGE":
        expect(1)
        return Change(variable=str(arguments[0]))
    if op == "LAG":
        expect(2)
        return Lag(variable=str(arguments[0]), steps=int(_number(str(arguments[1]))))
    if op == "MEAN":
        expect(2)
        return Mean(variable=str(arguments[0]), window=int(_number(str(arguments[1]))))
    if op == "SUM":
        expect(2)
        return Sum(variable=str(arguments[0]), window=int(_number(str(arguments[1]))))
    if op == "CONST":
        expect(1)
        return Const(value=_number(str(arguments[0])))
    expect(2)
    left, right = arguments
    if not isinstance(left, Expr) or not isinstance(right, Expr):
        raise ExpressionError(f"{op} combines two expressions, not bare names")
    if op == "DIFFERENCE":
        return Difference(left, right)
    if op == "PRODUCT":
        return Product(left, right)
    if op == "RATIO":
        return Ratio(left, right)
    raise ExpressionError(f"{op!r} is not an approved primitive")


def parse(text: str) -> Expr:
    """Read the printed form back into a tree. Never executes anything."""
    tokens = _tokenise(text)
    if not tokens:
        raise ExpressionError("empty expression")
    node, position = _parse_tokens(tokens, 0)
    if position != len(tokens):
        raise ExpressionError(f"trailing input after expression: {' '.join(tokens[position:])}")
    return node


# ------------------------------------------------------------------ evaluation


def evaluate(expr: Expr, columns: Mapping[str, Sequence[float]]) -> list[float]:
    """The whole interpreter. A tree walk, and nothing else."""
    values = expr.evaluate(columns)
    return [v if math.isfinite(v) else 0.0 for v in values]
