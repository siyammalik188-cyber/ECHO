"""The predictors a strategy can select between.

Deliberately a tiny, fixed menu. A strategy chooses *which* of these to use and
with what parameters; nothing here writes or modifies code, and there is no path
by which ECHO invents a new predictor. Learning here means selecting and
parameterising, not self-modification.
"""

from __future__ import annotations

from typing import Any

LAPLACE_ALL = "laplace_all_history"
LAPLACE_WINDOW = "laplace_recent_window"

KINDS = (LAPLACE_ALL, LAPLACE_WINDOW)


def predict(kind: str, params: dict[str, Any], view) -> tuple[float, str]:
    """Return P(next outcome is A) plus a plain statement of the arithmetic.

    `view` is a `TimelineView` and cannot reach past its horizon, so no
    predictor can see the outcome it is predicting.
    """
    alpha = float(params.get("alpha", 1.0))
    observed = list(view.outcomes())

    if kind == LAPLACE_ALL:
        used = observed
        scope = f"all {len(observed)} prior observation(s)"
    elif kind == LAPLACE_WINDOW:
        window = int(params.get("window", 10))
        used = observed[-window:]
        scope = f"the most recent {len(used)} of {len(observed)} observation(s)"
    else:
        raise ValueError(f"unknown predictor kind {kind!r}")

    n = len(used)
    a = sum(1 for value in used if value)
    probability = (a + alpha) / (n + 2 * alpha)
    rationale = (
        f"{kind}: {a} of {n} were OUTCOME_A over {scope}; "
        f"({a} + {alpha}) / ({n} + {2 * alpha}) = {probability:.4f}."
    )
    return probability, rationale
