"""The observation side of the wall.

This module defines everything the discovery system is allowed to see: six
observed variables and an outcome, in time order, and nothing else. There is no
field here for the process that produced them, no parameter of that process, and
no handle back to whatever object generated the rows. An `ObservationSet` is a
table.

The generators live in `experiments/discovery_worlds.py` and are not imported
here, or anywhere under `echo/`. The evaluator may know the truth. ECHO gets the
table.

The other half of this module is `ChronologicalSplit`, which enforces the thing
that is easiest to get wrong and hardest to notice: the final holdout must not
influence anything. Not hypothesis generation, not ranking, not thresholds, not
promotion. It is not enough to intend that — asking for the test block during
search raises `HoldoutViolation`, and unlocking it closes the search for good.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .expressions import WARMUP_FLOOR

SCHEMA_VERSION = 1

TRAIN = "TRAIN"
VAL_A = "VAL_A"
VAL_B = "VAL_B"
TEST = "TEST"

#: The order matters and is the whole design: every block is strictly later in
#: time than the one before it. There is no random split anywhere in ECHO 5.
BLOCK_ORDER: tuple[str, ...] = (TRAIN, VAL_A, VAL_B, TEST)


class HoldoutViolation(RuntimeError):
    """Raised when something reaches for the final test block too early."""


@dataclass(frozen=True)
class Block:
    """A half-open range of row indices, `[start, stop)`."""

    name: str
    start: int
    stop: int

    def __len__(self) -> int:
        return max(0, self.stop - self.start)

    def indices(self) -> range:
        return range(self.start, self.stop)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "start": self.start, "stop": self.stop, "rows": len(self)}


@dataclass(frozen=True)
class ObservationSet:
    """Observed variables and outcomes. No truth, by construction."""

    id: str
    variable_names: tuple[str, ...]
    columns: tuple[tuple[float, ...], ...]
    outcomes: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.variable_names) != len(self.columns):
            raise ValueError("variable_names and columns disagree in length")
        lengths = {len(column) for column in self.columns} | {len(self.outcomes)}
        if len(lengths) > 1:
            raise ValueError(f"ragged observation set: lengths {sorted(lengths)}")

    def __len__(self) -> int:
        return len(self.outcomes)

    def column(self, name: str) -> tuple[float, ...]:
        try:
            return self.columns[self.variable_names.index(name)]
        except ValueError:
            raise KeyError(f"no observed variable {name!r}") from None

    def as_mapping(self) -> Mapping[str, Sequence[float]]:
        """The form the expression interpreter reads."""
        return {name: column for name, column in zip(self.variable_names, self.columns)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "variable_names": list(self.variable_names),
            "columns": [list(column) for column in self.columns],
            "outcomes": list(self.outcomes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservationSet":
        version = payload.get("schema_version")
        if version is not None and version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported observation schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        return cls(
            id=str(payload["id"]),
            variable_names=tuple(payload["variable_names"]),
            columns=tuple(tuple(float(v) for v in column) for column in payload["columns"]),
            outcomes=tuple(bool(v) for v in payload["outcomes"]),
        )

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "ObservationSet":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class ChronologicalSplit:
    """Four blocks in time order, with the last one locked.

    `TRAIN` fits a hypothesis. `VAL_A` ranks the candidates. `VAL_B` confirms
    the winner against a baseline — it is never used for ranking, so it is an
    honest check rather than a second chance to pick. `TEST` is reported once,
    after every decision has already been made.

    Rows before `warmup` are excluded from every block: the longest expression
    in the language needs that much history, so this is what keeps candidates
    comparable. Every candidate is scored on exactly the same rows, whatever it
    happens to reference.
    """

    def __init__(
        self,
        rows: int,
        *,
        train_stop: int,
        val_a_stop: int,
        val_b_stop: int,
        warmup: int = WARMUP_FLOOR,
        train_rows: int | None = None,
    ) -> None:
        bounds = [warmup, train_stop, val_a_stop, val_b_stop, rows]
        if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
            raise ValueError(f"block boundaries must strictly increase, got {bounds}")
        self.rows = rows
        self.warmup = warmup
        # A training budget shortens TRAIN from its *end*, leaving the three
        # later blocks untouched. That is what makes sample efficiency
        # measurable: every budget is judged on identical validation and
        # holdout rows, so the only thing that varies is how much history the
        # search had. A gap opens between TRAIN and VAL_A at small budgets,
        # which is harmless — the blocks stay in time order either way.
        train_end = train_stop
        if train_rows is not None:
            if train_rows < 1:
                raise ValueError(f"train_rows must be positive, got {train_rows}")
            train_end = min(train_stop, warmup + train_rows)
        self.train_budget = train_rows
        self._blocks = {
            TRAIN: Block(TRAIN, warmup, train_end),
            VAL_A: Block(VAL_A, train_stop, val_a_stop),
            VAL_B: Block(VAL_B, val_a_stop, val_b_stop),
            TEST: Block(TEST, val_b_stop, rows),
        }
        self._holdout_unlocked = False
        self._search_closed = False

    # ------------------------------------------------------------------ state

    @property
    def holdout_unlocked(self) -> bool:
        return self._holdout_unlocked

    @property
    def search_closed(self) -> bool:
        """True once the holdout has been read. Search may not resume."""
        return self._search_closed

    def searchable_blocks(self) -> tuple[Block, ...]:
        return tuple(self._blocks[name] for name in (TRAIN, VAL_A, VAL_B))

    # ----------------------------------------------------------------- access

    def block(self, name: str) -> Block:
        if name not in self._blocks:
            raise KeyError(f"no block named {name!r}")
        if name == TEST and not self._holdout_unlocked:
            raise HoldoutViolation(
                "the TEST block is sealed. It may not influence hypothesis "
                "generation, ranking, thresholds or promotion. Call "
                "unlock_holdout() only after every decision has been made."
            )
        return self._blocks[name]

    def unlock_holdout(self) -> Block:
        """Open the final holdout. Search is closed permanently by doing so."""
        self._holdout_unlocked = True
        self._search_closed = True
        return self._blocks[TEST]

    def require_open_search(self, what: str = "this operation") -> None:
        if self._search_closed:
            raise HoldoutViolation(
                f"{what} is not allowed after the holdout has been read — "
                "anything decided now would be a decision informed by the test set"
            )

    # ------------------------------------------------------------------- misc

    def describe(self) -> list[dict[str, Any]]:
        return [self._blocks[name].to_dict() for name in BLOCK_ORDER]

    def outcomes(self, observations: ObservationSet, block: Block) -> list[bool]:
        return [observations.outcomes[t] for t in block.indices()]
