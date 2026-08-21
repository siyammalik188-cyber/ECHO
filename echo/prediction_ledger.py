"""The prediction ledger: what was predicted, with what information, and what happened.

Append-only in intent and in enforcement. A prediction can be added once and
evaluated once. There is no update path, no delete, and no way to restate an old
probability — the ledger exists specifically so that a prediction cannot be made
to look better after the fact.

It answers, for any prediction: what did I predict, what probability did I
assign, what information did I have, what actually happened, and how wrong was I.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from .prediction import (
    Experience,
    Observation,
    Prediction,
    PredictionRecord,
    Timeline,
)

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "predictions.json"


class PredictionLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records: dict[str, PredictionRecord] = {}
        self._order: list[str] = []  # issue order, never reordered
        self._experiences: list[Experience] = []
        self._observations: dict[str, Observation] = {}

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Path | str) -> "PredictionLedger":
        ledger = cls(path)
        if not ledger.path.is_file():
            return ledger

        with ledger.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported prediction schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        for record in payload.get("observations", []):
            observation = Observation.from_dict(record)
            ledger._observations[observation.id] = observation
        for record in payload.get("predictions", []):
            prediction_record = PredictionRecord.from_dict(record)
            ledger._records[prediction_record.id] = prediction_record
            ledger._order.append(prediction_record.id)
        ledger._experiences = [
            Experience.from_dict(e) for e in payload.get("experiences", [])
        ]
        return ledger

    @classmethod
    def in_directory(cls, directory: Path | str) -> "PredictionLedger":
        return cls.load(Path(directory) / DEFAULT_FILENAME)

    # ----------------------------------------------------------------- write

    def remember_timeline(self, timeline: Timeline) -> None:
        """Copy the timeline in, so 'what did I know' survives a restart."""
        for observation in timeline.all():
            self._observations.setdefault(observation.id, observation)

    def add(self, prediction: Prediction) -> PredictionRecord:
        if prediction.id in self._records:
            raise ValueError(f"prediction {prediction.id} is already in the ledger")
        record = PredictionRecord(prediction=prediction)
        self._records[prediction.id] = record
        self._order.append(prediction.id)
        return record

    def evaluate(
        self,
        prediction_id: str,
        outcome: bool,
        outcome_label: str,
        evaluation_time: int,
    ) -> PredictionRecord:
        """Attach an outcome and file an experience. Refuses a second attempt.

        The stored `Prediction` object is reused unchanged — only the wrapper
        around it is replaced — so the issued probability cannot move.
        """
        record = self._records.get(prediction_id)
        if record is None:
            raise KeyError(f"no prediction {prediction_id!r} in this ledger")

        issued_before = record.prediction  # the exact object, for the check below
        evaluated = record.evaluate(outcome, outcome_label, evaluation_time)

        # Belt and braces: the frozen prediction must be the identical instance.
        assert evaluated.prediction is issued_before, "evaluation replaced the prediction"

        self._records[prediction_id] = evaluated
        self._experiences.append(Experience.from_record(evaluated))
        return evaluated

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "predictions": [self._records[pid].to_dict() for pid in self._order],
            "experiences": [e.to_dict() for e in self._experiences],
            "observations": [o.to_dict() for o in self.observations()],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)
        return self.path

    # ------------------------------------------------------------------ read

    def get(self, prediction_id: str) -> PredictionRecord | None:
        return self._records.get(prediction_id)

    def records(self) -> tuple[PredictionRecord, ...]:
        """Every prediction in the order it was issued. A copy."""
        return tuple(self._records[pid] for pid in self._order)

    def evaluated(self) -> tuple[PredictionRecord, ...]:
        return tuple(r for r in self.records() if r.is_evaluated)

    def pending(self) -> tuple[PredictionRecord, ...]:
        return tuple(r for r in self.records() if not r.is_evaluated)

    def experiences(self) -> tuple[Experience, ...]:
        return tuple(self._experiences)

    def observations(self) -> tuple[Observation, ...]:
        return tuple(sorted(self._observations.values(), key=lambda o: (o.at, o.id)))

    def information_at(self, prediction_id: str) -> tuple[Observation, ...]:
        """The observations a given prediction actually had in front of it."""
        record = self._records.get(prediction_id)
        if record is None:
            return ()
        return tuple(
            self._observations[oid]
            for oid in record.information_available_at_prediction_time
            if oid in self._observations
        )

    def history(self, proposition: str | None = None) -> list[dict[str, Any]]:
        """The plain-language answer to 'what did I predict and how did it go'."""
        out = []
        for record in self.records():
            if proposition is not None and record.proposition != proposition:
                continue
            out.append(
                {
                    "prediction_id": record.id,
                    "proposition": record.proposition,
                    "predicted_probability": record.predicted_probability,
                    "prediction_time": record.prediction_time,
                    "information_count": len(
                        record.information_available_at_prediction_time
                    ),
                    "status": record.evaluation_status.value,
                    "actual_outcome": (
                        None
                        if record.evaluation is None
                        else record.evaluation.outcome_label
                    ),
                    "prediction_error": record.prediction_error,
                }
            )
        return out

    def scored(self) -> tuple[list[float], list[bool]]:
        """Probabilities and outcomes for every evaluated prediction, in order."""
        evaluated = self.evaluated()
        return (
            [r.predicted_probability for r in evaluated],
            [r.evaluation.actual_outcome for r in evaluated],  # type: ignore[union-attr]
        )

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[PredictionRecord]:
        return iter(self.records())
