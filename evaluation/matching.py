"""Deterministic matching of proposed memories against pre-registered labels.

The matcher is intentionally dumb: normalise the text, then require that every
`must_include` group contributes at least one of its alternatives as a
substring. No model is used to judge a model. That costs some sensitivity —
a correct paraphrase that avoids every listed alternative will be scored as a
miss — but it keeps the measurement reproducible and impossible to quietly
tune in the model's favour after the fact.

Consequences worth stating plainly when reading results:
  - Recall is a LOWER bound. A wording the labels did not anticipate reads as a
    false negative even when the memory is good.
  - Precision on unlabelled extractions is a JUDGEMENT, not a measurement: a
    proposal matching nothing is counted against the model, which is correct
    for the negative cases (where nothing should be produced) and harsher than
    a human would be on the positive ones.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    'ACME-4471' -> 'acme4471'; 'paediatric nurse.' -> 'paediatric nurse'.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.lower()
    no_punct = _PUNCT.sub("", lowered)
    return _SPACE.sub(" ", no_punct).strip()


def matches(groups: list[list[str]], content: str) -> bool:
    """True if, for every group, at least one alternative appears in `content`.

    An empty alternative string matches anything — used by the negative cases to
    say "no memory at all is acceptable here".
    """
    if not groups:
        return False
    haystack = normalise(content)
    for group in groups:
        if not any(normalise(alt) in haystack for alt in group):
            return False
    return True


def in_band(value: float, band: list[float] | tuple[float, float]) -> bool:
    low, high = band
    return low <= value <= high


def band_deviation(value: float, band: list[float] | tuple[float, float]) -> float:
    """Distance outside the band; 0.0 when inside it."""
    low, high = band
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0
