"""Reference-row selectors for delta-dataset construction.

The delta-learning pipeline pairs every "cheap" calculation against a
per-ICSD "reference" calculation at maximum precision. The definition of
"reference" varies by dataset:

* **aims** — FHI-aims sweep: `binary_precision == 11` AND `k_point_density == 8`.
* **exciting** — HU-Berlin Oasis sweep: `APW_precision == 1.0` AND
  `k_point_density == 8`. `rmt_scaling` is ignored (held fixed per-sweep;
  only lowered on muffin-tin collisions, not a convergence axis).

Selector callables take a row-like object (ase.db Row OR plain dict) and
return True iff that row should be used as the reference. They never raise.

Add a new selector by:

    1. Write a function `def my_selector(row) -> bool`.
    2. Register it in `SELECTORS` and document its fields in `SELECTOR_FIELDS`.
    3. Invoke via `--reference_selector=my` in `create_delta_dataset.py`.
"""
from __future__ import annotations

from typing import Any, Callable


def _get(row: Any, key: str) -> Any:
    """Pull `key` off an ase.db Row, a Row-with-key_value_pairs, or a dict."""
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, key):
        try:
            return getattr(row, key)
        except AttributeError:
            pass
    kvp = getattr(row, "key_value_pairs", None)
    if isinstance(kvp, dict):
        return kvp.get(key)
    return None


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def aims_selector(row: Any) -> bool:
    """True iff this row is the FHI-aims reference (binary_precision=11, k=8)."""
    prec = _as_int(_get(row, "binary_precision"))
    k = _as_float(_get(row, "k_point_density"))
    if prec is None or k is None:
        return False
    return prec == 11 and abs(k - 8.0) < 1e-5


def exciting_selector(row: Any) -> bool:
    """True iff this row is the exciting reference (APW_precision=1.0, k=8).

    Prefers `APWprecision_input` (from input.xml) then falls back to
    `APW_precision_path` (from the NOMAD mainfile regex). Same for k-density:
    `k_point_density` (already aims-style key, if present) then
    `k_point_density_path`. `rmt_scaling` is NOT checked — it's held fixed
    per sweep and isn't a convergence axis for the reference.
    """
    prec = (
        _as_float(_get(row, "APWprecision_input"))
        or _as_float(_get(row, "APW_precision_path"))
    )
    k = (
        _as_float(_get(row, "k_point_density"))
        or _as_float(_get(row, "k_point_density_path"))
    )
    if prec is None or k is None:
        return False
    return abs(prec - 1.0) < 1e-5 and abs(k - 8.0) < 1e-5


SELECTORS: dict[str, Callable[[Any], bool]] = {
    "aims": aims_selector,
    "exciting": exciting_selector,
}

SELECTOR_FIELDS: dict[str, tuple[str, ...]] = {
    "aims": ("binary_precision", "k_point_density"),
    "exciting": (
        "APWprecision_input", "APW_precision_path",
        "k_point_density", "k_point_density_path",
    ),
}


def get_selector(name: str) -> Callable[[Any], bool]:
    """Look up a selector by name; raises ValueError on unknown name."""
    try:
        return SELECTORS[name]
    except KeyError as e:
        known = ", ".join(sorted(SELECTORS))
        raise ValueError(
            f"Unknown reference selector: {name!r}. Known: {known}"
        ) from e
