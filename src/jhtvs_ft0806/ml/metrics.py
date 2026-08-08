"""Deterministic grouped metrics for reaction-property ensembles."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


class MetricsError(ValueError):
    """Raised when evaluation rows do not support a requested metric."""


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(observed: Sequence[float], predicted: Sequence[float]) -> float | None:
    if len(observed) != len(predicted) or len(observed) < 2:
        return None
    first = _rankdata(observed)
    second = _rankdata(predicted)
    if float(first.std()) < 1e-15 or float(second.std()) < 1e-15:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def top_k_recall(
    observed: Sequence[float],
    predicted: Sequence[float],
    *,
    fraction: float = 0.1,
    highest: bool,
) -> dict[str, float | int | None]:
    if len(observed) != len(predicted) or not observed:
        return {"count": len(observed), "k": 0, "recall": None}
    k = max(1, int(math.ceil(len(observed) * fraction)))
    direction = -1.0 if highest else 1.0
    truth = set(np.argsort(direction * np.asarray(observed))[:k].tolist())
    selected = set(np.argsort(direction * np.asarray(predicted))[:k].tolist())
    return {"count": len(observed), "k": k, "recall": len(truth & selected) / k}


def interval_coverage(
    observed: Sequence[float], predicted: Sequence[float], uncertainty: Sequence[float]
) -> dict[str, float | int | None]:
    if not (len(observed) == len(predicted) == len(uncertainty)) or not observed:
        return {"count": len(observed), "coverage_95": None, "mean_width_95": None}
    truth = np.asarray(observed, dtype=np.float64)
    mean = np.asarray(predicted, dtype=np.float64)
    std = np.asarray(uncertainty, dtype=np.float64)
    width = 1.96 * std
    return {
        "count": len(observed),
        "coverage_95": float(np.mean(np.abs(truth - mean) <= width)),
        "mean_width_95": float(np.mean(2.0 * width)),
    }


def coverage_accuracy_curve(
    absolute_errors: Sequence[float],
    uncertainty: Sequence[float],
    *,
    coverages: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> list[dict[str, float | int]]:
    if len(absolute_errors) != len(uncertainty) or not absolute_errors:
        return []
    errors = np.asarray(absolute_errors, dtype=np.float64)
    order = np.argsort(np.asarray(uncertainty, dtype=np.float64), kind="mergesort")
    curve: list[dict[str, float | int]] = []
    for coverage in coverages:
        count = max(1, min(len(order), int(math.ceil(float(coverage) * len(order)))))
        selected = errors[order[:count]]
        curve.append(
            {
                "coverage": count / len(order),
                "count": count,
                "mae_eV": float(selected.mean()),
            }
        )
    return curve


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    observed: list[float] = []
    predicted: list[float] = []
    uncertainty: list[float] = []
    for row in rows:
        truth = _finite(row.get("observed_final_eV"))
        mean = _finite(row.get("predicted_final_mean_eV"))
        std = _finite(row.get("predicted_final_std_eV"))
        if truth is None or mean is None or std is None:
            continue
        observed.append(truth)
        predicted.append(mean)
        uncertainty.append(std)
    errors = [abs(mean - truth) for mean, truth in zip(predicted, observed, strict=True)]
    return {
        "count": len(observed),
        "mae_eV": None if not errors else float(np.mean(errors)),
        "spearman": spearman_correlation(observed, predicted),
        "top_10pct_low": top_k_recall(observed, predicted, highest=False),
        "top_10pct_high": top_k_recall(observed, predicted, highest=True),
        "interval": interval_coverage(observed, predicted, uncertainty),
        "coverage_accuracy": coverage_accuracy_curve(errors, uncertainty),
    }


def _solvent_trends(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["parent_id"])].append(row)
    correlations: list[float] = []
    eligible = 0
    for group in groups.values():
        observed: list[float] = []
        predicted: list[float] = []
        for row in group:
            truth = _finite(row.get("observed_final_eV"))
            mean = _finite(row.get("predicted_final_mean_eV"))
            if truth is not None and mean is not None:
                observed.append(truth)
                predicted.append(mean)
        if len(observed) < 2:
            continue
        eligible += 1
        correlation = spearman_correlation(observed, predicted)
        if correlation is not None:
            correlations.append(correlation)
    return {
        "eligible_parents": eligible,
        "computed_parents": len(correlations),
        "mean_spearman": None if not correlations else float(np.mean(correlations)),
        "median_spearman": None if not correlations else float(np.median(correlations)),
    }


def summarize_evaluation(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    materialized = tuple(rows)
    by_split: dict[str, object] = {}
    for split in ("val", "test"):
        selected = tuple(row for row in materialized if row.get("split") == split)
        roles = sorted({str(row["role"]) for row in selected})
        classes = sorted({str(row["reaction_class"]) for row in selected})
        by_split[split] = {
            "overall": _summary(selected),
            "by_role": {
                role: _summary(tuple(row for row in selected if row["role"] == role))
                for role in roles
            },
            "by_reaction_class": {
                reaction_class: _summary(
                    tuple(
                        row
                        for row in selected
                        if row["reaction_class"] == reaction_class
                    )
                )
                for reaction_class in classes
            },
            "per_parent_solvent_trend": _solvent_trends(selected),
        }
    return {"status": "PASS", "splits": by_split}
