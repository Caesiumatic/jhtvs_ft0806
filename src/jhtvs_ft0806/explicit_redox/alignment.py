from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ReferenceAlignment:
    C_model_V: float
    mae_before_V: float
    mae_after_V: float
    residuals_V: tuple[float, ...]


def assert_disjoint_keys(calibration_keys: Sequence[str], validation_keys: Sequence[str]) -> None:
    overlap = sorted(set(calibration_keys) & set(validation_keys))
    if overlap:
        raise ValueError(f"calibration and validation overlap: {overlap}")


def fit_global_intercept(
    experimental_V: Sequence[float], raw_delta_F_eV: Sequence[float]
) -> ReferenceAlignment:
    experimental = np.asarray(experimental_V, dtype=np.float64)
    raw = np.asarray(raw_delta_F_eV, dtype=np.float64)
    if experimental.ndim != 1 or raw.ndim != 1 or experimental.size != raw.size or not experimental.size:
        raise ValueError("alignment requires equal non-empty vectors")
    if not np.all(np.isfinite(experimental)) or not np.all(np.isfinite(raw)):
        raise ValueError("alignment inputs must be finite")
    intercept = float(np.mean(experimental - raw))
    residuals = experimental - (raw + intercept)
    return ReferenceAlignment(
        C_model_V=intercept,
        mae_before_V=float(np.mean(np.abs(experimental - raw))),
        mae_after_V=float(np.mean(np.abs(residuals))),
        residuals_V=tuple(float(value) for value in residuals),
    )
