from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlockStatistics:
    mean_eV: float
    standard_deviation_eV: float
    block_standard_error_eV: float
    block_count: int
    skewness: float
    multimodal: bool


@dataclass(frozen=True)
class SeedMarcus:
    mu_lower_eV: float
    mu_oxidized_eV: float
    delta_F_ox_eV: float
    lambda_eV: float
    lower_block_se_eV: float
    oxidized_block_se_eV: float
    delta_F_block_se_eV: float


@dataclass(frozen=True)
class SystemMarcus:
    delta_F_ox_eV: float
    raw_voltage_V: float
    lambda_eV: float
    shell_seed_sd_eV: float
    shell_seed_sem_eV: float
    within_seed_block_se_eV: float


def _multimodal_histogram(values: NDArray[np.float64]) -> bool:
    if values.size < 100:
        return False
    counts, _ = np.histogram(values, bins=max(8, round(math.sqrt(values.size))))
    if counts.max() == 0:
        return False
    threshold = 0.20 * counts.max()
    peaks = [
        index
        for index in range(1, len(counts) - 1)
        if counts[index] > counts[index - 1]
        and counts[index] >= counts[index + 1]
        and counts[index] >= threshold
    ]
    return any(second - first >= 2 for first, second in zip(peaks, peaks[1:]))


def block_statistics(
    values_eV: Sequence[float],
    *,
    sample_interval_fs: float = 20.0,
    block_length_ps: float = 5.0,
) -> BlockStatistics:
    values = np.asarray(values_eV, dtype=np.float64)
    block_size = round(block_length_ps * 1000.0 / sample_interval_fs)
    if values.ndim != 1 or values.size == 0 or values.size % block_size:
        raise ValueError("trajectory samples do not form complete contiguous blocks")
    if not np.all(np.isfinite(values)):
        raise ValueError("gap samples contain non-finite values")
    blocks = values.reshape(-1, block_size).mean(axis=1)
    block_se = float(blocks.std(ddof=1) / math.sqrt(blocks.size)) if blocks.size > 1 else 0.0
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    centered = values - values.mean()
    scale = float(np.sqrt(np.mean(centered**2)))
    skewness = float(np.mean(centered**3) / scale**3) if scale else 0.0
    return BlockStatistics(
        mean_eV=float(values.mean()),
        standard_deviation_eV=standard_deviation,
        block_standard_error_eV=block_se,
        block_count=int(blocks.size),
        skewness=skewness,
        multimodal=_multimodal_histogram(values),
    )


def assemble_seed(lower_gaps_eV: Sequence[float], oxidized_gaps_eV: Sequence[float]) -> SeedMarcus:
    lower = block_statistics(lower_gaps_eV)
    oxidized = block_statistics(oxidized_gaps_eV)
    delta_f = 0.5 * (lower.mean_eV + oxidized.mean_eV)
    reorganization = 0.5 * (lower.mean_eV - oxidized.mean_eV)
    propagated = 0.5 * math.hypot(
        lower.block_standard_error_eV, oxidized.block_standard_error_eV
    )
    return SeedMarcus(
        mu_lower_eV=lower.mean_eV,
        mu_oxidized_eV=oxidized.mean_eV,
        delta_F_ox_eV=delta_f,
        lambda_eV=reorganization,
        lower_block_se_eV=lower.block_standard_error_eV,
        oxidized_block_se_eV=oxidized.block_standard_error_eV,
        delta_F_block_se_eV=propagated,
    )


def ev_per_electron_to_volts(value_eV: float, electrons: int = 1) -> float:
    if electrons <= 0:
        raise ValueError("electron count must be positive")
    return float(value_eV) / electrons


def assemble_system(seeds: Sequence[SeedMarcus]) -> SystemMarcus:
    if len(seeds) != 5:
        raise ValueError("system aggregation requires exactly five shell seeds")
    delta_f = np.asarray([seed.delta_F_ox_eV for seed in seeds])
    lambdas = np.asarray([seed.lambda_eV for seed in seeds])
    block_se = np.asarray([seed.delta_F_block_se_eV for seed in seeds])
    sd = float(delta_f.std(ddof=1))
    mean = float(delta_f.mean())
    return SystemMarcus(
        delta_F_ox_eV=mean,
        raw_voltage_V=ev_per_electron_to_volts(mean),
        lambda_eV=float(lambdas.mean()),
        shell_seed_sd_eV=sd,
        shell_seed_sem_eV=sd / math.sqrt(5.0),
        within_seed_block_se_eV=float(np.sqrt(np.mean(block_se**2))),
    )
