from __future__ import annotations

import pytest

from jhtvs_ft0806.ml.evaluation import EvaluationError, aggregate_member_outputs


def test_ensemble_is_aggregated_after_immutable_reaction_arithmetic() -> None:
    members = [
        {
            "final_residual_eV": float(index),
            "sp_residual_eV": float(index) + 0.5,
            "rt_correction_eV": -float(index),
        }
        for index in range(5)
    ]
    result = aggregate_member_outputs(members, immutable_baseline_eV=10.0)
    assert result["predicted_final_mean_eV"] == pytest.approx(12.0)
    assert result["predicted_final_std_eV"] == pytest.approx(2.0**0.5)
    assert result["predicted_sp_mean_eV"] == pytest.approx(12.5)
    assert result["predicted_rt_mean_eV"] == pytest.approx(-2.0)
    assert result["ensemble_member_count"] == 5


def test_ensemble_rejects_missing_members() -> None:
    with pytest.raises(EvaluationError, match="exactly five"):
        aggregate_member_outputs([], immutable_baseline_eV=0.0)
