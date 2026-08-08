from __future__ import annotations

import pytest

from jhtvs_ft0806.ml.metrics import (
    coverage_accuracy_curve,
    spearman_correlation,
    summarize_evaluation,
    top_k_recall,
)


def test_rank_and_top_k_metrics_are_deterministic_with_ties() -> None:
    assert spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)
    highest = top_k_recall([1, 2, 3, 4], [1, 3, 2, 4], fraction=0.5, highest=True)
    lowest = top_k_recall([1, 2, 3, 4], [1, 3, 2, 4], fraction=0.5, highest=False)
    assert highest == {"count": 4, "k": 2, "recall": 0.5}
    assert lowest == {"count": 4, "k": 2, "recall": 0.5}


def test_coverage_accuracy_sorts_by_uncertainty_without_freezing_a_threshold() -> None:
    curve = coverage_accuracy_curve(
        [0.4, 0.1, 0.3, 0.2], [0.4, 0.1, 0.3, 0.2], coverages=(0.5, 1.0)
    )
    assert curve[0] == {"coverage": 0.5, "count": 2, "mae_eV": pytest.approx(0.15)}
    assert curve[1] == {"coverage": 1.0, "count": 4, "mae_eV": pytest.approx(0.25)}


def test_evaluation_is_grouped_by_split_role_class_and_parent() -> None:
    rows = [
        {
            "split": "val",
            "role": "monomer",
            "reaction_class": "redox",
            "parent_id": "P1",
            "solvent_id": "S001",
            "observed_final_eV": 1.0,
            "predicted_final_mean_eV": 1.1,
            "predicted_final_std_eV": 0.1,
        },
        {
            "split": "val",
            "role": "monomer",
            "reaction_class": "redox",
            "parent_id": "P1",
            "solvent_id": "S002",
            "observed_final_eV": 2.0,
            "predicted_final_mean_eV": 1.9,
            "predicted_final_std_eV": 0.2,
        },
        {
            "split": "test",
            "role": "monomer_sigma",
            "reaction_class": "sigma_dimerization",
            "parent_id": "P2",
            "solvent_id": "S001",
            "observed_final_eV": -1.0,
            "predicted_final_mean_eV": -0.8,
            "predicted_final_std_eV": 0.2,
        },
    ]
    report = summarize_evaluation(rows)
    val = report["splits"]["val"]
    assert val["overall"]["count"] == 2
    assert val["overall"]["mae_eV"] == pytest.approx(0.1)
    assert val["per_parent_solvent_trend"]["computed_parents"] == 1
    assert val["per_parent_solvent_trend"]["mean_spearman"] == pytest.approx(1.0)
    assert report["splits"]["test"]["overall"]["count"] == 1
