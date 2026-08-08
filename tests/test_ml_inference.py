from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jhtvs_ft0806.ml.dataset import DatasetBundle, Normalization, ReactionExample, StateFeature
from jhtvs_ft0806.ml.inference import (
    InferenceError,
    load_abstention_policy,
    reaction_input_descriptor,
)
from jhtvs_ft0806.provenance import sha256_file


def _state(coefficient: int, vector: list[float]) -> StateFeature:
    return StateFeature(
        state_id=f"X{coefficient}",
        solvent_id="S001",
        coefficient=coefficient,
        geometry_hash="a" * 64,
        feature_cache_key="b" * 64,
        feature_path="x.npz",
        feature_sha256="c" * 64,
        base_energy_eV=float(coefficient),
        vector=np.asarray(vector, dtype=np.float64),
    )


def test_reaction_distance_descriptor_uses_exact_signed_aggregation() -> None:
    example = ReactionExample(
        reaction_id="R",
        reaction_class="redox",
        role="monomer",
        parent_id="P",
        solvent_id="S001",
        split="train",
        states=(_state(-1, [1.0, 2.0]), _state(1, [3.0, 5.0])),
        deltaE_base_MACE_rxn_eV=2.0,
        sp_residual_eV=0.0,
        rt_correction_eV=0.0,
        final_residual_eV=0.0,
        sp_mask=True,
        rt_mask=True,
        final_mask=True,
        row_weight=1.0,
        qc_status_sp="clean",
        qc_status_final="clean",
    )
    bundle = DatasetBundle(
        examples=(example,),
        feature_dimension=2,
        feature_mean=np.zeros(2),
        feature_std=np.ones(2),
        solvent_mean=np.zeros(8),
        solvent_std=np.ones(8),
        baseline_normalization=Normalization(mean=0.0, std=2.0, count=1),
        target_normalization={},
        dataset_sha256="d" * 64,
    )
    descriptor = reaction_input_descriptor(
        example, bundle=bundle, medium_vectors={"S001": np.arange(8.0)}
    )
    assert descriptor[:2].tolist() == [3.0, 5.0]
    assert descriptor[2:4].tolist() == [1.0, 2.0]
    assert descriptor[4:6].tolist() == [2.0, 3.0]
    assert descriptor[-1] == 1.0


def test_abstention_policy_requires_frozen_validation_hash(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")
    policy = tmp_path / "policy.json"
    payload = {
        "status": "FROZEN",
        "revision": "validation-policy-v1",
        "calibration_split": "val",
        "rule": "any_exceeds_or_upstream_qc",
        "disagreement_threshold_eV": 0.2,
        "representation_distance_threshold": 1.5,
        "validation_metrics_sha256": sha256_file(metrics),
    }
    policy.write_text(json.dumps(payload), encoding="utf-8")
    observed = load_abstention_policy(policy, validation_metrics_path=metrics)
    assert observed.disagreement_threshold_eV == 0.2
    metrics.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(InferenceError, match="hash mismatch"):
        load_abstention_policy(policy, validation_metrics_path=metrics)


def test_abstention_policy_rejects_nonpositive_threshold(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "revision": "validation-policy-v1",
                "calibration_split": "val",
                "rule": "any_exceeds_or_upstream_qc",
                "disagreement_threshold_eV": 0.0,
                "representation_distance_threshold": 1.5,
                "validation_metrics_sha256": sha256_file(metrics),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InferenceError, match="disagreement threshold"):
        load_abstention_policy(policy, validation_metrics_path=metrics)
