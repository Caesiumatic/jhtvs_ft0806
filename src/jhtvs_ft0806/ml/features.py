"""Rotation-invariant MACE-POLAR feature extraction.

The heavy MACE/PyTorch dependencies are imported only by the execution backend so
that specification validation and ORCA workflows remain usable without the ML
environment installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jhtvs_ft0806.provenance import content_hash, sha256_file


FEATURE_SCHEMA_REVISION = "polar1l_rotation_invariants_v1"
EXPECTED_MACE_VERSION = "0.3.16"
EXPECTED_MACE_SOURCE_COMMIT = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
EXPECTED_GRAPH_ELECTROSTATICS_VERSION = "0.4.0"
EXPECTED_GRAPH_ELECTROSTATICS_COMMIT = "0e21d5546c482d08388a08eb4d948e833227ce47"
EXPECTED_CHECKPOINT_NAME = "polar-1-l"
EXPECTED_CHECKPOINT_SHA256 = (
    "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b"
)
REQUIRED_POLAR_OUTPUTS = (
    "energy",
    "node_feats",
    "density_coefficients",
    "spin_density",
    "spin_charge_density",
    "dipole",
    "electrostatic_energy",
    "electron_energy",
)


class FeatureExtractionError(RuntimeError):
    """Raised when the frozen feature contract cannot be satisfied."""


def patch_incompatible_mace_lora_inference(module: Any) -> int:
    """Keep official LoRA math usable for e3nn linears with bias instructions.

    MACE 0.3.16's fused inference path assumes every ``o3.Linear`` instruction
    has a two-dimensional ``path_shape``.  PolarMACE contains biased linears
    whose bias instructions are one-dimensional.  The official gradient-enabled
    path is valid for those layers, so use the same activation-space expression
    under ``no_grad`` while preserving the injected adapter parameters exactly.
    """

    from mace.modules.lora import LoRAO3Linear

    patched = 0

    def activation_space_forward(wrapper: Any, values: Any) -> Any:
        wrapper._cached_merged_weight = None  # pylint: disable=protected-access
        return wrapper.base(values) + wrapper.scaling * wrapper.lora_B(
            wrapper.lora_A(values)
        )

    for child in module.modules():
        if not isinstance(child, LoRAO3Linear):
            continue
        if not any(
            len(instruction.path_shape) != 2
            for instruction in child.base.instructions
        ):
            continue
        child.forward = MethodType(activation_space_forward, child)
        patched += 1
    return patched


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    feature_vector: np.ndarray
    base_energy_eV: float
    output_shapes: Mapping[str, tuple[int, ...]]
    invariant_layout: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    checkpoint_name: str
    checkpoint_path: str
    checkpoint_sha256: str
    mace_version: str
    mace_source_commit: str
    mace_package_sha256: str
    graph_electrostatics_version: str
    graph_electrostatics_commit: str
    graph_electrostatics_package_sha256: str
    default_dtype: str

    def to_dict(self) -> dict[str, str]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "mace_version": self.mace_version,
            "mace_source_commit": self.mace_source_commit,
            "mace_package_sha256": self.mace_package_sha256,
            "graph_electrostatics_version": self.graph_electrostatics_version,
            "graph_electrostatics_commit": self.graph_electrostatics_commit,
            "graph_electrostatics_package_sha256": self.graph_electrostatics_package_sha256,
            "default_dtype": self.default_dtype,
        }


def feature_cache_key(
    *,
    checkpoint_sha256: str,
    geometry_sha256: str,
    formal_charge: int,
    multiplicity: int,
    feature_schema_revision: str = FEATURE_SCHEMA_REVISION,
) -> str:
    """Return the frozen content-addressed feature key."""

    return content_hash(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "geometry_sha256": geometry_sha256,
            "formal_charge": int(formal_charge),
            "multiplicity": int(multiplicity),
            "feature_schema_revision": feature_schema_revision,
        }
    )


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FeatureExtractionError(f"non-finite values in {name}")
    return array


def _sum_mean(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise FeatureExtractionError("molecular summary requires a non-empty vector")
    return float(values.sum()), float(values.mean())


def multipole_invariants(
    multipoles: Any,
    *,
    prefix: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize monopoles and l=1 norms without retaining orientation.

    MACE-POLAR multipoles use one monopole followed by three l=1 components for
    the frozen checkpoint. Sum and mean pooling are molecular summaries and are
    applied identically to charge, spin-density, and each spin channel.
    """

    values = _finite_array(multipoles, name=prefix)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 4:
        raise FeatureExtractionError(
            f"{prefix} must have shape (n_atoms, >=4), observed {values.shape}"
        )
    monopole = values[:, 0]
    l1_norm = np.linalg.norm(values[:, 1:4], axis=1)
    mono_sum, mono_mean = _sum_mean(monopole)
    norm_sum, norm_mean = _sum_mean(l1_norm)
    names = (
        f"{prefix}_monopole_sum",
        f"{prefix}_monopole_mean",
        f"{prefix}_l1_norm_sum",
        f"{prefix}_l1_norm_mean",
    )
    return np.asarray((mono_sum, mono_mean, norm_sum, norm_mean)), names


def pool_even_scalar_channels(
    node_features: Any,
    *,
    layer_widths: Sequence[int],
    even_scalar_indices: Sequence[Sequence[int]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Pool checkpoint-irrep-selected 0e node channels per interaction."""

    features = _finite_array(node_features, name="node_feats")
    if features.ndim != 2 or features.shape[0] == 0:
        raise FeatureExtractionError(
            f"node_feats must have shape (n_atoms, n_features), observed {features.shape}"
        )
    if len(layer_widths) != len(even_scalar_indices) or not layer_widths:
        raise FeatureExtractionError("invalid interaction-layer irrep plan")
    if sum(int(width) for width in layer_widths) != features.shape[1]:
        raise FeatureExtractionError(
            "node feature width does not match checkpoint interaction irreps"
        )

    pooled: list[np.ndarray] = []
    names: list[str] = []
    offset = 0
    for layer_index, (width, indices) in enumerate(
        zip(layer_widths, even_scalar_indices, strict=True)
    ):
        width = int(width)
        layer = features[:, offset : offset + width]
        offset += width
        selected = np.asarray(tuple(int(index) for index in indices), dtype=np.int64)
        if selected.size == 0:
            raise FeatureExtractionError(f"interaction {layer_index} has no 0e channels")
        if selected.min() < 0 or selected.max() >= width:
            raise FeatureExtractionError(f"invalid 0e channel index for interaction {layer_index}")
        scalar = layer[:, selected]
        pooled.extend((scalar.sum(axis=0), scalar.mean(axis=0)))
        names.extend(
            f"node_0e_layer{layer_index}_sum_{index}" for index in range(scalar.shape[1])
        )
        names.extend(
            f"node_0e_layer{layer_index}_mean_{index}" for index in range(scalar.shape[1])
        )
    return np.concatenate(pooled).astype(np.float64), tuple(names)


def build_invariant_feature_record(
    outputs: Mapping[str, Any],
    *,
    layer_widths: Sequence[int],
    even_scalar_indices: Sequence[Sequence[int]],
    formal_charge: int,
    multiplicity: int,
) -> FeatureRecord:
    """Convert the required raw PolarMACE outputs into one invariant state vector."""

    missing = [name for name in REQUIRED_POLAR_OUTPUTS if outputs.get(name) is None]
    if missing:
        raise FeatureExtractionError(f"raw PolarMACE output missing: {missing}")

    node_features = _finite_array(outputs["node_feats"], name="node_feats")
    node_vector, node_names = pool_even_scalar_channels(
        node_features,
        layer_widths=layer_widths,
        even_scalar_indices=even_scalar_indices,
    )
    density = _finite_array(outputs["density_coefficients"], name="density_coefficients")
    spin_density = _finite_array(outputs["spin_density"], name="spin_density")
    spin_channels = _finite_array(outputs["spin_charge_density"], name="spin_charge_density")
    if spin_channels.ndim != 3 or spin_channels.shape[1] != 2:
        raise FeatureExtractionError(
            "spin_charge_density must have shape (n_atoms, 2, n_multipoles)"
        )
    if not (
        density.shape[0]
        == spin_density.shape[0]
        == spin_channels.shape[0]
        == node_features.shape[0]
    ):
        raise FeatureExtractionError("PolarMACE atom-level output counts disagree")

    blocks: list[np.ndarray] = [node_vector]
    names: list[str] = list(node_names)
    for values, prefix in (
        (density, "density"),
        (spin_density, "spin_density"),
        (spin_channels[:, 0, :], "spin_alpha"),
        (spin_channels[:, 1, :], "spin_beta"),
    ):
        vector, block_names = multipole_invariants(values, prefix=prefix)
        blocks.append(vector)
        names.extend(block_names)

    dipole = _finite_array(outputs["dipole"], name="dipole").reshape(-1, 3)
    if dipole.shape[0] != 1:
        raise FeatureExtractionError("feature extraction accepts one graph at a time")
    energy = _finite_array(outputs["energy"], name="energy").reshape(-1)
    electrostatic = _finite_array(
        outputs["electrostatic_energy"], name="electrostatic_energy"
    ).reshape(-1)
    electron = _finite_array(outputs["electron_energy"], name="electron_energy").reshape(-1)
    if energy.size != 1 or electrostatic.size != 1 or electron.size != 1:
        raise FeatureExtractionError("graph-level PolarMACE outputs must be scalar")
    global_values = np.asarray(
        (
            float(np.linalg.norm(dipole[0])),
            float(energy[0]),
            float(electrostatic[0]),
            float(electron[0]),
            float(density.shape[0]),
            float(formal_charge),
            float(multiplicity),
        ),
        dtype=np.float64,
    )
    blocks.append(global_values)
    names.extend(
        (
            "dipole_norm",
            "immutable_base_energy_eV",
            "electrostatic_energy_eV",
            "electron_energy_eV",
            "atom_count",
            "formal_charge",
            "multiplicity",
        )
    )
    vector = np.concatenate(blocks).astype(np.float64)
    if len(names) != vector.size or not np.all(np.isfinite(vector)):
        raise FeatureExtractionError("invalid invariant feature vector")
    shapes = {
        name: tuple(int(size) for size in np.asarray(outputs[name]).shape)
        for name in REQUIRED_POLAR_OUTPUTS
    }
    return FeatureRecord(
        feature_vector=vector,
        base_energy_eV=float(energy[0]),
        output_shapes=shapes,
        invariant_layout=tuple(names),
    )


def build_torch_invariant_feature_vector(
    outputs: Mapping[str, Any],
    *,
    layer_widths: Sequence[int],
    even_scalar_indices: Sequence[Sequence[int]],
    formal_charge: int,
    multiplicity: int,
    immutable_base_energy_eV: float | None = None,
) -> Any:
    """Build the same single-graph invariant vector with gradients preserved."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by the ML extra
        raise FeatureExtractionError("PyTorch is required for online LoRA features") from exc
    node = outputs.get("node_feats")
    if node is None or node.ndim != 2 or int(node.shape[0]) == 0:
        raise FeatureExtractionError("node_feats must contain one non-empty graph")
    matrix = build_torch_invariant_feature_matrix(
        outputs,
        atom_graph_index=torch.zeros(
            int(node.shape[0]), dtype=torch.long, device=node.device
        ),
        graph_count=1,
        layer_widths=layer_widths,
        even_scalar_indices=even_scalar_indices,
        formal_charges=(formal_charge,),
        multiplicities=(multiplicity,),
        immutable_base_energies_eV=(
            None
            if immutable_base_energy_eV is None
            else (immutable_base_energy_eV,)
        ),
    )
    return matrix[0]


def build_torch_invariant_feature_matrix(
    outputs: Mapping[str, Any],
    *,
    atom_graph_index: Any,
    graph_count: int,
    layer_widths: Sequence[int],
    even_scalar_indices: Sequence[Sequence[int]],
    formal_charges: Sequence[int],
    multiplicities: Sequence[int],
    immutable_base_energies_eV: Sequence[float] | None = None,
) -> Any:
    """Pool a graph batch into the exact per-graph invariant feature vectors."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by the ML extra
        raise FeatureExtractionError("PyTorch is required for online LoRA features") from exc
    missing = [name for name in REQUIRED_POLAR_OUTPUTS if outputs.get(name) is None]
    if missing:
        raise FeatureExtractionError(f"raw PolarMACE output missing: {missing}")

    node = outputs["node_feats"]
    if node.ndim != 2 or node.shape[0] == 0:
        raise FeatureExtractionError(
            f"node_feats must have shape (n_atoms, n_features), observed {tuple(node.shape)}"
        )
    if len(layer_widths) != len(even_scalar_indices) or not layer_widths:
        raise FeatureExtractionError("invalid interaction-layer irrep plan")
    if sum(int(width) for width in layer_widths) != int(node.shape[1]):
        raise FeatureExtractionError(
            "node feature width does not match checkpoint interaction irreps"
        )
    if graph_count <= 0 or len(formal_charges) != graph_count or len(multiplicities) != graph_count:
        raise FeatureExtractionError("batched online feature identity count mismatch")
    if atom_graph_index.ndim != 1 or int(atom_graph_index.numel()) != int(node.shape[0]):
        raise FeatureExtractionError("atom-to-graph index shape mismatch")
    if int(atom_graph_index.min()) < 0 or int(atom_graph_index.max()) >= graph_count:
        raise FeatureExtractionError("atom-to-graph index is out of range")
    counts = torch.bincount(atom_graph_index, minlength=graph_count).to(node.dtype)
    if bool((counts <= 0).any().detach().cpu()):
        raise FeatureExtractionError("batched online feature contains an empty graph")

    def pool(values: Any) -> tuple[Any, Any]:
        if values.ndim != 2 or int(values.shape[0]) != int(node.shape[0]):
            raise FeatureExtractionError("atom-level batch pool shape mismatch")
        sums = values.new_zeros((graph_count, int(values.shape[1])))
        sums.index_add_(0, atom_graph_index, values)
        return sums, sums / counts.unsqueeze(1)

    blocks: list[Any] = []
    offset = 0
    for layer_index, (width, indices) in enumerate(
        zip(layer_widths, even_scalar_indices, strict=True)
    ):
        width = int(width)
        selected = tuple(int(index) for index in indices)
        if not selected or min(selected) < 0 or max(selected) >= width:
            raise FeatureExtractionError(
                f"invalid 0e channel indices for interaction {layer_index}"
            )
        layer = node[:, offset : offset + width]
        offset += width
        scalar = layer.index_select(
            1, torch.as_tensor(selected, dtype=torch.long, device=node.device)
        )
        scalar_sum, scalar_mean = pool(scalar)
        blocks.extend((scalar_sum, scalar_mean))

    density = outputs["density_coefficients"]
    spin_density = outputs["spin_density"]
    spin_channels = outputs["spin_charge_density"]
    if spin_channels.ndim != 3 or int(spin_channels.shape[1]) != 2:
        raise FeatureExtractionError(
            "spin_charge_density must have shape (n_atoms, 2, n_multipoles)"
        )
    if not (
        int(density.shape[0])
        == int(spin_density.shape[0])
        == int(spin_channels.shape[0])
        == int(node.shape[0])
    ):
        raise FeatureExtractionError("PolarMACE atom-level output counts disagree")

    def multipole_block(values: Any, *, name: str) -> Any:
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 4:
            raise FeatureExtractionError(
                f"{name} must have shape (n_atoms, >=4), observed {tuple(values.shape)}"
            )
        monopole = values[:, 0]
        l1_norm = torch.linalg.vector_norm(values[:, 1:4], dim=1)
        sums, means = pool(torch.stack((monopole, l1_norm), dim=1))
        return torch.stack(
            (sums[:, 0], means[:, 0], sums[:, 1], means[:, 1]), dim=1
        )

    for values, name in (
        (density, "density"),
        (spin_density, "spin_density"),
        (spin_channels[:, 0, :], "spin_alpha"),
        (spin_channels[:, 1, :], "spin_beta"),
    ):
        blocks.append(multipole_block(values, name=name))

    dipole = outputs["dipole"].reshape(-1, 3)
    energy = outputs["energy"].reshape(-1)
    electrostatic = outputs["electrostatic_energy"].reshape(-1)
    electron = outputs["electron_energy"].reshape(-1)
    if any(
        size != graph_count
        for size in (
            int(dipole.shape[0]),
            int(energy.numel()),
            int(electrostatic.numel()),
            int(electron.numel()),
        )
    ):
        raise FeatureExtractionError("graph-level PolarMACE batch output count mismatch")
    baseline_energy = (
        energy
        if immutable_base_energies_eV is None
        else energy.new_tensor(tuple(float(value) for value in immutable_base_energies_eV))
    )
    if int(baseline_energy.numel()) != graph_count:
        raise FeatureExtractionError("immutable baseline energy count mismatch")
    constants = torch.stack(
        (
            counts,
            energy.new_tensor(tuple(float(value) for value in formal_charges)),
            energy.new_tensor(tuple(float(value) for value in multiplicities)),
        ),
        dim=1,
    )
    blocks.append(
        torch.cat(
            (
                torch.linalg.vector_norm(dipole, dim=1).unsqueeze(1),
                baseline_energy.unsqueeze(1),
                electrostatic.unsqueeze(1),
                electron.unsqueeze(1),
                constants,
            ),
            dim=1,
        )
    )
    matrix = torch.cat(blocks, dim=1)
    if matrix.ndim != 2 or int(matrix.shape[0]) != graph_count:
        raise FeatureExtractionError("invalid online invariant feature matrix shape")
    if not bool(torch.isfinite(matrix).all().detach().cpu()):
        raise FeatureExtractionError("invalid online invariant feature matrix")
    return matrix


def checkpoint_irrep_plan(model: Any) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Read interaction irreps from the loaded checkpoint and select exact 0e slices."""

    products = tuple(getattr(model, "products", ()))
    if not products:
        raise FeatureExtractionError("checkpoint has no interaction products")
    widths: list[int] = []
    indices_by_layer: list[tuple[int, ...]] = []
    for layer_index, product in enumerate(products):
        linear = getattr(product, "linear", None)
        irreps = getattr(linear, "irreps_out", None)
        if irreps is None:
            raise FeatureExtractionError(
                f"checkpoint interaction {layer_index} lacks output irreps"
            )
        widths.append(int(irreps.dim))
        indices: list[int] = []
        for (_, irrep), component_slice in zip(irreps, irreps.slices(), strict=True):
            if int(irrep.l) == 0 and int(irrep.p) == 1:
                indices.extend(range(component_slice.start, component_slice.stop))
        if not indices:
            raise FeatureExtractionError(
                f"checkpoint interaction {layer_index} has no even scalar output"
            )
        indices_by_layer.append(tuple(indices))
    return tuple(widths), tuple(indices_by_layer)


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise FeatureExtractionError(f"required package is not installed: {names[0]}")


class PolarMACEBackend:
    """Exact v0.3.16 raw-forward backend for feature extraction."""

    def __init__(self, *, checkpoint: str, device: str = "cpu") -> None:
        try:
            import torch
            import graph_longrange
            import mace
            from mace import data as mace_data
            from mace.calculators import mace_polar
            from mace.calculators.foundations_models import (
                download_mace_polar_checkpoint,
            )
            from mace.tools import AtomicNumberTable, torch_geometric
        except ImportError as exc:
            raise FeatureExtractionError(
                "install the project ML dependencies before extracting MACE-POLAR features"
            ) from exc

        mace_version = _package_version("mace-torch")
        if mace_version != EXPECTED_MACE_VERSION:
            raise FeatureExtractionError(
                f"mace-torch version drift: expected {EXPECTED_MACE_VERSION}, observed {mace_version}"
            )
        graph_version = _package_version("graph_longrange")
        if graph_version != EXPECTED_GRAPH_ELECTROSTATICS_VERSION:
            raise FeatureExtractionError(
                "graph_electrostatics version drift: "
                f"expected {EXPECTED_GRAPH_ELECTROSTATICS_VERSION}, observed {graph_version}"
            )

        torch.set_default_dtype(torch.float64)
        checkpoint_path = Path(download_mace_polar_checkpoint(checkpoint)).resolve()
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise FeatureExtractionError(
                "polar-1-l checkpoint hash drift: "
                f"expected {EXPECTED_CHECKPOINT_SHA256}, observed {checkpoint_sha256}"
            )
        model = mace_polar(
            model=str(checkpoint_path),
            device=device,
            default_dtype="float64",
            return_raw_model=True,
        )
        model = model.to(device).double()
        model.eval()
        self._torch = torch
        self._mace_data = mace_data
        self._batch_type = torch_geometric.Batch
        self._device = device
        self.model = model
        self.z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.cutoff = float(model.r_max.detach().cpu())
        self.available_heads = list(getattr(model, "heads", ["Default"]))
        self.head = "Default" if "Default" in self.available_heads else self.available_heads[-1]
        self.layer_widths, self.even_scalar_indices = checkpoint_irrep_plan(model)
        self.provenance = CheckpointProvenance(
            checkpoint_name=checkpoint,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            mace_version=mace_version,
            mace_source_commit=EXPECTED_MACE_SOURCE_COMMIT,
            mace_package_sha256=sha256_file(Path(mace.__file__).resolve()),
            graph_electrostatics_version=graph_version,
            graph_electrostatics_commit=EXPECTED_GRAPH_ELECTROSTATICS_COMMIT,
            graph_electrostatics_package_sha256=sha256_file(
                Path(graph_longrange.__file__).resolve()
            ),
            default_dtype="float64",
        )

    def build_graph(
        self,
        *,
        xyz_path: Path,
        formal_charge: int,
        multiplicity: int,
    ) -> Any:
        try:
            from ase.io import read as read_atoms
        except ImportError as exc:
            raise FeatureExtractionError("ASE is required for feature extraction") from exc

        atoms = read_atoms(xyz_path, index=0)
        return self.build_graph_from_atoms(
            atoms=atoms, formal_charge=formal_charge, multiplicity=multiplicity
        )

    def build_graph_from_atoms(
        self,
        *,
        atoms: Any,
        formal_charge: int,
        multiplicity: int,
    ) -> Any:
        atoms.info["charge"] = int(formal_charge)
        atoms.info["spin"] = int(multiplicity)
        atoms.info["external_field"] = np.zeros(3, dtype=np.float64)
        keyspec = self._mace_data.KeySpecification(
            info_keys={
                "total_spin": "spin",
                "total_charge": "charge",
                "external_field": "external_field",
            },
            arrays_keys={},
        )
        config = self._mace_data.config_from_atoms(
            atoms,
            key_specification=keyspec,
            head_name=self.head,
        )
        graph = self._mace_data.AtomicData.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            heads=self.available_heads,
        )
        graph = graph.to(self._device)
        batch = self._batch_type.from_data_list([graph]).to(self._device)
        batch._jhtvs_singleton_data = graph  # pylint: disable=protected-access
        return batch

    def forward_graph(self, batch: Any, *, training: bool) -> Mapping[str, Any]:
        """Run the raw checkpoint without detaching its official outputs."""

        return self.model(
            batch.to_dict(),
            training=training,
            compute_force=False,
            compute_stress=False,
        )

    def batch_graphs(self, graphs: Sequence[Any]) -> Any:
        if not graphs:
            raise FeatureExtractionError("cannot batch an empty graph sequence")
        data = []
        for graph in graphs:
            item = getattr(graph, "_jhtvs_singleton_data", None)
            if item is None:
                raise FeatureExtractionError(
                    "cached online graph lacks its singleton AtomicData source"
                )
            data.append(item)
        return self._batch_type.from_data_list(data).to(self._device)

    def enable_lora(self, *, rank: int = 4, alpha: float = 1.0) -> None:
        """Inject the official MACE adapters and leave only adapter weights trainable."""

        from mace.modules.lora import inject_lora

        inject_lora(self.model, rank=rank, alpha=alpha)
        self.lora_inference_compatibility_layers = (
            patch_incompatible_mace_lora_inference(self.model)
        )
        self.model.train()

    def extract_tensor(
        self,
        *,
        batch: Any,
        formal_charge: int,
        multiplicity: int,
        training: bool,
        immutable_base_energy_eV: float,
    ) -> Any:
        outputs = self.forward_graph(batch, training=training)
        return build_torch_invariant_feature_vector(
            outputs,
            layer_widths=self.layer_widths,
            even_scalar_indices=self.even_scalar_indices,
            formal_charge=formal_charge,
            multiplicity=multiplicity,
            immutable_base_energy_eV=immutable_base_energy_eV,
        )

    def extract_tensor_batch(
        self,
        *,
        graphs: Sequence[Any],
        formal_charges: Sequence[int],
        multiplicities: Sequence[int],
        training: bool,
        immutable_base_energies_eV: Sequence[float],
    ) -> Any:
        batch = self.batch_graphs(graphs)
        outputs = self.forward_graph(batch, training=training)
        return build_torch_invariant_feature_matrix(
            outputs,
            atom_graph_index=batch.batch,
            graph_count=len(graphs),
            layer_widths=self.layer_widths,
            even_scalar_indices=self.even_scalar_indices,
            formal_charges=formal_charges,
            multiplicities=multiplicities,
            immutable_base_energies_eV=immutable_base_energies_eV,
        )

    def extract(
        self,
        *,
        xyz_path: Path,
        formal_charge: int,
        multiplicity: int,
    ) -> FeatureRecord:
        batch = self.build_graph(
            xyz_path=xyz_path,
            formal_charge=formal_charge,
            multiplicity=multiplicity,
        )
        with self._torch.no_grad():
            outputs = self.forward_graph(batch, training=False)
        detached = {
            name: outputs[name].detach().cpu().numpy()
            for name in REQUIRED_POLAR_OUTPUTS
            if outputs.get(name) is not None
        }
        return build_invariant_feature_record(
            detached,
            layer_widths=self.layer_widths,
            even_scalar_indices=self.even_scalar_indices,
            formal_charge=formal_charge,
            multiplicity=multiplicity,
        )


def serialize_output_shapes(shapes: Mapping[str, Iterable[int]]) -> str:
    return json.dumps(
        {key: list(value) for key, value in sorted(shapes.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


def assert_rotation_invariant(a: Sequence[float], b: Sequence[float], *, atol: float = 1e-10) -> None:
    if len(a) != len(b) or any(
        not math.isclose(float(x), float(y), rel_tol=0.0, abs_tol=atol)
        for x, y in zip(a, b, strict=True)
    ):
        raise FeatureExtractionError("feature vector is not rotation invariant")
