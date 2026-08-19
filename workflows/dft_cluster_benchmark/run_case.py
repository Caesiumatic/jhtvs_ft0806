#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    from .common import (
        BASIS,
        FUNCTIONAL_IMPLEMENTATION,
        LIBXC_CORRELATION,
        LIBXC_EXCHANGE,
        METHOD,
        SOFTWARE,
        read_csv,
        read_xyz,
        repo_root,
        sha256_file,
    )
    from .orca_input import bias_payload, build_input
except ImportError:
    from common import (
        BASIS,
        FUNCTIONAL_IMPLEMENTATION,
        LIBXC_CORRELATION,
        LIBXC_EXCHANGE,
        METHOD,
        SOFTWARE,
        read_csv,
        read_xyz,
        repo_root,
        sha256_file,
    )
    from orca_input import bias_payload, build_input

VERSION_RE = re.compile(r"Program Version\s+([0-9]+(?:\.[0-9]+){1,2})", re.I)


def _markers_complete(output: Path, optimized: bool) -> bool:
    try:
        text = output.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return (
        "ORCA TERMINATED NORMALLY" in text
        and "SCF CONVERGED AFTER" in text
        and "SCF NOT CONVERGED" not in text
        and (not optimized or "THE OPTIMIZATION HAS CONVERGED" in text)
    )


def _run_orca(orca: str, state_dir: Path) -> None:
    with (state_dir / "orca.out").open("w", encoding="utf-8") as output:
        completed = subprocess.run([orca, "orca.inp"], cwd=state_dir, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"ORCA exited with status {completed.returncode}")


def _software_version(outputs: list[Path]) -> str:
    for output in outputs:
        try:
            match = VERSION_RE.search(output.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if match:
            return match.group(1)
    return "unparsed"


def run_case(row: dict[str, str], orca: str, run_root: Path) -> dict:
    resolved_orca = shutil.which(orca)
    if not resolved_orca:
        raise RuntimeError(f"ORCA executable not found: {orca}")
    input_xyz = repo_root() / row["input_xyz"]
    atom_count = len(read_xyz(input_xyz)[0])
    task_dir = run_root / row["task_id"]
    opt_dir, reduced_dir, oxidized_dir = (task_dir / name for name in ("reduced_opt", "reduced_sp", "oxidized_sp"))
    for state_dir in (opt_dir, reduced_dir, oxidized_dir):
        state_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "task": row,
        "status": "running",
        "method": METHOD,
        "basis": BASIS,
        "software": SOFTWARE,
        "functional_implementation": FUNCTIONAL_IMPLEMENTATION,
        "libxc_exchange": LIBXC_EXCHANGE,
        "libxc_correlation": LIBXC_CORRELATION,
        "orca_executable": resolved_orca,
        "input_geometry_sha256": sha256_file(input_xyz),
        "atom_count": atom_count,
        "restraint_bias": bias_payload(row, input_xyz),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "executed_this_invocation": {"reduced_opt": False, "reduced_sp": False, "oxidized_sp": False},
    }
    try:
        opt_input_xyz = opt_dir / "in.xyz"
        if not _markers_complete(opt_dir / "orca.out", optimized=True) or not (opt_dir / "orca.xyz").exists():
            if (opt_dir / "orca.out").exists():
                raise RuntimeError("existing reduced optimization is incomplete; preserve it and use an isolated recovery run")
            shutil.copy2(input_xyz, opt_input_xyz)
            (opt_dir / "orca.inp").write_text(build_input(row, "reduced_opt", input_xyz), encoding="utf-8")
            provenance["executed_this_invocation"]["reduced_opt"] = True
            _run_orca(resolved_orca, opt_dir)
        optimized_xyz = opt_dir / "orca.xyz"
        if not _markers_complete(opt_dir / "orca.out", optimized=True) or len(read_xyz(optimized_xyz)[0]) != atom_count:
            raise RuntimeError("reduced optimization did not converge with a valid optimized XYZ")

        for state, state_dir in (("reduced_sp", reduced_dir), ("oxidized_sp", oxidized_dir)):
            state_xyz = state_dir / "in.xyz"
            if not _markers_complete(state_dir / "orca.out", optimized=False):
                if (state_dir / "orca.out").exists():
                    raise RuntimeError(f"existing {state} is incomplete; preserve it and use an isolated recovery run")
                shutil.copy2(optimized_xyz, state_xyz)
                (state_dir / "orca.inp").write_text(build_input(row, state, optimized_xyz), encoding="utf-8")
                provenance["executed_this_invocation"][state] = True
                _run_orca(resolved_orca, state_dir)
            if not _markers_complete(state_dir / "orca.out", optimized=False):
                raise RuntimeError(f"{state} did not converge")
        hashes = {
            "optimized_geometry_sha256": sha256_file(optimized_xyz),
            "reduced_sp_input_geometry_sha256": sha256_file(reduced_dir / "in.xyz"),
            "oxidized_sp_input_geometry_sha256": sha256_file(oxidized_dir / "in.xyz"),
        }
        provenance.update(hashes)
        provenance["same_sp_geometry"] = len(set(hashes.values())) == 1
        if not provenance["same_sp_geometry"]:
            raise RuntimeError("reduced and oxidized SP geometry payloads differ")
        provenance["software_version"] = _software_version([opt_dir / "orca.out", reduced_dir / "orca.out", oxidized_dir / "orca.out"])
        provenance["status"] = "complete"
    except Exception as exc:
        provenance.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data/dft_cluster_benchmark/calculation_manifest.csv")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task-index", type=int)
    selector.add_argument("--task-id")
    parser.add_argument("--orca", default="orca")
    parser.add_argument("--run-root", type=Path, default=root / "runs/dft_cluster_benchmark")
    args = parser.parse_args()
    rows = read_csv(args.manifest.resolve())
    selected = [row for row in rows if int(row["task_index"]) == args.task_index] if args.task_index else [row for row in rows if row["task_id"] == args.task_id]
    if len(selected) != 1:
        raise SystemExit(f"expected exactly one DFT case, found {len(selected)}")
    result = run_case(selected[0], args.orca, args.run_root.resolve())
    print(json.dumps({"task_id": selected[0]["task_id"], "status": result["status"], "software_version": result["software_version"]}))


if __name__ == "__main__":
    main()
