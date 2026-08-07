"""Command-line entry points for the jhtvs_ft0806 workflow."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from jhtvs_ft0806 import __version__
from jhtvs_ft0806.geometry.resolution import resolve_geometries
from jhtvs_ft0806.hpc.accounting import collect_accounting, submission_status
from jhtvs_ft0806.hpc.submission import (
    execute_submission,
    prepare_submission,
    selected_ids_from_file,
)
from jhtvs_ft0806.labels.assembly import (
    PINNED_REFERENCE_CONVERSION_RELATIVE_PATH,
    assemble_labels,
)
from jhtvs_ft0806.orca.decks import build_decks
from jhtvs_ft0806.orca.preflight import audit_decks
from jhtvs_ft0806.orca.parser import parse_results
from jhtvs_ft0806.spec_validation import validate_spec

COMMANDS = (
    "validate-spec",
    "resolve-geometries",
    "scan-reuse",
    "build-decks",
    "audit-decks",
    "submit",
    "status",
    "collect-accounting",
    "parse-results",
    "assemble-labels",
    "extract-base-features",
    "train",
    "evaluate",
    "infer-fullspace",
)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_validate_spec(args: argparse.Namespace) -> int:
    report = validate_spec(args.spec_dir)
    rendered = report.to_json() + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    logging.getLogger(__name__).info(
        "spec validation %s with %d issue(s)",
        "passed" if report.ok else "failed",
        len(report.issues),
    )
    return 0 if report.ok else 1


def _not_implemented(args: argparse.Namespace) -> int:
    logging.getLogger(__name__).error(
        "command %s is registered but not implemented in the current stage",
        args.command,
    )
    return 2


def _run_resolve_geometries(args: argparse.Namespace) -> int:
    summary = resolve_geometries(
        spec_dir=args.spec_dir,
        tier1_run=args.tier1_run,
        run_dir=args.run_dir,
        index_path=args.index,
        n_conformers=args.n_conformers,
    )
    sys.stdout.write(json.dumps(summary.to_dict(), sort_keys=True, indent=2) + "\n")
    logging.getLogger(__name__).info(
        "geometry resolution: %d resolved, %d pending, %d failed",
        summary.resolved,
        summary.pending,
        summary.failed,
    )
    if summary.failed or (args.require_complete and summary.pending):
        return 1
    return 0


def _run_build_decks(args: argparse.Namespace) -> int:
    selected = set(args.job_id) if args.job_id else None
    summary = build_decks(
        spec_dir=args.spec_dir,
        geometry_index_path=args.geometry_index,
        run_dir=args.run_dir,
        manifest_path=args.manifest,
        selected_job_ids=selected,
    )
    sys.stdout.write(json.dumps(summary.to_dict(), sort_keys=True, indent=2) + "\n")
    logging.getLogger(__name__).info(
        "deck generation: %d ready, %d waiting for geometry, %d existing outputs",
        summary.ready,
        summary.waiting_geometry,
        summary.existing_outputs,
    )
    if args.require_complete and summary.waiting_geometry:
        return 1
    return 0


def _run_audit_decks(args: argparse.Namespace) -> int:
    selected = set(args.job_id) if args.job_id else None
    report = audit_decks(
        spec_dir=args.spec_dir,
        geometry_index_path=args.geometry_index,
        deck_manifest_path=args.manifest,
        selected_job_ids=selected,
        report_path=args.report,
    )
    sys.stdout.write(report.to_json() + "\n")
    logging.getLogger(__name__).info(
        "ORCA deck preflight %s for %s selected jobs",
        "passed" if report.ok else "failed",
        report.checks["selected_jobs"],
    )
    return 0 if report.ok else 1


def _run_submit(args: argparse.Namespace) -> int:
    selected = set(args.job_id or ())
    if args.job_id_file is not None:
        selected.update(selected_ids_from_file(args.job_id_file))
    plan = prepare_submission(
        submission_id=args.submission_id,
        selected_job_ids=selected,
        spec_dir=args.spec_dir,
        geometry_index_path=args.geometry_index,
        deck_manifest_path=args.manifest,
        submissions_root=args.submissions_root,
        accounting_path=args.accounting,
        ledger_path=args.ledger,
        runner_path=args.runner,
        budget_scope=args.budget_scope,
        queue=args.queue,
        parallel_environment=args.parallel_environment,
        nprocs=args.nprocs,
        max_concurrent=args.max_concurrent,
    )
    result = (
        execute_submission(
            plan=plan,
            runner_path=args.runner,
            spec_dir=args.spec_dir,
            accounting_path=args.accounting,
            ledger_path=args.ledger,
            budget_scope=args.budget_scope,
        )
        if args.execute
        else plan.to_dict()
    )
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    logging.getLogger(__name__).info(
        "ORCA submission %s: %d jobs in %d array tasks",
        "executed" if args.execute else "prepared without qsub",
        plan.job_count,
        plan.array_task_count,
    )
    return 0


def _run_status(args: argparse.Namespace) -> int:
    report = submission_status(
        repository_root=_repository_root(), ledger_path=args.ledger
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return 0


def _run_collect_accounting(args: argparse.Namespace) -> int:
    summary = collect_accounting(
        submission_id=args.submission_id,
        repository_root=_repository_root(),
        ledger_path=args.ledger,
        accounting_path=args.accounting,
        qacct_file=args.qacct_file,
        allow_partial=args.allow_partial,
    )
    sys.stdout.write(
        json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )
    return 0 if summary.ledger_status == "complete" else 1


def _run_parse_results(args: argparse.Namespace) -> int:
    summary = parse_results(
        spec_dir=args.spec_dir,
        geometry_index_path=args.geometry_index,
        deck_manifest_path=args.manifest,
        output_path=args.output,
    )
    sys.stdout.write(
        json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )
    return 1 if summary.scientific_stops else 0


def _run_assemble_labels(args: argparse.Namespace) -> int:
    summary = assemble_labels(
        spec_dir=args.spec_dir,
        state_results_path=args.state_results,
        baseline_state_energies_path=args.baseline_state_energies,
        state_sp_output_path=args.state_sp_output,
        reaction_sp_output_path=args.reaction_sp_output,
        reaction_final_output_path=args.reaction_final_output,
        reference_conversion_path=args.reference_project
        / PINNED_REFERENCE_CONVERSION_RELATIVE_PATH,
    )
    sys.stdout.write(
        json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )
    return 1 if summary.scientific_stops else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jhtvs-ft0806")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-spec", help="validate supplied scientific tables and invariants")
    validate.add_argument("--spec-dir", type=Path, default=_repository_root() / "spec")
    validate.add_argument("--report", type=Path)
    validate.set_defaults(handler=_run_validate_spec)

    resolve = subparsers.add_parser(
        "resolve-geometries",
        help="materialize same-run Tier-1 geometries and prepare/collect sigma preoptimization",
    )
    resolve.add_argument("--spec-dir", type=Path, default=_repository_root() / "spec")
    resolve.add_argument("--tier1-run", type=Path, required=True)
    resolve.add_argument(
        "--run-dir", type=Path, default=_repository_root() / "runs" / "geometry"
    )
    resolve.add_argument(
        "--index",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "geometry_index.csv",
    )
    resolve.add_argument("--n-conformers", type=int, default=100)
    resolve.add_argument("--require-complete", action="store_true")
    resolve.set_defaults(handler=_run_resolve_geometries)

    build = subparsers.add_parser(
        "build-decks", help="generate manifest-bound ORCA SP and Opt/Freq decks"
    )
    build.add_argument("--spec-dir", type=Path, default=_repository_root() / "spec")
    build.add_argument(
        "--geometry-index",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "geometry_index.csv",
    )
    build.add_argument(
        "--run-dir", type=Path, default=_repository_root() / "runs" / "orca"
    )
    build.add_argument(
        "--manifest",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "deck_manifest.csv",
    )
    build.add_argument("--job-id", action="append")
    build.add_argument("--require-complete", action="store_true")
    build.set_defaults(handler=_run_build_decks)

    audit = subparsers.add_parser(
        "audit-decks", help="re-render and hash-audit submit-ready ORCA decks"
    )
    audit.add_argument("--spec-dir", type=Path, default=_repository_root() / "spec")
    audit.add_argument(
        "--geometry-index",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "geometry_index.csv",
    )
    audit.add_argument(
        "--manifest",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "deck_manifest.csv",
    )
    audit.add_argument(
        "--report",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "orca_preflight.json",
    )
    audit.add_argument("--job-id", action="append")
    audit.set_defaults(handler=_run_audit_decks)

    submit = subparsers.add_parser(
        "submit", help="prepare and optionally qsub a budget-guarded ORCA job wave"
    )
    submit.add_argument("--submission-id", required=True)
    submit.add_argument("--job-id", action="append")
    submit.add_argument("--job-id-file", type=Path)
    submit.add_argument("--spec-dir", type=Path, default=_repository_root() / "spec")
    submit.add_argument(
        "--geometry-index",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "geometry_index.csv",
    )
    submit.add_argument(
        "--manifest",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "deck_manifest.csv",
    )
    submit.add_argument(
        "--submissions-root",
        type=Path,
        default=_repository_root() / "runs" / "hpc" / "submissions",
    )
    submit.add_argument(
        "--accounting",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "accounting.csv",
    )
    submit.add_argument(
        "--ledger",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "submission_ledger.csv",
    )
    submit.add_argument(
        "--runner", type=Path, default=_repository_root() / "hpc" / "run_orca.sh"
    )
    submit.add_argument(
        "--budget-scope",
        choices=("first_round", "whole_project"),
        default="first_round",
    )
    submit.add_argument("--queue", default="amd16smt")
    submit.add_argument("--parallel-environment", default="orte")
    submit.add_argument("--nprocs", type=int, default=8)
    submit.add_argument("--max-concurrent", type=int, default=8)
    submit.add_argument(
        "--execute",
        action="store_true",
        help="invoke qsub; without this flag only immutable inputs are prepared",
    )
    submit.set_defaults(handler=_run_submit)

    status = subparsers.add_parser(
        "status", help="summarize immutable submission outputs and ledger state"
    )
    status.add_argument(
        "--ledger",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "submission_ledger.csv",
    )
    status.set_defaults(handler=_run_status)

    accounting = subparsers.add_parser(
        "collect-accounting", help="collect complete SGE qacct evidence and actual core-hours"
    )
    accounting.add_argument("--submission-id", required=True)
    accounting.add_argument(
        "--ledger",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "submission_ledger.csv",
    )
    accounting.add_argument(
        "--accounting",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "accounting.csv",
    )
    accounting.add_argument("--qacct-file", type=Path)
    accounting.add_argument(
        "--allow-partial",
        action="store_true",
        help="account canceled/failed arrays even when qacct covers only started tasks",
    )
    accounting.set_defaults(handler=_run_collect_accounting)

    parse = subparsers.add_parser(
        "parse-results", help="parse ORCA outputs and retain raw values with QC status"
    )
    parse.add_argument("--spec-dir", type=Path, default=_repository_root() / "spec")
    parse.add_argument(
        "--geometry-index",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "geometry_index.csv",
    )
    parse.add_argument(
        "--manifest",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "deck_manifest.csv",
    )
    parse.add_argument(
        "--output",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "state_results.csv",
    )
    parse.set_defaults(handler=_run_parse_results)

    labels = subparsers.add_parser(
        "assemble-labels",
        help="assemble state, SP-reaction, and final-reaction labels",
    )
    labels.add_argument("--spec-dir", type=Path, default=_repository_root() / "spec")
    labels.add_argument(
        "--state-results",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "state_results.csv",
    )
    labels.add_argument(
        "--baseline-state-energies",
        type=Path,
        default=_repository_root()
        / "data"
        / "resolved"
        / "base_state_energies.csv",
    )
    labels.add_argument(
        "--state-sp-output",
        type=Path,
        default=_repository_root() / "data" / "resolved" / "state_sp_labels.csv",
    )
    labels.add_argument(
        "--reaction-sp-output",
        type=Path,
        default=_repository_root()
        / "data"
        / "resolved"
        / "reaction_sp_labels.csv",
    )
    labels.add_argument(
        "--reaction-final-output",
        type=Path,
        default=_repository_root()
        / "data"
        / "resolved"
        / "reaction_final_labels.csv",
    )
    labels.add_argument(
        "--reference-project",
        type=Path,
        default=_repository_root().parent / "20260707",
        help="path to the supplied project containing the pinned Ag/AgCl conversion",
    )
    labels.set_defaults(handler=_run_assemble_labels)

    for command in COMMANDS[2:]:
        if command in {
            "build-decks",
            "audit-decks",
            "submit",
            "status",
            "collect-accounting",
            "parse-results",
            "assemble-labels",
        }:
            continue
        subparser = subparsers.add_parser(command)
        subparser.set_defaults(handler=_not_implemented)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
