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
from jhtvs_ft0806.spec_validation import validate_spec

COMMANDS = (
    "validate-spec",
    "resolve-geometries",
    "scan-reuse",
    "build-decks",
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

    for command in COMMANDS[2:]:
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
